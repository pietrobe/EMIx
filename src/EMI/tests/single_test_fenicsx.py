import ufl
import time
import numpy.typing
import multiphenicsx.fem
import multiphenicsx.fem.petsc

import numpy    as np
import scipy    as sp
import dolfinx  as dfx

from ufl      import inner, grad
from sys      import argv
from mpi4py   import MPI
from pathlib  import Path
from petsc4py import PETSc
from dfx_mesh_creation import create_circle_mesh_gmsh, create_square_mesh_with_tags

start_time = time.perf_counter()

# Subdomain tags
INTRA = 1
EXTRA = 2

# Boundary tags
PARTIAL_OMEGA = 3
GAMMA         = 4
DEFAULT       = 5

# Options for the fenicsx form compiler optimization
cache_dir       = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args"  : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

def dump(thing, path):
    if isinstance(thing, PETSc.Vec):
        assert np.all(np.isfinite(thing.array))
        return np.save(path, thing.array)
    m = sp.sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
    assert np.all(np.isfinite(m.data))
    return np.save(path, np.c_[m.row, m.col, m.data])

def mpi_print(comm, stuff):
    print(f"MPI Rank: {comm.rank} {stuff}")

def calc_error_L2(u_h: dfx.fem.Function, u_exact: dfx.fem.Function, subdomain_id: int, degree_raise: int = 3) -> float:
        """ Calculate the L2 error for a solution approximated with finite elements.

        Parameters
        ----------
        u_h : dolfinx.fem Function
            The solution function approximated with finite elements.

        u_exact : dolfinx.fem Function
            The exact solution function.

        degree_raise : int, optional
            The amount of polynomial degrees that the approximated solution
            is refined, by default 3

        Returns
        -------
        error_global : float
            The L2 error norm.
        """
        # Create higher-order function space for solution refinement
        degree = u_h.function_space.ufl_element().degree()
        family = u_h.function_space.ufl_element().family()
        mesh   = u_h.function_space.mesh

        if u_h.function_space.element.signature().startswith('Vector'):
            # Create higher-order function space based on vector elements
            W = dfx.fem.FunctionSpace(mesh, ufl.VectorElement(family = family, 
                                      degree = (degree + degree_raise), cell = mesh.ufl_cell()))
        else:
            # Create higher-order funciton space based on finite elements
            W = dfx.fem.FunctionSpace(mesh, (family, degree + degree_raise))

        # Interpolate the approximate solution into the refined space
        u_W = dfx.fem.Function(W)
        u_W.interpolate(u_h)

        # Interpolate exact solution, special handling if exact solution
        # is a ufl expression
        u_exact_W = dfx.fem.Function(W)

        if isinstance(u_exact, ufl.core.expr.Expr):
            u_expr = dfx.fem.Expression(u_exact, W.element.interpolation_points())
            u_exact_W.interpolate(u_expr)
        else:
            u_exact_W.interpolate(u_exact)
        
        # Compute the error in the higher-order function space
        e_W = dfx.fem.Function(W)
        with e_W.vector.localForm() as e_W_loc, u_W.vector.localForm() as u_W_loc, u_exact_W.vector.localForm() as u_ex_W_loc:
            e_W_loc[:] = u_W_loc[:] - u_ex_W_loc[:]

        # Integrate the error
        error        = dfx.fem.form(inner(e_W, e_W) * dx(subdomain_id))
        error_local  = dfx.fem.assemble_scalar(error)

        return error_local


#----------------------------------------#
#     PARAMETERS AND SOLVER SETTINGS     #
#----------------------------------------#
# Space discretization parameters
N = int(argv[1])
P = 1

# Time discretization parameters
t          = 0.0
T          = 0.1
deltaT     = 0.1
time_steps = int(T / deltaT)

# Physical parameters
capacitance_membrane       = 1.0
conductivity_intracellular = 1.0
conductivity_extracellular = 1.0

# Flags
mesh_type      = 'square' # 'square' or 'circle'
write_mesh     = False
save_output    = False    
save_matrix    = False
direct_solver  = False
ksp_type       = 'cg'
pc_type        = 'hypre'
ds_solver_type = 'mumps'
ksp_rtol       = 1e-6

# Timers
solve_time    = 0
assemble_time = 0

#------------------------------------#
#        FUNCTION EXPRESSIONS        #
#------------------------------------#
# Membrane potential expression
class InitialMembranePotential:
    def __init__(self): pass

    def __call__(self, x: numpy.typing.NDArray) -> numpy.typing.NDArray:
        return np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

# Forcing factor expressions
class IntracellularSource:
    def __init__(self, t_0: float):
        self.t = t_0 # Initial time

    def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
        return 8 * np.pi ** 2 * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]) * (1.0 + np.exp(-self.t))

class ExtracellularSource:
    def __init__(self): pass

    def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
        return 8 * np.pi ** 2 * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

# Exact solution expressions
class uiExact:
    def __init__(self, t_0: float):
        self.t = t_0

    def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
        return np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]) * (1.0 + np.exp(-self.t))

class ueExact:
    def __init__(self): pass

    def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
        return np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

#-----------------------#
#          MESH         #
#-----------------------#
comm       = MPI.COMM_WORLD # MPI communicator
ghost_mode = dfx.mesh.GhostMode.shared_facet # How dofs are distributed in parallel

# Create mesh
assert (mesh_type == 'square' or mesh_type == 'circle')
if mesh_type == 'square':
    t1 = time.perf_counter()
    mesh, subdomains, boundaries = create_square_mesh_with_tags(N_cells = N, comm = comm, ghost_mode = ghost_mode)
    print(f"Create mesh time: {time.perf_counter() - t1:.2f}")
elif mesh_type == 'circle':
    raise NotImplementedError('Circle mesh implementation not completed.')
    partitioner = dfx.mesh.create_cell_partitioner(ghost_mode)
    mesh, subdomains, boundaries = create_circle_mesh_gmsh(comm = comm, partitioner = partitioner)

# Define integral measures
dx = ufl.Measure("dx", subdomain_data = subdomains) # Cell integrals
dS = ufl.Measure("dS", subdomain_data = boundaries) # Facet integrals
dS = dS(GAMMA) # Restrict facet integrals to gamma interface

# Mesh constants
dt      = dfx.fem.Constant(mesh, PETSc.ScalarType(deltaT))
C_M     = dfx.fem.Constant(mesh, PETSc.ScalarType(capacitance_membrane))
sigma_i = dfx.fem.Constant(mesh, PETSc.ScalarType(conductivity_intracellular))
sigma_e = dfx.fem.Constant(mesh, PETSc.ScalarType(conductivity_extracellular))

#------------------------------------------#
#     FUNCTION SPACES AND RESTRICTIONS     #
#------------------------------------------#
V  = dfx.fem.FunctionSpace(mesh, ("Lagrange", P)) # Space for functions defined on the entire mesh
V1 = dfx.fem.FunctionSpace(mesh, ("Lagrange", P)) # Intracellular space
V2 = dfx.fem.FunctionSpace(mesh, ("Lagrange", P)) # Extracellular space

print(f"MPI Rank {comm.rank}")
print(f"Size of local index map V1: {V1.dofmap.index_map.size_local}")
print(f"Size of global index map V1: {V1.dofmap.index_map.size_global}")
print(f"Number of ghost nodes V1: {V1.dofmap.index_map.num_ghosts}")
# print(f"Size of local index map V2: {V2.dofmap.index_map.size_local}")
# print(f"Size of global index map V2: {V2.dofmap.index_map.size_global}")
# print(f"Number of ghost nodes V2: {V2.dofmap.index_map.num_ghosts}")


# Trial and test functions
(ui, ue) = (ufl.TrialFunction(V1), ufl.TrialFunction(V2))
(vi, ve) = (ufl.TestFunction (V1), ufl.TestFunction (V2))

# Functions for storing the solutions and the exact solutions
ui_h, ui_ex = dfx.fem.Function(V1), dfx.fem.Function(V1)
ue_h, ue_ex = dfx.fem.Function(V2), dfx.fem.Function(V2)

ui_ex_expr = uiExact(t_0 = t)
ui_ex.interpolate(ui_ex_expr)

ue_ex_expr = ueExact()
ue_ex.interpolate(ue_ex_expr)

# Membrane potential function
v_expr = InitialMembranePotential()
v = dfx.fem.Function(V)
v.interpolate(v_expr)

# Membrane forcing term function
fg = dfx.fem.Function(V)

# Forcing term in the intracellular space
fi_expr = IntracellularSource(t_0 = t)
fi = dfx.fem.Function(V)
fi.interpolate(fi_expr)

# Forcing term in the extracellular space
fe_expr = ExtracellularSource()
fe = dfx.fem.Function(V)
fe.interpolate(fe_expr)

##### Restrictions #####
# Get indices of the cells of the intra- and extracellular subdomains
cells_Omega1 = subdomains.indices[subdomains.values == INTRA]
cells_Omega2 = subdomains.indices[subdomains.values == EXTRA]

# Get dofs of the intra- and extracellular subdomains
dofs_V1_Omega1 = dfx.fem.locate_dofs_topological(V1, subdomains.dim, cells_Omega1)
dofs_V2_Omega2 = dfx.fem.locate_dofs_topological(V2, subdomains.dim, cells_Omega2)

# Define the restrictions of the subdomains
restriction_V1_Omega1 = multiphenicsx.fem.DofMapRestriction(V1.dofmap, dofs_V1_Omega1)
restriction_V2_Omega2 = multiphenicsx.fem.DofMapRestriction(V2.dofmap, dofs_V2_Omega2)

restriction = [restriction_V1_Omega1, restriction_V2_Omega2]

print(f"Time 1: {time.perf_counter() - start_time:.2f}")

t_second = time.perf_counter()
#------------------------------------#
#        VARIATIONAL PROBLEM         #
#------------------------------------#
# First row of block bilinear form
a11 = dt * inner(sigma_i * grad(ui), grad(vi)) * dx(INTRA) + C_M * inner(ui('-'), vi('-')) * dS # ui terms
a12 = - C_M * inner(ue('+'), vi('-')) * dS # ue terms

# Second row of block bilinear form
a21 = - C_M * inner(ui('-'), ve('+')) * dS # ui terms
a22 = dt * inner(sigma_e * grad(ue), grad(ve)) * dx(EXTRA) + C_M * inner(ue('+'), ve('+')) * dS # ue terms

# Define boundary conditions
zero = dfx.fem.Constant(mesh, PETSc.ScalarType(0.0)) # Grounded exterior boundary BC

facets_partial_Omega = boundaries.indices[boundaries.values == PARTIAL_OMEGA] # Get indices of the facets on the exterior boundary
dofs_V2_partial_Omega = dfx.fem.locate_dofs_topological(V2, boundaries.dim, facets_partial_Omega) # Get the dofs on the exterior boundary facets
bce = dfx.fem.dirichletbc(zero, dofs_V2_partial_Omega, V2) # Set Dirichlet BC on exterior boundary facets

bcs = [bce]

# Assemble block form
a = [[a11, a12],
     [a21, a22]]

# Convert form to dolfinx form
a = dfx.fem.form(a, jit_options = jit_parameters)


#---------------------------#
#      MATRIX ASSEMBLY      #
#---------------------------#
t1 = time.perf_counter() # Timestamp for assembly time-lapse

# Assemble the block linear system matrix
A = multiphenicsx.fem.petsc.assemble_matrix_block(a, bcs = bcs, restriction = (restriction, restriction))
A.assemble()

assemble_time += time.perf_counter() - t1 # Add time lapsed to total assembly time

if save_matrix:
    # Write system matrix A to file
    print("Saving A in npy format...")   
    dump(A, 'output/Amat')
    # use then in /output with MATLAB:
    # addpath('some_path/scripts')
    # addpath('some_path/scripts/npy-matlab-master/npy-matlab/')
    # data = readNPY('Amat.npy'); A = create_sparse_mat_from_data(data);
    exit()

# Configure Krylov solver
ksp = PETSc.KSP()
ksp.create(comm)
ksp.setOperators(A)
ksp.setTolerances(rtol = ksp_rtol)

# Set options based on direct/iterative solution method
if direct_solver:
    if comm.rank == 0:
        print("Using a direct solver ...")
    #ksp.setType("preonly")
    #ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType(ds_solver_type)
else:
    if comm.rank == 0:
        print("Using an iterative solver ...")
    opt = PETSc.Options()
    opt['ksp_converged_reason'] = None
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)

ksp.setFromOptions()


# Create output files
if save_output:
    out_ui = dfx.io.XDMFFile(mesh.comm, "ui_mphx_square.xdmf", "w")
    out_ue = dfx.io.XDMFFile(mesh.comm, "ue_mphx_square.xdmf", "w")
    out_v  = dfx.io.XDMFFile(mesh.comm, "v_mphx_square.xdmf" , "w")

    out_ui.write_mesh(mesh)
    out_ue.write_mesh(mesh)
    out_v.write_mesh(mesh)


print(f"Time 2: {time.perf_counter() - t_second:.2f}")

#---------------------------------#
#        SOLUTION TIMELOOP        #
#---------------------------------#
for i in range(time_steps):

    # Increment time
    t += deltaT

    # Update time-dependent expressions
    fi_expr.t = t
    fi.interpolate(fi_expr)

    # Update and assemble vector that is the RHS of the linear system
    t1 = time.perf_counter() # Timestamp for assembly time-lapse

    # Forcing term on membrane
    # Get local + ghost values of vectors using .localForm() (for parallel)
    with fg.vector.localForm() as fg_local, v.vector.localForm() as v_local:
        fg_local[:] = v_local[:] - (deltaT / capacitance_membrane) * v_local[:]
    
    Li = dt * inner(fi, vi) * dx(INTRA) + C_M * inner(fg, vi('-')) * dS # Linear form intracellular space
    Le = dt * inner(fe, ve) * dx(EXTRA) - C_M * inner(fg, ve('+')) * dS # Linear form extracellular space

    L = [Li, Le] # Block linear form
    L = dfx.fem.form(L, jit_options = jit_parameters) # Convert form to dolfinx form

    b = multiphenicsx.fem.petsc.assemble_vector_block(L, a, bcs = bcs, restriction = restriction) # Assemble RHS vector
    
    assemble_time += time.perf_counter() - t1 # Add time lapsed to total assembly time
    
    if i == 0:
        # Create solution vector
        sol_vec = multiphenicsx.fem.petsc.create_vector_block(L, restriction = restriction)
    
    # Solve the system
    t1 = time.perf_counter() # Timestamp for solver time-lapse
    ksp.solve(b, sol_vec)

    # Update ghost values
    sol_vec.ghostUpdate(addv = PETSc.InsertMode.INSERT, mode = PETSc.ScatterMode.FORWARD)

    solve_time += time.perf_counter() - t1 # Add time lapsed to total solver time

    # Extract sub-components of solution
    with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(sol_vec, [V1.dofmap, V2.dofmap], restriction) as ui_ue_wrapper:
        for ui_ue_wrapper_local, component in zip(ui_ue_wrapper, (ui_h, ue_h)): 
            with component.vector.localForm() as component_local:
                component_local[:] = ui_ue_wrapper_local

    # Update membrane potential
    # Get local + ghost values using .localForm() (in case code is run in parallel)
    with v.vector.localForm() as v_local, ui_h.vector.localForm() as ui_h_local, ue_h.vector.localForm() as ue_h_local:
        v_local[:]  = ui_h_local[:] - ue_h_local[:]
    
    if save_output:
        out_ui.write_function(ui_h, t)
        out_ue.write_function(ue_h, t)
        out_v.write_function(v, t)

# Update time of exact ui expression
ui_ex_expr.t = t
ui_ex.interpolate(ui_ex_expr)

#------------------------------#
#         POST PROCESS         #
#------------------------------#
# Error analysis
L2_error_i_local = calc_error_L2(u_h = ui_h, u_exact = ui_ex, subdomain_id = INTRA) # Local L2 error (squared) of intracellular electric potential
L2_error_e_local = calc_error_L2(u_h = ue_h, u_exact = ue_ex, subdomain_id = EXTRA) # Local L2 error (squared) of extracellular electric potential

L2_error_i_global = np.sqrt(comm.allreduce(L2_error_i_local, op = MPI.SUM)) # Global L2 error of intracellular electric potential
L2_error_e_global = np.sqrt(comm.allreduce(L2_error_e_local, op = MPI.SUM)) # Global L2 error of extracellular electric potential

# Sum local assembly and solve times to get global values
max_local_assemble_time = comm.allreduce(assemble_time, op = MPI.MAX) # Global assembly time
max_local_solve_time    = comm.allreduce(solve_time   , op = MPI.MAX) # Global solve time


# mpi_print(comm, f"Local assembly time: {assemble_time}")
# mpi_print(comm, f"Local solve time: {solve_time}")


print(comm.gather(assemble_time, root = 0))
print(comm.gather(solve_time, root = 0))

if comm.rank == 0:
    # Print stuff
    print("\n#-----------INFO-----------#\n")
    print("MPI size = ", comm.size)
    print("dt = ", deltaT)
    print("N = ", N)
    print("P = ", P)

    print("\n#----------ERRORS----------#\n")
    print(f"L2 error norm intracellular: {L2_error_i_global:.2e}")
    print(f"L2 error norm extracellular: {L2_error_e_global:.2e}")

    print("\n#-------TIME ELAPSED-------#\n")
    print(f"Max assembly time: {max_local_assemble_time:.3f} seconds\n")
    print(f"Max solve time: {max_local_solve_time:.3f} seconds\n")

    print("#--------------------------#")

    print(f"Script time elapsed: {time.perf_counter() - start_time:.3f} seconds")

# Write solutions to file
if save_output:
    out_ui.close()
    out_ue.close()

if write_mesh:
    with dfx.io.XDMFFile(mesh.comm, "dfx_square_mesh.xdmf", "w") as mesh_file:
        mesh_file.write_mesh(mesh)
        mesh_file.write_meshtags(boundaries)
        #mesh_file.write_meshtags(subdomains)