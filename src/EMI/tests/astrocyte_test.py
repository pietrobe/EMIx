import time
import sys

import numpy             as np 
import scipy.sparse      as sparse
import matplotlib.pyplot as plt

from dolfin       import *
from multiphenics import *
from petsc4py     import PETSc
from sys          import argv

parameters["ghost_mode"] = "shared_facet" # required by dS

# flags
save_output   = True
direct_solver = False
ksp_type      = 'cg'
pc_type       = 'hypre'
ksp_rtol      = 1e-6
dt = float(argv[1])
P = 1
max_amg_iter = 1

# MESHES #
# Mesh
mesh = Mesh(MPI.comm_world)

if MPI.comm_world.rank == 0: print('Reading mesh...')
mesh = Mesh("data/astrocyte_mesh.xml")
subdomains = MeshFunction("size_t", mesh, "data/astrocyte_mesh_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/astrocyte_mesh_facet_region.xml")

m_conversion_factor = 0.01
mesh.coordinates()[:] *= m_conversion_factor

# print MPI size
if MPI.comm_world.rank == 0:
    print('MPI size =', MPI.size(MPI.comm_world))
    print('#Mesh cells =', mesh.num_cells())
    print('Loading sub meshes and restrictions...') 

# Restrictions
interior  = MeshRestriction(mesh, "data/astrocyte_mesh_interior_restriction.rtc.xml")
exterior  = MeshRestriction(mesh, "data/astrocyte_mesh_exterior_restriction.rtc.xml")
interface = MeshRestriction(mesh, "data/astrocyte_mesh_interface_restriction.rtc.xml")

# forcing term on interface
f_gamma = Expression("sin(2*pi*x[0]) * sin(2*pi*x[1])", degree=4)

if MPI.comm_world.rank == 0: print('Creating FEM objects...') 

# FUNCTION SPACES #
# Function spaces
V = FunctionSpace(mesh, "Lagrange", P)

# Block function space
W = BlockFunctionSpace([V, V], restrict=[interior, exterior])

# TRIAL/TEST FUNCTIONS #
uu = BlockTrialFunction(W)
vv = BlockTestFunction(W)

(ui, ue) = block_split(uu)
(vi, ve) = block_split(vv)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)
dS = Measure("dS")(subdomain_data=boundaries)
dS = dS(5) # restrict to the interface, which has facet ID equal to 5


if MPI.comm_world.rank == 0: print('Assembling forms...') 
# ASSEMBLE #
# Interior mesh has subdomain id = 3
# Exterior mesh has subdomain id = 1
a11 = dt * inner(grad(ui), grad(vi))*dx(3) + inner(ui('-'), vi('-'))*dS
a22 = dt * inner(grad(ue), grad(ve))*dx(1) + inner(ue('+'), ve('+'))*dS
a12 = - inner(ue('-'), vi('+'))*dS
a21 = - inner(ui('+'), ve('-'))*dS

a = [[a11, a12],
    [a21, a22]]

f1 = -inner(f_gamma, vi('-'))*dS
f2 =  inner(f_gamma, ve('+'))*dS

f =  [f1, f2]

if MPI.comm_world.rank == 0: print('Setting boundary conditions...') 
# Outer shell of exterior mesh has facet tag 4
bce = DirichletBC(W.sub(1), Constant(0.), boundaries, 4)

bcs = BlockDirichletBC([None, bce])

if MPI.comm_world.rank == 0: print('Assembling linear system...') 
t1 = time.perf_counter() 
A = block_assemble(a)
F = block_assemble(f)
bcs.apply(A)
bcs.apply(F)
assemble_time = time.perf_counter() - t1

#from IPython import embed;embed()
U = BlockFunction(W)

if MPI.comm_world.rank == 0: print('Solving the system...')

if not direct_solver:

    X  = U.block_vector()
    A_ = as_backend_type(A).mat()
    F_ = as_backend_type(F).vec()
    x  = as_backend_type(X).vec()
        
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_, A_)
    ksp.setType(ksp_type)

    opts = PETSc.Options()
    opts.setValue('ksp_view', None)
    opts.setValue('ksp_monitor_true_residual', None)
    opts.setValue('ksp_rtol', ksp_rtol)
    opts.setValue('ksp_converged_reason', None)
    opts.setValue('pc_type', pc_type)
    opts.setValue("ksp_norm_type", 'unpreconditioned')

    opts.setValue("pc_hypre_boomeramg_max_iter", max_amg_iter)  
    
    # opts.setValue("pc_hypre_boomeramg_nodal_coarsen", 2)  
    opts.setValue("pc_hypre_boomeramg_strong_threshold", 0.9)    

    # PETSc.Options().setValue("pc_hypre_boomeramg_interp_type", "standard")
    # PETSc.Options().setValue("pc_hypre_boomeramg_coarsen_type", "Ruge-Stueben")
    # opts.setValue("-pc_hypre_boomeramg_relax_weight_all", 0.8)
    # opts.setValue("-pc_hypre_boomeramg_outer_relax_weight_all", 0.5)

    ksp.setFromOptions()

    t1 = time.perf_counter()

    ksp.solve(F_, x)

    U.block_vector().apply("")
    U.apply("to subfunctions")            
else:

    t1 = time.perf_counter()
    block_solve(A, U.block_vector(), F, linear_solver = 'mumps')

solve_time = time.perf_counter() - t1

if save_output:
    if MPI.comm_world.rank == 0: print('Writing output...')
    U[0].rename('u_i', '')
    U[1].rename('u_e', '')
    out_i = XDMFFile(MPI.comm_world, "output/astrocyte_sol_i.xdmf")
    out_e = XDMFFile(MPI.comm_world, "output/astrocyte_sol_e.xdmf")  
    out_i.write(U[0])
    out_e.write(U[1])
    
    if MPI.comm_world.size == 1:
        out_i.write(subdomains)
        out_e.write(subdomains)

if MPI.comm_world.rank == 0:
    print("~~~~~~~~~~~~~~ Info ~~~~~~~~~~~~~~")
    print("dt =", dt)
    # print("N =", N)
    print("P =", P)
    if direct_solver:
        print("Solver: direct")
    else:
        print('Solver: [', ksp_type,'+', pc_type, ']')
    
    print("~~~~~~~~~~~~~~ Timing ~~~~~~~~~~~~~~")
    print('Assemble:', assemble_time ,   's')
    print('Solve:   ', solve_time, 's')



