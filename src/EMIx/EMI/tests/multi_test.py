from numpy import isclose
from dolfin import *
from multiphenics import *
import time
from petsc4py import PETSc
import scipy.sparse as sparse
import numpy as np 
from sys import argv

parameters["ghost_mode"] = "shared_facet" # required by dS
# parameters['form_compiler']['optimize']           = True
# parameters['form_compiler']['cpp_optimize']       = True
# parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

def dump(thing, path):
            if isinstance(thing, PETSc.Vec):
                assert np.all(np.isfinite(thing.array))
                return np.save(path, thing.array)
            m = sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
            assert np.all(np.isfinite(m.data))
            return np.save(path, np.c_[m.row, m.col, m.data])


# space discretization parameters
N = int(argv[1])
P = 1

# time discretization parameters
t          = 0.0
T          = 0.1
dt         = 0.1 #0.01/N
time_steps = 1   #int(T / dt)

# physical parameters
C_M     = 1.0
sigma_i = 1.0
sigma_e = 1.0

# flags
save_output   = False    
save_matrix   = False
direct_solver = False
ksp_type      = 'gmres'
pc_type       = 'ilu'
ksp_rtol      = 1e-6

# timers 
solve_time    = 0
assemble_time = 0

# initial membrane potential V
v = Expression("sin(2*pi*x[0]) * sin(2*pi*x[1])", degree = 4)

# forcing factors
source_i = Expression("8*pi*pi*sin(2*pi*x[0]) * sin(2*pi*x[1]) * (1.0 + exp(-t))", degree = 4, t = t)
source_e = Expression("8*pi*pi*sin(2*pi*x[0]) * sin(2*pi*x[1])",                   degree = 4)

if save_output:
    # output files
    out_i = XDMFFile(MPI.comm_world, "output/sol_i.xdmf")
    out_e = XDMFFile(MPI.comm_world, "output/sol_e.xdmf") 
    out_g = XDMFFile(MPI.comm_world, "output/sol_g.xdmf") 

    out_i.parameters['functions_share_mesh' ] = True
    out_i.parameters['rewrite_function_mesh'] = False
    out_e.parameters['functions_share_mesh' ] = True
    out_e.parameters['rewrite_function_mesh'] = False
    out_g.parameters['functions_share_mesh' ] = True
    out_g.parameters['rewrite_function_mesh'] = False

# MESHES #
# Mesh
mesh = Mesh("data/square" + str(N) + ".xml")
subdomains = MeshFunction("size_t", mesh, "data/square_physical_region" + str(N) + ".xml")
boundaries = MeshFunction("size_t", mesh, "data/square_facet_region"    + str(N) + ".xml")
# Restrictions
omega_i = MeshRestriction(mesh, "data/square_restriction_om_i"      + str(N) + ".rtc.xml")
omega_e = MeshRestriction(mesh, "data/square_restriction_om_e"      + str(N) + ".rtc.xml")
gamma   = MeshRestriction(mesh, "data/square_restriction_interface" + str(N) + ".rtc.xml")

# # Mesh
# mesh = Mesh("data/circle.xml")
# subdomains = MeshFunction("size_t", mesh, "data/circle_physical_region.xml")
# boundaries = MeshFunction("size_t", mesh, "data/circle_facet_region.xml")
# # Restrictions
# left = MeshRestriction(mesh, "data/circle_restriction_left.rtc.xml")
# right = MeshRestriction(mesh, "data/circle_restriction_right.rtc.xml")
# interface = MeshRestriction(mesh, "data/circle_restriction_interface.rtc.xml")

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, "Lagrange", P)
# Block function space
W  = BlockFunctionSpace([V, V, V], restrict=[omega_i, omega_e, gamma])
# W  = BlockFunctionSpace([V, V, V], restrict=[left, right, interface])

# TRIAL/TEST FUNCTIONS #
uu = BlockTrialFunction(W)
vv = BlockTestFunction(W)

(ui, ue, Im) = block_split(uu)
(vi, ve, vg) = block_split(vv)

# MEASURES #
dx  = Measure("dx")(subdomain_data=subdomains)
dS  = Measure("dS")(subdomain_data=boundaries)
dxS = dS(2) # restrict to the interface, which has facet ID equal to 2

# weak form of equation for Omega_i
a00 = inner(sigma_i*grad(ui), grad(vi))*dx(1)
a02 = inner(Im('-'), vi('-'))*dxS

# weak form of equation for Omega_e
a11 =   inner(sigma_e*grad(ue), grad(ve))*dx(2)
a12 = - inner(Im('+'), ve('+'))*dxS

a20 =   inner(ui('-'), vg('-'))*dxS
a21 = - inner(ue('+'), vg('+'))*dxS
#a22 = - (dt/C_M) * inner(Im, vg)*dxg   
a22 = - (dt/C_M) * inner(Im('-'), vg('-'))*dxS

a = [[a00,  0 , a02],
     [  0, a11, a12],
     [a20, a21, a22]]

bc_e = DirichletBC(W.sub(1), Constant(0.0), boundaries, 1)
bcs  = BlockDirichletBC([None, bc_e, None])

# ASSEMBLE #
t1 = time.perf_counter() 
A = block_assemble(a)
bcs.apply(A)
assemble_time += time.perf_counter() - t1

if save_matrix:
    # Write A
    print("Saving A in npy format...")   
    dump(A.mat(),'output/Amat')
    # use then in /output with MATLAB:
    # addpath('some_path/scripts')
    # addpath('some_path/scripts/npy-matlab-master/npy-matlab/')
    # data = readNPY('Amat.npy'); A = create_sparse_mat_from_data(data);
    exit()

U = BlockFunction(W)

# save output
U[0].rename('u_i', '')
U[1].rename('u_e', '')
U[2].rename('I_m', '')

if not direct_solver:

    A_ = as_backend_type(A).mat()
        
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_, A_)
    ksp.setType(ksp_type)

    opts = PETSc.Options()
    opts.setValue('ksp_view', None)
    opts.setValue('ksp_monitor_true_residual', None)
    opts.setValue('ksp_rtol', ksp_rtol)
    opts.setValue('ksp_converged_reason', None)
   # opts.setValue('pc_type', pc_type)
    ksp.setFromOptions()

    X = U.block_vector()
    x = as_backend_type(X).vec()

# Time-stepping
for i in range(time_steps):

    if MPI.comm_world.rank == 0:
        print('Time step', i + 1)   

    # update current time
    t += dt
    
    # update source term 
    source_i.t = t   

    # update rhs
    t1 = time.perf_counter()    
    fi = inner(source_i, vi)*dx(1)
    fe = inner(source_e, ve)*dx(2)
    
    rhs = v - (dt/C_M) * v
    fg =  inner(rhs, vg('-'))*dxS

    f =  [fi, fe, fg]

    F = block_assemble(f)
    bcs.apply(F)

    if not direct_solver:
        F_ = as_backend_type(F).vec()

    assemble_time += time.perf_counter() - t1

    # SOLVE
    t1 = time.perf_counter()    
    
    if direct_solver:
        block_solve(A, U.block_vector(), F)
    else:
        ksp.solve(F_, x)
        U.apply("to subfunctions")            

    solve_time += time.perf_counter() - t1

    # update membrane potential        
    v = U[0] - U[1]

    if save_output:
        out_i.write(U[0], t)
        out_e.write(U[1], t)   
        out_g.write(U[2], t)    

# ERROR
ui_exact = Expression("(1 + exp(-t)) * sin(2*pi*x[0]) * sin(2*pi*x[1])", degree = 4, t = t)
ue_exact = Expression("sin(2*pi*x[0]) * sin(2*pi*x[1])", degree = 4)
Im_exact = Expression("0", degree = 4)

err_i = inner(U[0] - ui_exact, U[0] - ui_exact)*dx(1)
err_e = inner(U[1] - ue_exact, U[1] - ue_exact)*dx(2)
err_g = inner(U[2] - Im_exact, U[2] - Im_exact)*dx(2)

L2_norm_i = sqrt(assemble(err_i))
L2_norm_e = sqrt(assemble(err_e))
L2_norm_g = sqrt(assemble(err_g))

if MPI.comm_world.rank == 0:
    print("~~~~~~~~~~~~~~ Info ~~~~~~~~~~~~~~")
    print("dt =", dt)
    print("N =", N)
    if direct_solver:
        print("solver: direct")
    else:
        print('solver: [', ksp_type,'+', pc_type, ']')

    print("~~~~~~~~~~~~~~ Errors ~~~~~~~~~~~~~~")
    print('L2 error interior:', L2_norm_i)
    print('L2 error exterior:', L2_norm_e)
    print('L2 error membrane:', L2_norm_g)

    print("~~~~~~~~~~~~~~ Timing ~~~~~~~~~~~~~~")
    print('Assemble:', solve_time ,   's')
    print('Solve:   ', assemble_time, 's')



