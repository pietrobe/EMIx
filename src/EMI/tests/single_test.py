from dolfin import *
from multiphenics import *
import time
from petsc4py import PETSc
import scipy.sparse as sparse
import numpy as np 
from sys import argv
import sys

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
P = int(argv[2])

use_dirichlet_bc = True

# time discretization parameters
t          = 0.0
T          = 3
dt         = float(argv[3])
time_steps = 10 #int(T / dt)

# physical parameters
C_M     = 1.0
sigma_i = 1.0
sigma_e = 1.0

# flags
save_output   = True   
save_matrix   = False
direct_solver = False
ksp_type      = 'cg'
pc_type       = 'ilu' #gamg
ksp_rtol      = 1e-6
ds_type       = 'mumps'
norm_type     = 'unpreconditioned'

# timers 
solve_time    = 0
assemble_time = 0

# initial membrane potential V
v = Expression("sin(2*pi*x[0]) * sin(2*pi*x[1])", degree = 4)

# forcing factors
source_i = Expression("0*8*pi*pi*sin(2*pi*x[0]) * sin(2*pi*x[1]) * (1.0 + exp(-t))", degree = 4, t = t)
source_e = Expression("0*8*pi*pi*sin(2*pi*x[0]) * sin(2*pi*x[1])",  degree = 4)

if save_output:
    # output files
    out_i = XDMFFile(MPI.comm_world, "output/sol_i.xdmf")
    out_e = XDMFFile(MPI.comm_world, "output/sol_e.xdmf") 

    out_i.parameters['functions_share_mesh' ] = True
    out_i.parameters['rewrite_function_mesh'] = False
    out_e.parameters['functions_share_mesh' ] = True
    out_e.parameters['rewrite_function_mesh'] = False

# MESHES #
# Mesh
mesh = Mesh("../data/square/square" + str(N) + ".xml")
subdomains = MeshFunction("size_t", mesh, "../data/square/square_physical_region" + str(N) + ".xml")
boundaries = MeshFunction("size_t", mesh, "../data/square/square_facet_region" + str(N) + ".xml")
# Restrictions
omega_i = MeshRestriction(mesh, "../data/square/square_restriction_om_i" + str(N) + ".rtc.xml")
omega_e = MeshRestriction(mesh, "../data/square/square_restriction_om_e" + str(N) + ".rtc.xml")
# gamma   = MeshRestriction(mesh, "data/circle_restriction_interface.rtc" + str(N) + ".xml")

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, "Lagrange", P)
# Block function space
# if use_dirichlet_bc:
    
W = BlockFunctionSpace([V, V], restrict=[omega_i, omega_e])

# else:

#     # Lagrange multiplier space    
#     P1 = FiniteElement("Lagrange", mesh.ufl_cell(), P)
#     R  = FiniteElement("Real",     mesh.ufl_cell(), 0)
#     Ve = FunctionSpace(mesh, P1 * R)

#     # function spaces               
#     W  = BlockFunctionSpace([V, Ve], restrict=[omega_i, omega_e])


# TRIAL/TEST FUNCTIONS #
uu = BlockTrialFunction(W)
vv = BlockTestFunction(W)

# if use_dirichlet_bc:      
(ui, ue) = block_split(uu)
(vi, ve) = block_split(vv)
# else:

#     (ui, ue_tot) = block_split(uu)
#     (vi, ve_tot) = block_split(vv)
    
#     # extracellular potential
#     ue = ue_tot[0]  # unknown
#     ve = ve_tot[0]  # test function

#     # Lagrange mult.
#     _c = ue_tot[1]   # unknown
#     _d = ve_tot[1]   # test function


# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
dS = Measure("dS")(subdomain_data=boundaries)
dS = dS(2) # restrict to the interface, which has facet ID equal to 2

# ASSEMBLE #
a11 = inner(sigma_i*grad(ui), grad(vi))*dx(1) + (C_M/dt) * inner(ui('-'), vi('-'))*dS
a22 = inner(sigma_e*grad(ue), grad(ve))*dx(2) + (C_M/dt) * inner(ue('+'), ve('+'))*dS
a12 = - (C_M/dt) * inner(ue('+'), vi('-'))*dS
a21 = - (C_M/dt) * inner(ui('-'), ve('+'))*dS

# if not use_dirichlet_bc:                  
#     a22 += _c*ve*dx(2) + _d*ue*dx(2) # test for MMS

a = [[a11, a12],
    [ a21, a22]]

# prec
# p1 = inner(sigma_i*grad(ui), grad(vi))*dx(1) + (C_M/dt) * inner(ui('-'), vi('-'))*dS
# p2 = inner(sigma_e*grad(ue), grad(ve))*dx(2) + (C_M/dt) * inner(ue('+'), ve('+'))*dS

# p = [[p1, 0],
#     [ 0, p2]]

# Prec = block_assemble(p)


# from IPython import embed;embed()
# ASSEMBLE #
t1 = time.perf_counter() 
A = block_assemble(a)

# Enforce dirichlet_bc
if use_dirichlet_bc: 
    bc_e = DirichletBC(W.sub(1), Constant(0.), boundaries, 1)
    bcs  = BlockDirichletBC([None,bc_e])
    bcs.apply(A)

    # bcs.apply(Prec)
    # P_ = as_backend_type(Prec).mat()

assemble_time += time.perf_counter() - t1

if save_matrix:
    # Write A
    print("Saving A in npy format...")   
    dump(A.mat(),'output/Amat_' + str(dt))
    # use then in /output with MATLAB:
    # addpath('some_path/scripts')
    # addpath('some_path/scripts/npy-matlab-master/npy-matlab/')
    # data = readNPY('Amat.npy'); A = create_sparse_mat_from_data(data);

    # out_string = 'output/rhs.m'
    # original_stdout = sys.stdout
    # np.set_printoptions(threshold=sys.maxsize, linewidth=1000000000)
    # with open(out_string, 'w') as ff:
    #     sys.stdout = ff
    #     print("f = ", f.get_local(), ";")
    # sys.stdout = original_stdout # Reset the standard output to its original value

    exit()

U = BlockFunction(W)

# save output
U[0].rename('u_i', '')
U[1].rename('u_e', '')

if not direct_solver:

    A_ = as_backend_type(A).mat()
        
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_, A_)
    # ksp.setOperators(A_, P_)
    ksp.setType(ksp_type)

    opts = PETSc.Options()
    # opts.setValue('ksp_view', None)
    # opts.setValue('ksp_monitor_true_residual', None)
    #opts.setValue('ksp_view_eigenvalues', None)
    #opts.setValue('ksp_monitor_singular_value', None)    
    opts.setValue('ksp_rtol', ksp_rtol)
    opts.setValue('ksp_converged_reason', None)
    opts.setValue('pc_type', pc_type)
    opts.setValue("ksp_norm_type", norm_type)

    # for IC
    # opts.setValue("pc_factor_in_place", None)
    # opts.setValue("-pc_factor_levels", 4)

    # PETSc.Options().setValue("pc_hypre_boomeramg_relax_type_all", "jacobi")
    # PETSc.Options().setValue("pc_hypre_boomeramg_interp_type", "standard")
    # PETSc.Options().setValue("pc_hypre_boomeramg_coarsen_type", "Ruge-Stueben")
    # PETSc.Options().setValue("-pc_hypre_boomeramg_relax_weight_all", 0.8)
    # PETSc.Options().setValue("-pc_hypre_boomeramg_outer_relax_weight_all", 0.5)

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
    fg = v #- (dt/C_M) * v
    fi = inner(source_i, vi)*dx(1) + (C_M/dt) * inner(fg, vi('-'))*dS
    fe = inner(source_e, ve)*dx(2) - (C_M/dt) * inner(fg, ve('+'))*dS
    f =  [fi, fe]
    F = block_assemble(f)
    
    if use_dirichlet_bc: bcs.apply(F)

    if save_matrix:
        dump(as_backend_type(F).vec(),'output/rhs')
        exit()
    
    if not direct_solver:
        F_ = as_backend_type(F).vec()

    assemble_time += time.perf_counter() - t1

    # SOLVE
    t1 = time.perf_counter()    
    
    if direct_solver:
        block_solve(A, U.block_vector(), F, linear_solver = ds_type)
    else:
        ksp.solve(F_, x)
        U.block_vector().apply("")
        U.apply("to subfunctions")            

    solve_time_t = time.perf_counter() - t1    
    print("Solve time = " + str(solve_time_t))    
    solve_time += solve_time_t
    
    v = U[0] - U[1]

    if save_output:
        out_i.write(U[0], t)
        out_e.write(U[1], t)
        
#if MPI.comm_world.size == 1: # JD debugging this
 #    out_i.write(subdomains)  
  #   out_e.write(subdomains)

# ERROR
ui_exact = Expression("(1 + exp(-t)) * sin(2*pi*x[0]) * sin(2*pi*x[1])", degree = 4, t = t)
ue_exact = Expression("sin(2*pi*x[0]) * sin(2*pi*x[1])", degree = 4)

err_i = inner(U[0] - ui_exact, U[0] - ui_exact)*dx(1)
err_e = inner(U[1] - ue_exact, U[1] - ue_exact)*dx(2)
L2_norm_i = sqrt(assemble(err_i))
L2_norm_e = sqrt(assemble(err_e))

if MPI.comm_world.rank == 0:
    print("~~~~~~~~~~~~~~ Info ~~~~~~~~~~~~~~")
    print("dt =", dt)
    print("N =", N)
    print("P =", P)
    if direct_solver:
        print("direct solver:", ds_type)
    else:
        print('solver: [', ksp_type,'+', pc_type, ']')

    print("~~~~~~~~~~~~~~ Errors ~~~~~~~~~~~~~~")
    print('L2 error interior:', L2_norm_i)
    print('L2 error exterior:', L2_norm_e)

    print("~~~~~~~~~~~~~~ Timing ~~~~~~~~~~~~~~")
    print('Assemble:', assemble_time ,   's')
    print('Solve:   ', solve_time, 's')

# print('L2 error interior: ', errornorm(ui_exact, U[0], 'L2'))
# print('L2 error exterior: ', errornorm(ue_exact, U[1], 'L2'))



