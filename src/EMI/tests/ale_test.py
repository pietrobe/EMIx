from dolfin import *
from multiphenics import *
import time
from petsc4py import PETSc
import scipy.sparse as sparse
import numpy as np 
from sys import argv

parameters["ghost_mode"] = "shared_facet" # required by dS
parameters['form_compiler']['optimize']           = True
parameters['form_compiler']['cpp_optimize']       = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

def dump(thing, path):
            if isinstance(thing, PETSc.Vec):
                assert np.all(np.isfinite(thing.array))
                return np.save(path, thing.array)
            m = sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
            assert np.all(np.isfinite(m.data))
            return np.save(path, np.c_[m.row, m.col, m.data])


# space discretization parameters
P = 1

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
# Restrictions #TODO

# FUNCTION SPACES #
V = FunctionSpace(mesh, "Lagrange", P)    
W = BlockFunctionSpace([V, V], restrict=[omega_i, omega_e])

# TRIAL/TEST FUNCTIONS #
uu = BlockTrialFunction(W)
vv = BlockTestFunction(W)

(ui, ue) = block_split(uu)
(vi, ve) = block_split(vv)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
dS = Measure("dS")(subdomain_data=boundaries)
dS = dS(2) # restrict to the interface, which has facet ID equal to 2

# ASSEMBLE #
a11 = inner(sigma_i*grad(ui), grad(vi))*dx(1) + (C_M/dt) * inner(ui('-'), vi('-'))*dS
a22 = inner(sigma_e*grad(ue), grad(ve))*dx(2) + (C_M/dt) * inner(ue('+'), ve('+'))*dS
a12 = - (C_M/dt) * inner(ue('+'), vi('-'))*dS
a21 = - (C_M/dt) * inner(ui('-'), ve('+'))*dS

a = [[a11, a12],
    [ a21, a22]]

A = block_assemble(a)

# Enforce BC
bc_e = DirichletBC(W.sub(1), Constant(0.), boundaries, 1)
bcs  = BlockDirichletBC([None,bc_e])
bcs.apply(A)

if save_matrix:
    # Write A
    print("Saving A in npy format...")   
    dump(A.mat(),'output/Amat_' + str(dt))
    
    exit()

U = BlockFunction(W)

# save output
U[0].rename('u_i', '')
U[1].rename('u_e', '')

# Time-stepping
for i in range(time_steps):

    if MPI.comm_world.rank == 0:
        print('Time step', i + 1)   

    # update current time
    t += dt
    
    # update source term 
    source_i.t = t   

    # update rhs    
    fg = v #- (dt/C_M) * v
    fi = inner(source_i, vi)*dx(1) + (C_M/dt) * inner(fg, vi('-'))*dS
    fe = inner(source_e, ve)*dx(2) - (C_M/dt) * inner(fg, ve('+'))*dS
    f =  [fi, fe]
    F = block_assemble(f)
    
    if use_dirichlet_bc: bcs.apply(F)

    # save rhs
    # if save_matrix:
    #     dump(as_backend_type(F).vec(),'output/rhs')
    #     exit()            

    # SOLVE            
    block_solve(A, U.block_vector(), F, linear_solver = 'mumps')
    
    v = U[0] - U[1]

    if save_output:
        out_i.write(U[0], t)
        out_e.write(U[1], t)
        
#if MPI.comm_world.size == 1: # JD debugging this
  #   out_i.write(subdomains)  
  #   out_e.write(subdomains)

