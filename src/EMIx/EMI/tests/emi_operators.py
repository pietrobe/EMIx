from numpy import isclose
from dolfin import *
from multiphenics import *

parameters["ghost_mode"] = "shared_facet" # required by dS
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["allow_extrapolation"] = True

import numpy as np 
import scipy.sparse as sparse
from petsc4py     import PETSc
from sys import argv

def dump(thing, path):
            if isinstance(thing, PETSc.Vec):
                assert np.all(np.isfinite(thing.array))
                return np.save(path, thing.array)
            m = sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
            assert np.all(np.isfinite(m.data))
            return np.save(path, np.c_[m.row, m.col, m.data])


class Omega_e(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]<=0.25 or x[0]>=0.75 or x[1]<=0.25 or x[1]>=0.75

class Omega_i(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]>=0.25 and x[0]<=0.75 and x[1]>=0.25 and x[1]<=0.75
    
class Gamma(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 0.25) or near(x[0], 0.75)) and x[1]>=0.25 and x[1]<=0.75 or \
               (near(x[1], 0.25) or near(x[1], 0.75)) and x[0]>=0.25 and x[0]<=0.75

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


####################################

N = int(argv[1])
ksp_type   = 'cg'
pc_type    = 'ilu'
verbose    = True

####################################

# MESH
# mesh = UnitSquareMesh.create(N, N, CellType.Type.quadrilateral)
mesh = UnitSquareMesh(N,N)

subdomains = MeshFunction("size_t", mesh, 2)
boundaries = MeshFunction("size_t", mesh, 1)

#### subdomains ####
omega_i_tag = 1
omega_e_tag = 2

Omega_i().mark(subdomains,omega_i_tag)
Omega_e().mark(subdomains,omega_e_tag)

om_i = Omega_i()
om_e = Omega_e()

omega_i = MeshRestriction(mesh, om_i)
omega_e = MeshRestriction(mesh, om_e)

#### boundaries ####
boundary_tag = 1 
gamma_tag    = 2 

Boundary().mark( boundaries, boundary_tag)
Gamma().mark(    boundaries, gamma_tag)

on_boundary  = Boundary()
on_interface = Gamma()

on_boundary.mark( boundaries, 1)
on_interface.mark(boundaries, 2)

####################################

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, "Lagrange", 1)
# Block function space
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
a11 = inner(grad(ui), grad(vi))*dx(1) 
a22 = inner(grad(ue), grad(ve))*dx(2) 

m11 = inner(ui('-'), vi('-'))*dS
m22 = inner(ue('+'), ve('+'))*dS

m12 = - inner(ue('+'), vi('-'))*dS
m21 = - inner(ui('-'), ve('+'))*dS

a = [[a11+m11, m12],
     [m21, a22+m22]]

p = [[a11+m11, 0],
     [0, a22+m22]]

# ASSEMBLE #
A = block_assemble(a)
P = block_assemble(p)

# SAVE MATRICES
# print("Saving MATLAB data...")   
# dump(A.mat() ,'output/Amat')
# dump(P.mat() ,'output/Pmat')

# read data from MATLAB with: 
# data = readNPY('Amat.npy'); A = create_sparse_mat_from_data(data);

####################################

# SOLVE #

# solution
wh  = BlockFunction(W)

# rhs
# f_gamma = Expression('sin(x[0]+x[1])', degree=1)
f_gamma = Expression("sin(2*pi*x[0]) * sin(2*pi*x[1])", degree = 4)
fi = -inner(f_gamma, vi('-'))*dS
fe =  inner(f_gamma, ve('+'))*dS
f =  [fi, fe]
F = block_assemble(f)

# boundary conditions
bc_e = DirichletBC(W.sub(1), Constant(0.), boundaries, 1)
bcs  = BlockDirichletBC([None,bc_e])
bcs.apply(A)
bcs.apply(P)
bcs.apply(F)

# setup solver
X  = wh.block_vector()
x = as_backend_type(X).vec()            
     
ksp = PETSc.KSP().create()
ksp.setType(ksp_type)
pc = ksp.getPC()     
pc.setType(pc_type)

PETScOptions.set("ksp_converged_reason")
PETScOptions.set("ksp_norm_type", 'unpreconditioned')

if verbose:
    PETScOptions.set("ksp_monitor_true_residual")
#     PETScOptions.set("ksp_view")

ksp.setFromOptions() 

# set up PETSc structures                    
A_ = as_backend_type(A).mat()
F_ = as_backend_type(F).vec()                
P_ = as_backend_type(P).mat()

ksp.setOperators(A_, A_)           
# ksp.setOperators(A_, P_)           
                    
ksp.solve(F_, x)
               



