from dolfin import *
import numpy as np
from petsc4py import PETSc
import scipy.sparse as sparse


n = 20
# square mesh
mesh = UnitSquareMesh(n, n)
# define interior domain
interior = CompiledSubDomain('std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25')
# create mesh function
subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

# mark interior and exterior domain
for cell in cells(mesh):
    x = cell.midpoint().array()
    subdomains[cell] = int(interior.inside(x, False))
assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

# create exterior mesh
exter_mesh = MeshView.create(subdomains, 0)
# create interior mesh
inter_mesh = MeshView.create(subdomains, 1)

Wi = FunctionSpace(inter_mesh, 'CG', 1)
We = FunctionSpace(exter_mesh, 'CG', 1)
W = MixedFunctionSpace(Wi, We)

wh = Function(W)
wi = Function(Wi)
we = Function(We)

wi.interpolate(Expression("x[0] + 2*x[1]", degree=1))
we.interpolate(Expression("x[0] + 2*x[1]", degree=1))

# from IPython import embed;embed()

# assemble rhs and a 
dxi = Measure("dx", domain=inter_mesh)
dxe = Measure("dx", domain=exter_mesh)

(ui, ue) = TrialFunctions(W)
(vi, ve) =  TestFunctions(W)

ai = 1 * ds(domain=inter_mesh)
ae = 1 * ds(domain=exter_mesh)

a = ae + ai

print(assemble(ae))
print(assemble(ai))

exit()

Li = inner(wi, vi) * dxi
Le = inner(we, ve) * dxe

L = Le + Li

system = assemble_mixed_system(a == L, wh)
matrix_blocks = system[0]
rhs_blocks    = system[1]
sol_blocks    = system[2]

# Update the system and convert again to AIJ format 
A = PETScNestMatrix(matrix_blocks)
b = Vector()
w = Vector()
A.init_vectors(b, rhs_blocks)
A.init_vectors(w, sol_blocks)   
A.convert_to_aij()

# get the matrix norm
norm = A.norm("frobenius")
print("Frobenius norm of A:", norm)

# wi_vec = wi.vector()    
# we_vec = we.vector()

# w_petsc = as_backend_type(w).vec()
# wi_petsc = w_petsc.getNestSubVecs()[0]
# we_petsc = w_petsc.getNestSubVecs()[1]  

# wi_vec.set_local(wi_petsc.array)                    
# we_vec.set_local(we_petsc.array)                
    
# wi_vec.apply('insert')    
# we_vec.apply('insert')
# wh.sub(0).assign(wi)
# wh.sub(1).assign(we)

# # rename for Paraview convenience
# wh.sub(0).rename('wi', '')
# wh.sub(1).rename('we', '')
# wi.rename('wi', '')
# we.rename('we', '')

# # write vectors
# XDMFFile("output/wh_i.xdmf").write(wi)
# XDMFFile("output/wh_e.xdmf").write(we)

# # save mesh partition
# XDMFFile("output/partition_i.xdmf").write(MeshFunction("size_t", inter_mesh, inter_mesh.topology().dim(), inter_mesh.mpi_comm().rank))
# XDMFFile("output/partition_e.xdmf").write(MeshFunction("size_t", exter_mesh, exter_mesh.topology().dim(), exter_mesh.mpi_comm().rank))




