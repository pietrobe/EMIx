from numpy import isclose
from dolfin import *
from multiphenics import *
parameters["ghost_mode"] = "shared_facet" # required by dS

# EMI example on a circle, and constants set to unity

# MESHES #
# Mesh
mesh = Mesh("data/circle.xml")
subdomains = MeshFunction("size_t", mesh, "data/circle_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/circle_facet_region.xml")
# Restrictions
left  = MeshRestriction(mesh, "data/circle_restriction_left.rtc.xml")
right = MeshRestriction(mesh, "data/circle_restriction_right.rtc.xml")
# interface = MeshRestriction(mesh, "data/circle_restriction_interface.rtc.xml")

# forcing term on interface
f_gamma = Expression('10*sin(x[0]+x[1])', degree=1)

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, "Lagrange", 2)
# Block function space
W = BlockFunctionSpace([V, V], restrict=[left, right])

# TRIAL/TEST FUNCTIONS #
u1u2l = BlockTrialFunction(W)
v1v2m = BlockTestFunction(W)

(u1, u2) = block_split(u1u2l)
(v1, v2) = block_split(v1v2m)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)
dS = Measure("dS")(subdomain_data=boundaries)
dS = dS(2) # restrict to the interface, which has facet ID equal to 2

# ASSEMBLE #
a11 = inner(grad(u1), grad(v1))*dx(1) + inner(u1('-'), v1('-'))*dS
a22 = inner(grad(u2), grad(v2))*dx(2) + inner(u2('+'), v2('+'))*dS
a12 = - inner(u2('-'), v1('+'))*dS
a21 = - inner(u1('+'), v2('-'))*dS

a = [[a11, a12],
    [a21, a22]]

f1 = -inner(f_gamma, v1('-'))*dS
f2 =  inner(f_gamma, v2('+'))*dS

f =  [f1, f2]

bc1 = DirichletBC(W.sub(0), Constant(0.), boundaries, 1)
bc2 = DirichletBC(W.sub(1), Constant(0.), boundaries, 1)
bcs = BlockDirichletBC([bc1,bc2])

# SOLVE #
A = block_assemble(a)
F = block_assemble(f)
bcs.apply(A)
bcs.apply(F)

U = BlockFunction(W)
block_solve(A, U.block_vector(), F)

# save output
U[0].rename('u_1', '')
U[1].rename('u_2', '')
out1 = XDMFFile(MPI.comm_world, "output/sol_1.xdmf")
out2 = XDMFFile(MPI.comm_world, "output/sol_2.xdmf")  
out1.write(U[0])  
out2.write(U[1])
if MPI.comm_world.size == 1: # Jurgen debugging this
    out1.write(subdomains)
    out2.write(subdomains)
