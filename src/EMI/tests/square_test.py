from numpy import isclose
from dolfin import *
from multiphenics import *
parameters["ghost_mode"] = "shared_facet" # required by dS

# MESHES #
# Mesh
mesh = Mesh("data/square16.xml")
subdomains = MeshFunction("size_t", mesh, "data/square_physical_region16.xml")
boundaries = MeshFunction("size_t", mesh, "data/square_facet_region16.xml")
# Restrictions
omega_i = MeshRestriction(mesh, "data/square_restriction_om_i16.rtc.xml")
omega_e = MeshRestriction(mesh, "data/square_restriction_om_e16.rtc.xml")

# forcing term on interface
f_gamma = Expression('sin(x[0]+x[1])', degree=1)

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
a11 = inner(grad(ui), grad(vi))*dx(1) + inner(ui('-'), vi('-'))*dS
a22 = inner(grad(ue), grad(ve))*dx(2) + inner(ue('+'), ve('+'))*dS
a12 = - inner(ue('+'), vi('-'))*dS
a21 = - inner(ui('-'), ve('+'))*dS

a = [[a11, a12],
    [a21, a22]]

fi = -inner(f_gamma, vi('-'))*dS
fe =  inner(f_gamma, ve('+'))*dS

f =  [fi, fe]

bc_e = DirichletBC(W.sub(1), Constant(0.), boundaries, 1)
bcs  = BlockDirichletBC([None,bc_e])

# SOLVE #
A = block_assemble(a)
F = block_assemble(f)
bcs.apply(A)
bcs.apply(F)

U = BlockFunction(W)
block_solve(A, U.block_vector(), F)

# save output
U[0].rename('u_i', '')
U[1].rename('u_e', '')
out1 = XDMFFile(MPI.comm_world, "output/sol_i.xdmf")
out2 = XDMFFile(MPI.comm_world, "output/sol_e.xdmf")  
out1.write(U[0])
out2.write(U[1])
if MPI.comm_world.size == 1: # Jurgen debugging this
    out1.write(subdomains)
    out2.write(subdomains)

