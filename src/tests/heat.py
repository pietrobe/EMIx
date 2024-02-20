"""
FEniCS tutorial demo program: Diffusion of a Gaussian hill.

  u'= Laplace(u) + f  in a square domain
  u = u_D             on the boundary
  u = u_0             at t = 0

  u_D = f = 0

The initial condition u_0 is chosen as a Gaussian hill.
"""

from __future__ import print_function
from fenics import *
# from dolfin import *
# import dolfin
import time

T = 100.0          # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size

# Import mesh
mesh = Mesh()
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile("convex_hull16.xdmf") as xdmf:
    xdmf.read(mesh)
    xdmf.read(mvc, "name_to_read")

cell_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0), boundary)

# # Define initial value
# u_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))',
#                  degree=2, a=1)
u_0 = Expression('x[0] - 218', degree=2)

u_n = interpolate(u_0, V)

# diffusion
D0 = Constant(0.0)
D1 = Constant(100.0)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

# Define new measures associated with the interior domains
# dx = Measure("dx", domain=mesh)
dx =  Measure("dx", domain=mesh, subdomain_data=cell_markers)

# print(assemble(1*dx(1)))
# print(assemble(1*dx(3)))

F = u*v*dx + D0*dt*dot(grad(u), grad(v))*dx(1) + D1*dt*dot(grad(u), grad(v))*dx(3) - (u_n + dt*f)*v*dx

#F = u*v*dx + D0*dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx

a, L = lhs(F), rhs(F)

# Create VTK file for saving solution
vtkfile = File('heat_gaussian/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    print('Time step:', n) 

    # Update current time
    t += dt

    # Compute solution
    # solve(a == L, u, bc)
    solve(a == L, u)

    # if n == num_steps - 1:        
    # Save to file and plot solution
    vtkfile << (u, t)
    plot(u)

    # Update previous solution
    u_n.assign(u)

