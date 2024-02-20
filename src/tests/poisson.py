from dolfin import *
import time
from sys   import argv
from petsc4py     import PETSc

rank = MPI.rank(MPI.comm_world)

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

# Create mesh and define function space
# mesh = UnitSquareMesh(5000,5000)
nx = int(argv[1])
#mesh = RectangleMesh(Point(0, 0), Point(100, 100), nx, nx)
mesh = UnitSquareMesh(nx,nx)
V = FunctionSpace(mesh, "Lagrange", 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1 - DOLFIN_EPS
# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds


parameters["linear_algebra_backend"] = "PETSc"
solver = KrylovSolver("cg", "hypre_amg")
solver.parameters["monitor_convergence"] = True


# solver.parameters["relative_tolerance"] = 1.0e-10

cpu_time = time.process_time()
A, b = assemble_system(a, L, bc)
if rank == 0: print("Assembly time is: ",time.process_time()-cpu_time)
u = Function(V)

cpu_time = time.process_time()
solver.solve(A, u.vector(), b)


if rank == 0: 
	print(u.vector().size())
	print("Solve time is: ",time.process_time()-cpu_time)



	