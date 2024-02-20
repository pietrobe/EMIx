import petsc4py, sys
# petsc4py.init(sys.argv)

from dolfin import *
from multiphenics import *

from petsc4py import PETSc
print = PETSc.Sys.Print

def generate_subdomain_restriction(mesh, subdomains, subdomain_id):
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain id
    for c in cells(mesh):
        if subdomains[c] == subdomain_id:
            restriction[D][c] = True
            for d in range(D):
                for e in entities(c, d):
                    restriction[d][e] = True
    # Return
    return restriction

# -------

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["allow_extrapolation"] = True

class MElast(SubDomain):
    def inside(self, x, on_boundary):
        return x[1]>=0.5

class MPorous(SubDomain):
    def inside(self, x, on_boundary):
        return x[1]<=0.5  
    
class Inter(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.5)


mesh = UnitSquareMesh(128, 128, 'crossed')
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(998)
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(999)

porous = 0; elast = 1; interf = 16;

MPorous().mark(subdomains, porous)
MElast().mark(subdomains, elast)
Inter().mark(boundaries,interf)
    

n = FacetNormal(mesh)

OmE = generate_subdomain_restriction(mesh, subdomains, elast)
OmP = generate_subdomain_restriction(mesh, subdomains, porous)
    
dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh, subdomain_data=boundaries)

Vh = FunctionSpace(mesh, "CG", 1)
    
Hh = BlockFunctionSpace([Vh, Vh], restrict = [OmE, OmP])

trial = BlockTrialFunction(Hh)
u1, u2 = block_split(trial)
test  = BlockTestFunction(Hh)
v1, v2 = block_split(test)


A00 = (inner(u1, v1)*dx(elast) + inner(grad(u1), grad(v1))*dx(elast) + inner(u1('+'), v1('+'))*dS(interf))
A01 = -inner(u2('-'), v1('+'))*dS(interf)
A11 = (inner(u2, v2)*dx(porous) + inner(grad(u2), grad(v2))*dx(porous) + inner(u2('-'), v2('-'))*dS(interf))
A10 = -inner(u1('+'), v2('-'))*dS(interf)

b0 = inner(Expression('x[0]+x[1]*x[1]', degree=2), v1)*dx(elast)
b1 = inner(Constant(2), v2)*dx(porous)

rhs = [b0, b1]

lhs = [[A00, A01],
       [A10, A11]]


bcU1 = DirichletBC(Hh.sub(0), Expression('x[0]+x[1]', degree=1), 'on_boundary')
bcs  = BlockDirichletBC([bcU1, None])# this will only prescribe u.n

AA = block_assemble(lhs)
FF = block_assemble(rhs)
bcs.apply(AA); bcs.apply(FF)
    
sol = BlockFunction(Hh)

AA_ = as_backend_type(AA).mat()
FF_ = as_backend_type(FF).vec()
    
ksp = PETSc.KSP().create()
ksp.setOperators(AA_, AA_)
ksp.setType('cg')

opts = PETSc.Options()
opts.setValue('ksp_view', None)
opts.setValue('ksp_converged_reason', None)
opts.setValue('ksp_monitor_true_residual', None)
opts.setValue('ksp_rtol', 1E-12)
opts.setValue('ksp_max_it', 1000)
opts.setValue('ksp_converged_reason', None)

opts.setValue('pc_type', 'hypre')
    
ksp.setFromOptions()
ksp.setUp()

X = sol.block_vector()
x = as_backend_type(X).vec()

ksp.solve(FF_, x)
sol.apply("to subfunctions")    

u1h, u2h = block_split(sol)

File('u1h.pvd') << u1h  # NOTE: it plots as if the function was defined everywhere
File('u2h.pvd') << u2h  # check with Marius Causemann how to do subdomain

print('|x|', x.norm(2))
print('|b|', FF_.norm(2))

