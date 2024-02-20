from dolfin import *

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

dxi = Measure("dx", domain=inter_mesh)
dxe = Measure("dx", domain=exter_mesh)

ai = 1 * Measure("ds", domain=inter_mesh)
ae = 1 * Measure("ds", domain=exter_mesh)

print(assemble(ae))
print(assemble(ai))
