# import time
import meshio

read = meshio.read('../data/Ale_test/_.msh')
from IPython import embed

embed()
from dolfin       import *
# from multiphenics import *
# from mpi4py import MPI as _MPI
# import numpy as np

mesh = Mesh(MPI.comm_world)

# XDMF
with XDMFFile("../data/Ale_test/test.xdmf") as f:
    f.read(mesh)

print(mesh.num_entities(0), mesh.num_vertices(),mesh.num_entities_global(0), flush=True)
print(mesh.num_entities(3), mesh.num_cells(), mesh.num_entities_global(3), flush=True)

V = FunctionSpace(mesh, "Lagrange", 1)
#print(V.dim())
#a = interpolate(Constant(1.0), V)
u = Function(V)
print(len(u.vector().get_local()))


# print("Function:",
#      V.dofmap().global_dimension(),uV.vector().size())
# mesh.init(3, 0)
# c_to_v = mesh.topology()(3, 0)
# vertices = []
# for cell in cells(mesh):
#     vertices.append(c_to_v(cell.index()))
# all_vertices_local = np.unique(np.hstack(vertices))
# print(len(all_vertices_local))
"""
    subdomains = MeshFunction("size_t", mesh, 3, 0)
    f.read(subdomains)
    
# Restrictions
interior  = MeshRestriction(mesh, "../KNPEMI/data/interior_restriction.rtc.xdmf")
exterior  = MeshRestriction(mesh, "../KNPEMI/data/exterior_restriction.rtc.xdmf")

# mesh = Mesh("../KNPEMI/data/mesh.xml")
# subdomains = MeshFunction("size_t", mesh, "../KNPEMI/data/physical_region.xml")
# # boundaries = MeshFunction("size_t", mesh, "../KNPEMI/data/astrocyte_mesh_facet_region.xml")

# # Restrictions
# interior  = MeshRestriction(mesh, "../KNPEMI/data/interior_restriction.rtc.xml")
# exterior  = MeshRestriction(mesh, "../KNPEMI/data/exterior_restriction.rtc.xml")


# FUNCTION SPACES #
V = FunctionSpace(mesh, "Lagrange", 1)
uV = Function(V)
print(V.dofmap().global_dimension(),uV.vector().size())
# Block function space
W = BlockFunctionSpace([V, V], restrict=[interior, exterior])

U = BlockFunction(W)
#print(W.block_dofmap().global_dimension(), U.block_vector().size())

#U.block_vector().apply("")
for i in range(2):
    time.sleep(mesh.mpi_comm().rank)
    print(i, U[i].vector().size(), U[i].function_space().dofmap().global_dimension(), flush=True)

#U.apply("to subfunctions") 

"""