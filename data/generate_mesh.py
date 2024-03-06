from dolfin       import *
from multiphenics import *
import meshio
import numpy as np

# Helper function to generate subdomain restriction based on a gmsh subdomain id
def generate_subdomain_restriction(mesh, subdomains, subdomain_ids):
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain id
    for c in cells(mesh):
        if subdomains[c] in subdomain_ids:
            restriction[D][c] = True
            for d in range(D):
                for e in entities(c, d):
                    restriction[d][e] = True
    # Return
    return restriction

# Helper function to generate interface restriction based on a pair of gmsh subdomain ids
def generate_interface_restriction(mesh, subdomains, subdomain_ids):
    assert isinstance(subdomain_ids, set)
    assert len(subdomain_ids) == 2
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain ids (except the mesh function corresponding to dimension D, as it is trivially false)
    for f in facets(mesh):
        subdomains_ids_f = set(subdomains[c] for c in cells(f))
        assert len(subdomains_ids_f) in (1, 2)
        if subdomains_ids_f == subdomain_ids:
            restriction[D - 1][f] = True
            for d in range(D - 1):
                for e in entities(f, d):
                    restriction[d][e] = True
    # Return
    return restriction


def mark_internal_interface(mesh, subdomains, boundaries, exterior_id):
    # set internal interface
    for f in facets(mesh):        
        domains = []
        for c in cells(f):                        
            domains.append(subdomains[c])            
        
        # if a facet is between two cells with different tags 
        # and one tag is ECS, we tag the facet with the second tag
        if len(domains) == 2 and domains[0] != domains[1]:

            if domains[0]==exterior_id[0]:
                boundaries[f] = domains[1]

            if domains[1]==exterior_id[0]:
                boundaries[f] = domains[0]
            

def mark_external_boundaries(mesh, subdomains, boundaries, subdomain_ids, new_boundary_id):
    for f in facets(mesh):
        if not f.exterior():
            continue
        c = list(cells(f))[0]
        if subdomains[c] in subdomain_ids:
            boundaries[f] = new_boundary_id


# for meshio
# pip3 install --no-cache-dir --no-binary=h5py h5py meshio

msh_file  = "_.msh"   # input mesh
xdmf_file = "test.xdmf"

# convert msh to xdmf removing extra points with meshio
msh_mesh = meshio.read(msh_file)

points     = msh_mesh.points
cell_array = msh_mesh.cells[0].data
marker     = msh_mesh.cell_data_dict["gmsh:physical"]["tetra"]

needless_points = np.setdiff1d(np.arange(0, points.shape[0]), cell_array)
print(needless_points)
points = np.delete(points, needless_points, axis=0)
new_indices = np.zeros_like(cell_array, dtype=np.float64)
for i, cell in enumerate(cell_array):
    for j, vertex in enumerate(cell):
        new_indices[i,j] = vertex - np.sum(vertex>needless_points)

m = meshio.Mesh(points,[("tetra", new_indices)], cell_data={"label": [marker.ravel()]})
m.write(xdmf_file)
####################################

# set input geometry 
exterior_subdomain_id = [1]
interior_subdomain_id = [2,3]
boundary_id  = 1

# refinment level (0 no refinement)
refine = 1
print("Refinment level: " + str(refine))

out_mesh = "mesh"   + str(refine) + ".xdmf"
out_face = "facets" + str(refine) + ".xdmf"

# Import mesh
mesh = Mesh(MPI.comm_world)

print("Reading mesh from " + xdmf_file)

infile = XDMFFile(xdmf_file)
infile.read(mesh)
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
infile.read(subdomains, "label")
# subdomains.rename("gmsh:physical","subdomains")
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
# hdf.read(boundaries, "/boundaries")

while refine > 0:
    print("Refining mesh...")
    mesh = adapt(mesh)
    subdomains = adapt(subdomains, mesh)
    boundaries = adapt(boundaries, mesh)
    refine -= 1

# Generate subdomain restrictions
interior_restriction = generate_subdomain_restriction(mesh, subdomains, interior_subdomain_id)
exterior_restriction = generate_subdomain_restriction(mesh, subdomains, exterior_subdomain_id)

# Generate interface restriction
mark_internal_interface( mesh, subdomains, boundaries, exterior_subdomain_id)
mark_external_boundaries(mesh, subdomains, boundaries, exterior_subdomain_id, boundary_id)

# Write xdmf
print("Writing restrictions")
interior_restriction._write("interior_restriction" + str(refine) + ".rtc.xdmf")
exterior_restriction._write("exterior_restriction" + str(refine) + ".rtc.xdmf")

print("Writing subdomains...")
with XDMFFile(out_mesh) as f:
    f.write(subdomains)
print("Writing boundaries...")
with XDMFFile(out_face) as f:
    f.write(boundaries)


