import os 
from fenics import *
from pathlib import Path
import numpy as np

# script to download and read meshes from https://zenodo.org/record/13373950

ACCESS_TOKEN = ("eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU3Yzc3OTZmLWE3ZjAtNGU5YS1hNGM4LTE2OD"+
                "gxNThmZTc5NiIsImRhdGEiOnt9LCJyYW5kb20iOiJjNTVkMWNiNWQzNzlmOTdlM2MxZ"+
                "jE0MWJkYTc0NTg3YyJ9.FG9Pa3T5Z5e_4SyWgD7P7tiHcypae00izZG_NR9n1dF8WFX"+
                "cJQFakmEuK3uaftvJHN4GlXsyKw70MKAEXnhUkQ")

record_id = "13373950"

def download_mesh(ncells, size, meshdir):
    
    url = (f"https://zenodo.org/records/{record_id}/files/volmesh_size+{size}-dx+20_ncells+{ncells}.zip?token={ACCESS_TOKEN}")
    os.system(f"mkdir -p {meshdir}")
    os.system(f"wget -O {meshdir}/mesh.zip {url} && cd {meshdir} && unzip mesh.zip && rm mesh.zip")


def read_mesh(meshfile, facetfile):
    mesh = Mesh()
    infile = XDMFFile(meshfile)
    infile.read(mesh)
    gdim = mesh.geometric_dimension()
    labels = MeshFunction("size_t", mesh, gdim)
    infile.read(labels, "label")
    infile.close()
    infile = XDMFFile(facetfile)
    gdim = mesh.geometric_dimension()
    boundary_marker = MeshFunction("size_t", mesh, gdim - 1)
    infile.read(boundary_marker, "boundaries")
    infile.close()
    return mesh, labels, boundary_marker

def generate_subdomain_restriction(mesh, subdomains, subdomain_ids):
    from multiphenics import MeshRestriction
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

if __name__ == "__main__":
    ncells = 50
    size = 5000
    meshfile = Path(f"meshes/ncells_{ncells}_size_{size}/mesh.xdmf")
    if not meshfile.exists():
        download_mesh(ncells, size, meshfile.parent)
    mesh, labels, boundary_marker = read_mesh(str(meshfile), f"{meshfile.parent}/facets.xdmf")
    
    ecs_ids = [1]
    cell_ids = range(2, 2 + ncells)
    # Generate subdomain restrictions
    interior_restriction = generate_subdomain_restriction(mesh, labels, cell_ids)
    exterior_restriction = generate_subdomain_restriction(mesh, labels, ecs_ids)
    
    interior_restriction._write(f"{meshfile.parent}/interior_restriction.rtc.xdmf")
    exterior_restriction._write(f"{meshfile.parent}/exterior_restriction.rtc.xdmf")
