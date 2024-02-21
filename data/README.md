## How to generate EMI meshes from surfaces files

### Dependencies 
- meshio:
`pip3 install --no-cache-dir --no-binary=h5py h5py meshio`

### Generate .msh file
Given a collection of surfaces files (e.g. .stl) generate an .msh file with intra/extra tags with fTetWild running:

```
/path_to_fTetWild/fTetWild/build/FloatTetwild_bin --csg csg.json
```
csg.json file example for two surfaces:

`{"operation":"union", "right": "intra.stl","left": "extra.stl"}`

The order in the .json file can be important for correct tagging.
The accuracy of the mesh to representing curved sty geometries can be controlled with the -e option (https://github.com/wildmeshing/fTetWild).

### Generate data structures (multiphenics restrictions and xdmd files)
### Setup
In *generate_mesh.py* set tags as in your input .msh file, typically
```
exterior_subdomain_id = [1]
interior_subdomain_id = [2,3]
```

in this case with two interior tags (i.e. cell types).

### Run (only in serial)
`python3 generate_mesh.py`
