from meshlib import mrmeshpy

input_mesh  = "astro"
offset = 10

# load closed mesh
mesh = mrmeshpy.loadMesh(input_mesh + ".stl")

# setup offset parameters
params = mrmeshpy.OffsetParameters()
# params.voxelSize = 0.1 # control refinement

# create positive offset mesh
ecs_mesh = mrmeshpy.offsetMesh(mesh, offset, params)

# save results
mrmeshpy.saveMesh(ecs_mesh, input_mesh + "_" + str(offset) + ".stl")

# extracted from:
# https://stackoverflow.com/questions/74862109/extruding-an-stl-from-a-binary-png-image-with-python