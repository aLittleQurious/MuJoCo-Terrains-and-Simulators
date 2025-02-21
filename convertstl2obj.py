import trimesh

# Load the STL file
mesh = trimesh.load_mesh("assets/turtlev1/assets/turtlev1/meshes/frontleftflipperlink.STL")

# Export as OBJ
mesh.export("assets/turtlev1/assets/turtlev1/meshes/frontleftflipperlink.obj")
print("Conversion complete: frontrightflipperlink.obj created")
