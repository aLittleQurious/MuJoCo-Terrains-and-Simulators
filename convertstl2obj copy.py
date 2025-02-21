import trimesh

# Load the STL file
mesh = trimesh.load_mesh("assets/turtlev1/frontrightflipperlink.STL")

# Export as OBJ
mesh.export("assets/turtlev1/frontrightflipperlink.obj")
print("Conversion complete: frontrightflipperlink.obj created")
