
import open3d as o3d
import numpy as np

# Create a simple cube mesh
mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
mesh.compute_vertex_normals()

# Color it
mesh.paint_uniform_color([0.1, 0.7, 0.7])

# Save
o3d.io.write_triangle_mesh("assets/lobby_cube.obj", mesh)
print("Created assets/lobby_cube.obj")
