
import numpy as np
from da3d.viewing.mesh import DepthMeshViewer

def test_metric_projection_coordinates_fixed():
    print("\n--- Metric Projection Fix Verification ---")
    width, height = 3, 3
    focal_length = 100.0
    
    viewer = DepthMeshViewer(
        use_metric_depth=True,
        focal_length_x=focal_length,
        focal_length_y=focal_length,
        principal_point_x=1.0, # Center at (1,1)
        principal_point_y=1.0,
        metric_depth_scale=1.0
    )
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Depth input: constant 2.0 meters
    # If invert_depth=True, this is 2.0m
    depth_input = np.full((height, width), 2.0, dtype=np.float32)
    
    mesh = viewer.create_mesh_from_depth(
        image, 
        depth_input, 
        subsample=1, 
        invert_depth=True, # Treat input as meters directly
        smooth_mesh=False
    )
    
    vertices = np.asarray(mesh.vertices)
    
    # Check Right-Middle pixel (x=2, y=1)
    # Center (cx, cy) = (1.0, 1.0)
    # dx = 2 - 1 = 1
    # dy = 1 - 1 = 0
    # Z = 2.0
    # X = dx * Z / fx = 1 * 2.0 / 100.0 = 0.02
    # Y = dy * Z / fy = 0 * 2.0 / 100.0 = 0.0
    
    # Expected vertex in Open3D (Right-Handed, Y-Up? No, Open3D usually uses:
    # Camera looks down -Z. X is right, Y is up (or down depending on convention).
    # The previous code had: [y_3d, -x_3d, -z_scaled] which was rotated.
    # We want standard pinhole: [x, -y, -z] (if y is down in image, -y is up in 3D)
    # or [x, y, -z] if we want y down.
    # Usually for 3D viewers:
    # X = Right
    # Y = Up (so image y (down) needs to be flipped)
    # Z = Back (so camera looks down -Z)
    
    # So we expect:
    # x_final = X = 0.02
    # y_final = -Y = 0.0
    # z_final = -Z = -2.0
    
    # Index for (2,1) in row-major 3x3 grid: 1*3 + 2 = 5
    rm_vertex = vertices[5]
    print(f"Right-middle vertex (2,1): {rm_vertex}")
    
    # Check X coordinate (should be ~0.02, NOT 0.0)
    if abs(rm_vertex[0] - 0.02) < 1e-5:
        print("PASS: X coordinate correct (No rotation)")
    else:
        print(f"FAIL: X coordinate incorrect. Expected 0.02, got {rm_vertex[0]}")

    # Check Y coordinate (should be 0.0)
    if abs(rm_vertex[1] - 0.0) < 1e-5:
        print("PASS: Y coordinate correct")
    else:
        print(f"FAIL: Y coordinate incorrect. Expected 0.0, got {rm_vertex[1]}")

    # Check Z coordinate (should be -2.0, NOT -2.0 * scale)
    if abs(rm_vertex[2] - (-2.0)) < 1e-5:
        print("PASS: Z coordinate correct (Metric scale)")
    else:
        print(f"FAIL: Z coordinate incorrect. Expected -2.0, got {rm_vertex[2]}")

def test_metric_inverse_depth():
    print("\n--- Metric Inverse Depth Test ---")
    viewer = DepthMeshViewer(
        use_metric_depth=True,
        focal_length_x=100.0,
        focal_length_y=100.0
    )
    
    # Input: 0.5. If inverse depth, Z = 1/0.5 = 2.0m
    depth_input = np.full((3, 3), 0.5, dtype=np.float32)
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    
    mesh = viewer.create_mesh_from_depth(
        image, depth_input, subsample=1, invert_depth=False
    )
    
    vertices = np.asarray(mesh.vertices)
    z_val = -vertices[0, 2] # Convert back to positive distance
    
    print(f"Input 0.5 -> Z output: {z_val}")
    
    if abs(z_val - 2.0) < 1e-5:
        print("PASS: Inverse depth calculation correct")
    else:
        print(f"FAIL: Inverse depth calculation incorrect. Expected 2.0, got {z_val}")

if __name__ == "__main__":
    test_metric_projection_coordinates_fixed()
    test_metric_inverse_depth()
