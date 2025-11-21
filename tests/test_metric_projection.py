
import numpy as np
from da3d.viewing.mesh import DepthMeshViewer

def test_metric_projection_coordinates():
    print("\n--- Metric Projection Test ---")
    width, height = 3, 3
    focal_length = 1.0
    
    viewer = DepthMeshViewer(
        use_metric_depth=True,
        focal_length_x=focal_length,
        focal_length_y=focal_length,
        principal_point_x=1.0,
        principal_point_y=1.0,
        metric_depth_scale=1.0
    )
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_input = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 3.0, 2.0],
        [1.0, 2.0, 1.0]
    ], dtype=np.float32)
    
    mesh = viewer.create_mesh_from_depth(
        image, 
        depth_input, 
        subsample=1, 
        invert_depth=True,
        smooth_mesh=False
    )
    
    vertices = np.asarray(mesh.vertices)
    
    # Check Right-Middle pixel (2,1)
    # x=2, y=1, z=2.0
    # x_3d = 2, y_3d = 0
    # Scale = 3.0
    # x_scaled = 6, y_scaled = 0
    # Rotated (swap X/Y): [y_scaled, -x_scaled, -z_scaled]
    # [0, -6, -6]
    rm_vertex = vertices[5]
    print(f"Right-middle vertex (2,1): {rm_vertex}")
    
    if abs(rm_vertex[0] - 0) < 1e-5 and abs(rm_vertex[1] - (-6)) < 1e-5:
        print("PASS: Rotation verified (Right -> Down)")
    else:
        print("FAIL: Rotation incorrect")

def test_relative_raw_depth():
    print("\n--- Relative Raw Depth Test ---")
    width, height = 10, 10
    
    # Create viewer with raw depth enabled
    # Set min/max percentiles to something that would clip if enabled
    viewer = DepthMeshViewer(
        use_metric_depth=False,
        use_raw_depth=True,
        depth_min_percentile=10, # Should be ignored
        depth_max_percentile=90, # Should be ignored
        depth_scale=1.0
    )
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create depth with outliers
    depth_input = np.linspace(0, 100, 100).reshape(10, 10).astype(np.float32)
    # Min=0, Max=100.
    # If clamped (10-90%), range would be 10-90.
    
    mesh = viewer.create_mesh_from_depth(
        image, 
        depth_input, 
        subsample=1, 
        invert_depth=False,
        smooth_mesh=False
    )
    
    vertices = np.asarray(mesh.vertices)
    z_values = -vertices[:, 2] # Invert back to positive depth
    
    print(f"Z Range: {z_values.min():.2f} to {z_values.max():.2f}")
    
    # In raw mode, we expect full range (scaled by depth_scale * z_scale_factor)
    # z_scale_factor = max(w,h)*0.5 = 5.0
    # depth_scale = 1.0
    # z = depth * 5.0
    # Expected range: 0 to 500
    
    expected_max = 100.0 * 5.0
    
    if abs(z_values.max() - expected_max) < 1e-3:
        print("PASS: Raw depth range preserved (no clamping)")
    else:
        print(f"FAIL: Range mismatch. Expected max ~{expected_max}, got {z_values.max()}")

if __name__ == "__main__":
    test_metric_projection_coordinates()
    test_relative_raw_depth()
