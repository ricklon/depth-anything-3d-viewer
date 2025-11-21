
import numpy as np
import matplotlib.pyplot as plt
from da3d.viewing.mesh import DepthMeshViewer
import os

def verify_metric_mode():
    print("\n--- Verifying Metric Mode ---")
    
    # Test Case 1: Focal Length Sensitivity
    # If we double focal length, FOV decreases, so X/Y spread should decrease (for same depth)
    # X = (x - cx) * Z / fx
    viewer_low_f = DepthMeshViewer(use_metric_depth=True, focal_length_x=100, focal_length_y=100)
    viewer_high_f = DepthMeshViewer(use_metric_depth=True, focal_length_x=200, focal_length_y=200)
    
    depth_input = np.full((3, 3), 2.0, dtype=np.float32) # 2 meters
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    
    mesh_low = viewer_low_f.create_mesh_from_depth(image, depth_input, subsample=1, invert_depth=True)
    mesh_high = viewer_high_f.create_mesh_from_depth(image, depth_input, subsample=1, invert_depth=True)
    
    pts_low = np.asarray(mesh_low.vertices)
    pts_high = np.asarray(mesh_high.vertices)
    
    # Check X spread (max X - min X)
    spread_low = pts_low[:, 0].max() - pts_low[:, 0].min()
    spread_high = pts_high[:, 0].max() - pts_high[:, 0].min()
    
    print(f"Focal Length 100 -> X Spread: {spread_low:.4f}")
    print(f"Focal Length 200 -> X Spread: {spread_high:.4f}")
    
    if spread_high < spread_low:
        print("PASS: Higher focal length reduced X spread (Correct behavior)")
    else:
        print("FAIL: Focal length did not affect spread as expected")

    # Test Case 2: Metric Scale Accuracy
    # Input 2.0m -> Output Z should be -2.0m
    z_val = pts_low[0, 2]
    print(f"Input Depth 2.0m -> Output Z: {z_val:.4f}")
    if abs(z_val - (-2.0)) < 1e-5:
        print("PASS: Metric scale is correct (1.0 = 1 meter)")
    else:
        print("FAIL: Metric scale incorrect")

    return pts_low

def verify_relative_mode():
    print("\n--- Verifying Relative Mode ---")
    
    # Test Case 1: Depth Scaling
    # depth_scale=1.0 vs 2.0
    viewer_scale_1 = DepthMeshViewer(use_metric_depth=False, depth_scale=1.0)
    viewer_scale_2 = DepthMeshViewer(use_metric_depth=False, depth_scale=2.0)
    
    # Gradient depth 0 to 1
    depth_input = np.linspace(0, 1, 9).reshape(3, 3).astype(np.float32)
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    
    mesh_1 = viewer_scale_1.create_mesh_from_depth(image, depth_input, subsample=1, invert_depth=False)
    mesh_2 = viewer_scale_2.create_mesh_from_depth(image, depth_input, subsample=1, invert_depth=False)
    
    pts_1 = np.asarray(mesh_1.vertices)
    pts_2 = np.asarray(mesh_2.vertices)
    
    z_range_1 = pts_1[:, 2].max() - pts_1[:, 2].min()
    z_range_2 = pts_2[:, 2].max() - pts_2[:, 2].min()
    
    print(f"Scale 1.0 -> Z Range: {z_range_1:.4f}")
    print(f"Scale 2.0 -> Z Range: {z_range_2:.4f}")
    
    if abs(z_range_2 - 2 * z_range_1) < 1e-3:
        print("PASS: Depth scale is linear")
    else:
        print("FAIL: Depth scale is not linear")

    return pts_1

def generate_visual_plot(metric_pts, relative_pts):
    print("\n--- Generating Visual Plot ---")
    
    fig = plt.figure(figsize=(12, 6))
    
    # Plot Metric Mode (Top View)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(metric_pts[:, 0], metric_pts[:, 2], metric_pts[:, 1], c='b', marker='o')
    ax1.set_title("Metric Mode (Blue)\nShould be flat plane at Z=-2.0")
    ax1.set_xlabel("X (Right)")
    ax1.set_ylabel("Z (Back)") # Swapped for plotting convention
    ax1.set_zlabel("Y (Up)")
    
    # Plot Relative Mode (Side View)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(relative_pts[:, 0], relative_pts[:, 2], relative_pts[:, 1], c='r', marker='^')
    ax2.set_title("Relative Mode (Red)\nShould show gradient")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_zlabel("Y")
    
    output_path = "verification_plot.png"
    plt.savefig(output_path)
    print(f"Saved plot to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    metric_pts = verify_metric_mode()
    relative_pts = verify_relative_mode()
    generate_visual_plot(metric_pts, relative_pts)
