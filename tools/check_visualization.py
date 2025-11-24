"""
Quick script to understand what we're showing the Vision Agent
"""
import numpy as np
import cv2
from pathlib import Path
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

output_dir = Path('test_outputs_multicam/camera_0_logi_cam_c920e')

# Load data
image = cv2.imread(str(output_dir / 'captured_frame.png'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
depth = np.load(str(output_dir / 'depth_raw.npy'))

print("DEPTH DATA SUMMARY")
print("="*60)
print(f"Image: {image.shape} - {image.dtype}")
print(f"Depth: {depth.shape} - {depth.dtype}")
print(f"\nDepth range: [{depth.min():.4f}, {depth.max():.4f}]")
print(f"Depth mean: {depth.mean():.4f}")
print(f"Depth median: {np.median(depth):.4f}")

# Check mesh generation
from da3d.viewing.mesh import DepthMeshViewer

print("\n" + "="*60)
print("MESH GENERATION (subsample=2, as used in mode comparison)")
print("="*60)

viewer = DepthMeshViewer(
    use_metric_depth=True,
    focal_length_x=470.4,
    focal_length_y=470.4
)

mesh = viewer.create_mesh_from_depth(image, depth, subsample=2, invert_depth=False)
vertices = np.asarray(mesh.vertices)

print(f"Total vertices in mesh: {len(vertices):,}")
print("Vertices shown in plot_mesh: 10,000 (downsampled)")
print(f"Reduction factor: {len(vertices) / 10000:.1f}x")

print("\nVertex coordinates (meters):")
print(f"  X: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
print(f"  Y: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
print(f"  Z: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")

print("\n" + "="*60)
print("TUNED VISUALIZATION (subsample=1, full resolution)")
print("="*60)

viewer2 = DepthMeshViewer(
    use_metric_depth=True,
    focal_length_x=470.4,
    focal_length_y=470.4,
    display_mode="pointcloud"
)

pcd = viewer2.create_mesh_from_depth(image, depth, subsample=1, invert_depth=False)
points = np.asarray(pcd.points)

print(f"Total points in point cloud: {len(points):,}")
print("Points shown in plot_point_cloud: 30,000 (downsampled)")
print(f"Reduction factor: {len(points) / 30000:.1f}x")

print("\nPoint coordinates (meters):")
print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

print("\n" + "="*60)
print("VISUALIZATION ISSUE ANALYSIS")
print("="*60)
print("\nPROBLEM: Heavy downsampling for matplotlib plots")
print(f"  - Mode comparison: {len(vertices):,} → 10,000 points ({len(vertices)/10000:.1f}x reduction)")
print(f"  - Tuned viz: {len(points):,} → 30,000 points ({len(points)/30000:.1f}x reduction)")
print("\nThis explains 'not crisp' and 'hard to make out' feedback!")
print("\nSOLUTION OPTIONS:")
print("  1. Use Open3D screenshots instead of matplotlib")
print("  2. Increase matplotlib point limit")
print("  3. Show actual mesh surface, not just scattered points")
