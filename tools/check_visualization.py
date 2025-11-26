"""
Quick script to understand what we're showing the Vision Agent
"""
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

output_dir = Path('test_outputs_multicam/camera_0_logi_cam_c920e')
# Fallback if specific dir doesn't exist
if not output_dir.exists():
    output_dir = Path('test_outputs')

# Load data
image_path = output_dir / 'captured_frame.png'
depth_path = output_dir / 'depth_raw.npy'

if not image_path.exists() or not depth_path.exists():
    print(f"Error: Could not find test data in {output_dir}")
    sys.exit(1)

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
depth = np.load(str(depth_path))

print("DEPTH DATA SUMMARY")
print("="*60)
print(f"Image: {image.shape} - {image.dtype}")
print(f"Depth: {depth.shape} - {depth.dtype}")
print(f"\nDepth range: [{depth.min():.4f}, {depth.max():.4f}]")
print(f"Depth mean: {depth.mean():.4f}")
print(f"Depth median: {np.median(depth):.4f}")
print(f"Depth std dev: {np.std(depth):.4f}")

# Percentiles
percentiles = [1, 5, 25, 50, 75, 95, 99]
p_vals = np.percentile(depth, percentiles)
print("\nPercentiles:")
for p, val in zip(percentiles, p_vals):
    print(f"  {p}%: {val:.4f}")

# Generate Histogram
print("\nGenerating depth distribution histogram...")
plt.figure(figsize=(10, 6))
plt.hist(depth.flatten(), bins=100, color='blue', alpha=0.7)
plt.title("Depth Value Distribution (Raw Output)")
plt.xlabel("Depth Value (Inverse Depth/Disparity)")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
hist_path = output_dir / 'depth_histogram.png'
plt.savefig(hist_path)
print(f"Saved histogram to {hist_path}")
plt.close()

# Check mesh generation
from da3d.viewing.mesh import DepthMeshViewer

print("\n" + "="*60)
print("MESH GENERATION ANALYSIS")
print("="*60)

viewer = DepthMeshViewer(
    use_metric_depth=True,
    focal_length_x=470.4,
    focal_length_y=470.4
)

# Analyze mesh statistics
mesh = viewer.create_mesh_from_depth(image, depth, subsample=2, invert_depth=False)
vertices = np.asarray(mesh.vertices)

print(f"Total vertices in mesh: {len(vertices):,}")
print("\nVertex coordinates (meters):")
print(f"  X: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
print(f"  Y: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
print(f"  Z: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")

# Check for outliers (Z-score)
z_vals = vertices[:, 2]
z_mean = np.mean(z_vals)
z_std = np.std(z_vals)
outliers = np.sum(np.abs(z_vals - z_mean) > 3 * z_std)
print(f"\nPotential outliers (>3 std dev): {outliers} ({outliers/len(z_vals)*100:.2f}%)")
