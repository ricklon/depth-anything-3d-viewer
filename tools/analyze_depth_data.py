import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Load test data
output_dir = Path('test_outputs_multicam/camera_0_logi_cam_c920e')
image = cv2.imread(str(output_dir / 'captured_frame.png'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
depth = np.load(str(output_dir / 'depth_raw.npy'))

print("="*80)
print("DATA ANALYSIS")
print("="*80)

print(f"\nImage shape: {image.shape}")
print(f"Depth shape: {depth.shape}")
print(f"Image dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
print(f"Depth dtype: {depth.dtype}, range: [{depth.min():.6f}, {depth.max():.6f}]")

print("\nDepth statistics:")
print(f"  Mean: {depth.mean():.6f}")
print(f"  Median: {np.median(depth):.6f}")
print(f"  Std: {depth.std():.6f}")
print(f"  5th percentile: {np.percentile(depth, 5):.6f}")
print(f"  95th percentile: {np.percentile(depth, 95):.6f}")

# Check if depth is inverse depth (disparity) or actual depth
print("\nDepth interpretation:")
print("  If this is DISPARITY (inverse depth): higher values = CLOSER")
print("  If this is DEPTH: higher values = FARTHER")
print("  Depth-Anything V2 outputs: INVERSE DEPTH (disparity)")

# Visualize depth distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original image
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Raw depth map
im1 = axes[0, 1].imshow(depth, cmap='viridis')
axes[0, 1].set_title("Raw Depth Map (as loaded)")
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1])

# Inverted depth map
im2 = axes[0, 2].imshow(depth.max() - depth, cmap='viridis')
axes[0, 2].set_title("Inverted Depth Map")
axes[0, 2].axis('off')
plt.colorbar(im2, ax=axes[0, 2])

# Depth histogram
axes[1, 0].hist(depth.flatten(), bins=100, edgecolor='black')
axes[1, 0].set_title("Depth Value Distribution")
axes[1, 0].set_xlabel("Depth Value")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].grid(True, alpha=0.3)

# Center crop analysis (where objects typically are)
h, w = depth.shape
center_crop = depth[h//4:3*h//4, w//4:3*w//4]
axes[1, 1].hist(center_crop.flatten(), bins=100, edgecolor='black', color='orange')
axes[1, 1].set_title("Center Region Depth Distribution")
axes[1, 1].set_xlabel("Depth Value")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].grid(True, alpha=0.3)

# Depth gradient (edges)
depth_grad_x = np.abs(np.gradient(depth, axis=1))
depth_grad_y = np.abs(np.gradient(depth, axis=0))
depth_grad = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
im3 = axes[1, 2].imshow(depth_grad, cmap='hot')
axes[1, 2].set_title("Depth Gradients (Edges)")
axes[1, 2].axis('off')
plt.colorbar(im3, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('depth_data_analysis.png', dpi=150)
print("\nSaved depth_data_analysis.png")

# Now let's check what the 3D mesh looks like
print("\n" + "="*80)
print("3D MESH GENERATION ANALYSIS")
print("="*80)

from da3d.viewing.mesh import DepthMeshViewer

# Test with metric depth (what we're using for evaluation)
viewer = DepthMeshViewer(
    use_metric_depth=True,
    focal_length_x=470.4,
    focal_length_y=470.4,
    metric_depth_scale=1.0
)

try:
    mesh = viewer.create_mesh_from_depth(
        image, 
        depth, 
        subsample=2,
        invert_depth=False,
        smooth_mesh=False
    )
except Exception as e:
    print(f"Error creating mesh: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

vertices = np.asarray(mesh.vertices)
print("\nMesh statistics:")
print(f"  Vertices: {len(vertices)}")
print(f"  Triangles: {len(mesh.triangles)}")

print("\nVertex coordinates (X, Y, Z):")
print(f"  X range: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}] meters")
print(f"  Y range: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}] meters")
print(f"  Z range: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}] meters")

print("\nCoordinate system:")
print("  X: Left (-) to Right (+)")
print("  Y: Top (-) to Bottom (+) [image coords, inverted in viewer]")
print("  Z: Camera (0) to Scene (+)")

# Check if Z values make sense for metric depth
print("\nMetric depth interpretation:")
print(f"  Z mean: {vertices[:, 2].mean():.3f}m")
print(f"  Z median: {np.median(vertices[:, 2]):.3f}m")
print("  Expected range for typical webcam scene: 0.5m - 5m")

# Analyze the bounding box
bbox_size = [
    vertices[:, 0].max() - vertices[:, 0].min(),
    vertices[:, 1].max() - vertices[:, 1].min(),
    vertices[:, 2].max() - vertices[:, 2].min()
]
print("\nBounding box size:")
print(f"  Width (X): {bbox_size[0]:.3f}m")
print(f"  Height (Y): {bbox_size[1]:.3f}m")
print(f"  Depth (Z): {bbox_size[2]:.3f}m")

# Calculate aspect ratio
aspect_ratio = bbox_size[0] / bbox_size[1]
print(f"\nAspect ratio (W/H): {aspect_ratio:.2f}")
print(f"  Image aspect ratio: {image.shape[1] / image.shape[0]:.2f}")
print("  Should be similar if projection is correct")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
