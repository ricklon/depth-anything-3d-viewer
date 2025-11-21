
import numpy as np
import matplotlib.pyplot as plt
from da3d.viewing.mesh import DepthMeshViewer
import cv2
import os
from pathlib import Path

def load_test_data(output_dir='./test_outputs'):
    image_path = Path(output_dir) / 'captured_frame.png'
    depth_path = Path(output_dir) / 'depth_raw.npy'
    
    if not image_path.exists() or not depth_path.exists():
        print(f"Test data not found in {output_dir}. Please run test_webcam_single_frame.py first.")
        return None, None

    print(f"Loading test data from {output_dir}...")
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = np.load(str(depth_path))
    
    return image, depth

def plot_point_cloud(mesh, title, output_filename):
    # Extract points and colors
    points = np.asarray(mesh.points)
    colors = np.asarray(mesh.colors)
    
    # Downsample for plotting speed and clarity (too many points clutter the plot)
    # For 640x480 = 300k points. Let's keep ~30k for the plot.
    if len(points) > 30000:
        indices = np.random.choice(len(points), 30000, replace=False)
        points = points[indices]
        colors = colors[indices]
        
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Isometric View
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with actual RGB colors
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=1, c=colors)
    
    ax.set_title("Metric Point Cloud (Best Quality)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    ax.set_zlabel("Y (meters)")
    
    # Set equal aspect ratio for true metric visualization
    # Matplotlib 3D doesn't have 'equal' aspect ratio easily, but we can set limits
    max_range = np.array([points[:,0].max()-points[:,0].min(), 
                          points[:,1].max()-points[:,1].min(), 
                          points[:,2].max()-points[:,2].min()]).max() / 2.0
    
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range) # Z is mapped to Y axis in plot for top-down feel? No, let's keep standard.
    ax.set_zlim(mid_y - max_range, mid_y + max_range)
    
    # View angle: Elevated and slightly rotated
    ax.view_init(elev=20, azim=-45)
    
    # Invert Y axis (image coordinates Y is down, but 3D world Y is usually Up)
    # Our mesh generation already handles -y.
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)
    print(f"Saved {output_filename}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate tuned visualization")
    parser.add_argument("--input-dir", type=str, default="./test_outputs", help="Input directory")
    parser.add_argument("--output-dir", type=str, default="./test_outputs", help="Output directory")
    args = parser.parse_args()

    image, depth = load_test_data(args.input_dir)
    if image is None:
        return
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Image resolution: {image.shape[1]}x{image.shape[0]}")

    # Optimal parameters for 640x480
    params = {
        "use_metric_depth": True,
        "focal_length_x": 470.4, # Correct for 640 width
        "focal_length_y": 470.4,
        "metric_depth_scale": 1.0,
        "display_mode": "pointcloud" # Point clouds often look cleaner for raw depth
    }

    print("Generating mesh with optimal parameters...")
    viewer = DepthMeshViewer(**params)
    
    # Create point cloud
    # We use invert_depth=False because the model outputs disparity (1/depth)
    # and we want the viewer to convert it to depth (meters).
    pcd = viewer.create_mesh_from_depth(
        image, 
        depth, 
        subsample=1, # Full resolution for point cloud
        invert_depth=False,
        smooth_mesh=False
    )
    
    plot_point_cloud(pcd, "Tuned Metric Point Cloud (640x480)", str(Path(args.output_dir) / "tuned_visualization.png"))

if __name__ == "__main__":
    main()
