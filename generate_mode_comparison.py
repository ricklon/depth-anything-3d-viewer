
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

def plot_mesh(mesh, title, output_filename):
    vertices = np.asarray(mesh.vertices)
    
    # Downsample for plotting speed if needed (increased limit for better quality)
    max_points = 50000  # Increased from 10000
    if len(vertices) > max_points:
        indices = np.random.choice(len(vertices), max_points, replace=False)
        vertices = vertices[indices]
        
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    
    # Top View (X-Z plane) - Looking down from above
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1], s=0.5, c=vertices[:, 2], cmap='viridis', alpha=0.8)
    ax1.set_title("Top View (X-Z)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_zlabel("Y (m)")
    ax1.view_init(elev=90, azim=-90)
    # Set equal aspect ratio for proper metric visualization
    ax1.set_box_aspect([1, 1, 0.5])
    
    # Side View (Y-Z plane) - Looking from the side
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1], s=0.5, c=vertices[:, 2], cmap='viridis', alpha=0.8)
    ax2.set_title("Side View (Y-Z)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_zlabel("Y (m)")
    ax2.view_init(elev=0, azim=0)
    ax2.set_box_aspect([1, 1, 0.5])
    
    # Front View (X-Y plane) - Looking at the scene from camera position
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1], s=0.5, c=vertices[:, 2], cmap='viridis', alpha=0.8)
    ax3.set_title("Front View (X-Y)")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_zlabel("Y (m)")
    ax3.view_init(elev=0, azim=-90)
    ax3.set_box_aspect([1, 1, 0.5])
    
    # Remove grid lines for cleaner visualization
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)  # Increase DPI for better quality
    plt.close(fig)
    print(f"Saved {output_filename}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate mode comparison")
    parser.add_argument("--input-dir", type=str, default="./test_outputs", help="Input directory")
    parser.add_argument("--output-dir", type=str, default="./test_outputs", help="Output directory")
    args = parser.parse_args()

    image, depth = load_test_data(args.input_dir)
    if image is None:
        return
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    modes = [
        {
            "name": "Metric Mode (Default)",
            "filename": "comparison_metric.png",
            "params": {
                "use_metric_depth": True,
                "focal_length_x": 470.4,
                "focal_length_y": 470.4,
                "metric_depth_scale": 1.0
            },
            "invert_depth": False # Model outputs disparity (1/depth), so we want False to convert it to depth
        },
        {
            "name": "Relative Mode (Default)",
            "filename": "comparison_relative.png",
            "params": {
                "use_metric_depth": False,
                "depth_scale": 0.5
            },
            "invert_depth": False
        },
        {
            "name": "Metric Mode (High FOV / Low Focal Length)",
            "filename": "comparison_metric_high_fov.png",
            "params": {
                "use_metric_depth": True,
                "focal_length_x": 200.0,
                "focal_length_y": 200.0,
                "metric_depth_scale": 1.0
            },
            "invert_depth": False
        },
        {
            "name": "Metric Mode (Low FOV / High Focal Length)",
            "filename": "comparison_metric_low_fov.png",
            "params": {
                "use_metric_depth": True,
                "focal_length_x": 1000.0,
                "focal_length_y": 1000.0,
                "metric_depth_scale": 1.0
            },
            "invert_depth": False
        },
        {
            "name": "Metric Mode (Inverted - Incorrect)",
            "filename": "comparison_metric_inverted.png",
            "params": {
                "use_metric_depth": True,
                "focal_length_x": 470.4,
                "focal_length_y": 470.4,
                "metric_depth_scale": 1.0
            },
            "invert_depth": True # This treats disparity as distance (Wrong)
        }
    ]

    for mode in modes:
        print(f"Processing {mode['name']}...")
        viewer = DepthMeshViewer(**mode['params'])
        mesh = viewer.create_mesh_from_depth(
            image, 
            depth, 
            subsample=1, # Full resolution for best quality
            invert_depth=mode['invert_depth'],
            smooth_mesh=False
        )
        plot_mesh(mesh, mode['name'], str(Path(args.output_dir) / mode['filename']))

if __name__ == "__main__":
    main()
