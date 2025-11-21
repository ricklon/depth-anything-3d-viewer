
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
    
    # Downsample for plotting speed if needed
    if len(vertices) > 10000:
        indices = np.random.choice(len(vertices), 10000, replace=False)
        vertices = vertices[indices]
        
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    
    # Top View (X-Z plane)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1], s=1, c=vertices[:, 2], cmap='viridis')
    ax1.set_title("Top View (X-Z)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.set_zlabel("Y")
    ax1.view_init(elev=90, azim=-90) # Look down Y axis? No, Z is depth. 
    # Open3D: Y is Up/Down. Z is Depth. X is Left/Right.
    # Top view usually means looking down Y.
    
    # Side View (Y-Z plane)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1], s=1, c=vertices[:, 2], cmap='viridis')
    ax2.set_title("Side View (Y-Z)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_zlabel("Y")
    ax2.view_init(elev=0, azim=0) # Look along X?
    
    # Front View (X-Y plane)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1], s=1, c=vertices[:, 2], cmap='viridis')
    ax3.set_title("Front View (X-Y)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_zlabel("Y")
    ax3.view_init(elev=0, azim=-90) # Look along Z?
    
    plt.tight_layout()
    plt.savefig(output_filename)
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
            subsample=4, # Downsample for speed
            invert_depth=mode['invert_depth'],
            smooth_mesh=False
        )
        plot_mesh(mesh, mode['name'], str(Path(args.output_dir) / mode['filename']))

if __name__ == "__main__":
    main()
