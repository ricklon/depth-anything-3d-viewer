
import numpy as np
import cv2
import os
from pathlib import Path
import sys
import open3d as o3d
import matplotlib.pyplot as plt

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from da3d.viewing.mesh import DepthMeshViewer

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

def capture_view(vis, view_name):
    """Capture a specific view from the visualizer."""
    ctr = vis.get_view_control()
    
    if view_name == "Top":
        # Top View (X-Z plane) - Looking down Y axis
        ctr.set_front([0, -1, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, -1])
        ctr.set_zoom(0.7)
    elif view_name == "Side":
        # Side View (Y-Z plane) - Looking down X axis
        ctr.set_front([-1, 0, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.7)
    elif view_name == "Front":
        # Front View (X-Y plane) - Looking down -Z axis
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.7)
        
    # Let the renderer update
    for _ in range(5):
        vis.poll_events()
        vis.update_renderer()
        
    # Capture image
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Add label
    cv2.putText(image, f"{view_name} View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return image

def plot_mesh_open3d(mesh, title, output_filename):
    """Render mesh using Open3D for high quality output."""
    print(f"Rendering {title} with Open3D...")
    
    # Initialize visualizer (offscreen if possible, but standard works on Windows)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600, visible=False)
    
    # Add geometry
    vis.add_geometry(mesh)
    
    # Set background to light gray for better visibility
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.9, 0.9, 0.9])
    opt.point_size = 2.0
    
    # Capture views
    img_top = capture_view(vis, "Top")
    img_side = capture_view(vis, "Side")
    img_front = capture_view(vis, "Front")
    
    vis.destroy_window()
    
    # Combine images horizontally
    combined = np.hstack([img_top, img_side, img_front])
    
    # Add title
    # Add a white bar at the top for the title
    h, w, c = combined.shape
    title_bar = np.ones((60, w, c), dtype=np.uint8) * 255
    cv2.putText(title_bar, title, (w//2 - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    final_image = np.vstack([title_bar, combined])
    
    cv2.imwrite(output_filename, final_image)
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
                "metric_depth_scale": 1.0,
                "display_mode": "pointcloud" # Use pointcloud for crisper visualization
            },
            "invert_depth": False
        },
        {
            "name": "Relative Mode (Default)",
            "filename": "comparison_relative.png",
            "params": {
                "use_metric_depth": False,
                "depth_scale": 0.5,
                "display_mode": "pointcloud"
            },
            "invert_depth": False
        },
        {
            "name": "Metric Mode (High FOV)",
            "filename": "comparison_metric_high_fov.png",
            "params": {
                "use_metric_depth": True,
                "focal_length_x": 200.0,
                "focal_length_y": 200.0,
                "metric_depth_scale": 1.0,
                "display_mode": "pointcloud"
            },
            "invert_depth": False
        },
        {
            "name": "Metric Mode (Low FOV)",
            "filename": "comparison_metric_low_fov.png",
            "params": {
                "use_metric_depth": True,
                "focal_length_x": 1000.0,
                "focal_length_y": 1000.0,
                "metric_depth_scale": 1.0,
                "display_mode": "pointcloud"
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
                "metric_depth_scale": 1.0,
                "display_mode": "pointcloud"
            },
            "invert_depth": True
        }
    ]

    for mode in modes:
        try:
            print(f"Processing {mode['name']}...")
            viewer = DepthMeshViewer(**mode['params'])
            mesh = viewer.create_mesh_from_depth(
                image, 
                depth, 
                subsample=1, # Full resolution for best quality
                invert_depth=mode['invert_depth'],
                smooth_mesh=False
            )
            plot_mesh_open3d(mesh, mode['name'], str(Path(args.output_dir) / mode['filename']))
        except Exception as e:
            print(f"Error processing {mode['name']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
