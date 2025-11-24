import numpy as np
import open3d as o3d
import cv2
import argparse
from pathlib import Path
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from da3d.viewing.mesh import DepthMeshViewer

def load_test_data(output_dir):
    image_path = Path(output_dir) / 'captured_frame.png'
    depth_path = Path(output_dir) / 'depth_raw.npy'
    
    if not image_path.exists() or not depth_path.exists():
        print(f"Test data not found in {output_dir}")
        return None, None

    print(f"Loading test data from {output_dir}...")
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = np.load(str(depth_path))
    
    return image, depth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="./test_outputs_multicam/camera_0_logi_cam_c920e")
    parser.add_argument("--output-dir", type=str, default="./test_outputs_multicam/camera_0_logi_cam_c920e")
    args = parser.parse_args()
    
    image, depth = load_test_data(args.input_dir)
    if image is None:
        return
        
    height, width = image.shape[:2]
    
    # Create viewer
    viewer = DepthMeshViewer(
        use_metric_depth=True,
        focal_length_x=470.4,
        focal_length_y=470.4,
        metric_depth_scale=1.0,
        display_mode='pointcloud' # Safer than mesh
    )
    
    # Create geometry
    pcd = viewer.create_mesh_from_depth(
        image, 
        depth, 
        subsample=1,
        invert_depth=False,
        smooth_mesh=False,
        use_sor=True
    )
    
    print("Creating visible window...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Vision Agent Capture", width=width, height=height, visible=True)
    vis.add_geometry(pcd)
    
    # Set view
    ctr = vis.get_view_control()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 470.4, 470.4, width/2-0.5, height/2-0.5)
    extrinsic = np.eye(4)
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic
    camera_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    
    output_path = str(Path(args.output_dir) / "open3d_camera_view.png")
    
    def capture_image(vis):
        print(f"Capturing to {output_path}...")
        vis.capture_screen_image(output_path, do_render=True)
        return False

    vis.register_key_callback(ord("C"), capture_image)
    
    print("\n=== Interactive Verification ===")
    print(f"1. Press 'C' to capture the view to {output_path}")
    print("2. Press 'Q' to close the window")
    print("==============================\n")
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
