import argparse
import os
import time
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import sys

# Ensure parent Video-Depth-Anything is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
github_dir = os.path.dirname(project_root)

# Add project root to sys.path
sys.path.append(project_root)

video_depth_sibling = os.path.join(github_dir, 'Video-Depth-Anything')
if os.path.exists(os.path.join(video_depth_sibling, 'video_depth_anything')):
    sys.path.insert(0, video_depth_sibling)

try:
    from da3d.viewing.mesh import DepthMeshViewer
except ImportError:
    # If running from root, add current dir to path
    sys.path.insert(0, os.getcwd())
    from da3d.viewing.mesh import DepthMeshViewer

def run_visual_test(args):
    """
    Runs a visual test by:
    1. Capturing/Loading a test frame
    2. Generating a 3D mesh with current settings
    3. Rendering views (Front, Side, Top)
    4. Saving these views for Vision Agent review
    """
    print(f"Running visual test for commit: {args.commit_id}")
    output_dir = Path(args.output_dir) / args.commit_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Test Data
    # We use a static test image to ensure consistency across runs
    test_image_path = Path("tests/data/test_image.jpg")
    test_depth_path = Path("tests/data/test_depth.npy")
    
    if not test_image_path.exists() or not test_depth_path.exists():
        print("Test data not found. Please run 'python tests/generate_test_data.py' first.")
        # Create dummy test data if needed for first run
        create_dummy_test_data(test_image_path, test_depth_path)

    image = cv2.imread(str(test_image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = np.load(str(test_depth_path))

    # 2. Generate Mesh
    print("Generating 3D mesh...")
    viewer = DepthMeshViewer(
        use_metric_depth=True,
        focal_length_x=470.4,
        focal_length_y=470.4,
        metric_depth_scale=1.0,
        display_mode='pointcloud'
    )
    
    # Apply current settings (simulating CLI defaults)
    pcd = viewer.create_mesh_from_depth(
        image,
        depth,
        subsample=2,
        use_sor=True,
        sor_neighbors=50,
        sor_std_ratio=1.0,
        smooth_mesh=True # Enable smoothing as per recent changes
    )

    # 3. Render Views
    print("Rendering views...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=600)
    vis.add_geometry(pcd)
    
    # Helper to capture view
    def capture_view(name, front, lookat, up):
        ctr = vis.get_view_control()
        ctr.set_front(front)
        ctr.set_lookat(lookat)
        ctr.set_up(up)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(output_dir / f"{name}.png"))

    capture_view("view_front", [0, 0, -1], [0, 0, 0], [0, 1, 0])
    capture_view("view_side", [1, 0, 0], [0, 0, 0], [0, 1, 0])
    capture_view("view_top", [0, -1, 0], [0, 0, 0], [0, 0, -1])
    
    vis.destroy_window()
    print(f"Views saved to {output_dir}")
    
    # 4. Generate Report
    generate_report(output_dir, args.commit_id)

def create_dummy_test_data(img_path, depth_path):
    print("Creating dummy test data...")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    # Create a gradient image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(480):
        img[i, :, :] = (i // 2, 100, 255 - i // 2)
    cv2.imwrite(str(img_path), img)
    
    # Create a gradient depth map (planar ramp)
    depth = np.linspace(0.5, 5.0, 640 * 480).reshape(480, 640).astype(np.float32)
    np.save(str(depth_path), depth)

def generate_report(output_dir, commit_id):
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("# Visual Test Report\n")
        f.write(f"**Commit:** {commit_id}\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Rendered Views\n")
        f.write("| Front | Side | Top |\n")
        f.write("|-------|------|-----|\n")
        f.write("| ![Front](view_front.png) | ![Side](view_side.png) | ![Top](view_top.png) |\n")
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit-id', type=str, default="current", help="Commit ID or tag for labeling")
    parser.add_argument('--output-dir', type=str, default="test_results", help="Directory to save results")
    args = parser.parse_args()
    run_visual_test(args)
