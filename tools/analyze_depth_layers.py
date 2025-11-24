import sys
import os
import cv2
import torch
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure parent Video-Depth-Anything is in path
# Ensure parent Video-Depth-Anything is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
github_dir = os.path.dirname(project_root)

# Add project root to sys.path so we can import da3d
sys.path.append(project_root)

# Check for Video-Depth-Anything in sibling directory
video_depth_sibling = os.path.join(github_dir, 'Video-Depth-Anything')
if os.path.exists(os.path.join(video_depth_sibling, 'video_depth_anything')):
    sys.path.insert(0, video_depth_sibling)
elif os.path.exists(os.path.join(project_root, 'Video-Depth-Anything')):
    sys.path.insert(0, os.path.join(project_root, 'Video-Depth-Anything'))

try:
    from video_depth_anything.video_depth import VideoDepthAnything
    from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream
    from da3d.viewing.mesh import DepthMeshViewer
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def capture_and_analyze(args):
    # Load Model
    print(f"Loading model {args.encoder}...")
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    checkpoint_path = Path(f'./checkpoints/metric_video_depth_anything_{args.encoder}.pth')
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    model = VideoDepthAnythingStream(**model_configs[args.encoder])
    model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()

    # Open Camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nPress SPACE to capture a frame for analysis...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Preview", frame)
        key = cv2.waitKey(1)
        if key == ord(' '):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    print("Capturing and processing...")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Inference
    with torch.no_grad():
        depth = model.infer_video_depth_one(frame_rgb, input_size=518, device=DEVICE)

    # Save data for analysis
    np.save('debug_depth.npy', depth)
    cv2.imwrite('debug_rgb.png', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    
    # Generate visualizations
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(frame_rgb)
    plt.title("RGB Image")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(depth, cmap='inferno')
    plt.title("Raw Depth Map")
    plt.colorbar(label='Depth (meters)')
    plt.axis('off')
    
    plt.subplot(133)
    plt.hist(depth.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title("Depth Distribution")
    plt.xlabel("Depth (m)")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig('depth_analysis.png')
    print("Saved analysis to depth_analysis.png")

    # Generate multi-view 3D visualization
    viewer = DepthMeshViewer(
        use_metric_depth=True,
        focal_length_x=470.4, # Default
        focal_length_y=470.4,
        metric_depth_scale=1.0,
        display_mode='pointcloud'
    )
    
    pcd = viewer.create_mesh_from_depth(
        frame_rgb,
        depth,
        subsample=2,
        use_sor=True,
        sor_neighbors=50,
        sor_std_ratio=1.0
    )
    
    # Save point cloud
    o3d.io.write_point_cloud("debug_pcd.ply", pcd)
    print("Saved point cloud to debug_pcd.ply")
    
    # Render views
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=600)
    vis.add_geometry(pcd)
    
    # Front View
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("view_front.png")
    
    # Side View (Right)
    ctr.set_front([1, 0, 0])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("view_side.png")
    
    # Top View
    ctr.set_front([0, -1, 0])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, -1])
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("view_top.png")
    
    vis.destroy_window()
    print("Saved views: view_front.png, view_side.png, view_top.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-id', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='vits')
    args = parser.parse_args()
    capture_and_analyze(args)
