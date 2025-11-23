import sys
import os
import cv2
import torch
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path

# Ensure parent Video-Depth-Anything is in path
parent_dir = os.path.dirname(os.path.abspath(__file__))
github_dir = os.path.dirname(parent_dir)

# Check for Video-Depth-Anything in sibling directory
video_depth_sibling = os.path.join(github_dir, 'Video-Depth-Anything')
if os.path.exists(os.path.join(video_depth_sibling, 'video_depth_anything')):
    sys.path.insert(0, video_depth_sibling)
elif os.path.exists(os.path.join(parent_dir, 'Video-Depth-Anything')):
    sys.path.insert(0, os.path.join(parent_dir, 'Video-Depth-Anything'))

try:
    from video_depth_anything.video_depth import VideoDepthAnything
    from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream
    from da3d.viewing.mesh import DepthMeshViewer
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    parser = argparse.ArgumentParser(description="Verify metric scale by measuring distances on a captured frame.")
    parser.add_argument('--camera-id', type=int, default=1, help='Camera ID (default: 1)')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--high-quality', action='store_true', help='Use high quality settings (960p, metric, SOR)')
    
    args = parser.parse_args()

    # Default settings
    max_res = 480
    sor_neighbors = 50
    sor_std_ratio = 1.0
    focal_length_x = 470.4
    
    if args.high_quality:
        print("Using High Quality Settings")
        max_res = 960
        sor_neighbors = 100
        sor_std_ratio = 0.5

    # Load Model
    print(f"Loading model {args.encoder}...")
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    # Assume checkpoints are in ./checkpoints
    checkpoint_path = Path(f'./checkpoints/video_depth_anything_{args.encoder}.pth')
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

    # Set resolution if high quality
    if args.high_quality:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "="*50)
    print("SCALE VERIFICATION TOOL")
    print("="*50)
    print("1. Align your camera with the object (caliper/mat).")
    print("2. Press SPACE to capture and measure.")
    print("3. Press Q to quit.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        cv2.imshow("Preview - Press SPACE to Capture", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord(' '):
            print("Capturing...")
            
            # Process Frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            
            # Resize for inference if needed
            if max(h, w) > max_res:
                scale = max_res / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame_rgb_resized = cv2.resize(frame_rgb, (new_w, new_h))
            else:
                frame_rgb_resized = frame_rgb

            # Use infer_video_depth_one for consistency with CLI
            depth = model.infer_video_depth_one(frame_rgb_resized, input_size=args.input_size, device=DEVICE)

            # Generate Point Cloud
            viewer = DepthMeshViewer(
                use_metric_depth=True,
                focal_length_x=focal_length_x,
                focal_length_y=focal_length_x, # Square pixels assumption
                metric_depth_scale=1.0, # Revert to 1.0 for baseline
                display_mode='pointcloud'
            )

            pcd = viewer.create_mesh_from_depth(
                frame_rgb_resized,
                depth,
                subsample=1,
                use_sor=True,
                sor_neighbors=sor_neighbors,
                sor_std_ratio=sor_std_ratio
            )

            print("\n" + "-"*30)
            print("OPENING 3D VIEW FOR MEASUREMENT")
            print("-"*30)
            print("INSTRUCTIONS:")
            print("1. Hold SHIFT + LEFT CLICK to pick a point.")
            print("2. Pick exactly TWO points (start and end of measurement).")
            print("3. Close the window to see the result.")
            print("-"*30)

            # Visualization with Picking
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name='Pick Two Points to Measure')
            vis.add_geometry(pcd)
            
            # Set view
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
            ctr.set_zoom(0.8)
            
            vis.run() # Blocks until closed
            vis.destroy_window()

            picked_indices = vis.get_picked_points()
            if len(picked_indices) >= 2:
                p1_idx = picked_indices[-2]
                p2_idx = picked_indices[-1]
                
                points = np.asarray(pcd.points)
                p1 = points[p1_idx]
                p2 = points[p2_idx]
                
                dist = np.linalg.norm(p2 - p1)
                
                print("\n" + "*"*40)
                print(f"MEASUREMENT RESULT:")
                print(f"Point 1: {p1}")
                print(f"Point 2: {p2}")
                print(f"Distance: {dist:.4f} meters")
                print(f"          {dist*100:.2f} cm")
                print(f"          {dist*1000:.2f} mm")
                print("*"*40 + "\n")
            else:
                print("\n[WARNING] You picked fewer than 2 points. Please try again.\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
