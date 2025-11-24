import cv2
import torch
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Ensure parent Video-Depth-Anything is in path
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
    from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def capture_test_case(args):
    output_dir = Path("tests/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
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

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n==================================================")
    print("CAPTURE TEST CASE")
    print("==================================================")
    print("1. Align your camera with a test object (e.g. caliper, box).")
    print("2. Press SPACE to capture.")
    print("3. Press Q to quit.")
    print("==================================================")
    
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

    # Save data
    img_path = output_dir / "test_image.jpg"
    depth_path = output_dir / "test_depth.npy"
    
    cv2.imwrite(str(img_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    np.save(str(depth_path), depth)
    
    print(f"Saved test image to {img_path}")
    print(f"Saved test depth to {depth_path}")
    print("\nNow run 'uv run tests/visual_test.py' to generate the report.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-id', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='vits')
    args = parser.parse_args()
    capture_test_case(args)
