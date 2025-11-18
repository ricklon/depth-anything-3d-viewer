#!/usr/bin/env python3
"""
Capture a single frame from webcam with depth map for testing.
Saves RGB image and depth map as numpy array.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add Video-Depth-Anything to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_depth_path = os.path.join(parent_dir, 'Video-Depth-Anything')
if os.path.exists(os.path.join(video_depth_path, 'video_depth_anything')):
    sys.path.insert(0, video_depth_path)

from video_depth_anything.video_depth_stream import VideoDepthAnythingStream


def capture_frame(
    camera_id: int = 0,
    encoder: str = 'vits',
    metric: bool = True,
    output_prefix: str = 'frame',
    checkpoint_dir: str = 'checkpoints'
):
    """
    Capture a single frame with depth estimation.

    Args:
        camera_id: Camera device ID
        encoder: Model size (vits, vitb, vitl)
        metric: Use metric depth model
        output_prefix: Prefix for output files
        checkpoint_dir: Directory containing model checkpoints
    """
    print(f"Initializing depth model ({encoder})...")

    # Determine checkpoint path
    model_type = f"metric_video_depth_anything_{encoder}.pth" if metric else f"video_depth_anything_{encoder}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, model_type)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Run download_metric_weights.ps1 to download model weights")
        return

    # Initialize depth model
    depth_stream = VideoDepthAnythingStream(
        encoder=encoder,
        checkpoint_path=checkpoint_path,
        max_depth=20.0 if metric else None,
        device='cuda'
    )

    # Open webcam
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    print("\nPress SPACE to capture, ESC to quit")

    captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get depth map
        depth = depth_stream.infer_depth(frame)

        # Normalize depth for visualization
        depth_vis = depth.copy()
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        # Show preview
        display = np.hstack([frame, depth_colored])
        cv2.imshow('Capture Frame - SPACE to capture, ESC to quit', display)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            # Save files
            timestamp = Path(output_prefix).stem

            rgb_path = f"{output_prefix}_rgb.jpg"
            depth_path = f"{output_prefix}_depth.npy"
            depth_vis_path = f"{output_prefix}_depth_vis.png"

            cv2.imwrite(rgb_path, frame)
            np.save(depth_path, depth)
            cv2.imwrite(depth_vis_path, depth_colored)

            print(f"\nCaptured!")
            print(f"  RGB image: {rgb_path}")
            print(f"  Depth data: {depth_path}")
            print(f"  Depth visualization: {depth_vis_path}")
            print(f"  Depth range: {depth.min():.3f}m to {depth.max():.3f}m (mean: {depth.mean():.3f}m)")
            print(f"  Image size: {frame.shape[1]}x{frame.shape[0]}")

            captured = True
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured:
        print(f"\nTo view in 3D:")
        print(f"  uv run da3d view3d {rgb_path} {depth_path} --metric --focal-length-x 476 --focal-length-y 476")
        print("\nTo compare different focal lengths:")
        print(f"  # Wrong focal length (stretched):")
        print(f"  uv run da3d view3d {rgb_path} {depth_path} --metric --focal-length-x 1430 --focal-length-y 1430")
        print(f"  # Correct focal length:")
        print(f"  uv run da3d view3d {rgb_path} {depth_path} --metric --focal-length-x 476 --focal-length-y 476")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Capture webcam frame with depth map')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'],
                        help='Model size')
    parser.add_argument('--metric', action='store_true', help='Use metric depth model')
    parser.add_argument('--output', type=str, default='test_frame',
                        help='Output prefix (default: test_frame)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')

    args = parser.parse_args()

    capture_frame(
        camera_id=args.camera_id,
        encoder=args.encoder,
        metric=args.metric,
        output_prefix=args.output,
        checkpoint_dir=args.checkpoint_dir
    )
