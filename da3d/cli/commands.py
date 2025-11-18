#!/usr/bin/env python3
# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Command-line interface for Video Depth Anything.
Provides easy access to depth estimation for videos and webcam input.
"""

import argparse
import os
import sys
import numpy as np
import torch
import cv2
import time
from pathlib import Path

# Video-Depth-Anything imports (from original repo - must be in PYTHONPATH)
from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream
from utils.dc_utils import read_video_frames, save_video

# Da3d package imports (from this package)
from da3d.projection import DepthProjector, InteractiveParallaxController
from da3d.viewing import DepthMeshViewer, view_depth_3d, RealTime3DViewer


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def video_command(args):
    """Process a video file for depth estimation."""
    print(f"Processing video: {args.input}")
    print(f"Model: {args.encoder}, Metric: {args.metric}, Streaming: {args.streaming}")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cpu':
        print("Warning: CUDA not available. Running on CPU will be very slow.")

    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
    checkpoint_path = Path(args.checkpoints_dir) / f'{checkpoint_name}_{args.encoder}.pth'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Please download checkpoints using: bash get_weights.sh")
        sys.exit(1)

    # Load model
    if args.streaming:
        video_depth_anything = VideoDepthAnythingStream(**MODEL_CONFIGS[args.encoder])
    else:
        video_depth_anything = VideoDepthAnything(**MODEL_CONFIGS[args.encoder], metric=args.metric)

    video_depth_anything.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # Process video
    if args.streaming:
        process_video_streaming(args, video_depth_anything, DEVICE)
    else:
        process_video_batch(args, video_depth_anything, DEVICE)

    print(f"[OK] Processing complete! Output saved to: {args.output_dir}")


def process_video_batch(args, model, device):
    """Process video in batch mode."""
    frames, target_fps = read_video_frames(args.input, args.max_len, args.target_fps, args.max_res)
    print(f"Loaded {len(frames)} frames at {target_fps} fps")

    depths, fps = model.infer_video_depth(frames, target_fps, input_size=args.input_size, device=device, fp32=args.fp32)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = Path(args.input).stem

    if not args.depth_only:
        processed_video_path = Path(args.output_dir) / f'{video_name}_src.mp4'
        save_video(frames, str(processed_video_path), fps=fps)
        print(f"  Saved source video: {processed_video_path}")

    depth_vis_path = Path(args.output_dir) / f'{video_name}_depth.mp4'
    save_video(depths, str(depth_vis_path), fps=fps, is_depths=True, grayscale=args.grayscale)
    print(f"  Saved depth video: {depth_vis_path}")

    if args.save_npz:
        depth_npz_path = Path(args.output_dir) / f'{video_name}_depths.npz'
        np.savez_compressed(depth_npz_path, depths=depths)
        print(f"  Saved depth data: {depth_npz_path}")

    if args.save_exr:
        save_exr_depths(depths, args.output_dir, video_name)

    if args.metric:
        save_point_clouds(frames, depths, args)


def process_video_streaming(args, model, device):
    """Process video in streaming mode."""
    cap = cv2.VideoCapture(args.input)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    height, width = original_height, original_width
    if args.max_res > 0 and max(original_height, original_width) > args.max_res:
        scale = args.max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)

    fps = original_fps if args.target_fps < 0 else args.target_fps
    stride = max(round(original_fps / fps), 1)

    depths = []
    frame_count = 0
    start = time.time()

    print(f"Processing {total_frames} frames (sampling every {stride} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (args.max_len > 0 and frame_count >= args.max_len):
            break
        if frame_count % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if args.max_res > 0 and max(original_height, original_width) > args.max_res:
                frame = cv2.resize(frame, (width, height))

            depth = model.infer_video_depth_one(frame, input_size=args.input_size, device=device, fp32=args.fp32)
            depths.append(depth)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Progress: {frame_count}/{total_frames} frames")

    cap.release()
    elapsed = time.time() - start
    print(f"Processing took {elapsed:.2f}s ({len(depths)/elapsed:.2f} fps)")

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = Path(args.input).stem
    depth_vis_path = Path(args.output_dir) / f'{video_name}_depth_streaming.mp4'
    depths = np.stack(depths, axis=0)
    save_video(depths, str(depth_vis_path), fps=fps, is_depths=True, grayscale=args.grayscale)
    print(f"  Saved depth video: {depth_vis_path}")


def webcam_command(args):
    """Real-time depth estimation from webcam."""
    print(f"Starting webcam depth estimation...")
    print(f"Model: {args.encoder}, Camera: {args.camera_id}")
    print("Press 'q' to quit, 's' to save current frame, 'r' to start/stop recording")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cpu':
        print("Warning: CUDA not available. Webcam mode will be very slow on CPU.")

    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
    checkpoint_path = Path(args.checkpoints_dir) / f'{checkpoint_name}_{args.encoder}.pth'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Please download checkpoints using: bash get_weights.sh")
        sys.exit(1)

    # Load streaming model for webcam
    model = VideoDepthAnythingStream(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()

    # Open webcam - try different backends for Windows compatibility
    cap = None
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto"),
    ]

    for backend, name in backends:
        print(f"Trying {name} backend...")
        cap = cv2.VideoCapture(args.camera_id, backend)
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, test_frame = cap.read()
            if ret:
                print(f"[OK] Successfully opened camera using {name}")
                break
            else:
                print(f"  {name} opened but couldn't read frames")
                cap.release()
                cap = None
        else:
            print(f"  {name} failed to open")

    if cap is None or not cap.isOpened():
        print(f"\nError: Could not open camera {args.camera_id} with any backend")
        print("Please check:")
        print("  1. Camera is connected and not in use by another application")
        print("  2. Camera permissions are granted")
        print("  3. Try a different --camera-id (0, 1, 2, etc.)")
        sys.exit(1)

    # Set camera resolution if specified
    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    recording = False
    recorded_frames = []
    frame_count = 0
    fps_display = 0

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nWebcam active. Controls:")
    print("  q - Quit")
    print("  s - Save current frame")
    print("  r - Toggle recording")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Convert and process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if args.max_res > 0:
            h, w = frame_rgb.shape[:2]
            if max(h, w) > args.max_res:
                scale = args.max_res / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

        # Infer depth
        with torch.no_grad():
            depth = model.infer_video_depth_one(frame_rgb, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

        # Debug: Print depth statistics on first few frames
        if frame_count < 3:
            print(f"Frame {frame_count}: depth shape={depth.shape}, min={depth.min():.4f}, max={depth.max():.4f}, mean={depth.mean():.4f}")

        # Visualize depth - ensure proper normalization
        depth_min = depth.min()
        depth_max = depth.max()

        # Avoid division by zero
        if depth_max - depth_min < 1e-8:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
            print(f"Warning: Flat depth map detected (min={depth_min:.4f}, max={depth_max:.4f})")
        else:
            depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

        if not args.grayscale:
            import matplotlib.cm as cm
            colormap = cm.get_cmap("inferno")
            # Apply colormap to normalized depth
            depth_colored = colormap(depth_norm / 255.0)
            depth_vis = (depth_colored[:, :, :3] * 255).astype(np.uint8)
        else:
            depth_vis = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

        # Resize depth to match frame for side-by-side display
        depth_vis = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))

        # Create side-by-side display
        combined = np.hstack([frame, depth_vis])

        # Add info overlay
        fps_display = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        cv2.putText(combined, f'FPS: {fps_display:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if recording:
            cv2.circle(combined, (combined.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(combined, 'REC', (combined.shape[1] - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Video Depth Anything - Webcam', combined)

        if recording:
            recorded_frames.append((frame_rgb, depth))

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            frame_path = Path(args.output_dir) / f'frame_{timestamp}.png'
            depth_path = Path(args.output_dir) / f'depth_{timestamp}.png'
            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(depth_path), depth_vis)
            print(f"  Saved frame to {frame_path}")
        elif key == ord('r'):
            # Toggle recording
            if recording:
                print(f"  Stopped recording. Saving {len(recorded_frames)} frames...")
                save_recording(recorded_frames, args.output_dir, args.grayscale)
                recorded_frames = []
            else:
                print("  Started recording...")
            recording = not recording

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    if recorded_frames:
        print(f"Saving final recording of {len(recorded_frames)} frames...")
        save_recording(recorded_frames, args.output_dir, args.grayscale)

    print("Webcam session ended.")


def save_recording(frames, output_dir, grayscale=False):
    """Save recorded webcam frames as video."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = Path(output_dir) / f'recording_{timestamp}.mp4'
    depth_path = Path(output_dir) / f'recording_{timestamp}_depth.mp4'

    rgb_frames = np.array([f[0] for f in frames])
    depth_frames = np.array([f[1] for f in frames])

    save_video(rgb_frames, str(video_path), fps=15)
    save_video(depth_frames, str(depth_path), fps=15, is_depths=True, grayscale=grayscale)

    print(f"  Saved recording: {video_path}")
    print(f"  Saved depth: {depth_path}")


def save_exr_depths(depths, output_dir, video_name):
    """Save depth maps as EXR files."""
    import OpenEXR
    import Imath

    depth_exr_dir = Path(output_dir) / f'{video_name}_depths_exr'
    depth_exr_dir.mkdir(exist_ok=True)

    for i, depth in enumerate(depths):
        output_exr = depth_exr_dir / f"frame_{i:05d}.exr"
        header = OpenEXR.Header(depth.shape[1], depth.shape[0])
        header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        exr_file = OpenEXR.OutputFile(str(output_exr), header)
        exr_file.writePixels({"Z": depth.tobytes()})
        exr_file.close()

    print(f"  Saved EXR depths: {depth_exr_dir}")


def save_point_clouds(frames, depths, args):
    """Save point clouds from metric depth."""
    try:
        import open3d as o3d
    except ImportError:
        print("  Warning: open3d not installed. Skipping point cloud generation.")
        print("  Install with: uv sync --extra metric")
        return

    width, height = depths[0].shape[-1], depths[0].shape[-2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - width / 2) / args.focal_length_x
    y = (y - height / 2) / args.focal_length_y

    for i, (color_image, depth) in enumerate(zip(frames, depths)):
        z = np.array(depth)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(color_image).reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pc_path = Path(args.output_dir) / f'point_{i:04d}.ply'
        o3d.io.write_point_cloud(str(pc_path), pcd)

    print(f"  Saved {len(frames)} point clouds")


def screen_command(args):
    """Real-time depth estimation from screen capture."""
    try:
        import mss
    except ImportError:
        print("Error: mss (screen capture library) not installed.")
        print("Install with: uv pip install mss")
        sys.exit(1)

    print(f"Starting screen capture depth estimation...")
    print(f"Model: {args.encoder}, Monitor: {args.monitor}, Target FPS: {args.fps}")
    print("Press 'q' to quit, 's' to save current frame, 'r' to start/stop recording")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cpu':
        print("Warning: CUDA not available. Screen capture mode will be very slow on CPU.")

    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
    checkpoint_path = Path(args.checkpoints_dir) / f'{checkpoint_name}_{args.encoder}.pth'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Please download checkpoints using: bash get_weights.sh")
        sys.exit(1)

    # Load streaming model for screen capture
    model = VideoDepthAnythingStream(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()

    # Setup screen capture
    sct = mss.mss()

    # Determine capture region
    if args.region:
        # Parse region string "x,y,width,height"
        try:
            x, y, width, height = map(int, args.region.split(','))
            monitor = {"top": y, "left": x, "width": width, "height": height}
            print(f"Capturing region: x={x}, y={y}, width={width}, height={height}")
        except:
            print(f"Error: Invalid region format. Use 'x,y,width,height' (e.g., '0,0,1920,1080')")
            sys.exit(1)
    else:
        # Capture specified monitor
        if args.monitor < 1 or args.monitor > len(sct.monitors) - 1:
            print(f"Error: Monitor {args.monitor} not found. Available monitors: 1-{len(sct.monitors) - 1}")
            sys.exit(1)
        monitor = sct.monitors[args.monitor]
        print(f"Capturing monitor {args.monitor}: {monitor['width']}x{monitor['height']}")

    recording = False
    recorded_frames = []
    frame_count = 0
    fps_display = 0
    target_frame_time = 1.0 / args.fps

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nScreen capture active. Controls:")
    print("  q - Quit")
    print("  s - Save current frame")
    print("  r - Toggle recording")

    last_frame_time = time.time()

    while True:
        loop_start = time.time()

        # Capture screen
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)[:, :, :3]  # Remove alpha channel, BGR format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Resize if needed
        h, w = frame_rgb.shape[:2]

        # Check if exact dimensions are specified
        if args.width > 0 and args.height > 0:
            frame_rgb = cv2.resize(frame_rgb, (args.width, args.height))
        elif args.width > 0:
            scale = args.width / w
            new_h = int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (args.width, new_h))
        elif args.height > 0:
            scale = args.height / h
            new_w = int(w * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, args.height))
        elif args.max_res > 0 and max(h, w) > args.max_res:
            scale = args.max_res / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

        # Infer depth
        with torch.no_grad():
            depth = model.infer_video_depth_one(frame_rgb, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

        # Debug: Print depth statistics on first few frames
        if frame_count < 3:
            print(f"Frame {frame_count}: depth shape={depth.shape}, min={depth.min():.4f}, max={depth.max():.4f}, mean={depth.mean():.4f}")

        # Visualize depth - ensure proper normalization
        depth_min = depth.min()
        depth_max = depth.max()

        # Avoid division by zero
        if depth_max - depth_min < 1e-8:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
            if frame_count < 3:
                print(f"Warning: Flat depth map detected (min={depth_min:.4f}, max={depth_max:.4f})")
        else:
            depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

        if not args.grayscale:
            import matplotlib.cm as cm
            colormap = cm.get_cmap("inferno")
            # Apply colormap to normalized depth
            depth_colored = colormap(depth_norm / 255.0)
            depth_vis = (depth_colored[:, :, :3] * 255).astype(np.uint8)
        else:
            depth_vis = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

        # Convert frame_rgb to BGR for display
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Resize depth to match frame for side-by-side display
        depth_vis = cv2.resize(depth_vis, (frame_bgr.shape[1], frame_bgr.shape[0]))

        # Create side-by-side display
        combined = np.hstack([frame_bgr, depth_vis])

        # Add info overlay
        elapsed = time.time() - last_frame_time
        fps_display = 1.0 / elapsed if elapsed > 0 else 0
        last_frame_time = time.time()

        cv2.putText(combined, f'FPS: {fps_display:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if recording:
            cv2.circle(combined, (combined.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(combined, 'REC', (combined.shape[1] - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Video Depth Anything - Screen Capture', combined)

        if recording:
            recorded_frames.append((frame_rgb, depth))

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            frame_path = Path(args.output_dir) / f'screen_{timestamp}.png'
            depth_path = Path(args.output_dir) / f'depth_{timestamp}.png'
            cv2.imwrite(str(frame_path), frame_bgr)
            cv2.imwrite(str(depth_path), depth_vis)
            print(f"  Saved frame to {frame_path}")
        elif key == ord('r'):
            # Toggle recording
            if recording:
                print(f"  Stopped recording. Saving {len(recorded_frames)} frames...")
                save_recording(recorded_frames, args.output_dir, args.grayscale)
                recorded_frames = []
            else:
                print("  Started recording...")
            recording = not recording

        frame_count += 1

        # Frame rate limiting
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < target_frame_time:
            time.sleep(target_frame_time - loop_elapsed)

    cv2.destroyAllWindows()

    if recorded_frames:
        print(f"Saving final recording of {len(recorded_frames)} frames...")
        save_recording(recorded_frames, args.output_dir, args.grayscale)

    print("Screen capture session ended.")


def screen3d_command(args):
    """Screen capture with 2.5D parallax effect and optional virtual camera output."""
    try:
        import mss
    except ImportError:
        print("Error: mss (screen capture library) not installed.")
        print("Install with: uv sync")
        sys.exit(1)

    print(f"Starting 3D screen capture with parallax effect...")
    print(f"Model: {args.encoder}, Monitor: {args.monitor}, Target FPS: {args.fps}")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cpu':
        print("Warning: CUDA not available. 3D mode will be very slow on CPU.")

    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
    checkpoint_path = Path(args.checkpoints_dir) / f'{checkpoint_name}_{args.encoder}.pth'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Please download checkpoints using: bash get_weights.sh")
        sys.exit(1)

    # Load streaming model
    model = VideoDepthAnythingStream(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()

    # Setup screen capture
    sct = mss.mss()

    # Determine capture region
    if args.region:
        try:
            x, y, width, height = map(int, args.region.split(','))
            monitor = {"top": y, "left": x, "width": width, "height": height}
            print(f"Capturing region: x={x}, y={y}, width={width}, height={height}")
        except:
            print(f"Error: Invalid region format. Use 'x,y,width,height'")
            sys.exit(1)
    else:
        if args.monitor < 1 or args.monitor > len(sct.monitors) - 1:
            print(f"Error: Monitor {args.monitor} not found. Available: 1-{len(sct.monitors) - 1}")
            sys.exit(1)
        monitor = sct.monitors[args.monitor]
        print(f"Capturing monitor {args.monitor}: {monitor['width']}x{monitor['height']}")

    # Setup virtual camera if requested
    vcam = None
    if args.virtual_cam:
        try:
            import pyvirtualcam
            # Determine output resolution
            output_w = min(monitor['width'], args.max_res) if args.max_res > 0 else monitor['width']
            output_h = min(monitor['height'], args.max_res) if args.max_res > 0 else monitor['height']

            vcam = pyvirtualcam.Camera(width=output_w, height=output_h, fps=args.fps)
            print(f"[OK] Virtual camera started: {output_w}x{output_h}@{args.fps}fps")
            print(f"  You can now add 'OBS Virtual Camera' as a source in OBS Studio")
        except Exception as e:
            print(f"Warning: Could not initialize virtual camera: {e}")
            print("Continuing without virtual camera output...")
            vcam = None

    # Initialize 3D projector and controller
    projector = None
    controller = InteractiveParallaxController(
        auto_rotate=args.auto_rotate,
        auto_speed=0.5
    )
    controller.scale_z = args.depth_scale
    controller.lighting_intensity = args.lighting
    controller.invert_depth = args.invert_depth

    recording = False
    recorded_frames = []
    frame_count = 0
    target_frame_time = 1.0 / args.fps
    show_displacement = args.show_displacement  # Can be toggled with 'b' key
    show_depth = False  # Can be toggled with 'n' key
    displacement_grayscale = args.displacement_gray  # Can be toggled with 'g' key
    show_arrows = True  # Can be toggled with 'h' key

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n3D Screen capture active. Controls:")
    print("  q - Quit")
    print("  s - Save current frame")
    print("  r - Toggle recording / Reset (hold)")
    print("  t - Toggle auto-rotate")
    print("  w/a/s/d - Manual rotation (up/left/down/right)")
    print("  z/x - Decrease/increase 3D depth effect")
    print("  c/v - Decrease/increase lighting intensity")
    print("  [/] or -/= - Zoom out/in")
    print("  b - Toggle displacement visualization")
    print("  n - Toggle depth map view")
    print("  g - Toggle grayscale displacement (faster)")
    print("  h - Toggle displacement arrows")
    print("  i - Toggle invert depth (reverse parallax)")
    if args.mouse_control:
        print("  Move mouse - Control parallax angle")

    last_frame_time = time.time()

    # Mouse callback for interactive control
    mouse_pos = [0, 0]
    window_name = 'Video Depth Anything - 3D Screen Capture'

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and args.mouse_control:
            mouse_pos[0] = x
            mouse_pos[1] = y

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        loop_start = time.time()

        # Capture screen
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)[:, :, :3]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Resize if needed
        display_frame = frame_rgb
        h, w = frame_rgb.shape[:2]

        # Check if exact dimensions are specified
        if args.width > 0 and args.height > 0:
            display_frame = cv2.resize(frame_rgb, (args.width, args.height))
        elif args.width > 0:
            # Only width specified, maintain aspect ratio
            scale = args.width / w
            new_h = int(h * scale)
            display_frame = cv2.resize(frame_rgb, (args.width, new_h))
        elif args.height > 0:
            # Only height specified, maintain aspect ratio
            scale = args.height / h
            new_w = int(w * scale)
            display_frame = cv2.resize(frame_rgb, (new_w, args.height))
        elif args.max_res > 0 and max(h, w) > args.max_res:
            # Use max_res as before
            scale = args.max_res / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            display_frame = cv2.resize(frame_rgb, (new_w, new_h))

        # Infer depth
        with torch.no_grad():
            depth = model.infer_video_depth_one(display_frame, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

        # Initialize projector on first frame
        if projector is None:
            h, w = display_frame.shape[:2]
            projector = DepthProjector(w, h, focal_length=500)
            print(f"Initialized 3D projector: {w}x{h}")

        # Debug: Print depth stats on first few frames
        if frame_count < 3:
            print(f"\n=== Frame {frame_count} Debug ===")
            print(f"Depth shape: {depth.shape}")
            print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
            print(f"Depth mean: {depth.mean():.4f}")
            print(f"Rotation: ({controller.rotation_x:.1f}, {controller.rotation_y:.1f})")
            print(f"Depth scale: {controller.scale_z:.2f}")
            if controller.rotation_x == 0 and controller.rotation_y == 0:
                print("⚠️  WARNING: No rotation! Displacement will be ZERO.")
                print("   Press W/A/S/D to rotate manually, or enable --auto-rotate")

        # Update controller
        if args.mouse_control:
            h, w = display_frame.shape[:2]
            controller.update_from_mouse(mouse_pos[0], mouse_pos[1], w * 2, h)  # *2 for side-by-side

        if args.auto_rotate:
            controller.update_auto_rotate(target_frame_time)

        # Create 3D projection with parallax
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Optional: Add grid overlay to original to see displacement better
        test_frame = display_frame.copy()

        # Draw test grid if enabled - this makes displacement VERY obvious
        if args.test_grid:
            h, w = test_frame.shape[:2]
            grid_spacing = 40
            # Draw vertical lines
            for x in range(0, w, grid_spacing):
                cv2.line(test_frame, (x, 0), (x, h), (0, 255, 0), 2)
            # Draw horizontal lines
            for y in range(0, h, grid_spacing):
                cv2.line(test_frame, (0, y), (w, y), (0, 255, 0), 2)
            # Add center crosshair
            cv2.line(test_frame, (w//2, 0), (w//2, h), (255, 0, 0), 3)
            cv2.line(test_frame, (0, h//2), (w, h//2), (255, 0, 0), 3)

        if frame_count % 30 == 0:  # Every 30 frames, print if displacement is working
            if hasattr(projector, 'last_displacement_x') and projector.last_displacement_x is not None:
                disp_mag = np.sqrt(projector.last_displacement_x**2 + projector.last_displacement_y**2)
                avg_disp = disp_mag.mean()
                max_disp = disp_mag.max()
                print(f"[Frame {frame_count}] Avg displacement: {avg_disp:.2f}px, Max: {max_disp:.2f}px, Rotation: ({controller.rotation_x:.1f}, {controller.rotation_y:.1f})")

        projected_3d = projector.project_with_parallax(
            test_frame,
            depth_normalized,
            rotation_x=controller.rotation_x,
            rotation_y=controller.rotation_y,
            scale_z=controller.scale_z,
            lighting_intensity=controller.lighting_intensity,
            zoom=controller.zoom,
            invert_depth=controller.invert_depth
        )

        # Create visualization: original on left, 3D projection on right
        frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        projected_bgr = cv2.cvtColor(projected_3d, cv2.COLOR_RGB2BGR)

        # Create depth map visualization
        depth_norm_vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
        import matplotlib.cm as cm
        colormap = cm.get_cmap("inferno")
        depth_colored = (colormap(depth_norm_vis / 255.0)[:, :, :3] * 255).astype(np.uint8)
        depth_vis_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

        # Build display based on toggles
        panels = []
        labels = []

        # Always include original and 3D projection
        panels.append(('Original', frame_bgr))

        if show_depth:
            panels.append(('Depth Map', depth_vis_bgr))

        if show_displacement:
            displacement_vis = projector.get_displacement_visualization(
                grayscale=displacement_grayscale,
                show_arrows=show_arrows
            )
            disp_label = 'Displacement' + (' (Gray)' if displacement_grayscale else '')
            panels.append((disp_label, displacement_vis))

        panels.append(('3D Projection', projected_bgr))

        # Arrange panels in grid layout (2x2 or 2x1)
        num_panels = len(panels)

        if num_panels <= 2:
            # Simple horizontal layout
            combined = np.hstack([p[1] for p in panels])
            # Add labels
            panel_x = 0
            for label, panel in panels:
                cv2.putText(combined, label, (panel_x + 10, combined.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panel_x += panel.shape[1]
        else:
            # Grid layout (2 per row)
            rows = []
            for i in range(0, num_panels, 2):
                if i + 1 < num_panels:
                    # Two panels in this row
                    row_panels = [panels[i][1], panels[i + 1][1]]
                    row = np.hstack(row_panels)
                    # Add labels
                    cv2.putText(row, panels[i][0], (10, row.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(row, panels[i + 1][0], (panels[i][1].shape[1] + 10, row.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # One panel in this row (pad with black)
                    black_panel = np.zeros_like(panels[i][1])
                    row = np.hstack([panels[i][1], black_panel])
                    cv2.putText(row, panels[i][0], (10, row.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                rows.append(row)

            combined = np.vstack(rows)

        # Add info overlay
        elapsed = time.time() - last_frame_time
        fps_display = 1.0 / elapsed if elapsed > 0 else 0
        last_frame_time = time.time()

        info_text = [
            f'FPS: {fps_display:.1f}',
            f'Rot: ({controller.rotation_x:.1f}, {controller.rotation_y:.1f})',
            f'Depth: {controller.scale_z:.2f}',
            f'Light: {controller.lighting_intensity:.2f}',
            f'Zoom: {controller.zoom:.2f}x',
        ]

        y_pos = 30
        for text in info_text:
            cv2.putText(combined, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25

        if recording:
            cv2.circle(combined, (combined.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(combined, 'REC', (combined.shape[1] - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if args.auto_rotate:
            cv2.putText(combined, 'AUTO', (combined.shape[1] - 150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if controller.invert_depth:
            cv2.putText(combined, 'INVERTED', (combined.shape[1] - 180, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Output to virtual camera (only the 3D projection)
        if vcam is not None:
            try:
                # Convert to RGB for virtual camera
                vcam_frame = cv2.cvtColor(projected_bgr, cv2.COLOR_BGR2RGB)
                vcam.send(vcam_frame)
            except Exception as e:
                print(f"Warning: Virtual camera error: {e}")

        cv2.imshow(window_name, combined)

        if recording:
            recorded_frames.append((projected_3d, depth))

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            # Toggle displacement visualization
            show_displacement = not show_displacement
            print(f"  Displacement visualization: {'ON' if show_displacement else 'OFF'}")
        elif key == ord('n'):
            # Toggle depth map view
            show_depth = not show_depth
            print(f"  Depth map view: {'ON' if show_depth else 'OFF'}")
        elif key == ord('g'):
            # Toggle grayscale displacement
            displacement_grayscale = not displacement_grayscale
            print(f"  Displacement grayscale: {'ON' if displacement_grayscale else 'OFF'}")
        elif key == ord('h'):
            # Toggle displacement arrows
            show_arrows = not show_arrows
            print(f"  Displacement arrows: {'ON' if show_arrows else 'OFF'}")
        elif key == ord('i'):
            # Toggle invert depth (handled by controller)
            handled = controller.handle_key(key)
            if handled:
                print(f"  Invert depth: {'ON' if controller.invert_depth else 'OFF'}")
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            frame_path = Path(args.output_dir) / f'screen3d_{timestamp}.png'
            depth_path = Path(args.output_dir) / f'depth_{timestamp}.png'
            cv2.imwrite(str(frame_path), projected_bgr)

            # Save depth as colormap
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
            import matplotlib.cm as cm
            colormap = cm.get_cmap("inferno")
            depth_colored = (colormap(depth_norm / 255.0)[:, :, :3] * 255).astype(np.uint8)
            cv2.imwrite(str(depth_path), cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
            print(f"  Saved 3D frame to {frame_path}")
        elif key == ord('r'):
            if recording:
                print(f"  Stopped recording. Saving {len(recorded_frames)} frames...")
                save_recording(recorded_frames, args.output_dir, grayscale=False)
                recorded_frames = []
            else:
                print("  Started recording...")
            recording = not recording
        else:
            controller.handle_key(key)

        frame_count += 1

        # Frame rate limiting
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < target_frame_time:
            time.sleep(target_frame_time - loop_elapsed)

    cv2.destroyAllWindows()

    if vcam is not None:
        vcam.close()
        print("Virtual camera closed.")

    if recorded_frames:
        print(f"Saving final recording of {len(recorded_frames)} frames...")
        save_recording(recorded_frames, args.output_dir, grayscale=False)

    print("3D screen capture session ended.")


def view3d_command(args):
    """View a depth map as an interactive 3D mesh."""
    try:
        import open3d as o3d
    except ImportError:
        print("Error: open3d not installed.")
        print("Install with: uv sync --extra metric")
        sys.exit(1)

    print(f"Loading 3D mesh viewer...")
    print(f"Image: {args.image}")
    print(f"Depth: {args.depth}")

    if args.metric:
        print(f"Metric depth mode enabled")
        print(f"  Focal length: fx={args.focal_length_x:.1f}, fy={args.focal_length_y:.1f}")
        if args.principal_point_x is not None and args.principal_point_y is not None:
            print(f"  Principal point: cx={args.principal_point_x:.1f}, cy={args.principal_point_y:.1f}")
        else:
            print(f"  Principal point: image center (auto)")

    viewer = DepthMeshViewer(
        depth_scale=args.depth_scale,
        max_depth_threshold=args.depth_threshold,
        depth_min_percentile=args.depth_min_percentile,
        depth_max_percentile=args.depth_max_percentile,
        display_mode=args.display_mode,
        use_metric_depth=args.metric,
        focal_length_x=args.focal_length_x if args.metric else None,
        focal_length_y=args.focal_length_y if args.metric else None,
        principal_point_x=args.principal_point_x,
        principal_point_y=args.principal_point_y,
        metric_depth_scale=args.metric_depth_scale if hasattr(args, 'metric_depth_scale') else 1.0
    )

    viewer.process_and_view(
        args.image,
        args.depth,
        subsample=args.subsample,
        invert_depth=args.invert_depth,
        smooth_mesh=not args.no_smooth,
        show_wireframe=args.wireframe,
        background_color=tuple(float(x) for x in args.background.split(','))
    )


def webcam3d_command(args):
    """Real-time 3D mesh viewer from webcam feed."""
    try:
        import open3d as o3d
    except ImportError:
        print("Error: open3d not installed.")
        print("Install with: uv sync --extra metric")
        sys.exit(1)

    print(f"Starting real-time 3D webcam viewer...")
    print(f"Model: {args.encoder}, Camera: {args.camera_id}")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cpu':
        print("Warning: CUDA not available. 3D webcam mode will be very slow on CPU.")

    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
    checkpoint_path = Path(args.checkpoints_dir) / f'{checkpoint_name}_{args.encoder}.pth'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Please download checkpoints using: bash get_weights.sh")
        sys.exit(1)

    # Load streaming model
    model = VideoDepthAnythingStream(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()

    # Open webcam
    cap = None
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto"),
    ]

    for backend, name in backends:
        print(f"Trying {name} backend...")
        cap = cv2.VideoCapture(args.camera_id, backend)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"[OK] Successfully opened camera using {name}")
                break
            else:
                cap.release()
                cap = None

    if cap is None or not cap.isOpened():
        print(f"\nError: Could not open camera {args.camera_id}")
        sys.exit(1)

    # Set camera resolution if specified
    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    # Initialize 3D viewer
    viewer_3d = None
    frame_count = 0

    print("\nWebcam 3D active. Close the 3D window to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Convert and process
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize if needed
            if args.max_res > 0:
                h, w = frame_rgb.shape[:2]
                if max(h, w) > args.max_res:
                    scale = args.max_res / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

            # Infer depth
            with torch.no_grad():
                depth = model.infer_video_depth_one(frame_rgb, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

            # Initialize viewer on first frame
            if viewer_3d is None:
                h, w = frame_rgb.shape[:2]
                viewer_3d = RealTime3DViewer(
                    depth_scale=args.depth_scale,
                    subsample=args.subsample,
                    smooth_mesh=args.smooth,
                    max_depth_threshold=args.depth_threshold,
                    depth_min_percentile=args.depth_min_percentile,
                    depth_max_percentile=args.depth_max_percentile,
                    background_color=tuple(float(x) for x in args.background.split(',')),
                    display_mode=args.display_mode,
                    use_raw_depth=args.raw_depth,
                    use_metric_depth=args.metric,
                    focal_length_x=args.focal_length_x if args.metric else None,
                    focal_length_y=args.focal_length_y if args.metric else None,
                    principal_point_x=args.principal_point_x if hasattr(args, 'principal_point_x') else None,
                    principal_point_y=args.principal_point_y if hasattr(args, 'principal_point_y') else None,
                    metric_depth_scale=args.metric_depth_scale if hasattr(args, 'metric_depth_scale') else 1.0
                )
                viewer_3d.initialize(width=1280, height=720)
                print(f"Initialized 3D viewer: {w}x{h} ({args.display_mode} mode)")
                if args.metric:
                    print(f"Metric depth: fx={args.focal_length_x:.1f}, fy={args.focal_length_y:.1f}")
                print(f"Depth range: {args.depth_min_percentile}%-{args.depth_max_percentile}% percentile")

            # Update 3D mesh
            viewer_3d.update_mesh(frame_rgb, depth, invert_depth=args.invert_depth)

            # Check if window closed
            if viewer_3d.should_close():
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        if viewer_3d:
            viewer_3d.close()

    print("Webcam 3D session ended.")


def screen3d_viewer_command(args):
    """Real-time 3D mesh viewer from screen capture."""
    try:
        import open3d as o3d
        import mss
    except ImportError as e:
        print(f"Error: Required library not installed: {e}")
        print("Install with: uv sync --extra metric")
        sys.exit(1)

    print(f"Starting real-time 3D screen viewer...")
    print(f"Model: {args.encoder}, Monitor: {args.monitor}")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cpu':
        print("Warning: CUDA not available. 3D screen mode will be very slow on CPU.")

    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
    checkpoint_path = Path(args.checkpoints_dir) / f'{checkpoint_name}_{args.encoder}.pth'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Please download checkpoints using: bash get_weights.sh")
        sys.exit(1)

    # Load streaming model
    model = VideoDepthAnythingStream(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()

    # Setup screen capture
    sct = mss.mss()

    # Determine capture region
    if args.region:
        try:
            x, y, width, height = map(int, args.region.split(','))
            monitor = {"top": y, "left": x, "width": width, "height": height}
            print(f"Capturing region: x={x}, y={y}, width={width}, height={height}")
        except:
            print(f"Error: Invalid region format. Use 'x,y,width,height'")
            sys.exit(1)
    else:
        if args.monitor < 1 or args.monitor > len(sct.monitors) - 1:
            print(f"Error: Monitor {args.monitor} not found. Available: 1-{len(sct.monitors) - 1}")
            sys.exit(1)
        monitor = sct.monitors[args.monitor]
        print(f"Capturing monitor {args.monitor}: {monitor['width']}x{monitor['height']}")

    # Initialize 3D viewer
    viewer_3d = None
    frame_count = 0
    target_frame_time = 1.0 / args.fps

    print("\nScreen 3D capture active. Close the 3D window to exit.")

    try:
        while True:
            loop_start = time.time()

            # Capture screen
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)[:, :, :3]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            # Resize if needed
            h, w = frame_rgb.shape[:2]
            if args.max_res > 0 and max(h, w) > args.max_res:
                scale = args.max_res / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

            # Infer depth
            with torch.no_grad():
                depth = model.infer_video_depth_one(frame_rgb, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

            # Initialize viewer on first frame
            if viewer_3d is None:
                h, w = frame_rgb.shape[:2]
                viewer_3d = RealTime3DViewer(
                    depth_scale=args.depth_scale,
                    subsample=args.subsample,
                    smooth_mesh=args.smooth,
                    max_depth_threshold=args.depth_threshold,
                    depth_min_percentile=args.depth_min_percentile,
                    depth_max_percentile=args.depth_max_percentile,
                    background_color=tuple(float(x) for x in args.background.split(',')),
                    display_mode=args.display_mode,
                    use_raw_depth=args.raw_depth,
                    use_metric_depth=args.metric,
                    focal_length_x=args.focal_length_x if args.metric else None,
                    focal_length_y=args.focal_length_y if args.metric else None,
                    principal_point_x=args.principal_point_x if hasattr(args, 'principal_point_x') else None,
                    principal_point_y=args.principal_point_y if hasattr(args, 'principal_point_y') else None,
                    metric_depth_scale=args.metric_depth_scale if hasattr(args, 'metric_depth_scale') else 1.0
                )
                viewer_3d.initialize(width=1280, height=720)
                print(f"Initialized 3D viewer: {w}x{h} ({args.display_mode} mode)")
                if args.metric:
                    print(f"Metric depth: fx={args.focal_length_x:.1f}, fy={args.focal_length_y:.1f}")
                print(f"Depth range: {args.depth_min_percentile}%-{args.depth_max_percentile}% percentile")

            # Update 3D mesh
            viewer_3d.update_mesh(frame_rgb, depth, invert_depth=args.invert_depth)

            # Check if window closed
            if viewer_3d.should_close():
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

            # Frame rate limiting
            loop_elapsed = time.time() - loop_start
            if loop_elapsed < target_frame_time:
                time.sleep(target_frame_time - loop_elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if viewer_3d:
            viewer_3d.close()

    print("Screen 3D capture session ended.")


def demo_command(args):
    """Launch Gradio web demo."""
    try:
        import gradio as gr
    except ImportError:
        print("Error: gradio not installed.")
        print("Install with: uv sync --extra demo")
        sys.exit(1)

    print("Launching Gradio demo...")
    print("Note: This will start a web server. Use Ctrl+C to stop.")

    # Import and run the app
    from app import construct_demo
    demo = construct_demo()
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


def main():
    parser = argparse.ArgumentParser(
        description='Video Depth Anything - Consistent Video Depth Estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video file
  uv run vda video input.mp4 -o outputs/

  # Use webcam for real-time depth
  uv run vda webcam

  # Capture screen with depth estimation
  uv run vda screen

  # Capture screen with 2.5D parallax effect
  uv run vda screen3d --auto-rotate

  # Stream 3D screen to OBS Virtual Camera
  uv run vda screen3d --virtual-cam --mouse-control

  # Process video with metric depth
  uv run vda video input.mp4 --metric

  # Launch web demo
  uv run vda demo

  # View depth map in true 3D
  uv run vda view3d image.jpg depth.png

  # Real-time 3D webcam viewer
  uv run vda webcam3d

  # Real-time 3D screen capture viewer
  uv run vda screen3d-viewer
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Screen capture command
    screen_parser = subparsers.add_parser('screen', help='Real-time depth estimation from screen capture')
    screen_parser.add_argument('-o', '--output-dir', type=str, default='./screen_outputs',
                              help='Output directory for saved frames/recordings')
    screen_parser.add_argument('--monitor', type=int, default=1, help='Monitor number to capture (1 = primary)')
    screen_parser.add_argument('--region', type=str, default=None,
                              help='Capture region as "x,y,width,height" (e.g., "0,0,1920,1080")')
    screen_parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'],
                              help='Model size (vits recommended for real-time)')
    screen_parser.add_argument('--metric', action='store_true', help='Use metric depth model')
    screen_parser.add_argument('--input-size', type=int, default=518, help='Input size for model')
    screen_parser.add_argument('--max-res', type=int, default=640, help='Maximum resolution for screen frames')
    screen_parser.add_argument('--width', type=int, default=-1, help='Exact output width (-1 for auto)')
    screen_parser.add_argument('--height', type=int, default=-1, help='Exact output height (-1 for auto)')
    screen_parser.add_argument('--fps', type=int, default=10, help='Target FPS for screen capture')
    screen_parser.add_argument('--fp32', action='store_true', help='Use FP32 precision')
    screen_parser.add_argument('--grayscale', action='store_true', help='Display grayscale depth')
    screen_parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints', help='Checkpoints directory')
    screen_parser.set_defaults(func=screen_command)

    # Screen 3D command (with parallax effect and virtual camera output)
    screen3d_parser = subparsers.add_parser('screen3d', help='Screen capture with 2.5D parallax effect')
    screen3d_parser.add_argument('-o', '--output-dir', type=str, default='./screen3d_outputs',
                                help='Output directory for saved frames/recordings')
    screen3d_parser.add_argument('--monitor', type=int, default=1, help='Monitor number to capture (1 = primary)')
    screen3d_parser.add_argument('--region', type=str, default=None,
                                help='Capture region as "x,y,width,height" (e.g., "0,0,1920,1080")')
    screen3d_parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'],
                                help='Model size (vits recommended for real-time)')
    screen3d_parser.add_argument('--metric', action='store_true', help='Use metric depth model')
    screen3d_parser.add_argument('--input-size', type=int, default=518, help='Input size for model')
    screen3d_parser.add_argument('--max-res', type=int, default=640, help='Maximum resolution (scales down if larger)')
    screen3d_parser.add_argument('--width', type=int, default=-1, help='Exact output width (-1 for auto)')
    screen3d_parser.add_argument('--height', type=int, default=-1, help='Exact output height (-1 for auto)')
    screen3d_parser.add_argument('--fps', type=int, default=10, help='Target FPS')
    screen3d_parser.add_argument('--fp32', action='store_true', help='Use FP32 precision')
    screen3d_parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints', help='Checkpoints directory')
    screen3d_parser.add_argument('--virtual-cam', action='store_true', help='Output to OBS Virtual Camera')
    screen3d_parser.add_argument('--auto-rotate', action='store_true', help='Enable automatic rotation')
    screen3d_parser.add_argument('--depth-scale', type=float, default=0.5, help='3D effect strength (0.1-2.0)')
    screen3d_parser.add_argument('--lighting', type=float, default=0.5, help='3D lighting intensity (0.0-1.0)')
    screen3d_parser.add_argument('--mouse-control', action='store_true', help='Use mouse for parallax control')
    screen3d_parser.add_argument('--show-displacement', action='store_true', help='Show displacement overlay (debug)')
    screen3d_parser.add_argument('--displacement-gray', action='store_true', help='Start with grayscale displacement')
    screen3d_parser.add_argument('--test-grid', action='store_true', help='Overlay test grid to verify displacement')
    screen3d_parser.add_argument('--invert-depth', action='store_true', help='Invert depth (near becomes far, "pop out" effect)')
    screen3d_parser.set_defaults(func=screen3d_command)

    # Video command
    video_parser = subparsers.add_parser('video', help='Process video file')
    video_parser.add_argument('input', type=str, help='Input video path')
    video_parser.add_argument('-o', '--output-dir', type=str, default='./outputs', help='Output directory')
    video_parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'],
                             help='Model size')
    video_parser.add_argument('--metric', action='store_true', help='Use metric depth model')
    video_parser.add_argument('--streaming', action='store_true', help='Use streaming mode (lower memory)')
    video_parser.add_argument('--input-size', type=int, default=518, help='Input size for model')
    video_parser.add_argument('--max-res', type=int, default=1280, help='Maximum resolution')
    video_parser.add_argument('--max-len', type=int, default=-1, help='Maximum number of frames (-1 for all)')
    video_parser.add_argument('--target-fps', type=int, default=-1, help='Target FPS (-1 for original)')
    video_parser.add_argument('--fp32', action='store_true', help='Use FP32 precision')
    video_parser.add_argument('--grayscale', action='store_true', help='Save grayscale depth')
    video_parser.add_argument('--save-npz', action='store_true', help='Save depth as NPZ')
    video_parser.add_argument('--save-exr', action='store_true', help='Save depth as EXR')
    video_parser.add_argument('--depth-only', action='store_true', help='Only save depth, not source video')
    video_parser.add_argument('--focal-length-x', type=float, default=470.4, help='Focal length X (for metric depth)')
    video_parser.add_argument('--focal-length-y', type=float, default=470.4, help='Focal length Y (for metric depth)')
    video_parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints', help='Checkpoints directory')
    video_parser.set_defaults(func=video_command)

    # Webcam command
    webcam_parser = subparsers.add_parser('webcam', help='Real-time webcam depth estimation')
    webcam_parser.add_argument('-o', '--output-dir', type=str, default='./webcam_outputs',
                              help='Output directory for saved frames/recordings')
    webcam_parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    webcam_parser.add_argument('--camera-width', type=int, default=-1, help='Camera width (-1 for default)')
    webcam_parser.add_argument('--camera-height', type=int, default=-1, help='Camera height (-1 for default)')
    webcam_parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'],
                              help='Model size (vits recommended for webcam)')
    webcam_parser.add_argument('--metric', action='store_true', help='Use metric depth model')
    webcam_parser.add_argument('--input-size', type=int, default=518, help='Input size for model')
    webcam_parser.add_argument('--max-res', type=int, default=640, help='Maximum resolution for webcam frames')
    webcam_parser.add_argument('--fp32', action='store_true', help='Use FP32 precision')
    webcam_parser.add_argument('--grayscale', action='store_true', help='Display grayscale depth')
    webcam_parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints', help='Checkpoints directory')
    webcam_parser.set_defaults(func=webcam_command)

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Launch Gradio web demo')
    demo_parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host')
    demo_parser.add_argument('--port', type=int, default=7860, help='Server port')
    demo_parser.add_argument('--share', action='store_true', help='Create public share link')
    demo_parser.set_defaults(func=demo_command)

    # View3D command
    view3d_parser = subparsers.add_parser('view3d', help='View depth map as interactive 3D mesh or point cloud')
    view3d_parser.add_argument('image', type=str, help='Path to RGB image')
    view3d_parser.add_argument('depth', type=str, help='Path to depth map (PNG, JPG, or .npy)')
    view3d_parser.add_argument('--depth-scale', type=float, default=0.5,
                              help='Z-displacement scale (0.1-2.0, where 1.0 = depth spans half image width, default: 0.5)')
    view3d_parser.add_argument('--depth-threshold', type=float, default=0.95,
                              help='Filter out background pixels > this percentile (0-1, default: 0.95)')
    view3d_parser.add_argument('--depth-min-percentile', type=float, default=0.0,
                              help='Clamp near depth values at this percentile (0-100, default: 0, reduces extremes)')
    view3d_parser.add_argument('--depth-max-percentile', type=float, default=100.0,
                              help='Clamp far depth values at this percentile (0-100, default: 100, reduces extremes)')
    view3d_parser.add_argument('--subsample', type=int, default=2,
                              help='Downsample factor for performance (1=full, 2=half, 4=quarter, default: 2)')
    view3d_parser.add_argument('--display-mode', type=str, default='mesh', choices=['mesh', 'pointcloud'],
                              help='Display mode: mesh (triangle mesh) or pointcloud (point cloud)')
    view3d_parser.add_argument('--invert-depth', action='store_true',
                              help='Invert depth values (1=near, 0=far)')
    view3d_parser.add_argument('--no-smooth', action='store_true',
                              help='Disable mesh smoothing (mesh mode only)')
    view3d_parser.add_argument('--wireframe', action='store_true',
                              help='Start in wireframe mode (mesh mode only)')
    view3d_parser.add_argument('--background', type=str, default='0.1,0.1,0.1',
                              help='Background color as R,G,B (0-1 range, default: 0.1,0.1,0.1)')
    view3d_parser.add_argument('--metric', action='store_true',
                              help='Use metric depth mode (depth values in meters, requires --focal-length-x/y)')
    view3d_parser.add_argument('--focal-length-x', type=float, default=470.4,
                              help='Camera focal length X in pixels (default: 470.4, required for --metric)')
    view3d_parser.add_argument('--focal-length-y', type=float, default=470.4,
                              help='Camera focal length Y in pixels (default: 470.4, required for --metric)')
    view3d_parser.add_argument('--principal-point-x', type=float, default=None,
                              help='Principal point X in pixels (default: image center)')
    view3d_parser.add_argument('--principal-point-y', type=float, default=None,
                              help='Principal point Y in pixels (default: image center)')
    view3d_parser.add_argument('--metric-depth-scale', type=float, default=0.01,
                              help='Scale factor for metric depth values (default: 0.01, try 0.001-0.1 if depth range seems wrong)')
    view3d_parser.set_defaults(func=view3d_command)

    # Webcam3D command
    webcam3d_parser = subparsers.add_parser('webcam3d', help='Real-time 3D mesh/point cloud viewer from webcam')
    webcam3d_parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    webcam3d_parser.add_argument('--camera-width', type=int, default=-1, help='Camera width (-1 for default)')
    webcam3d_parser.add_argument('--camera-height', type=int, default=-1, help='Camera height (-1 for default)')
    webcam3d_parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'],
                                help='Model size (vits recommended for real-time)')
    webcam3d_parser.add_argument('--metric', action='store_true', help='Use metric depth model')
    webcam3d_parser.add_argument('--input-size', type=int, default=518, help='Input size for model')
    webcam3d_parser.add_argument('--max-res', type=int, default=480,
                                help='Maximum resolution for frames (lower = faster)')
    webcam3d_parser.add_argument('--fp32', action='store_true', help='Use FP32 precision')
    webcam3d_parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints',
                                help='Checkpoints directory')
    webcam3d_parser.add_argument('--depth-scale', type=float, default=0.5,
                                help='Z-displacement scale (0.1-2.0, where 1.0 = depth spans half image width, default: 0.5)')
    webcam3d_parser.add_argument('--depth-min-percentile', type=float, default=0.0,
                                help='Clamp near depth at this percentile (default: 0, preserves foreground)')
    webcam3d_parser.add_argument('--depth-max-percentile', type=float, default=95.0,
                                help='Clamp far depth at this percentile (default: 95, reduces background extremes while preserving detail)')
    webcam3d_parser.add_argument('--depth-threshold', type=float, default=1.0,
                                help='Filter pixels beyond this depth percentile (0-1, default: 1.0 keeps all pixels, 0.95 removes farthest 5%%)')
    webcam3d_parser.add_argument('--raw-depth', action='store_true',
                                help='Use raw depth values without normalization (preserves more detail, requires lower --depth-scale like 0.1-0.3)')
    webcam3d_parser.add_argument('--display-mode', type=str, default='mesh', choices=['mesh', 'pointcloud'],
                                help='Display mode: mesh (triangle mesh) or pointcloud (point cloud)')
    webcam3d_parser.add_argument('--subsample', type=int, default=3,
                                help='Geometry downsample factor for performance (2-4 recommended, default: 3)')
    webcam3d_parser.add_argument('--smooth', action='store_true',
                                help='Enable mesh smoothing (slower, mesh mode only)')
    webcam3d_parser.add_argument('--invert-depth', action='store_true',
                                help='Invert depth values')
    webcam3d_parser.add_argument('--background', type=str, default='0.1,0.1,0.1',
                                help='Background color as R,G,B (0-1 range)')
    webcam3d_parser.add_argument('--focal-length-x', type=float, default=470.4,
                                help='Camera focal length X in pixels (default: 470.4, used with --metric)')
    webcam3d_parser.add_argument('--focal-length-y', type=float, default=470.4,
                                help='Camera focal length Y in pixels (default: 470.4, used with --metric)')
    webcam3d_parser.add_argument('--principal-point-x', type=float, default=None,
                                help='Principal point X in pixels (default: image center)')
    webcam3d_parser.add_argument('--principal-point-y', type=float, default=None,
                                help='Principal point Y in pixels (default: image center)')
    webcam3d_parser.add_argument('--metric-depth-scale', type=float, default=0.01,
                                help='Scale factor for metric depth values (default: 0.01, try 0.001-0.1 if depth range seems wrong)')
    webcam3d_parser.set_defaults(func=webcam3d_command)

    # Screen3D Viewer command
    screen3d_viewer_parser = subparsers.add_parser('screen3d-viewer',
                                                   help='Real-time 3D mesh/point cloud viewer from screen capture')
    screen3d_viewer_parser.add_argument('--monitor', type=int, default=1,
                                       help='Monitor number to capture (1 = primary)')
    screen3d_viewer_parser.add_argument('--region', type=str, default=None,
                                       help='Capture region as "x,y,width,height"')
    screen3d_viewer_parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'],
                                       help='Model size (vits recommended for real-time)')
    screen3d_viewer_parser.add_argument('--metric', action='store_true', help='Use metric depth model')
    screen3d_viewer_parser.add_argument('--input-size', type=int, default=518, help='Input size for model')
    screen3d_viewer_parser.add_argument('--max-res', type=int, default=480,
                                       help='Maximum resolution (lower = faster)')
    screen3d_viewer_parser.add_argument('--fps', type=int, default=10, help='Target FPS')
    screen3d_viewer_parser.add_argument('--fp32', action='store_true', help='Use FP32 precision')
    screen3d_viewer_parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints',
                                       help='Checkpoints directory')
    screen3d_viewer_parser.add_argument('--depth-scale', type=float, default=0.5,
                                       help='Z-displacement scale (0.1-2.0, where 1.0 = depth spans half image width, default: 0.5)')
    screen3d_viewer_parser.add_argument('--depth-min-percentile', type=float, default=5.0,
                                       help='Clamp near depth at this percentile (default: 5, reduces extremes)')
    screen3d_viewer_parser.add_argument('--depth-max-percentile', type=float, default=95.0,
                                       help='Clamp far depth at this percentile (default: 95, reduces extremes)')
    screen3d_viewer_parser.add_argument('--depth-threshold', type=float, default=1.0,
                                       help='Filter pixels beyond this depth percentile (0-1, default: 1.0 keeps all pixels, 0.95 removes farthest 5%%)')
    screen3d_viewer_parser.add_argument('--raw-depth', action='store_true',
                                       help='Use raw depth values without normalization (preserves more detail, requires lower --depth-scale like 0.1-0.3)')
    screen3d_viewer_parser.add_argument('--display-mode', type=str, default='mesh', choices=['mesh', 'pointcloud'],
                                       help='Display mode: mesh (triangle mesh) or pointcloud (point cloud)')
    screen3d_viewer_parser.add_argument('--subsample', type=int, default=3,
                                       help='Geometry downsample factor (2-4 recommended, default: 3)')
    screen3d_viewer_parser.add_argument('--smooth', action='store_true',
                                       help='Enable mesh smoothing (slower, mesh mode only)')
    screen3d_viewer_parser.add_argument('--invert-depth', action='store_true',
                                       help='Invert depth values')
    screen3d_viewer_parser.add_argument('--background', type=str, default='0.1,0.1,0.1',
                                       help='Background color as R,G,B (0-1 range)')
    screen3d_viewer_parser.add_argument('--focal-length-x', type=float, default=470.4,
                                       help='Camera focal length X in pixels (default: 470.4, used with --metric)')
    screen3d_viewer_parser.add_argument('--focal-length-y', type=float, default=470.4,
                                       help='Camera focal length Y in pixels (default: 470.4, used with --metric)')
    screen3d_viewer_parser.add_argument('--principal-point-x', type=float, default=None,
                                       help='Principal point X in pixels (default: image center)')
    screen3d_viewer_parser.add_argument('--principal-point-y', type=float, default=None,
                                       help='Principal point Y in pixels (default: image center)')
    screen3d_viewer_parser.add_argument('--metric-depth-scale', type=float, default=0.01,
                                       help='Scale factor for metric depth values (default: 0.01, try 0.001-0.1 if depth range seems wrong)')
    screen3d_viewer_parser.set_defaults(func=screen3d_viewer_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
