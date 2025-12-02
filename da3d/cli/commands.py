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
import threading
from pathlib import Path

# Video-Depth-Anything imports (from original repo - must be in PYTHONPATH)
from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream
from utils.dc_utils import read_video_frames, save_video

# Da3d package imports (from this package)

from da3d.projection import DepthProjector, InteractiveParallaxController
from da3d.viewing import DepthMeshViewer, RealTime3DViewer
from da3d.viewing.gui_viewer import GUI3DViewer
from da3d.config import (
    MODEL_CONFIGS, 
    DEFAULT_FOCAL_LENGTH_X, 
    DEFAULT_FOCAL_LENGTH_Y,
    HQ_MAX_RES,
    HQ_INPUT_SIZE,
    HQ_SUBSAMPLE,
    HQ_SOR_NEIGHBORS,
    HQ_SOR_STD_RATIO,
    DEFAULT_SOR_NEIGHBORS,
    DEFAULT_SOR_STD_RATIO
)
from da3d.estimators.vda_estimator import VDAEstimator
from da3d.estimators.da3_estimator import DA3Estimator

def get_estimator(args, streaming=False):
    """Factory to get the correct estimator."""
    model_type = getattr(args, 'model_type', 'vda')

    # Get device with MPS fallback control
    enable_mps_fallback = getattr(args, 'mps_fallback', True)
    device = get_device(enable_mps_fallback=enable_mps_fallback)

    if model_type == 'da3':
        # DA3 doesn't support streaming mode in the same way, so we just init the estimator
        # We might need to map 'encoder' args if they differ
        config = {'encoder': args.encoder} # DA3 uses 'encoder' arg too?
        estimator = DA3Estimator(device=device)
        estimator.load_model(config)
        return estimator
    else:
        # VDA
        config = {
            'encoder': args.encoder,
            'checkpoint_path': None, # VDAEstimator loads based on encoder name usually
            'metric': args.metric,
            'input_size': args.input_size
        }
        # We need to construct the checkpoint path manually if VDAEstimator expects it
        # Or VDAEstimator can handle it.
        # Let's look at VDAEstimator again. It expects 'checkpoint_path' in config.

        checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
        checkpoint_path = Path(args.checkpoints_dir) / f'{checkpoint_name}_{args.encoder}.pth'

        config['checkpoint_path'] = str(checkpoint_path)

        estimator = VDAEstimator(device=device, streaming=streaming)
        estimator.load_model(config)
        return estimator


def get_device(enable_mps_fallback=True):
    """Get the best available device (CUDA, MPS, or CPU).

    Args:
        enable_mps_fallback: If True, automatically enable PYTORCH_ENABLE_MPS_FALLBACK
                           for MPS devices (default: True for convenience)
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if enable_mps_fallback:
            # Enable MPS fallback for unsupported operations (e.g., bicubic interpolation)
            # This allows operations not yet implemented on MPS to fall back to CPU
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        return 'mps'
    else:
        return 'cpu'


def video_command(args):
    """Process a video file for depth estimation."""
    print(f"Processing video: {args.input}")
    print(f"Model: {args.encoder}, Metric: {args.metric}, Streaming: {args.streaming}")

    DEVICE = get_device()
    if DEVICE == 'cpu':
        print("Warning: CUDA/MPS not available. Running on CPU will be very slow.")
    elif DEVICE == 'mps':
        print("Using Apple Metal Performance Shaders (MPS) acceleration.")

    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
    checkpoint_path = Path(args.checkpoints_dir) / f'{checkpoint_name}_{args.encoder}.pth'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please download checkpoints using: bash get_weights.sh")
        sys.exit(1)

    # Load model
    estimator = get_estimator(args, streaming=args.streaming)
    model = estimator # We'll need to adapt the process functions to use estimator
    
    # Refactoring process_video_* to use estimator is a bigger change.
    # For now, let's keep the old code for VDA if model_type is vda to minimize risk,
    # OR fully switch.
    # The plan said "Integrate estimators into commands".
    
    # If we are using VDA, we can extract the underlying model for compatibility
    # with existing process functions if we don't want to rewrite them all yet.
    if isinstance(estimator, VDAEstimator):
        video_depth_anything = estimator.model
    else:
        print("Error: Video processing currently only supports VDA model.")
        sys.exit(1)

    # video_depth_anything.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    # video_depth_anything = video_depth_anything.to(DEVICE).eval()

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
    print("Starting webcam depth estimation...")
    print(f"Model: {args.encoder}, Camera: {args.camera_id}")
    print("Press 'q' to quit, 's' to save current frame, 'r' to start/stop recording")

    DEVICE = get_device()
    if DEVICE == 'cpu':
        print("Warning: CUDA/MPS not available. Webcam mode will be very slow on CPU.")
    elif DEVICE == 'mps':
        print("Using Apple Metal Performance Shaders (MPS) acceleration.")

    # Load streaming model for webcam
    estimator = get_estimator(args, streaming=True)
    
    # For webcam, we can use the generic infer_depth interface!
    # But the existing code uses model.infer_video_depth_one directly.
    # Let's adapt the loop below.


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
            depth, confidence = estimator.infer_depth(frame_rgb)

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

    print("Starting screen capture depth estimation...")
    print(f"Model: {args.encoder}, Monitor: {args.monitor}, Target FPS: {args.fps}")
    print("Press 'q' to quit, 's' to save current frame, 'r' to start/stop recording")

    DEVICE = get_device()
    if DEVICE == 'cpu':
        print("Warning: CUDA/MPS not available. Screen capture mode will be very slow on CPU.")
    elif DEVICE == 'mps':
        print("Using Apple Metal Performance Shaders (MPS) acceleration.")

    # Load streaming model for screen capture
    estimator = get_estimator(args, streaming=True)


    # Setup screen capture
    sct = mss.mss()

    # Determine capture region
    if args.region:
        # Parse region string "x,y,width,height"
        try:
            x, y, width, height = map(int, args.region.split(','))
            monitor = {"top": y, "left": x, "width": width, "height": height}
            print(f"Capturing region: x={x}, y={y}, width={width}, height={height}")
        except ValueError:
            print("Error: Invalid region format. Use 'x,y,width,height' (e.g., '0,0,1920,1080')")
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
            depth, confidence = estimator.infer_depth(frame_rgb)

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

    print("Starting 3D screen capture with parallax effect...")
    print(f"Model: {args.encoder}, Monitor: {args.monitor}, Target FPS: {args.fps}")

    DEVICE = get_device()
    if DEVICE == 'cpu':
        print("Warning: CUDA/MPS not available. 3D mode will be very slow on CPU.")
    elif DEVICE == 'mps':
        print("Using Apple Metal Performance Shaders (MPS) acceleration.")

    # Load streaming model
    estimator = get_estimator(args, streaming=True)


    # Setup screen capture
    sct = mss.mss()

    # Determine capture region
    if args.region:
        try:
            x, y, width, height = map(int, args.region.split(','))
            monitor = {"top": y, "left": x, "width": width, "height": height}
            print(f"Capturing region: x={x}, y={y}, width={width}, height={height}")
        except ValueError:
            print("Error: Invalid region format. Use 'x,y,width,height'")
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
            print("  You can now add 'OBS Virtual Camera' as a source in OBS Studio")
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
        depth, confidence = estimator.infer_depth(display_frame)


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

    print("Loading 3D mesh viewer...")
    print("Loading 3D mesh viewer...")
    print(f"Image: {args.image}")
    
    depth_map = args.depth
    if depth_map is None:
        print(f"Depth map not provided. Inferring using model: {args.model_type} ({args.encoder})...")
        
        # Load image
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            sys.exit(1)
            
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            sys.exit(1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load estimator
        DEVICE = get_device()
        if DEVICE == 'cpu':
            print("Warning: CUDA/MPS not available. Inference will be slow.")
            
        estimator = get_estimator(args, streaming=False)
        
        # Run inference
        print("Running inference...")
        depth, confidence = estimator.infer_depth(image)
        
        # Save inferred depth temporarily or just pass it? 
        # DepthMeshViewer expects a path or numpy array?
        # Looking at DepthMeshViewer.process_and_view signature:
        # process_and_view(self, image_path, depth_path_or_array, ...)
        # It seems it might handle array if I check the code, but the docstring said path.
        # Let's check DepthMeshViewer later. For now assume it handles array or we save it.
        
        # Let's save it to a temp file to be safe and consistent with existing usage
        output_dir = Path("temp_depths")
        output_dir.mkdir(exist_ok=True)
        depth_map = str(output_dir / f"{Path(args.image).stem}_depth.png")
        
        # Normalize for saving as PNG (DepthMeshViewer might expect normalized or raw?)
        # If metric, we should save as .npy or .exr to preserve values.
        # If relative, PNG is fine.
        
        if args.metric:
            depth_map = str(output_dir / f"{Path(args.image).stem}_depth.npy")
            np.save(depth_map, depth)
            print(f"Saved temporary metric depth to {depth_map}")
        else:
            # Normalize to 0-255
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-8:
                depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)
            cv2.imwrite(depth_map, depth_norm)
            print(f"Saved temporary depth to {depth_map}")
            
    print(f"Depth: {depth_map}")

    if args.metric:
        print("Metric depth mode enabled")
        print(f"  Focal length: fx={args.focal_length_x:.1f}, fy={args.focal_length_y:.1f}")
        if args.principal_point_x is not None and args.principal_point_y is not None:
            print(f"  Principal point: cx={args.principal_point_x:.1f}, cy={args.principal_point_y:.1f}")
        else:
            print("  Principal point: image center (auto)")

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
        depth_map,
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

    print("Starting real-time 3D webcam viewer...")
    print(f"Model: {args.encoder}, Camera: {args.camera_id}")

    DEVICE = get_device()
    if DEVICE == 'cpu':
        print("Warning: CUDA/MPS not available. 3D webcam mode will be very slow on CPU.")
    elif DEVICE == 'mps':
        print("Using Apple Metal Performance Shaders (MPS) acceleration.")

    # Load streaming model
    estimator = get_estimator(args, streaming=True)

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

    # Apply High Quality Preset if requested
    if args.high_quality:
        print("\n[INFO] High Quality Preset Enabled")
        print(f"  - Resolution: {HQ_MAX_RES}p")
        print(f"  - Input Size: {HQ_INPUT_SIZE} (High Detail)")
        print(f"  - Subsample: {HQ_SUBSAMPLE}")
        print("  - Metric Depth: Enabled")
        print(f"  - SOR: Neighbors={HQ_SOR_NEIGHBORS}, Std Ratio={HQ_SOR_STD_RATIO}")
        args.max_res = HQ_MAX_RES
        args.input_size = HQ_INPUT_SIZE
        args.subsample = HQ_SUBSAMPLE
        args.metric = True
        args.sor_neighbors = HQ_SOR_NEIGHBORS
        args.sor_std_ratio = HQ_SOR_STD_RATIO
        # Ensure focal lengths are set if not manually overridden
        if args.focal_length_x == DEFAULT_FOCAL_LENGTH_X:
             args.focal_length_x = DEFAULT_FOCAL_LENGTH_X # Keep default or adjust if needed

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
                depth, confidence = estimator.infer_depth(frame_rgb)

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
                    metric_depth_scale=args.metric_depth_scale if hasattr(args, 'metric_depth_scale') else 1.0,
                    sor_neighbors=args.sor_neighbors,
                    sor_std_ratio=args.sor_std_ratio
                )
                viewer_3d.initialize(width=1280, height=720)
                print(f"Initialized 3D viewer: {w}x{h} ({args.display_mode} mode)")
                if args.metric:
                    print(f"Metric depth: fx={args.focal_length_x:.1f}, fy={args.focal_length_y:.1f}")
                print(f"Depth range: {args.depth_min_percentile}%-{args.depth_max_percentile}% percentile")

                # Register 'X' key for "Capture and View"
                def capture_and_view(vis):
                    if viewer_3d.current_image is not None:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        temp_dir = Path("temp_captures")
                        temp_dir.mkdir(exist_ok=True)
                        image_path = temp_dir / f"capture_{timestamp}.png"
                        
                        # Save RGB image
                        cv2.imwrite(str(image_path), cv2.cvtColor(viewer_3d.current_image, cv2.COLOR_RGB2BGR))
                        print(f"\n[INFO] Captured frame to {image_path}")
                        print("[INFO] Launching high-quality 3D viewer with DA3...")
                        
                        # Launch view3d in a separate process
                        import subprocess
                        subprocess.Popen([
                            "da3d", "view3d", str(image_path), 
                            "--model-type", "da3", 
                            "--encoder", "da3-large"
                        ])
                        return False # Don't update geometry in this callback

                viewer_3d.register_key_callback(ord('X'), capture_and_view)
                print("Controls: [X] Capture and View with DA3, [Q] Quit")

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
    except ImportError:
        print("Error: open3d not installed.")
        print("Install with: uv sync --extra metric")
        sys.exit(1)

    try:
        import mss
    except ImportError:
        print("Error: mss not installed.")
        print("Install with: uv sync")
        sys.exit(1)

    print("Starting real-time 3D screen viewer...")
    print(f"Model: {args.encoder}, Monitor: {args.monitor}")

    DEVICE = get_device()
    if DEVICE == 'cpu':
        print("Warning: CUDA/MPS not available. 3D mode will be very slow on CPU.")
    elif DEVICE == 'mps':
        print("Using Apple Metal Performance Shaders (MPS) acceleration.")

    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
    checkpoint_path = Path(args.checkpoints_dir) / f'{checkpoint_name}_{args.encoder}.pth'

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please download checkpoints using: bash get_weights.sh")
        sys.exit(1)

    # Apply High Quality Preset if requested
    if args.high_quality:
        print("\n[INFO] High Quality Preset Enabled")
        print("  - Resolution: 960p")
        print("  - Subsample: 2")
        print("  - Metric Depth: Enabled")
        print("  - SOR: Neighbors=100, Std Ratio=0.5")
        args.max_res = 960
        args.subsample = 2
        args.metric = True
        args.sor_neighbors = 100
        args.sor_std_ratio = 0.5
        # Ensure focal lengths are set if not manually overridden
        if args.focal_length_x == 470.4: # Default value
             args.focal_length_x = 470.4 # Keep default or adjust if needed


    # Load streaming model
    estimator = get_estimator(args, streaming=True)

    # Setup screen capture
    sct = mss.mss()

    # Determine capture region
    if args.region:
        try:
            x, y, width, height = map(int, args.region.split(','))
            monitor = {"top": y, "left": x, "width": width, "height": height}
            print(f"Capturing region: x={x}, y={y}, width={width}, height={height}")
        except:
            print("Error: Invalid region format. Use 'x,y,width,height'")
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

    if args.gui:
        print("\nStarting GUI 3D Viewer...")
        viewer = GUI3DViewer(window_name="Depth Anything 3D - Screen Viewer")
        
        # Configure initial settings from args
        viewer.depth_scale = args.depth_scale
        viewer.subsample = args.subsample
        viewer.display_mode = args.display_mode
        viewer.use_metric = args.metric
        viewer.sor_neighbors = args.sor_neighbors
        viewer.sor_std_ratio = args.sor_std_ratio
        if args.metric:
            viewer.focal_length_x = args.focal_length_x
            viewer.focal_length_y = args.focal_length_y
        
        # Thread-local storage for mss
        thread_data = threading.local()

        def data_provider():
            loop_start = time.time()
            
            # Initialize mss for this thread if needed
            if not hasattr(thread_data, 'sct'):
                import mss
                thread_data.sct = mss.mss()

            # Capture
            try:
                screenshot = thread_data.sct.grab(monitor)
            except Exception as e:
                print(f"Capture error: {e}")
                return None

            frame = np.array(screenshot)[:, :, :3]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            
            # Resize
            h, w = frame_rgb.shape[:2]
            if args.max_res > 0 and max(h, w) > args.max_res:
                scale = args.max_res / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
                
            # Infer
            with torch.no_grad():
                depth, _ = estimator.infer_depth(frame_rgb)
                
            # FPS Limit
            loop_elapsed = time.time() - loop_start
            if loop_elapsed < target_frame_time:
                time.sleep(target_frame_time - loop_elapsed)
                
            return frame_rgb, depth
            
        try:
            viewer.start(data_provider)
        except KeyboardInterrupt:
            pass
        print("GUI session ended.")
        return

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
                depth, confidence = estimator.infer_depth(frame_rgb)

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
                    metric_depth_scale=args.metric_depth_scale if hasattr(args, 'metric_depth_scale') else 1.0,
                    sor_neighbors=args.sor_neighbors,
                    sor_std_ratio=args.sor_std_ratio
                )
                viewer_3d.initialize(width=1280, height=720)
                print(f"Initialized 3D viewer: {w}x{h} ({args.display_mode} mode)")
                if args.metric:
                    print(f"Metric depth: fx={args.focal_length_x:.1f}, fy={args.focal_length_y:.1f}")
                print(f"Depth range: {args.depth_min_percentile}%-{args.depth_max_percentile}% percentile")

                # Register 'X' key for "Capture and View"
                def capture_and_view(vis):
                    if viewer_3d.current_image is not None:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        temp_dir = Path("temp_captures")
                        temp_dir.mkdir(exist_ok=True)
                        image_path = temp_dir / f"capture_{timestamp}.png"
                        
                        # Save RGB image
                        cv2.imwrite(str(image_path), cv2.cvtColor(viewer_3d.current_image, cv2.COLOR_RGB2BGR))
                        print(f"\n[INFO] Captured frame to {image_path}")
                        print("[INFO] Launching high-quality 3D viewer with DA3...")
                        
                        # Launch view3d in a separate process
                        import subprocess
                        subprocess.Popen([
                            "da3d", "view3d", str(image_path), 
                            "--model-type", "da3", 
                            "--encoder", "da3-large"
                        ])
                        return False

                viewer_3d.register_key_callback(ord('X'), capture_and_view)
                print("Controls: [X] Capture and View with DA3, [Q] Quit")

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


def projector_preview_command(args):
    """Run projection preview."""
    from da3d.projection.engine import ProjectionEngine
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
        
    engine = ProjectionEngine(args.config)
    engine.run_preview(args.show)

def projector_calibrate_command(args):
    """Run projector calibration UI."""
    from da3d.projection.calibration import CalibrationApp
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
        
    app = CalibrationApp(args.config, args.projector)
    app.run()

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
        description='Video Depth Anything 3D - CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video file
  da3d video input.mp4 -o outputs/

  # Use webcam for real-time depth
  da3d webcam

  # Capture screen with depth estimation
  da3d screen

  # Capture screen with 2.5D parallax effect
  da3d screen3d --auto-rotate

  # Stream 3D screen to OBS Virtual Camera
  da3d screen3d --virtual-cam --mouse-control

  # Process video with metric depth
  da3d video input.mp4 --metric

  # Launch web demo
  da3d demo

  # View depth map in true 3D
  da3d view3d image.jpg depth.png

  # Real-time 3D webcam viewer
  da3d webcam3d

  # Real-time 3D screen capture viewer
  da3d screen3d-viewer
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
    screen3d_parser.add_argument('--no-mps-fallback', dest='mps_fallback', action='store_false', default=True,
                                help='Disable automatic MPS fallback for unsupported operations (macOS only)')
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
    video_parser.add_argument('--no-mps-fallback', dest='mps_fallback', action='store_false', default=True,
                             help='Disable automatic MPS fallback for unsupported operations (macOS only)')
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
    webcam_parser.add_argument('--no-mps-fallback', dest='mps_fallback', action='store_false', default=True,
                              help='Disable automatic MPS fallback for unsupported operations (macOS only)')
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
    view3d_parser.add_argument('depth', type=str, nargs='?', default=None, help='Path to depth map (optional, inferred if not provided)')
    view3d_parser.add_argument('--model-type', type=str, default='vda', choices=['vda', 'da3'], help='Model type to use for inference')
    view3d_parser.add_argument('--encoder', type=str, default='vitl', help='Model encoder/size')
    view3d_parser.add_argument('--input-size', type=int, default=518, help='Input size for model')
    view3d_parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints', help='Checkpoints directory')
    
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
                                help='Clamp far depth (background) at this percentile (default: 0, preserves background)')
    webcam3d_parser.add_argument('--depth-max-percentile', type=float, default=100.0,
                                help='Clamp near depth (foreground) at this percentile (default: 100, preserves foreground details)')
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
    webcam3d_parser.add_argument('--metric-depth-scale', type=float, default=1.0,
                                help='Scale factor for metric depth values (default: 1.0, assumes 1/depth output)')
    webcam3d_parser.add_argument('--sor-neighbors', type=int, default=50,
                                help='Number of neighbors for Statistical Outlier Removal (default: 50)')
    webcam3d_parser.add_argument('--sor-std-ratio', type=float, default=1.0,
                                help='Standard deviation ratio for Statistical Outlier Removal (default: 1.0, lower = more aggressive)')
    webcam3d_parser.add_argument('--high-quality', action='store_true',
                                help='Enable high-quality preset (1024p input, subsample=2, metric depth, optimized SOR)')
    webcam3d_parser.add_argument('--no-mps-fallback', dest='mps_fallback', action='store_false', default=True,
                                help='Disable automatic MPS fallback for unsupported operations (macOS only)')
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
                                       help='Clamp far depth (background) at this percentile (default: 5, reduces background noise)')
    screen3d_viewer_parser.add_argument('--depth-max-percentile', type=float, default=100.0,
                                       help='Clamp near depth (foreground) at this percentile (default: 100, preserves foreground details)')
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
    screen3d_viewer_parser.add_argument('--metric-depth-scale', type=float, default=0.005,
                                       help='Scale factor for metric depth values (default: 0.005, adjusted for typical webcam)')
    screen3d_viewer_parser.add_argument('--sor-neighbors', type=int, default=50,
                                       help='Number of neighbors for Statistical Outlier Removal (default: 50)')
    screen3d_viewer_parser.add_argument('--sor-std-ratio', type=float, default=1.0,
                                       help='Standard deviation ratio for Statistical Outlier Removal (default: 1.0, lower = more aggressive)')
    screen3d_viewer_parser.add_argument('--high-quality', action='store_true',
                                       help='Enable high-quality preset (960p, subsample=2, metric depth, optimized SOR)')
    screen3d_viewer_parser.add_argument('--gui', action='store_true',
                                       help='Enable experimental GUI controls')
    screen3d_viewer_parser.add_argument('--no-mps-fallback', dest='mps_fallback', action='store_false', default=True,
                                       help='Disable automatic MPS fallback for unsupported operations (macOS only)')
    screen3d_viewer_parser.set_defaults(func=screen3d_viewer_command)

    # Projector Preview
    preview_parser = subparsers.add_parser('projector-preview', help='Preview projection show')
    preview_parser.add_argument('--config', type=str, required=True, help='Path to projection config YAML')
    preview_parser.add_argument('--show', type=str, required=True, help='Name of show to preview')
    preview_parser.set_defaults(func=projector_preview_command)

    # Projector Calibrate
    calibrate_parser = subparsers.add_parser('projector-calibrate', help='Calibrate projector surfaces')
    calibrate_parser.add_argument('--config', type=str, required=True, help='Path to projection config YAML')
    calibrate_parser.add_argument('--projector', type=str, required=True, help='Name of projector to calibrate')
    calibrate_parser.set_defaults(func=projector_calibrate_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
