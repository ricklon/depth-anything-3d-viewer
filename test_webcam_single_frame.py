#!/usr/bin/env python3
"""
Single-frame webcam test to identify depth-to-3D mapping errors.

This script captures ONE webcam frame and tests all 3D visualization variations:
1. Relative depth (normalized 0-1)
2. Metric depth with perspective projection
3. Different depth scales
4. Different percentile ranges
5. Mesh vs point cloud modes

This helps identify where the depth-to-3D mapping might be going wrong.
"""

import os
import sys
import numpy as np
import cv2
import torch
import time
from pathlib import Path

# Add Video-Depth-Anything to path (same logic as da3d/cli/main.py)
parent_dir = Path(__file__).parent.parent
video_depth_sibling = parent_dir / 'Video-Depth-Anything'
if (video_depth_sibling / 'video_depth_anything').exists():
    sys.path.insert(0, str(video_depth_sibling))
elif (parent_dir / 'video_depth_anything').exists():
    sys.path.insert(0, str(parent_dir))

# Import Video-Depth-Anything
try:
    from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream
except ImportError as e:
    print("=" * 70)
    print("ERROR: Could not import Video-Depth-Anything")
    print("=" * 70)
    print(f"\nDetails: {e}\n")
    print("Please ensure Video-Depth-Anything is cloned as a sibling directory:")
    print("  cd ..")
    print("  git clone https://github.com/DepthAnything/Video-Depth-Anything")
    print("=" * 70)
    sys.exit(1)

# Import our viewing module
from da3d.viewing.mesh import DepthMeshViewer

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def capture_webcam_frame(camera_id=0, max_res=640):
    """Capture a single frame from webcam."""
    print(f"\n{'='*60}")
    print("STEP 1: Capturing webcam frame")
    print(f"{'='*60}")

    # Try different backends for Windows
    cap = None
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto"),
    ]

    for backend, name in backends:
        print(f"  Trying {name} backend...")
        cap = cv2.VideoCapture(camera_id, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  [OK] Successfully opened camera using {name}")
                break
            else:
                cap.release()
                cap = None
        else:
            cap = None

    if cap is None:
        print(f"  [ERROR] Could not open camera {camera_id}")
        return None

    # Read a few frames to let camera adjust (auto-exposure, etc.)
    print("  Waiting for camera to adjust...")
    for i in range(10):
        ret, frame = cap.read()
        time.sleep(0.05)

    # Capture the actual frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  [ERROR] Failed to capture frame")
        return None

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize if needed
    h, w = frame_rgb.shape[:2]
    if max(h, w) > max_res:
        scale = max_res / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        print(f"  Resized: {w}x{h} -> {new_w}x{new_h}")
    else:
        print(f"  Size: {w}x{h}")

    print(f"  [OK] Captured frame: {frame_rgb.shape}")
    return frame_rgb


def generate_depth(frame_rgb, encoder='vits', use_metric=False, checkpoints_dir='./checkpoints'):
    """Generate depth map from image."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Generating {'metric' if use_metric else 'relative'} depth map")
    print(f"{'='*60}")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {DEVICE}")

    # Load model
    checkpoint_name = 'metric_video_depth_anything' if use_metric else 'video_depth_anything'
    checkpoint_path = Path(checkpoints_dir) / f'{checkpoint_name}_{encoder}.pth'

    if not checkpoint_path.exists():
        print(f"  [ERROR] Checkpoint not found: {checkpoint_path}")
        return None

    print(f"  Loading model: {checkpoint_path}")
    model = VideoDepthAnythingStream(**MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()

    # Generate depth
    print("  Inferring depth...")
    with torch.no_grad():
        depth = model.infer_video_depth_one(frame_rgb, input_size=518, device=DEVICE)

    print(f"  [OK] Depth generated: {depth.shape}")
    print(f"       Range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"       Mean: {depth.mean():.4f}, Std: {depth.std():.4f}")

    # Analyze depth distribution
    percentiles = [0, 5, 25, 50, 75, 95, 100]
    print(f"       Percentiles: ", end="")
    for p in percentiles:
        val = np.percentile(depth, p)
        print(f"{p}%={val:.3f} ", end="")
    print()

    return depth


def analyze_3d_mapping(depth, frame_rgb):
    """Analyze the depth-to-3D coordinate mapping for potential issues."""
    print(f"\n{'='*60}")
    print("STEP 3: Analyzing depth-to-3D mapping")
    print(f"{'='*60}")

    h, w = depth.shape
    print(f"  Image size: {w}x{h}")

    # Check for NaN or Inf values
    if np.any(np.isnan(depth)):
        print("  [WARNING] NaN values in depth map!")
    if np.any(np.isinf(depth)):
        print("  [WARNING] Inf values in depth map!")

    # Analyze depth normalization
    depth_min = depth.min()
    depth_max = depth.max()
    depth_range = depth_max - depth_min

    print(f"\n  Depth statistics:")
    print(f"    Min: {depth_min:.6f}")
    print(f"    Max: {depth_max:.6f}")
    print(f"    Range: {depth_range:.6f}")

    if depth_range < 1e-8:
        print("  [ERROR] Depth range too small - will cause division by zero!")
        return

    # Simulate the normalization that happens in create_mesh_from_depth
    depth_normalized = (depth - depth_min) / (depth_max - depth_min)

    print(f"\n  After normalization (0-1):")
    print(f"    Min: {depth_normalized.min():.6f}")
    print(f"    Max: {depth_normalized.max():.6f}")

    # Check what happens at different depth scales
    print(f"\n  3D Z-coordinate analysis at different depth_scale values:")
    for scale in [0.1, 0.3, 0.5, 0.8, 1.0]:
        # Calculate z_scale_factor as in the code
        max_dim = max(w, h)
        z_scale_factor = max_dim * 0.5
        z_values = depth_normalized * scale * z_scale_factor
        print(f"    scale={scale}: Z range = [{z_values.min():.1f}, {z_values.max():.1f}] pixels")

    # Test metric depth calculation
    print(f"\n  Metric depth perspective projection test:")
    # Typical webcam focal length
    fx = fy = 470.4
    cx, cy = w / 2.0, h / 2.0

    # Create coordinate grids
    x = np.arange(w)
    y = np.arange(h)
    x_grid, y_grid = np.meshgrid(x, y)

    # Assuming depth is already in meters (typical metric depth output)
    z = depth

    # Perspective projection
    x_3d = (x_grid - cx) * z / fx
    y_3d = (y_grid - cy) * z / fy

    print(f"    Focal length: {fx}")
    print(f"    Principal point: ({cx:.1f}, {cy:.1f})")
    print(f"    X range: [{x_3d.min():.3f}, {x_3d.max():.3f}] meters")
    print(f"    Y range: [{y_3d.min():.3f}, {y_3d.max():.3f}] meters")
    print(f"    Z range: [{z.min():.3f}, {z.max():.3f}] meters")

    # Check if metric depth values make sense
    if depth.max() > 100:
        print("  [WARNING] Depth values > 100 - might not be in meters!")
        print("           Try using --metric-depth-scale to adjust")

    # Analyze aspect ratio handling
    print(f"\n  Aspect ratio analysis:")
    aspect_ratio = w / h
    print(f"    Aspect ratio: {aspect_ratio:.3f}")

    if w >= h:
        x_centered = np.arange(w) - w / 2
        y_centered = (np.arange(h) - h / 2) * aspect_ratio
    else:
        x_centered = (np.arange(w) - w / 2) / aspect_ratio
        y_centered = np.arange(h) - h / 2

    print(f"    X centered range: [{x_centered.min():.1f}, {x_centered.max():.1f}]")
    print(f"    Y centered range: [{y_centered.min():.1f}, {y_centered.max():.1f}]")


def test_3d_variations(frame_rgb, depth, output_dir='./test_outputs'):
    """Test different 3D visualization variations and show them sequentially."""
    print(f"\n{'='*60}")
    print("STEP 4: Testing 3D visualization variations")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    h, w = depth.shape

    # Define test variations
    variations = [
        {
            'name': '1. Relative depth (default settings)',
            'params': {
                'depth_scale': 0.5,
                'depth_min_percentile': 0.0,
                'depth_max_percentile': 100.0,
                'display_mode': 'mesh',
                'use_metric_depth': False,
            }
        },
        {
            'name': '2. Relative depth (webcam defaults: 0-95%)',
            'params': {
                'depth_scale': 0.5,
                'depth_min_percentile': 0.0,
                'depth_max_percentile': 95.0,
                'display_mode': 'mesh',
                'use_metric_depth': False,
            }
        },
        {
            'name': '3. Relative depth (RAW - proportional)',
            'params': {
                'depth_scale': 0.05,  # Much smaller - raw values are ~0-6, not 0-1
                'depth_min_percentile': 0.0,
                'depth_max_percentile': 95.0,
                'display_mode': 'mesh',
                'use_metric_depth': False,
                'use_raw_depth': True,
            }
        },
        {
            'name': '4. Relative depth (RAW - stronger scale)',
            'params': {
                'depth_scale': 0.1,  # Stronger depth effect
                'depth_min_percentile': 0.0,
                'depth_max_percentile': 95.0,
                'display_mode': 'mesh',
                'use_metric_depth': False,
                'use_raw_depth': True,
            }
        },
        {
            'name': '5. Relative depth (normalized - for comparison)',
            'params': {
                'depth_scale': 0.5,
                'depth_min_percentile': 0.0,
                'depth_max_percentile': 95.0,
                'display_mode': 'mesh',
                'use_metric_depth': False,
                'use_raw_depth': False,
            }
        },
        {
            'name': '6. Metric depth (default scale)',
            'params': {
                'depth_scale': 0.5,  # ignored for metric
                'display_mode': 'mesh',
                'use_metric_depth': True,
                'focal_length_x': 470.4,
                'focal_length_y': 470.4,
                'metric_depth_scale': 1.0,
            }
        },
        {
            'name': '7. Metric depth (scaled 0.1x)',
            'params': {
                'depth_scale': 0.5,
                'display_mode': 'mesh',
                'use_metric_depth': True,
                'focal_length_x': 470.4,
                'focal_length_y': 470.4,
                'metric_depth_scale': 0.1,
            }
        },
        {
            'name': '8. Metric depth (scaled 0.01x)',
            'params': {
                'depth_scale': 0.5,
                'display_mode': 'mesh',
                'use_metric_depth': True,
                'focal_length_x': 470.4,
                'focal_length_y': 470.4,
                'metric_depth_scale': 0.01,
            }
        },
        {
            'name': '9. Metric depth (point cloud)',
            'params': {
                'depth_scale': 0.5,
                'display_mode': 'pointcloud',
                'use_metric_depth': True,
                'focal_length_x': 470.4,
                'focal_length_y': 470.4,
                'metric_depth_scale': 0.1,
            }
        },
        {
            'name': '10. Inverted depth (near becomes far)',
            'params': {
                'depth_scale': 0.5,
                'depth_min_percentile': 0.0,
                'depth_max_percentile': 95.0,
                'display_mode': 'mesh',
                'use_metric_depth': False,
            },
            'invert_depth': True,
        },
    ]

    print(f"\n  Will test {len(variations)} variations")
    print("  Close each 3D viewer window to proceed to the next variation")
    print("  Press 'q' or ESC in the 3D viewer to close it\n")

    # Save the captured frame and depth for reference
    cv2.imwrite(str(Path(output_dir) / 'captured_frame.png'), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    # Save depth as colored visualization
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    import matplotlib.cm as cm
    colormap = cm.get_cmap("inferno")
    depth_colored = (colormap(depth_norm / 255.0)[:, :, :3] * 255).astype(np.uint8)
    cv2.imwrite(str(Path(output_dir) / 'depth_colormap.png'), cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))

    # Also save raw depth as .npy
    np.save(str(Path(output_dir) / 'depth_raw.npy'), depth)

    print(f"  Saved test files to {output_dir}/")
    print(f"    - captured_frame.png")
    print(f"    - depth_colormap.png")
    print(f"    - depth_raw.npy")

    # Test each variation
    for i, var in enumerate(variations):
        print(f"\n  Testing variation {var['name']}...")
        params = var['params']

        # Print parameters
        for key, val in params.items():
            print(f"    {key}: {val}")

        try:
            # Create viewer with these parameters
            viewer = DepthMeshViewer(**params)

            # Create mesh
            invert = var.get('invert_depth', False)
            mesh = viewer.create_mesh_from_depth(
                frame_rgb.copy(),
                depth.copy(),
                subsample=2,
                invert_depth=invert,
                smooth_mesh=False  # Disable smoothing for faster testing
            )

            # Count vertices/triangles
            if hasattr(mesh, 'vertices'):
                n_vertices = len(mesh.vertices)
                n_triangles = len(mesh.triangles) if hasattr(mesh, 'triangles') else 0
                print(f"    Mesh created: {n_vertices} vertices, {n_triangles} triangles")
            elif hasattr(mesh, 'points'):
                n_points = len(mesh.points)
                print(f"    Point cloud created: {n_points} points")

            # Display
            print(f"    Opening 3D viewer...")
            viewer.view_mesh(
                mesh,
                window_name=f"Test {i+1}: {var['name']}",
                width=1024,
                height=768,
                show_wireframe=False
            )

        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")


def main():
    """Main test function."""
    print("\n" + "="*60)
    print(" SINGLE FRAME WEBCAM 3D MAPPING TEST")
    print("="*60)
    print("\nThis test will:")
    print("  1. Capture one frame from webcam")
    print("  2. Generate depth map (relative depth)")
    print("  3. Analyze the depth-to-3D coordinate mapping")
    print("  4. Test different 3D visualization variations")
    print("\nClose each 3D viewer window to proceed to the next test.")
    print("="*60)

    # Configuration
    CAMERA_ID = 0
    MAX_RES = 640
    ENCODER = 'vits'
    CHECKPOINTS_DIR = './checkpoints'
    OUTPUT_DIR = './test_outputs'

    # Step 1: Capture webcam frame
    frame_rgb = capture_webcam_frame(camera_id=CAMERA_ID, max_res=MAX_RES)
    if frame_rgb is None:
        print("\n[FATAL] Could not capture webcam frame. Exiting.")
        return

    # Step 2: Generate relative depth (metric depth requires different model)
    depth = generate_depth(
        frame_rgb,
        encoder=ENCODER,
        use_metric=False,
        checkpoints_dir=CHECKPOINTS_DIR
    )
    if depth is None:
        print("\n[FATAL] Could not generate depth. Exiting.")
        return

    # Step 3: Analyze depth-to-3D mapping
    analyze_3d_mapping(depth, frame_rgb)

    # Step 4: Test 3D variations
    test_3d_variations(frame_rgb, depth, output_dir=OUTPUT_DIR)

    print("\n" + "="*60)
    print(" TEST COMPLETE")
    print("="*60)
    print(f"\nTest files saved to: {OUTPUT_DIR}/")
    print("You can re-view any variation using:")
    print(f"  da3d view3d {OUTPUT_DIR}/captured_frame.png {OUTPUT_DIR}/depth_raw.npy [options]")


if __name__ == '__main__':
    main()
