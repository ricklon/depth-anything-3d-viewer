#!/usr/bin/env python3
"""
Diagnostic script to identify specific depth-to-3D mapping errors.

This script focuses on the mathematical transformations and prints detailed
diagnostics to find exactly where the mapping might be going wrong.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent dirs for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_depth_pattern():
    """Create a synthetic depth map with known patterns for testing."""
    h, w = 480, 640

    # Create a gradient depth map (left=near, right=far)
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

    # Horizontal gradient
    depth_horizontal = X.copy()

    # Vertical gradient
    depth_vertical = Y.copy()

    # Radial (center=near, edges=far)
    cx, cy = 0.5, 0.5
    depth_radial = np.sqrt((X - cx)**2 + (Y - cy)**2) / np.sqrt(0.5)
    depth_radial = np.clip(depth_radial, 0, 1)

    # Sphere (center bulges out)
    depth_sphere = 1.0 - np.sqrt(np.maximum(0, 1 - 4*((X - 0.5)**2 + (Y - 0.5)**2)))

    return {
        'horizontal': depth_horizontal,
        'vertical': depth_vertical,
        'radial': depth_radial,
        'sphere': depth_sphere,
    }


def test_relative_depth_mapping(depth, width, height, depth_scale=0.5):
    """Test the relative depth coordinate mapping (orthographic projection)."""
    print(f"\n{'='*60}")
    print("RELATIVE DEPTH MAPPING TEST")
    print(f"{'='*60}")

    # Step 1: Percentile clamping (using defaults)
    depth_min_pct = 0.0
    depth_max_pct = 100.0
    d_min = np.percentile(depth, depth_min_pct)
    d_max = np.percentile(depth, depth_max_pct)

    print(f"\nStep 1: Percentile clamping ({depth_min_pct}%-{depth_max_pct}%)")
    print(f"  Original range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"  Clamped range: [{d_min:.4f}, {d_max:.4f}]")

    depth_clamped = np.clip(depth, d_min, d_max)

    # Step 2: Normalize to 0-1
    if d_max - d_min > 1e-8:
        depth_normalized = (depth_clamped - d_min) / (d_max - d_min)
    else:
        depth_normalized = np.zeros_like(depth_clamped)
        print("  [ERROR] Depth range too small!")

    print("\nStep 2: Normalization to 0-1")
    print(f"  Normalized range: [{depth_normalized.min():.4f}, {depth_normalized.max():.4f}]")

    # Step 3: Create centered coordinates
    aspect_ratio = width / height
    max_dim = max(width, height)

    print("\nStep 3: Coordinate centering")
    print(f"  Aspect ratio: {aspect_ratio:.3f}")
    print(f"  Max dimension: {max_dim}")

    if width >= height:
        x_range = [-width/2, width/2]
        y_range = [(-height/2) * aspect_ratio, (height/2) * aspect_ratio]
        print("  Mode: Landscape/Square")
    else:
        x_range = [(-width/2) / aspect_ratio, (width/2) / aspect_ratio]
        y_range = [-height/2, height/2]
        print("  Mode: Portrait")

    print(f"  X range: [{x_range[0]:.1f}, {x_range[1]:.1f}]")
    print(f"  Y range: [{y_range[0]:.1f}, {y_range[1]:.1f}]")

    # Step 4: Z scaling
    z_scale_factor = max_dim * 0.5
    z_values = depth_normalized * depth_scale * z_scale_factor

    print("\nStep 4: Z scaling")
    print(f"  depth_scale: {depth_scale}")
    print(f"  z_scale_factor: {z_scale_factor}")
    print(f"  Z range: [{z_values.min():.1f}, {z_values.max():.1f}]")

    # Check coordinate space ratios
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    z_span = z_values.max() - z_values.min()

    print("\nCoordinate space analysis:")
    print(f"  X span: {x_span:.1f}")
    print(f"  Y span: {y_span:.1f}")
    print(f"  Z span: {z_span:.1f}")
    print(f"  X:Y ratio: {x_span/y_span:.3f}")
    print(f"  Z:X ratio: {z_span/x_span:.3f}")
    print(f"  Z:Y ratio: {z_span/y_span:.3f}")

    # Check specific corner and center points
    print("\nSample 3D points:")

    # Center point
    cy, cx = height // 2, width // 2
    print(f"  Center [{cx}, {cy}]:")
    print(f"    Depth: {depth[cy, cx]:.4f}")
    print(f"    Normalized: {depth_normalized[cy, cx]:.4f}")
    print(f"    Z: {z_values[cy, cx]:.2f}")

    # Corners
    corners = [
        ("Top-left", 0, 0),
        ("Top-right", 0, width-1),
        ("Bottom-left", height-1, 0),
        ("Bottom-right", height-1, width-1),
    ]

    for name, y, x in corners:
        print(f"  {name} [{x}, {y}]:")
        print(f"    Depth: {depth[y, x]:.4f}, Normalized: {depth_normalized[y, x]:.4f}, Z: {z_values[y, x]:.2f}")

    return z_values


def test_metric_depth_mapping(depth, width, height, fx=470.4, fy=470.4, metric_scale=1.0):
    """Test the metric depth coordinate mapping (perspective projection)."""
    print(f"\n{'='*60}")
    print("METRIC DEPTH MAPPING TEST")
    print(f"{'='*60}")

    # Step 1: Scale depth values
    depth_scaled = depth * metric_scale

    print("\nStep 1: Metric scaling")
    print(f"  metric_depth_scale: {metric_scale}")
    print(f"  Original range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"  Scaled range: [{depth_scaled.min():.4f}, {depth_scaled.max():.4f}] meters")

    # Step 2: Perspective projection
    cx, cy = width / 2.0, height / 2.0

    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Perspective projection: X = (x - cx) * Z / fx
    z = depth_scaled
    x_3d = (x_grid - cx) * z / fx
    y_3d = (y_grid - cy) * z / fy

    print("\nStep 2: Perspective projection")
    print(f"  Focal length: fx={fx}, fy={fy}")
    print(f"  Principal point: cx={cx:.1f}, cy={cy:.1f}")
    print(f"  FOV horizontal: {2 * np.arctan(width / (2 * fx)) * 180 / np.pi:.1f} degrees")
    print(f"  FOV vertical: {2 * np.arctan(height / (2 * fy)) * 180 / np.pi:.1f} degrees")

    print("\nResulting 3D coordinates:")
    print(f"  X range: [{x_3d.min():.3f}, {x_3d.max():.3f}] meters")
    print(f"  Y range: [{y_3d.min():.3f}, {y_3d.max():.3f}] meters")
    print(f"  Z range: [{z.min():.3f}, {z.max():.3f}] meters")

    # Check coordinate ratios
    x_span = x_3d.max() - x_3d.min()
    y_span = y_3d.max() - y_3d.min()
    z_span = z.max() - z.min()

    print("\nCoordinate space analysis:")
    print(f"  X span: {x_span:.3f}m")
    print(f"  Y span: {y_span:.3f}m")
    print(f"  Z span: {z_span:.3f}m")

    if y_span > 0:
        print(f"  X:Y ratio: {x_span/y_span:.3f}")
    if x_span > 0:
        print(f"  Z:X ratio: {z_span/x_span:.3f}")

    # Sample points
    print("\nSample 3D points:")

    cy_idx, cx_idx = height // 2, width // 2
    print(f"  Center pixel [{cx_idx}, {cy_idx}]:")
    print(f"    3D: ({x_3d[cy_idx, cx_idx]:.4f}, {y_3d[cy_idx, cx_idx]:.4f}, {z[cy_idx, cx_idx]:.4f})")

    # Edge point
    print(f"  Edge pixel [{width-1}, {cy_idx}]:")
    print(f"    3D: ({x_3d[cy_idx, width-1]:.4f}, {y_3d[cy_idx, width-1]:.4f}, {z[cy_idx, width-1]:.4f})")

    return x_3d, y_3d, z


def compare_mappings(depth, width, height):
    """Compare relative and metric depth mappings side by side."""
    print(f"\n{'='*60}")
    print("MAPPING COMPARISON")
    print(f"{'='*60}")

    # Test with same depth_scale but relative vs metric
    depth_scale = 0.5
    fx = fy = 470.4

    # Relative mapping
    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    max_dim = max(width, height)
    z_scale_factor = max_dim * 0.5
    z_relative = depth_norm * depth_scale * z_scale_factor

    # Metric mapping (with different scales)
    for metric_scale in [1.0, 0.1, 0.01]:
        z_metric = depth * metric_scale

        print(f"\nMetric scale {metric_scale}:")
        print(f"  Relative Z: [{z_relative.min():.3f}, {z_relative.max():.3f}]")
        print(f"  Metric Z: [{z_metric.min():.3f}, {z_metric.max():.3f}]")

        # What metric_scale would give similar Z range as relative?
        if z_metric.max() - z_metric.min() > 0:
            ideal_scale = (z_relative.max() - z_relative.min()) / (depth.max() - depth.min())
            print(f"  To match relative Z span, use metric_scale = {ideal_scale:.6f}")


def diagnose_common_issues(depth, width, height):
    """Print diagnostics for common depth-to-3D mapping issues."""
    print(f"\n{'='*60}")
    print("COMMON ISSUE DIAGNOSTICS")
    print(f"{'='*60}")

    # Issue 1: Depth values too large
    if depth.max() > 10:
        print(f"\n[!] Depth values are large (max={depth.max():.2f})")
        print("    For metric depth, values should typically be 0-10 meters")
        print("    Solutions:")
        print("      - Use --metric-depth-scale with small value (0.001-0.1)")
        print("      - Check if depth is in mm instead of meters (divide by 1000)")

    # Issue 2: Depth range too narrow
    depth_range = depth.max() - depth.min()
    if depth_range < 0.1:
        print(f"\n[!] Depth range is narrow (range={depth_range:.4f})")
        print("    This will result in flat 3D geometry")
        print("    Solutions:")
        print("      - Increase --depth-scale (try 1.0 or higher)")
        print("      - Use --raw-depth option")
        print("      - Check depth estimation model output")

    # Issue 3: Aspect ratio mismatch
    aspect = width / height
    if aspect > 2.0 or aspect < 0.5:
        print(f"\n[!] Extreme aspect ratio ({aspect:.2f})")
        print("    This may cause distorted 3D geometry")
        print("    The Y coordinates are scaled by aspect ratio for correction")

    # Issue 4: Zero or negative depth
    if depth.min() <= 0:
        print(f"\n[!] Zero or negative depth values (min={depth.min():.4f})")
        print("    Metric depth projection will fail (division by zero)")
        print("    Solution: Add small offset to depth values")

    # Issue 5: Focal length mismatch
    # Check if using default focal length
    print("\n[i] Camera intrinsics check:")
    print("    Default focal length (470.4) assumes ~60 degree FOV")
    print("    If your webcam has different FOV, adjust --focal-length-x/y")
    print("    Typical webcam: 60-90 degree FOV")
    print(f"    For 90 deg FOV on {width}px: fx = {width / (2 * np.tan(45 * np.pi / 180)):.1f}")
    print(f"    For 60 deg FOV on {width}px: fx = {width / (2 * np.tan(30 * np.pi / 180)):.1f}")

    print("\n")


def main():
    """Run diagnostic tests."""
    print("\n" + "="*60)
    print(" DEPTH-TO-3D MAPPING DIAGNOSTIC")
    print("="*60)

    # Test with synthetic depth patterns first
    print("\n--- Testing with synthetic depth patterns ---")
    patterns = create_test_depth_pattern()

    for name, depth in patterns.items():
        print(f"\n\n### Testing '{name}' pattern ###")
        h, w = depth.shape

        # Run relative depth test
        test_relative_depth_mapping(depth, w, h, depth_scale=0.5)

        # Run metric depth test
        test_metric_depth_mapping(depth, w, h, fx=470.4, fy=470.4, metric_scale=0.1)

        # Only test first pattern in detail
        if name == 'sphere':
            compare_mappings(depth, w, h)
            diagnose_common_issues(depth, w, h)

    print("\n--- To test with real webcam data ---")
    print("Run: python test_webcam_single_frame.py")
    print("Then check the depth_raw.npy file generated\n")


if __name__ == '__main__':
    main()
