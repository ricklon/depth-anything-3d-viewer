# Metric Depth Implementation - Complete Guide

## Overview

Successfully implemented metric depth support for accurate 3D reconstruction with real-world scale using perspective projection and camera intrinsics.

## Key Insight: Focal Length Must Match Resolution

**CRITICAL:** The focal length in pixels must be scaled proportionally to the resolution the depth model processes.

### The Problem We Solved

The depth model downsamples input images for processing. If you provide a focal length for the camera's native resolution but the model runs at a lower resolution, the perspective projection will be completely wrong, causing extreme stretching.

**Example:**
- Camera native resolution: 1920x1080
- Camera focal length: 1430 pixels (at 1920px wide)
- Depth model processes at: 160x120 (downsampled)
- **Wrong:** Using focal_length_x=1430 → Creates 6.4° FOV (telescope view), extreme stretching
- **Correct:** Scale to 119 pixels → `1430 × (160/1920) = 119` → Creates 78° FOV (normal webcam)

### Formula

```
focal_length_at_target_res = focal_length_native × (target_width / native_width)
```

## Files Modified

### 1. Core Library: `da3d/viewing/mesh.py`

**Changes:**
- Added `use_metric_depth`, `focal_length_x/y`, `principal_point_x/y`, `metric_depth_scale` parameters
- Removed percentile clamping for metric depth (use raw values directly)
- Implemented perspective projection: `X = (x - cx) * Z / fx`, `Y = (y - cy) * Z / fy`
- Added debug output showing resolution, FOV, and depth statistics
- For metric mode: depth threshold is absolute meters (not percentile)

**Key code (lines 145-165):**
```python
if self.use_metric_depth:
    # Use provided principal point or default to image center
    cx = self.principal_point_x if self.principal_point_x is not None else w / 2.0
    cy = self.principal_point_y if self.principal_point_y is not None else h / 2.0

    # Perspective projection with real depth in meters
    z = depth_normalized  # Already in meters
    x_3d = (x_grid - cx) * z / self.focal_length_x
    y_3d = (y_grid - cy) * z / self.focal_length_y

    points = np.stack([
        x_3d.flatten(),
        -y_3d.flatten(),
        z.flatten()
    ], axis=1)
```

### 2. CLI: `da3d/cli/commands.py`

**Added arguments to `view3d`, `webcam3d`, `screen3d-viewer`:**
```bash
--metric                    # Enable metric depth mode
--focal-length-x FLOAT      # Focal length X in pixels (default: 470.4)
--focal-length-y FLOAT      # Focal length Y in pixels (default: 470.4)
--principal-point-x FLOAT   # Principal point X (default: image center)
--principal-point-y FLOAT   # Principal point Y (default: image center)
--metric-depth-scale FLOAT  # Scale factor for depth values (default: 1.0)
```

### 3. New Files

**`capture_depth_frame.py`**
- Utility to capture webcam frame with depth for testing
- Saves RGB image, depth array (.npy), and depth visualization
- Provides example commands with correct focal lengths

**`download_metric_weights.ps1`**
- Downloads metric depth model checkpoints
- Corrected URLs for HuggingFace: `Metric-Video-Depth-Anything-{Small,Base,Large}`

**`calibrate_webcam.py`**
- OpenCV checkerboard-based camera calibration
- Extracts precise focal lengths and principal points

## Webcam Calibration Data

### Logitech C920

**Physical specs:**
- Focal length: 3.67mm
- Sensor size: 1/2.9" (~4.9mm wide)
- Native resolution: 1920x1080

**Calculated focal length (native resolution):**
```
focal_pixels = (3.67mm × 1920px) / 4.9mm ≈ 1430 pixels
```

**Focal lengths for common processing resolutions:**

| Processing Resolution | Focal Length X | Focal Length Y | FOV     |
|-----------------------|----------------|----------------|---------|
| 1920x1080 (native)    | 1430 px        | 1430 px        | 78°     |
| 854x480               | 637 px         | 637 px         | 78°     |
| 640x360               | 476 px         | 476 px         | 78°     |
| 480x270               | 357 px         | 357 px         | 78°     |
| 320x180               | 238 px         | 238 px         | 78°     |
| 160x90                | 119 px         | 119 px         | 78°     |

**Note:** FOV stays constant - only focal length in pixels changes with resolution.

### Generic Webcam Defaults

For unknown webcams, use default: `focal_length = 0.7 × image_width`
- At 640px wide: `focal_length = 448 px`
- At 480px wide: `focal_length = 336 px`

## Usage Examples

### 1. Live Webcam 3D (Logitech C920)

Check what resolution the depth model is processing:
```bash
uv run da3d webcam3d --metric --encoder vits --focal-length-x 119 --focal-length-y 119
```

The debug output will show: `[DEBUG] Image size: 160x120, Focal length: 119.0px`

If processing at higher resolution:
```bash
# For 640px wide processing
uv run da3d webcam3d --metric --encoder vits \
  --focal-length-x 476 --focal-length-y 476 \
  --max-res 640
```

### 2. Capture Test Frame

```bash
# Capture frame with metric depth
uv run python capture_depth_frame.py --metric --encoder vits --output test_frame
```

Output:
- `test_frame_rgb.jpg` - RGB image
- `test_frame_depth.npy` - Raw depth data
- `test_frame_depth_vis.png` - Depth visualization

### 3. View Captured Frame

```bash
# View with correct focal length for 640px processing
uv run da3d view3d test_frame_rgb.jpg test_frame_depth.npy \
  --metric --focal-length-x 476 --focal-length-y 476

# Compare with wrong focal length (shows stretching)
uv run da3d view3d test_frame_rgb.jpg test_frame_depth.npy \
  --metric --focal-length-x 1430 --focal-length-y 1430
```

### 4. Depth Inversion

If depth appears inverted (background in front):
```bash
uv run da3d webcam3d --metric --encoder vits \
  --focal-length-x 476 --focal-length-y 476 \
  --invert-depth
```

This applies reciprocal: `depth = 1.0 / (depth + epsilon)` for disparity→depth conversion.

### 5. Precise Calibration

For best accuracy, calibrate your specific camera:
```bash
uv run python calibrate_webcam.py

# Use the output values:
uv run da3d webcam3d --metric --encoder vits \
  --focal-length-x 1425.3 --focal-length-y 1427.8 \
  --principal-point-x 962.1 --principal-point-y 541.3
```

**Remember:** Scale these values to match the processing resolution!

## Debug Output Interpretation

When running with `--metric`, you'll see:
```
[DEBUG] Image size: 640x480, Focal length: 476.0px
[DEBUG] Raw depth - min: 0.203m, max: 1.405m, mean: 0.746m
[DEBUG] After scale (1.0x): min: 0.203m, max: 1.405m
[DEBUG] Horizontal FOV: 77.3 degrees
```

**What to check:**
- **Image size:** This is the resolution the depth model processes (after downsampling)
- **Focal length:** Should be scaled to match image width
- **FOV:** Should be ~70-90° for typical webcams
  - If FOV < 20°: Focal length too high (stretched)
  - If FOV > 120°: Focal length too low (compressed)
- **Depth range:** Typical indoor scenes are 0.2-3.0 meters

## How Metric Depth Works

### Relative Depth (Original)
```python
# Orthographic projection
X = x - width/2
Y = y - height/2
Z = normalized_depth * depth_scale * (width * 0.5)
```
- Depth normalized to 0-1
- Arbitrary scale via `depth_scale`
- Good for visualization, not geometrically accurate

### Metric Depth (New)
```python
# Perspective projection
Z = depth_in_meters (absolute, not normalized)
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
```
- Depth values in meters (preserved from model)
- Real-world scale from camera intrinsics
- Geometrically accurate 3D geometry

## Common Issues & Solutions

### 1. Everything is extremely stretched
**Cause:** Focal length too high for processing resolution
**Fix:** Scale focal length: `focal_new = focal_native × (width_processed / width_native)`

### 2. Depth is inverted (background in front)
**Cause:** Model outputs disparity instead of depth
**Fix:** Add `--invert-depth` flag (applies reciprocal)

### 3. Very low resolution / blocky appearance
**Cause:** Depth model defaulting to low resolution for speed
**Fix:** Add `--max-res 640` or `--max-res 854`

### 4. Everything appears flat / no depth
**Cause:** Focal length too low
**Fix:** Check FOV in debug output, should be 70-90° for webcams

## Performance Recommendations

| Resolution | Focal Length (C920) | Speed      | Quality  | Command                    |
|------------|---------------------|------------|----------|----------------------------|
| 160x90     | 119 px              | ~20 FPS    | Low      | (default, no --max-res)    |
| 320x180    | 238 px              | ~15 FPS    | Fair     | `--max-res 320`            |
| 480x270    | 357 px              | ~12 FPS    | Good     | `--max-res 480`            |
| 640x360    | 476 px              | ~8 FPS     | Better   | `--max-res 640` (recommended) |
| 854x480    | 637 px              | ~5 FPS     | Best     | `--max-res 854`            |

## Model Checkpoints

Metric depth requires specific model weights:

```powershell
# Download with PowerShell script
.\download_metric_weights.ps1

# Select option 1 for vits (recommended for real-time)
```

**Checkpoint paths:**
- Small (vits): `checkpoints/metric_video_depth_anything_vits.pth` (28.4M params)
- Base (vitb): `checkpoints/metric_video_depth_anything_vitb.pth` (113.1M params)
- Large (vitl): `checkpoints/metric_video_depth_anything_vitl.pth` (381.8M params)

## Testing Workflow

1. **Capture test frame:**
   ```bash
   uv run python capture_depth_frame.py --metric --encoder vits --output test
   ```

2. **Test with different focal lengths:**
   ```bash
   # Wrong (stretched)
   uv run da3d view3d test_rgb.jpg test_depth.npy --metric --focal-length-x 1430 --focal-length-y 1430

   # Correct
   uv run da3d view3d test_rgb.jpg test_depth.npy --metric --focal-length-x 476 --focal-length-y 476
   ```

3. **Compare depth ranges:**
   ```bash
   # Check raw depth values
   python -c "import numpy as np; d = np.load('test_depth.npy'); print(f'Range: {d.min():.3f}m to {d.max():.3f}m')"
   ```

4. **Verify FOV is reasonable:**
   Check debug output for `Horizontal FOV: XX degrees` - should be 70-90° for webcams.

## Technical Notes

### Why No Percentile Clamping?
Metric depth values are real measurements in meters. Clamping to percentiles would lose the metric accuracy. Instead:
- Use raw depth values directly
- Filter by absolute distance threshold (e.g., `--depth-threshold 5.0` for 5 meters)
- The `metric_depth_scale` parameter is for fine-tuning only (usually keep at 1.0)

### Coordinate Systems
- **X-axis:** Horizontal (left to right)
- **Y-axis:** Vertical (flipped: -Y is up in 3D space so image appears right-side up)
- **Z-axis:** Depth (away from camera)
- Origin: Camera optical center

### Principal Point
Defaults to image center if not specified. Only needs manual setting if:
- Using calibrated camera with off-center optical axis
- Camera has significant lens distortion
- Need maximum accuracy for measurements

## Summary

Metric depth works correctly when:
1. ✅ Using metric depth model checkpoints (`Metric-Video-Depth-Anything-*`)
2. ✅ Focal length scaled to match processing resolution
3. ✅ FOV is reasonable (70-90° for webcams)
4. ✅ Depth values preserved as meters (no percentile clamping)
5. ✅ Perspective projection applied with camera intrinsics

The key breakthrough was realizing that focal length must be specified in pixels **at the resolution the depth model processes**, not the camera's native resolution.
