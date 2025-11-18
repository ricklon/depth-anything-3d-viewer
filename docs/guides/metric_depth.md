# Metric Depth 3D Viewing

This guide explains how to use metric depth for accurate 3D reconstruction with real-world scale.

## What is Metric Depth?

**Relative depth** (default mode):
- Depth values are normalized to 0-1 range
- Only represents relative distances (closer vs farther)
- Uses orthographic projection (no perspective)
- Arbitrary scale controlled by `--depth-scale`

**Metric depth** (new mode):
- Depth values are in **real-world units (meters)**
- Represents actual distances from camera
- Uses **perspective projection** with camera intrinsics
- Creates geometrically accurate 3D reconstructions

## Why Use Metric Depth?

Metric depth provides:
1. **Accurate 3D geometry** - Objects maintain real-world proportions
2. **Proper perspective** - Closer objects appear larger (like real cameras)
3. **Better extrusion quality** - Depth is scaled correctly based on camera properties
4. **Real-world measurements** - Can measure actual distances in the 3D view

## Quick Start

### View Static Depth Map with Metric Depth

```bash
# Basic metric depth viewing
da3d view3d image.jpg depth.npy --metric

# With custom focal length (if known)
da3d view3d image.jpg depth.npy --metric --focal-length-x 525.0 --focal-length-y 525.0

# With custom principal point
da3d view3d image.jpg depth.npy --metric \
  --focal-length-x 525.0 --focal-length-y 525.0 \
  --principal-point-x 320.0 --principal-point-y 240.0
```

### Real-time Webcam with Metric Depth

```bash
# Enable metric depth for webcam
da3d webcam3d --metric --focal-length-x 470.4 --focal-length-y 470.4

# Use metric depth model checkpoint
da3d webcam3d --metric --encoder vits
```

### Screen Capture with Metric Depth

```bash
# Screen capture with metric depth
da3d screen3d-viewer --metric --focal-length-x 470.4 --focal-length-y 470.4
```

## Camera Intrinsic Parameters

Metric depth requires camera intrinsic parameters:

### Focal Length

The focal length (in pixels) controls the field of view:

- **--focal-length-x**: Horizontal focal length (default: 470.4)
- **--focal-length-y**: Vertical focal length (default: 470.4)

**Finding your camera's focal length:**
- Check camera calibration data
- Estimate from field of view: `f = (width / 2) / tan(FOV / 2)`
- Default value (470.4) works well for typical webcams

### Principal Point

The principal point is where the optical axis intersects the image:

- **--principal-point-x**: X coordinate in pixels (default: image center)
- **--principal-point-y**: Y coordinate in pixels (default: image center)

**When to specify:**
- Usually defaults to image center work fine
- Specify if you have calibration data showing off-center optical axis
- Important for high-precision applications

## Perspective Projection Math

Metric depth uses the standard pinhole camera model:

```
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
Z = depth_value (in meters)
```

Where:
- `(x, y)` = pixel coordinates
- `(cx, cy)` = principal point (image center)
- `(fx, fy)` = focal lengths
- `Z` = metric depth in meters

This creates accurate 3D points that maintain real-world proportions.

## Model Checkpoints

Metric depth requires **metric-trained models**:

```bash
# Download metric depth models
cd checkpoints
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/metric_video_depth_anything_vitb.pth
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth
```

The CLI automatically selects the correct checkpoint when `--metric` is used.

## Comparison: Relative vs Metric Depth

### Relative Depth (Default)

```bash
da3d view3d image.jpg depth.png --depth-scale 0.5
```

**Characteristics:**
- Depth normalized to 0-1
- Orthographic projection (parallel lines stay parallel)
- Manual scale adjustment via `--depth-scale`
- Good for artistic visualization

### Metric Depth

```bash
da3d view3d image.jpg depth.npy --metric --focal-length-x 470.4 --focal-length-y 470.4
```

**Characteristics:**
- Depth in meters
- Perspective projection (perspective foreshortening)
- Automatic scale from camera intrinsics
- Accurate 3D reconstruction

## Depth File Formats

Metric depth works best with `.npy` files that preserve float values:

```python
import numpy as np

# Save metric depth from model output
depth_meters = model.predict(image)  # Shape: (H, W), values in meters
np.save('depth.npy', depth_meters)

# Load for viewing
depth = np.load('depth.npy')
```

**Avoid:**
- PNG/JPG for metric depth (quantizes to 0-255, loses precision)
- These work for relative depth but not metric depth

## Depth Range Control

Even with metric depth, you can control the visible range:

```bash
# Clamp depth to 5th-95th percentile (removes outliers)
da3d view3d image.jpg depth.npy --metric \
  --depth-min-percentile 5.0 \
  --depth-max-percentile 95.0

# Filter out far background (keep only closest 90% of pixels)
da3d view3d image.jpg depth.npy --metric --depth-threshold 0.9
```

**Note:** Percentile clamping preserves metric scale (doesn't normalize to 0-1).

## Performance Considerations

Metric depth has similar performance to relative depth:

```bash
# Fast real-time viewing
da3d webcam3d --metric --subsample 4 --max-res 320

# Quality viewing
da3d webcam3d --metric --subsample 2 --max-res 640 --smooth
```

The `--depth-scale` parameter is **ignored** in metric mode (scale comes from intrinsics).

## Troubleshooting

### Depth looks too flat or too exaggerated

**Problem:** Wrong focal length
**Solution:** Adjust focal length:

```bash
# Try different focal lengths
da3d view3d image.jpg depth.npy --metric --focal-length-x 300.0 --focal-length-y 300.0  # Wider FOV
da3d view3d image.jpg depth.npy --metric --focal-length-x 600.0 --focal-length-y 600.0  # Narrower FOV
```

### Objects appear distorted

**Problem:** Incorrect aspect ratio
**Solution:** Ensure `fx` and `fy` match your camera's pixel aspect ratio:

```bash
# Square pixels (typical)
--focal-length-x 470.4 --focal-length-y 470.4

# Non-square pixels (rare)
--focal-length-x 470.4 --focal-length-y 480.0
```

### Scene too small or too large

**Problem:** Depth values in wrong units
**Solution:** Ensure depth is in meters. If in mm, convert:

```python
depth_mm = np.load('depth_mm.npy')
depth_meters = depth_mm / 1000.0
np.save('depth_meters.npy', depth_meters)
```

## Example Workflows

### Workflow 1: Depth from Video-Depth-Anything

```bash
# Generate metric depth from video
da3d video input.mp4 --metric --encoder vitl --save-npz

# View first frame in 3D
da3d view3d outputs/input/input_0000.jpg outputs/input/depth_0000.npy --metric
```

### Workflow 2: Real-time Webcam with Calibrated Camera

```bash
# If you have camera calibration data
da3d webcam3d --metric \
  --focal-length-x 525.0 \
  --focal-length-y 525.0 \
  --principal-point-x 319.5 \
  --principal-point-y 239.5
```

### Workflow 3: Export Point Cloud with Metric Depth

```bash
# Video processing saves point clouds automatically with --metric
da3d video input.mp4 --metric --encoder vitb --focal-length-x 470.4 --focal-length-y 470.4

# Point clouds saved as: outputs/input/point_0000.ply
```

## Advanced: Custom Camera Calibration

If you have calibration data from OpenCV or similar:

```python
# From OpenCV calibration
import numpy as np

# Camera matrix from calibration
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

fx = K[0, 0]  # --focal-length-x
fy = K[1, 1]  # --focal-length-y
cx = K[0, 2]  # --principal-point-x
cy = K[1, 2]  # --principal-point-y
```

Then use in CLI:

```bash
da3d view3d image.jpg depth.npy --metric \
  --focal-length-x $fx \
  --focal-length-y $fy \
  --principal-point-x $cx \
  --principal-point-y $cy
```

## See Also

- [Real-time 3D Viewing](realtime_3d.md) - Webcam and screen capture
- [Static Viewing](static_viewing.md) - Viewing depth maps from files
- [Depth Tuning](depth_tuning.md) - Optimizing depth parameters
