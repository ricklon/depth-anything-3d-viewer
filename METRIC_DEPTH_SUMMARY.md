# Metric Depth Feature - Implementation Summary

## What Was Added

Added **metric depth support** for accurate 3D reconstruction with real-world scale to the Depth-Anything-3D viewer.

## Changes Made

### 1. Core Library ([da3d/viewing/mesh.py](da3d/viewing/mesh.py))

**DepthMeshViewer class:**
- Added `use_metric_depth` parameter
- Added `focal_length_x`, `focal_length_y` parameters (camera intrinsics)
- Added `principal_point_x`, `principal_point_y` parameters (optional)
- Implemented perspective projection for metric depth mode
- Preserves absolute depth values in meters (no normalization)

**RealTime3DViewer class:**
- Added same metric depth parameters
- Passes parameters through to DepthMeshViewer

### 2. CLI Interface ([da3d/cli/commands.py](da3d/cli/commands.py))

Added metric depth support to all 3D viewing commands:

**view3d command:**
- `--metric` - Enable metric depth mode
- `--focal-length-x` / `--focal-length-y` (default: 470.4)
- `--principal-point-x` / `--principal-point-y` (optional)

**webcam3d command:**
- Same arguments as view3d
- Works with real-time webcam depth estimation

**screen3d-viewer command:**
- Same arguments as view3d
- Works with real-time screen capture

### 3. Documentation

**New guide:** [docs/guides/metric_depth.md](docs/guides/metric_depth.md)
- Comprehensive explanation of metric vs relative depth
- Usage examples for all commands
- Camera calibration workflow
- Troubleshooting tips

**Updated:** [CLAUDE.md](CLAUDE.md)
- Documented new parameters
- Updated CLI architecture notes

### 4. Code Refactoring

**Renamed:** `da3d/cli/legacy.py` → `da3d/cli/commands.py`
- Removed "legacy" naming
- Updated all imports in `da3d/cli/main.py`
- Updated documentation references

## How It Works

### Relative Depth (Original, Default)

```python
# Orthographic projection
X = x - width/2
Y = y - height/2
Z = normalized_depth * depth_scale * (width * 0.5)
```

- Depth normalized to 0-1
- Arbitrary scale via `depth_scale`
- Good for visualization

### Metric Depth (New)

```python
# Perspective projection
Z = depth_in_meters (absolute)
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
```

- Depth in meters (preserved)
- Real-world scale from camera intrinsics
- Accurate 3D geometry

## Usage Examples

```bash
# View static depth with metric mode
da3d view3d image.jpg depth.npy --metric

# Real-time webcam with metric depth
da3d webcam3d --metric --encoder vits

# Screen capture with metric depth
da3d screen3d-viewer --metric

# Custom camera calibration
da3d view3d image.jpg depth.npy --metric \
  --focal-length-x 525.0 --focal-length-y 525.0 \
  --principal-point-x 320.0 --principal-point-y 240.0
```

## Benefits

1. **Accurate 3D reconstruction** - Real-world proportions preserved
2. **Better depth extrusion** - Geometrically correct perspective
3. **Real measurements** - Depth values have meaning (meters)
4. **Compatibility** - Works with existing metric depth models
5. **Backward compatible** - Relative depth mode still works (default)

## Technical Details

**Perspective projection implementation:**
- Located in `DepthMeshViewer.create_mesh_from_depth()` (lines 122-142)
- Uses same math as existing `save_point_clouds()` function
- Focal length in pixels controls field of view
- Principal point defaults to image center if not specified

**Depth value handling:**
- Metric mode skips 0-1 normalization (line 106)
- Preserves absolute meter values
- Still supports percentile clamping for outlier removal

**Model requirements:**
- Requires metric-trained Video-Depth-Anything models
- Checkpoint naming: `metric_video_depth_anything_{encoder}.pth`
- CLI automatically selects correct checkpoint when `--metric` flag used

## Files Modified

1. `da3d/viewing/mesh.py` - Core implementation
2. `da3d/cli/commands.py` - CLI arguments (renamed from legacy.py)
3. `da3d/cli/main.py` - Updated import
4. `CLAUDE.md` - Updated documentation
5. `docs/guides/metric_depth.md` - New comprehensive guide

## Files Created

- `docs/guides/metric_depth.md` - User guide for metric depth
- `METRIC_DEPTH_SUMMARY.md` - This file

## Testing

Verified implementation:
```bash
python -c "from da3d.viewing import DepthMeshViewer; \
           viewer = DepthMeshViewer(use_metric_depth=True, \
           focal_length_x=470.4, focal_length_y=470.4); \
           print('Metric depth initialized successfully')"
# Output: Metric depth initialized successfully
```

## Future Enhancements

Potential improvements:
1. Auto-detect metric vs relative depth from depth value range
2. Support for camera calibration file formats (e.g., OpenCV YAML)
3. Focal length estimation from EXIF data
4. Interactive focal length adjustment in viewer

## References

- Perspective projection: Standard pinhole camera model
- Based on existing `save_point_clouds()` implementation (commands.py:372-398)
- Compatible with Video-Depth-Anything metric models
