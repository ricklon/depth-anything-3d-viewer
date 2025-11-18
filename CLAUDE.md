# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Depth-Anything-3D Viewer is a Python package that extends Video-Depth-Anything with interactive 3D mesh visualization and real-time rendering capabilities. It transforms depth maps into interactive 3D geometry viewable from any angle, with support for real-time webcam/screen capture and 2.5D parallax effects.

**Package Name:** `depth-anything-3d` (installed as `da3d`)

## Build and Development Commands

### Installation

```bash
# Install package in editable mode
pip install -e .

# Or with uv (faster)
uv pip install -e .

# Install with all optional dependencies (screen capture, virtual cam, demo)
pip install -e ".[all]"

# Install development dependencies (tests, linting)
pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_mesh.py

# Run with coverage
pytest --cov=da3d tests/
```

### Code Quality

```bash
# Format code
black da3d/ tests/

# Sort imports
isort da3d/ tests/

# Type checking
mypy da3d/
```

### Running the CLI

```bash
# Main CLI entry point
da3d --help

# Real-time webcam 3D viewing
da3d webcam3d

# Screen capture in 3D
da3d screen3d-viewer

# View static depth map in 3D
da3d view3d image.jpg depth.png

# 2.5D parallax effects
da3d screen3d --auto-rotate
```

### Model Checkpoints

```bash
# Download required Video-Depth-Anything checkpoints
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
```

## Architecture

### Package Structure

```
da3d/
├── viewing/           # 3D mesh visualization
│   └── mesh.py       # DepthMeshViewer, RealTime3DViewer classes
├── projection/        # 2.5D parallax effects
│   └── parallax.py   # DepthProjector, InteractiveParallaxController classes
├── cli/              # Command-line interface
│   ├── main.py      # Entry point for `da3d` command
│   ├── commands.py  # Complete CLI implementation (to be modularized)
│   └── config.py    # Configuration utilities
└── __init__.py      # Package exports
```

### Key Components

#### 1. 3D Mesh Viewing (`da3d/viewing/mesh.py`)

**DepthMeshViewer:**
- Converts 2D images + depth maps into true 3D geometry (mesh or point cloud)
- Supports two display modes:
  - `mesh`: Triangle mesh with texture mapping (default)
  - `pointcloud`: Point cloud visualization
- Depth range control via percentile clamping to reduce extreme values
- Proportional depth scaling (Z-depth scales with X/Y dimensions)
- Grid-based mesh triangulation for efficient rendering
- Uses Open3D for visualization

**RealTime3DViewer:**
- Non-blocking Open3D visualization for continuous updates
- Used for webcam and screen capture 3D viewing
- Supports both mesh and point cloud modes
- Performance optimized with configurable subsampling and smoothing

**Key Parameters:**
- `depth_scale`: Z-displacement strength (0.1-2.0, where 1.0 = depth spans half image width) - ignored when `use_metric_depth=True`
- `depth_min_percentile` / `depth_max_percentile`: Clamp depth to percentile range
- `subsample`: Mesh resolution (1=full, 2=half, 3=balanced, 4=fast)
- `display_mode`: 'mesh' or 'pointcloud'
- `smooth_mesh`: Apply Laplacian smoothing (slower but cleaner)
- `use_metric_depth`: Use metric depth with camera intrinsics for accurate 3D reconstruction (NEW)
- `focal_length_x` / `focal_length_y`: Camera focal lengths in pixels (required for metric depth)
- `principal_point_x` / `principal_point_y`: Camera principal point (optional, defaults to image center)

#### 2. 2.5D Parallax Effects (`da3d/projection/parallax.py`)

**DepthProjector:**
- Creates parallax/3D effects by projecting 2D+depth into 3D space
- Applies rotation and projects back to 2D (perspective projection)
- Simulated 3D lighting based on surface normals
- Used for `screen3d` command (2.5D screen capture)
- Can create red-cyan anaglyph 3D images

**InteractiveParallaxController:**
- Mouse/keyboard control for interactive parallax
- Auto-rotation mode for cinematic effects
- Configurable depth scale, lighting, and zoom

#### 3. CLI Architecture (`da3d/cli/`)

**Current State:**
- `main.py`: Entry point that loads CLI commands
- `commands.py`: Complete CLI implementation (1500+ lines, all commands in one file)
- Imports Video-Depth-Anything from parent directory (expects it in PYTHONPATH)

**CLI imports:**
```python
# From Video-Depth-Anything (external dependency)
from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.video_depth_stream import VideoDepthAnythingStream
from utils.dc_utils import read_video_frames, save_video

# From this package
from da3d.projection import DepthProjector, InteractiveParallaxController
from da3d.viewing import DepthMeshViewer, view_depth_3d, RealTime3DViewer
```

**Commands implemented in commands.py:**
- `webcam3d`: Real-time webcam 3D mesh viewing
- `screen3d-viewer`: Screen capture 3D viewing
- `view3d`: Static depth map 3D viewing
- `screen3d`: 2.5D parallax screen capture
- Plus video processing commands

### Dependency Architecture

**Critical:** This package depends on the **Video-Depth-Anything** repository being available in PYTHONPATH. The original repo is not yet pip-installable.

**Current approach:**
- `main.py` adds parent directories to `sys.path` to find Video-Depth-Anything
- Provides helpful error message if imports fail
- User must clone Video-Depth-Anything alongside or set PYTHONPATH

**Model checkpoints:**
- Small (vits): 97MB, fastest, recommended for real-time
- Base (vitb): 388MB, balanced
- Large (vitl): 1.3GB, best quality
- Expected in `./checkpoints/` by default

### Depth Processing Pipeline

**For 3D Mesh Viewing:**
1. Load image and depth map
2. Apply percentile clamping to depth values (reduces extremes)
3. Normalize depth to 0-1 range
4. Create 3D point cloud: (X, Y, Z) where Z comes from depth
5. Scale Z proportionally to image width (depth_scale × width × 0.5)
6. For mesh mode: Connect adjacent pixels in grid pattern (2 triangles per quad)
7. For point cloud mode: Estimate normals for lighting
8. Render with Open3D

**For 2.5D Parallax:**
1. Load image and depth
2. Create 3D point cloud from 2D + depth
3. Apply rotation transformations (rotation_x, rotation_y)
4. Project back to 2D using perspective projection
5. Remap image pixels using new coordinates
6. Apply simulated lighting based on surface normals

### Performance Tiers

**Real-time viewing (webcam/screen):**
- Fast: `--subsample 4 --max-res 320` (15-20 FPS on RTX 3080)
- Balanced: `--subsample 3 --max-res 480` (8-12 FPS, default)
- Quality: `--subsample 2 --max-res 640 --smooth` (4-6 FPS)

### Default Depth Range Settings

Different commands have different default percentile ranges optimized for their use case:
- `webcam3d`: 0-95% (preserves foreground, reduces background extremes while preserving detail)
- `screen3d-viewer`: 5-95% (balanced for screen content)
- `view3d`: 0-100% (full control, no auto-clamping)

## Common Development Tasks

### Adding a New CLI Command

Currently all commands are in `da3d/cli/commands.py`. To add a new command:
1. Add command function (e.g., `def my_command(args)`)
2. Add argument parser in `create_parser()`
3. Add command to dispatch in `main()`

**Future:** CLI should be modularized into separate files (see PACKAGE_SUMMARY.md for plan).

### Modifying 3D Rendering

- Mesh creation: Edit `DepthMeshViewer.create_mesh_from_depth()` in `da3d/viewing/mesh.py`
- Triangulation: Edit `DepthMeshViewer._create_grid_mesh()`
- Real-time updates: Edit `RealTime3DViewer.update_mesh()`

### Modifying Parallax Effects

- Projection algorithm: Edit `DepthProjector.project_with_parallax()` in `da3d/projection/parallax.py`
- Lighting: Edit `DepthProjector._apply_lighting()`
- Controls: Edit `InteractiveParallaxController`

### Testing Depth Processing

```python
from da3d.viewing import view_depth_3d

# Quick test with different settings
view_depth_3d(
    "test_image.jpg",
    "test_depth.png",
    depth_scale=0.5,
    subsample=2,
    display_mode='mesh'  # or 'pointcloud'
)
```

## Important Implementation Notes

### Depth Scale Calculation

The depth scale is **proportional to image dimensions**:
```python
z_scale_factor = width * 0.5  # Z spans half width when depth_scale=1.0
z = depth_normalized * self.depth_scale * z_scale_factor
```

This ensures consistent 3D appearance across different image sizes.

### Mesh Triangulation

Uses grid-based triangulation:
- Each 2×2 quad of pixels creates 2 triangles
- Only creates triangles where all 4 vertices pass depth threshold
- Preserves image topology for proper texture mapping

### Point Cloud vs Mesh Mode

**Mesh mode (default):**
- Triangle mesh with RGB texture
- Shows continuous surfaces
- Supports wireframe mode
- Better for smooth surfaces

**Point cloud mode:**
- Individual 3D points
- Faster rendering
- Better for noisy depth maps
- Different aesthetic

### Video-Depth-Anything Integration

The package uses Video-Depth-Anything for depth estimation:
- Streaming mode: `VideoDepthAnythingStream` (frame-by-frame)
- Batch mode: `VideoDepthAnything` (temporal consistency)
- Must be imported from external repo (not bundled)

## Configuration Files

- `pyproject.toml`: Package configuration, dependencies, build system
  - Uses setuptools build backend
  - Line length: 100 (black/isort)
  - Python 3.9+ required
- `.gitignore`: Standard Python ignores + checkpoints, outputs

## Documentation

User-facing documentation in `docs/`:
- `docs/guides/realtime_3d.md`: Real-time webcam/screen 3D viewing
- `docs/guides/static_viewing.md`: Static depth map viewing
- `docs/guides/screen_capture.md`: Screen capture guide
- `docs/guides/depth_tuning.md`: Depth parameter tuning
- `docs/guides/depth_defaults.md`: Default settings explained

## Known Limitations

1. **Video-Depth-Anything dependency**: Not pip-installable yet, requires manual setup
2. **CLI modularization**: All commands in single `commands.py` file (planned to be split)
3. **GPU requirement**: CPU mode very slow for real-time viewing
4. **Open3D blocking**: Real-time viewer uses polling, not truly async

## Future Improvements Planned

From PACKAGE_SUMMARY.md:
1. Modularize CLI into separate command files
2. Add comprehensive unit tests
3. Create example scripts
4. Make Video-Depth-Anything dependency pip-installable or bundle it
5. Add checkpoint download script
