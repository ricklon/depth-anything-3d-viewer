# Depth-Anything-3D Viewer

🎥 **Interactive 3D mesh visualization and real-time rendering for Video-Depth-Anything**

Transform depth maps into stunning 3D visualizations with full camera control, real-time webcam/screen capture, and 2.5D parallax effects.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Features

- 🎮 **Real-time 3D Webcam** - View yourself in true 3D mesh or point cloud with live updates
- 🖥️ **Screen Capture 3D** - Turn any screen content into interactive 3D geometry
- 📊 **Static 3D Viewing** - Load and explore depth maps from any angle
- 🎨 **2.5D Parallax Effects** - Create cinematic depth effects for videos
- 🔷 **Mesh & Point Cloud Modes** - Choose between triangle mesh or point cloud visualization
- ⚡ **Performance Optimized** - GPU-accelerated with adjustable quality settings
- 🎯 **Depth Range Control** - Automatically reduce extreme values for better visualization
- 📏 **Proportional Depth Scaling** - Z-depth scales naturally with X/Y dimensions

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ricklon/depth-anything-3d-viewer
cd depth-anything-3d-viewer

# Install with pip
pip install -e .

# Or with uv (recommended)
uv pip install -e .

# Install with optional dependencies
pip install -e ".[all]"  # Screen capture + virtual cam + demo
```

### Download Depth Model Checkpoints

The package requires Video-Depth-Anything model checkpoints:

```bash
# Create checkpoints directory
mkdir checkpoints
cd checkpoints

# Download models (choose one or more)
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
```

### Usage

```bash
# View yourself in 3D mesh (webcam)
da3d webcam3d

# View as point cloud instead
da3d webcam3d --display-mode pointcloud

# Capture screen in 3D
da3d screen3d-viewer

# View static depth map
da3d view3d image.jpg depth.png

# View as point cloud
da3d view3d image.jpg depth.png --display-mode pointcloud

# 2.5D parallax screen capture
da3d screen3d --auto-rotate

# Get help
da3d --help
da3d webcam3d --help
```

## Commands

### `webcam3d` - Real-time Webcam 3D

View yourself as a live-updating 3D mesh:

```bash
# Basic usage (optimized defaults)
da3d webcam3d

# Point cloud mode (faster, different aesthetic)
da3d webcam3d --display-mode pointcloud

# Fast performance mode
da3d webcam3d --subsample 4 --max-res 320

# High quality mesh
da3d webcam3d --subsample 2 --max-res 640 --smooth

# Custom depth range (reduce background)
da3d webcam3d --depth-max-percentile 80
```

**Controls:**
- Mouse drag: Rotate camera 360°
- Mouse wheel: Zoom in/out
- Shift + drag: Pan camera
- Close window to exit

### `screen3d-viewer` - Screen Capture 3D

Turn your screen into an interactive 3D scene:

```bash
# Capture primary monitor
da3d screen3d-viewer

# Specific region
da3d screen3d-viewer --region 0,0,1920,1080

# Gaming mode (aggressive depth clamping)
da3d screen3d-viewer --depth-min-percentile 10 --depth-max-percentile 90

# Fast performance
da3d screen3d-viewer --subsample 4 --fps 15
```

### `view3d` - Static 3D Viewing

Explore static depth maps interactively:

```bash
# Basic viewing (mesh mode)
da3d view3d image.jpg depth.png

# View as point cloud
da3d view3d image.jpg depth.png --display-mode pointcloud

# Adjust depth range for better visualization
da3d view3d image.jpg depth.png --depth-min-percentile 5 --depth-max-percentile 95

# High quality with full resolution
da3d view3d image.jpg depth.png --subsample 1 --depth-scale 0.7

# Wireframe mesh mode
da3d view3d image.jpg depth.png --wireframe

# Point cloud with adjusted depth
da3d view3d image.jpg depth.png --display-mode pointcloud --depth-scale 0.8
```

### `screen3d` - 2.5D Parallax Effects

Real-time screen capture with cinematic parallax:

```bash
# Auto-rotating parallax
da3d screen3d --auto-rotate

# Mouse-controlled parallax
da3d screen3d --mouse-control

# With virtual camera output (for OBS)
da3d screen3d --virtual-cam --mouse-control

# Show displacement visualization (debug)
da3d screen3d --show-displacement --test-grid
```

## Python API

```python
from da3d.viewing import view_depth_3d, RealTime3DViewer
from da3d.projection import DepthProjector

# Quick 3D viewing (mesh mode)
view_depth_3d("image.jpg", "depth.png", depth_scale=0.5, subsample=2)

# View as point cloud
view_depth_3d("image.jpg", "depth.png", depth_scale=0.6, display_mode='pointcloud')

# Real-time 3D from custom source (mesh mode)
viewer = RealTime3DViewer(
    depth_scale=0.5,
    subsample=3,
    depth_min_percentile=0,
    depth_max_percentile=90,
    display_mode='mesh'  # or 'pointcloud'
)
viewer.initialize(width=1280, height=720)

# Your video processing loop
for frame, depth in your_video_source():
    viewer.update_mesh(frame, depth)
    if viewer.should_close():
        break

viewer.close()

# 2.5D Parallax effects
projector = DepthProjector(width, height)
projected = projector.project_with_parallax(
    image, depth,
    rotation_x=10,
    rotation_y=5,
    scale_z=0.5
)
```

## Key Parameters

### Depth Range Control

**Problem:** Raw depth maps often have extreme values that create stretched, unnatural 3D geometry.

**Solution:** Percentile clamping focuses on meaningful depth ranges:

```bash
--depth-min-percentile N  # Clamp near depth (0-100)
--depth-max-percentile N  # Clamp far depth (0-100)
```

**Defaults:**
- `webcam3d`: 0-90% (preserves foreground, simplifies background)
- `screen3d-viewer`: 5-95% (balanced for screen content)
- `view3d`: 0-100% (full control, no auto-clamping)

**Examples:**
```bash
# More aggressive background removal
da3d webcam3d --depth-max-percentile 80

# Focus on mid-range depth
da3d screen3d-viewer --depth-min-percentile 10 --depth-max-percentile 85

# Full depth range (no clamping)
da3d view3d image.jpg depth.png --depth-min-percentile 0 --depth-max-percentile 100
```

### Performance Tuning

```bash
--subsample N      # Mesh resolution (2=high, 3=balanced, 4=fast)
--max-res N        # Maximum frame resolution (lower = faster)
--encoder vits     # Model size (vits=fast, vitb=balanced, vitl=best)
--smooth           # Enable mesh smoothing (slower but cleaner)
```

**Performance Tiers:**
```bash
# Maximum FPS (15-20 FPS)
da3d webcam3d --subsample 4 --max-res 320

# Balanced (8-12 FPS, default)
da3d webcam3d --subsample 3 --max-res 480

# High Quality (4-6 FPS)
da3d webcam3d --subsample 2 --max-res 640 --smooth
```

### 3D Appearance

```bash
--depth-scale FLOAT     # Z-displacement strength (0.1-2.0, default: 0.5)
                        # 1.0 = depth spans half the image width
--display-mode MODE     # Visualization mode: 'mesh' or 'pointcloud'
--invert-depth          # Reverse depth direction
--wireframe             # Show mesh structure (mesh mode only)
--background R,G,B      # Background color (0-1 range)
```

**Depth Scale Guide:**
- `0.3` - Subtle depth effect
- `0.5` - Balanced (default)
- `0.7` - Pronounced depth
- `1.0` - Depth spans half image width
- `1.5+` - Very dramatic (may look stretched)

## Documentation

- 📖 [Getting Started Guide](docs/getting_started.md)
- 🎥 [Real-time 3D Viewing](docs/guides/realtime_3d.md)
- 🖼️ [Static 3D Viewing](docs/guides/static_viewing.md)
- 📺 [Screen Capture Guide](docs/guides/screen_capture.md)
- ⚙️ [Depth Tuning Guide](docs/guides/depth_tuning.md)
- 🎯 [Default Settings Explained](docs/guides/depth_defaults.md)
- 🔧 [API Reference](docs/api/viewing.md)

## Examples

See the `examples/` directory for complete Python examples:

- `01_basic_viewing.py` - Simple static depth viewing
- `02_webcam_3d.py` - Custom webcam 3D implementation
- `03_screen_3d.py` - Screen capture integration
- `04_custom_integration.py` - Integrate with your own depth estimator

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA (recommended) or CPU
- Open3D 0.18+ (for 3D viewing)
- OpenCV 4.8+
- NumPy, SciPy

### Optional Dependencies

- `mss` - Screen capture support
- `pyvirtualcam` - Virtual camera output
- `gradio` - Web demo interface

## Architecture

```
da3d/
├── viewing/           # 3D mesh viewing
│   ├── mesh.py       # Static mesh generation
│   └── realtime.py   # Real-time viewer
├── projection/        # 2.5D parallax
│   └── parallax.py   # Depth projection
├── cli/               # Command-line interface
│   ├── main.py       # Entry point
│   ├── webcam.py     # Webcam commands
│   ├── screen.py     # Screen commands
│   └── viewer.py     # Viewer commands
└── utils/             # Utilities
```

## Performance

| Configuration | FPS (RTX 3080) | Quality | Use Case |
|---------------|----------------|---------|----------|
| `--subsample 4 --max-res 320` | 15-20 | Medium | Fast preview |
| `--subsample 3 --max-res 480` | 8-12 | Good | Balanced (default) |
| `--subsample 2 --max-res 640` | 4-6 | High | Quality viewing |
| `--subsample 2 --smooth` | 2-4 | Highest | Maximum quality |

*Performance varies by GPU. CPU mode is significantly slower.*

## Troubleshooting

### Face/body cut off in webcam mode

The default 0-90% depth range should prevent this. If still cut off:
```bash
da3d webcam3d --depth-max-percentile 95
```

### Too much background noise

```bash
da3d webcam3d --depth-max-percentile 80
```

### Mesh looks too flat

```bash
da3d view3d image.jpg depth.png --depth-scale 0.8
```

### Depth looks too extreme/stretched

```bash
da3d view3d image.jpg depth.png --depth-scale 0.3
```

### Performance issues

```bash
da3d webcam3d --subsample 4 --max-res 320
```

### Model checkpoints not found

Ensure checkpoints are in `./checkpoints/` or specify path:
```bash
da3d webcam3d --checkpoints-dir /path/to/checkpoints
```

## Credits

Built on top of:
- [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) - Depth estimation model
- [Open3D](https://github.com/isl-org/Open3D) - 3D visualization
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) - Foundation model

## License

Apache-2.0 License - see [LICENSE](LICENSE) file for details.

Video-Depth-Anything-Small is Apache-2.0 licensed.
Video-Depth-Anything-Base/Large are CC-BY-NC-4.0 licensed (non-commercial use).

## Citation

If you use this in your research, please cite:

```bibtex
@article{VideoDepthAnything,
  title={Video Depth Anything: Consistent Video Depth Estimation},
  author={...},
  journal={...},
  year={2025}
}
```

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Support

- 📝 [Issues](https://github.com/ricklon/depth-anything-3d-viewer/issues)
- 💬 [Discussions](https://github.com/ricklon/depth-anything-3d-viewer/discussions)
