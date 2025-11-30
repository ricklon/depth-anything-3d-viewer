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

**Recommended: Using uv (faster, better dependency management)**

```bash
# Clone the repository
git clone https://github.com/ricklon/depth-anything-3d-viewer
cd depth-anything-3d-viewer

# Sync dependencies with uv
uv sync

# Install Video-Depth-Anything dependency
uv pip install -e ../Video-Depth-Anything

# Check installation status
uv run da3d status
```

**Alternative: Using pip**

```bash
# Install with pip
pip install -e .

# Install with optional dependencies
pip install -e ".[all]"  # Screen capture + virtual cam + demo

# Install Video-Depth-Anything (required dependency)
cd ..
git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything
pip install -e .
cd ../depth-anything-3d-viewer
```

### Download Depth Model Checkpoints

Download at least the Small model (recommended for real-time use):

```bash
# Create checkpoints directory
mkdir -p checkpoints
cd checkpoints

# Download Small model (112MB, fastest, recommended)
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth

# Optional: Download other models
# Base model (388MB, balanced)
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth

# Large model (1.3GB, best quality)
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth

cd ..
```

### Usage

**If using uv (recommended):**

```bash
# View yourself in 3D mesh (webcam)
uv run da3d webcam3d

# Use camera 1 instead of camera 0
uv run da3d webcam3d --camera-id 1

# View as point cloud instead
uv run da3d webcam3d --display-mode pointcloud

# Capture screen in 3D (interactive mesh)
uv run da3d screen3d-viewer

# Fast mode for better FPS
uv run da3d screen3d-viewer --subsample 4 --max-res 320

# View static depth map
uv run da3d view3d image.jpg depth.png

# 2.5D parallax screen capture
uv run da3d screen3d --auto-rotate

# Get help
uv run da3d --help
uv run da3d webcam3d --help
```

**If using pip:**

```bash
# Replace 'uv run da3d' with just 'da3d'
da3d webcam3d
da3d webcam3d --camera-id 1
da3d screen3d-viewer
```

## Quick Reference

| Command | Description | Common Options |
|---------|-------------|----------------|
| `uv run da3d webcam3d` | View webcam in interactive 3D mesh | `--camera-id 1` `--display-mode pointcloud` |
| `uv run da3d screen3d-viewer` | View desktop/screen in 3D mesh | `--subsample 4 --max-res 320` (fast mode) |
| `uv run da3d view3d img.jpg depth.png` | View static depth map in 3D | `--depth-scale 0.8` `--display-mode pointcloud` |
| `uv run da3d screen3d --auto-rotate` | 2.5D parallax screen effect | `--mouse-control` `--virtual-cam` |

**3D Viewer Controls (all commands):**
- **Mouse drag:** Rotate 3D view
- **Mouse wheel:** Zoom in/out
- **Shift + drag:** Pan camera
- **Q or ESC:** Exit

## Commands

### `webcam3d` - Real-time Webcam 3D

View yourself as a live-updating 3D mesh:

```bash
# Basic usage (optimized defaults, camera 0)
uv run da3d webcam3d

# Select specific camera (0, 1, 2, etc.)
uv run da3d webcam3d --camera-id 1

# Select camera with custom resolution
uv run da3d webcam3d --camera-id 1 --camera-width 1920 --camera-height 1080

# Point cloud mode (faster, different aesthetic)
uv run da3d webcam3d --display-mode pointcloud

# Fast performance mode
uv run da3d webcam3d --subsample 4 --max-res 320

# High quality mesh
uv run da3d webcam3d --subsample 2 --max-res 640 --smooth

# Custom depth range (reduce background)
uv run da3d webcam3d --depth-max-percentile 80

# Adjust depth effect strength
uv run da3d webcam3d --depth-scale 0.8
```

**Controls:**
- Mouse drag: Rotate camera 360°
- Mouse wheel: Zoom in/out
- Shift + drag: Pan camera
- **X: Capture current frame and view with high-quality DA3 model**
- Q or ESC: Close window and exit

**How to find your camera ID:**
```bash
# Test camera 0
uv run da3d webcam3d --camera-id 0

# Test camera 1
uv run da3d webcam3d --camera-id 1

# Continue testing until you find your camera
```

### `screen3d-viewer` - Screen Capture 3D

Turn your screen/desktop into an interactive 3D scene that you can rotate and explore:

```bash
# Capture primary monitor in 3D mesh
uv run da3d screen3d-viewer

# Fast mode (better FPS for real-time viewing)
uv run da3d screen3d-viewer --subsample 4 --max-res 320

# Balanced mode (good quality and performance)
uv run da3d screen3d-viewer --subsample 3 --max-res 480

# High quality (slower but better mesh)
uv run da3d screen3d-viewer --subsample 2 --max-res 640

# Point cloud mode instead of mesh
uv run da3d screen3d-viewer --display-mode pointcloud

# Gaming mode (aggressive depth clamping for better 3D)
uv run da3d screen3d-viewer --depth-min-percentile 10 --depth-max-percentile 90

# Adjust 3D depth effect
uv run da3d screen3d-viewer --depth-scale 0.8

# Custom background color (RGB 0-1 range)
uv run da3d screen3d-viewer --background 0.2,0.2,0.3

# Experimental GUI controls (adjust parameters in real-time)
uv run da3d screen3d-viewer --gui
```

**Controls:**
- Mouse drag: Rotate 3D view
- Mouse wheel: Zoom in/out
- Shift + drag: Pan camera
- **X: Capture current frame and view with high-quality DA3 model**
- Q or ESC: Close window and exit

**Use Cases:**
- View game footage in true 3D
- Explore photos/videos as interactive 3D scenes
- Create 3D presentations from 2D content
- Analyze depth in videos and streams

### `view3d` - Static 3D Viewing

Explore static depth maps interactively:

```bash
# Basic viewing (mesh mode)
uv run da3d view3d image.jpg depth.png

# View as point cloud
uv run da3d view3d image.jpg depth.png --display-mode pointcloud

# Adjust depth range for better visualization
uv run da3d view3d image.jpg depth.png --depth-min-percentile 5 --depth-max-percentile 95

# High quality with full resolution
uv run da3d view3d image.jpg depth.png --subsample 1 --depth-scale 0.7

# Wireframe mesh mode
uv run da3d view3d image.jpg depth.png --wireframe

# Point cloud with adjusted depth
uv run da3d view3d image.jpg depth.png --display-mode pointcloud --depth-scale 0.8
```

**Controls:**
- Same as webcam3d and screen3d-viewer
- Mouse drag to rotate, wheel to zoom, Shift+drag to pan
- Q or ESC to exit

### `screen3d` - 2.5D Parallax Effects

Real-time screen capture with cinematic parallax effects (not full 3D, but parallax projection):

```bash
# Auto-rotating parallax effect
uv run da3d screen3d --auto-rotate

# Mouse-controlled parallax
uv run da3d screen3d --mouse-control

# With virtual camera output (for OBS streaming)
uv run da3d screen3d --virtual-cam --mouse-control

# Show displacement visualization (debug mode)
uv run da3d screen3d --show-displacement --test-grid
```

**Note:** `screen3d` creates 2.5D parallax effects, while `screen3d-viewer` creates true interactive 3D meshes.

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
--depth-min-percentile N  # Clamp far depth/background (0-100)
--depth-max-percentile N  # Clamp near depth/foreground (0-100)
```

**Defaults:**
- `webcam3d`: 0-100% (full range, preserves foreground)
- `screen3d-viewer`: 5-100% (reduces background noise, preserves foreground)
- `view3d`: 0-100% (full control)

**Examples:**
```bash
# More aggressive background removal
uv run da3d webcam3d --depth-min-percentile 20

# Clip foreground artifacts (if objects get too close/distorted)
uv run da3d webcam3d --depth-max-percentile 95

# Full depth range (no clamping)
uv run da3d view3d image.jpg depth.png --depth-min-percentile 0 --depth-max-percentile 100
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
uv run da3d webcam3d --subsample 4 --max-res 320

# Balanced (8-12 FPS, default)
uv run da3d webcam3d --subsample 3 --max-res 480

# High Quality (4-6 FPS)
uv run da3d webcam3d --subsample 2 --max-res 640 --smooth
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

- Python 3.11+ (3.12 recommended)
- **NVIDIA GPU with CUDA** (highly recommended for real-time performance)
- PyTorch 2.0+ with CUDA support
- Open3D 0.18+ (for 3D viewing)
- OpenCV 4.8+
- NumPy, SciPy

### GPU Acceleration (Recommended)

**GPU acceleration provides 10-20x faster performance!**

The project supports:
- **NVIDIA GPUs** (CUDA) - Windows/Linux
- **Apple Silicon** (Metal/MPS) - macOS (M1/M2/M3)

The project is configured for CUDA 12.1 by default on Windows/Linux. After installation:

```bash
# Verify GPU is detected
uv run python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"MPS\" if hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available() else \"CPU only\"}')"
```

If you see "CPU only", follow the [GPU Setup Guide](GPU_SETUP.md) for detailed instructions.

**Quick GPU fix:**
```bash
# Remove CPU-only environment and reinstall with GPU support
rm -rf .venv
uv sync
```

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

### Check Installation Status

Run the diagnostic tool to verify your installation:

```bash
uv run da3d status
```

This will check:
- Python version compatibility
- All required dependencies
- GPU/CUDA availability
- Model checkpoints
- Optional dependencies

### Installation issues with xformers (Windows)

If you see build errors related to `xformers` during `uv sync`:

**Solution:** xformers is now optional and not required for this package. The package will install without it.

If you specifically need xformers for optimization, install it separately after the main installation:
```bash
uv pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

### 3D viewer window not appearing

The Open3D window should open automatically with the title "Real-Time 3D Depth Viewer" or "3D Depth Viewer":
- Check your taskbar for a new window
- Try Alt+Tab to switch between windows
- The window might be behind other windows
- On some systems, it may take a few seconds to appear

### Camera not found (webcam3d)

Test different camera IDs:
```bash
# Try camera 0
uv run da3d webcam3d --camera-id 0

# Try camera 1
uv run da3d webcam3d --camera-id 1

# Continue testing with 2, 3, etc.
```

### Face/body cut off in webcam mode

The default 0-90% depth range should prevent this. If still cut off:
```bash
uv run da3d webcam3d --depth-max-percentile 95
```

### Too much background noise

```bash
uv run da3d webcam3d --depth-max-percentile 80
```

### Mesh looks too flat

```bash
uv run da3d view3d image.jpg depth.png --depth-scale 0.8
```

### Depth looks too extreme/stretched

```bash
uv run da3d view3d image.jpg depth.png --depth-scale 0.3
```

### Performance issues (low FPS)

```bash
uv run da3d webcam3d --subsample 4 --max-res 320
uv run da3d screen3d-viewer --subsample 4 --max-res 320
```

### Model checkpoints not found

Ensure checkpoints are in `./checkpoints/` or specify path:
```bash
uv run da3d webcam3d --checkpoints-dir /path/to/checkpoints
```

Download checkpoints if missing:
```bash
cd checkpoints
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
cd ..
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
