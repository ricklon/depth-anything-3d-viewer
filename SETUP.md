# Setup Guide

Complete installation and setup instructions for Depth-Anything-3D Viewer.

## Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (recommended) or CPU
- Git

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/YourUsername/depth-anything-3d-viewer
cd depth-anything-3d-viewer

# Install the package
pip install -e .

# Or with all optional dependencies
pip install -e ".[all]"
```

### Method 2: uv (Faster)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager:

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/YourUsername/depth-anything-3d-viewer
cd depth-anything-3d-viewer

# Install with uv
uv pip install -e .

# Or with extras
uv pip install -e ".[all]"
```

### Method 3: From Source (Development)

```bash
# Clone repository
git clone https://github.com/YourUsername/depth-anything-3d-viewer
cd depth-anything-3d-viewer

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install in editable mode
pip install -e ".[dev]"
```

## Download Model Checkpoints

The 3D viewer requires Video-Depth-Anything model checkpoints.

### Quick Download

```bash
# Create checkpoints directory
mkdir checkpoints
cd checkpoints

# Download Small model (fastest, recommended for real-time)
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth

# Or download all models
bash ../scripts/download_checkpoints.sh
```

### Manual Download

Visit [Video-Depth-Anything Hugging Face](https://huggingface.co/depth-anything) and download:

- **Small (vits)** - 97MB, fastest, recommended for webcam/screen
- **Base (vitb)** - 388MB, balanced quality/speed
- **Large (vitl)** - 1.3GB, best quality

Place them in `./checkpoints/`:
```
checkpoints/
├── video_depth_anything_vits.pth
├── video_depth_anything_vitb.pth
└── video_depth_anything_vitl.pth
```

### Metric Depth Models (Optional)

For metric depth (actual distance values):

```bash
cd checkpoints
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth
```

## Optional Dependencies

### Screen Capture

For `screen3d-viewer` and `screen3d` commands:

```bash
pip install "depth-anything-3d[screen]"
# Or manually:
pip install mss
```

### Virtual Camera

To output 3D parallax to OBS virtual camera:

```bash
pip install "depth-anything-3d[virtual-cam]"
# Or manually:
pip install pyvirtualcam
```

### Web Demo

For Gradio web interface:

```bash
pip install "depth-anything-3d[demo]"
# Or manually:
pip install gradio
```

### All Optional Dependencies

```bash
pip install "depth-anything-3d[all]"
```

## Verify Installation

Test that everything is working:

```bash
# Check CLI is installed
da3d --help

# Should show:
# usage: da3d [-h] {webcam3d,screen3d-viewer,view3d,...}

# Verify models are found
da3d webcam3d --help
# Should NOT show "checkpoint not found" error
```

## Platform-Specific Notes

### Windows

1. **CUDA Setup:**
   ```powershell
   # Check CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Webcam Permissions:**
   - Windows may prompt for camera access
   - Grant permissions in Settings → Privacy → Camera

3. **Virtual Camera (Optional):**
   - Install [OBS Studio](https://obsproject.com/)
   - Install [OBS Virtual Camera plugin](https://obsproject.com/forum/resources/obs-virtualcam.949/)

### Linux

1. **CUDA Setup:**
   ```bash
   # Install CUDA drivers
   sudo apt install nvidia-cuda-toolkit

   # Verify
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Webcam Access:**
   ```bash
   # Add user to video group
   sudo usermod -a -G video $USER

   # Logout and login for changes to take effect
   ```

3. **Screen Capture Dependencies:**
   ```bash
   # Install system dependencies for mss
   sudo apt install python3-dev libx11-dev libxrandr-dev
   ```

### macOS

1. **Webcam Permissions:**
   - macOS will prompt for camera access
   - Grant in System Preferences → Security & Privacy → Camera

2. **Screen Recording:**
   - Grant screen recording permission
   - System Preferences → Security & Privacy → Screen Recording

## GPU vs CPU

### Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Performance Comparison

| Hardware | Real-time FPS | Quality Setting |
|----------|---------------|-----------------|
| **RTX 3080** | 15-20 | subsample=4, max-res=320 |
| **RTX 3060** | 10-15 | subsample=4, max-res=320 |
| **GTX 1660** | 5-10 | subsample=4, max-res=320 |
| **CPU (i7-10700)** | 1-2 | subsample=4, max-res=320 |

**Recommendation:** GPU strongly recommended for real-time viewing.

## Troubleshooting

### "CUDA out of memory"

Try lowering resolution:
```bash
da3d webcam3d --max-res 320 --subsample 4
```

### "Checkpoint not found"

Ensure checkpoints are downloaded to `./checkpoints/`:
```bash
ls checkpoints/
# Should show: video_depth_anything_vits.pth (or vitb/vitl)
```

Or specify custom path:
```bash
da3d webcam3d --checkpoints-dir /path/to/checkpoints
```

### "Could not open camera"

**Windows:**
- Close other applications using the camera
- Try different `--camera-id` (0, 1, 2)
- Check camera permissions

**Linux:**
- Ensure user is in video group
- Check `/dev/video*` devices exist

### "ModuleNotFoundError: No module named 'open3d'"

```bash
pip install open3d
```

### "ImportError: cannot import name 'VideoDepthAnything'"

The package depends on having Video-Depth-Anything codebase available. Two options:

**Option 1: Install from original repo (if they make it pip-installable)**
```bash
pip install video-depth-anything
```

**Option 2: Clone alongside (current workaround)**
```bash
# Clone original repo
cd ..
git clone https://github.com/DepthAnything/Video-Depth-Anything

# Add to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:../Video-Depth-Anything"

# Or on Windows PowerShell
$env:PYTHONPATH = "$env:PYTHONPATH;..\Video-Depth-Anything"
```

## Next Steps

1. **Quick Test:**
   ```bash
   da3d webcam3d
   ```

2. **Read Guides:**
   - [Getting Started](docs/getting_started.md)
   - [Webcam 3D Guide](docs/guides/realtime_3d.md)
   - [Depth Tuning](docs/guides/depth_tuning.md)

3. **Try Examples:**
   ```bash
   python examples/01_basic_viewing.py
   ```

## Uninstallation

```bash
pip uninstall depth-anything-3d
```

## Getting Help

- [Documentation](docs/)
- [GitHub Issues](https://github.com/YourUsername/depth-anything-3d-viewer/issues)
- [Discussions](https://github.com/YourUsername/depth-anything-3d-viewer/discussions)
