# GPU Acceleration Setup Guide

This guide helps you enable NVIDIA GPU acceleration for maximum performance with Depth-Anything-3D.

## Prerequisites

1. **NVIDIA GPU** with CUDA support (RTX series, GTX 10xx+, etc.)
2. **NVIDIA Drivers** installed and up-to-date
3. **CUDA Toolkit** (recommended but not required - PyTorch includes it)

## Check Your Current Setup

```bash
# Check if you have an NVIDIA GPU
nvidia-smi

# Check if PyTorch can see your GPU
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

If CUDA is **not available**, follow the steps below.

## Quick GPU Setup (Recommended)

The project is now configured to automatically use CUDA 12.1 when you sync:

```bash
# Sync with GPU support (CUDA 12.1)
uv sync

# Verify GPU is working
uv run python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not available\"}')"
```

**Expected output:**
```
GPU: NVIDIA GeForce RTX 3080
```
(or your GPU model)

**Note:** `xformers` is now an optional dependency. If you need it for optimization, install with:
```bash
uv pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

## Manual GPU Installation

If the automatic setup doesn't work, install manually:

### Option 1: CUDA 12.1 (Recommended for most GPUs)

```bash
# Install CUDA-enabled PyTorch
uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Optional: Install xformers for attention optimizations
uv pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

### Option 2: CUDA 11.8 (For older drivers)

```bash
# Update pyproject.toml [tool.uv.index] url to cu118
# Then install
uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Option 3: CUDA 12.4 (Latest, for newest GPUs)

```bash
# Update pyproject.toml [tool.uv.index] url to cu124
# Then install
uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Verify GPU Acceleration

Run this comprehensive test:

```bash
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"
```

## Performance Comparison

### CPU Mode (slow)
- Webcam 3D: 2-4 FPS
- Screen 3D: 1-3 FPS
- Processing time: ~500-1000ms per frame

### GPU Mode (fast)
- Webcam 3D: 15-30 FPS (depending on settings)
- Screen 3D: 10-20 FPS
- Processing time: ~30-100ms per frame

**10-20x faster with GPU!**

## Troubleshooting

### "CUDA out of memory" errors

Reduce resolution and subsample rate:
```bash
uv run da3d screen3d-viewer --subsample 4 --max-res 320
```

### GPU not detected after installation

1. Update NVIDIA drivers:
   - Visit: https://www.nvidia.com/Download/index.aspx
   - Download latest driver for your GPU

2. Restart your computer

3. Verify with `nvidia-smi`

### Multiple CUDA versions installed

```bash
# Check CUDA version supported by your driver
nvidia-smi

# Install matching PyTorch version
# For CUDA 12.x shown in nvidia-smi:
uv pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu121
```

### xFormers installation issues (Windows)

xFormers is now **optional** as it's not used by default. If you need it:

```bash
# Install pre-built wheel from PyTorch index
uv pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

If you see build errors, skip xformers - it's not required for this package.

## Optimal GPU Settings for Streaming

If using with OBS + NVIDIA Broadcast:

```bash
# Balance GPU usage between depth processing and streaming
uv run da3d screen3d-viewer \
  --display-mode pointcloud \
  --subsample 3 \
  --max-res 480 \
  --encoder vits
```

Monitor GPU usage:
- **Windows:** Task Manager → Performance → GPU
- **Linux:** `nvidia-smi -l 1` (updates every second)

Keep GPU usage at 80-90% for best balance between quality and stability.

## GPU Memory Requirements

| Configuration | VRAM Required | Recommended GPU |
|---------------|---------------|-----------------|
| `--encoder vits --max-res 320` | 2-3 GB | GTX 1060 6GB+ |
| `--encoder vits --max-res 480` | 3-4 GB | GTX 1660 Ti+ |
| `--encoder vitb --max-res 640` | 4-6 GB | RTX 2060+ |
| `--encoder vitl --max-res 1080` | 8-12 GB | RTX 3080+ |

## Force CPU Mode (if needed)

If you need to force CPU mode:

```bash
# Set environment variable
export CUDA_VISIBLE_DEVICES=""

# Or in PowerShell (Windows)
$env:CUDA_VISIBLE_DEVICES=""

# Then run
uv run da3d screen3d-viewer
```

## Next Steps

Once GPU is working:

1. Test performance:
   ```bash
   uv run da3d webcam3d --camera-id 0
   ```

2. Try high-quality settings:
   ```bash
   uv run da3d screen3d-viewer --subsample 2 --max-res 640 --encoder vitb
   ```

3. Optimize for your use case using the main README.md

## Support

If you're still having issues:
- Check: [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- Check: [PyTorch CUDA Docs](https://pytorch.org/get-started/locally/)
- Open an issue: https://github.com/ricklon/depth-anything-3d-viewer/issues
