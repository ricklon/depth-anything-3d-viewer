# Model Checkpoints Guide

This guide explains all the depth estimation models used by Depth-Anything-3D and how to obtain them.

## Model Overview

Depth-Anything-3D supports multiple depth estimation models for different use cases:

| Model | Size | Speed | Quality | Required For |
|-------|------|-------|---------|--------------|
| **Video-Depth-Anything Small (VDA-S)** | 116 MB | Fast | Good | Real-time viewing (default) |
| **Video-Depth-Anything Base (VDA-B)** | 388 MB | Medium | Better | Balanced quality |
| **Video-Depth-Anything Large (VDA-L)** | 1.5 GB | Slow | Best | High-quality offline |
| **Metric VDA Small** | ~120 MB | Fast | Good | Metric depth mode |
| **Depth-Anything-3 (DA3)** | ~2 GB | Slow | Excellent | X key capture |

## Required Models (Minimum)

### Video-Depth-Anything Small (VDA-S) ✅ REQUIRED

**Required for:** `webcam3d`, `screen3d-viewer`, `webcam`, `screen`, `video`

```bash
cd checkpoints
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
```

This is the **minimum required model** for the package to work.

## Optional Models (Enhanced Features)

### Video-Depth-Anything Base (VDA-B)

**Use for:** Better quality with moderate speed

```bash
cd checkpoints
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth
```

**Usage:**
```bash
uv run da3d webcam3d --encoder vitb
```

### Video-Depth-Anything Large (VDA-L)

**Use for:** Best quality for offline processing

```bash
cd checkpoints
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
```

**Usage:**
```bash
uv run da3d webcam3d --encoder vitl
uv run da3d video input.mp4 --encoder vitl
```

### Metric Video-Depth-Anything Small

**Required for:** `--metric` flag (accurate 3D reconstruction)

```bash
cd checkpoints
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth
```

**Usage:**
```bash
uv run da3d webcam3d --metric
uv run da3d webcam3d --metric --high-quality
```

**What is metric depth?**
- Uses camera intrinsics for accurate 3D proportions
- Provides real-world depth measurements
- Better for scientific applications
- Requires camera calibration data

### Depth-Anything-3 (DA3)

**Required for:** X key high-quality capture

**Status:** ⚠️ **NOT YET PUBLICLY RELEASED**

The Depth-Anything-3 model provides significantly better depth quality than Video-Depth-Anything. When available:

1. Download from Hugging Face (link TBD)
2. Place in `checkpoints/` directory
3. Install depth-anything-3 package:
   ```bash
   uv pip install depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3
   ```

**Current behavior:**
- If DA3 not found, X key capture falls back to VDA model
- Warning displayed: "depth_anything_3 not found. DA3Estimator will fail to load."
- This is **expected** and does not affect core functionality

## Download All Models Script

Create a helper script to download all models:

```bash
#!/bin/bash
# download_models.sh

CHECKPOINT_DIR="checkpoints"
mkdir -p "$CHECKPOINT_DIR"
cd "$CHECKPOINT_DIR"

echo "Downloading Video-Depth-Anything models..."

# Required: Small model
echo "1/3: Downloading Small model (116 MB)..."
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth

# Optional: Base model
echo "2/3: Downloading Base model (388 MB)..."
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth

# Optional: Large model
echo "3/3: Downloading Large model (1.5 GB)..."
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth

# Optional: Metric Small model
echo "4/4: Downloading Metric Small model (120 MB)..."
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth

echo "✅ Download complete!"
ls -lh
```

## Model Selection Guide

### For Real-Time Viewing (webcam, screen)
- **Fast (15-20 FPS):** `--encoder vits` (Small, default)
- **Balanced (8-12 FPS):** `--encoder vitb` (Base)
- **Quality (4-6 FPS):** `--encoder vitl` (Large)

### For Video Processing
- **Quick preview:** `--encoder vits`
- **Production:** `--encoder vitb` or `vitl`
- **With metric depth:** `--encoder vits --metric`

### For High-Quality Capture
1. Start real-time viewer: `uv run da3d webcam3d --encoder vits`
2. Press **X** to capture with DA3 (when available)
3. Or use: `--encoder vitl` for best real-time quality

## Checking Model Availability

```bash
# Check which models you have
ls -lh checkpoints/

# Verify model loads correctly
uv run python -c "
import torch
model = torch.load('checkpoints/video_depth_anything_vits.pth')
print('✅ Model loaded successfully')
print(f'Model keys: {list(model.keys())[:5]}...')
"
```

## Model Storage

**Recommended directory structure:**

```
depth-anything-3d-viewer/
├── checkpoints/                        # All model checkpoints
│   ├── video_depth_anything_vits.pth  # 116 MB ✅ Required
│   ├── video_depth_anything_vitb.pth  # 388 MB (optional)
│   ├── video_depth_anything_vitl.pth  # 1.5 GB (optional)
│   ├── metric_video_depth_anything_vits.pth  # 120 MB (optional)
│   └── depth_anything_3_*.pth         # TBD (optional, for X key)
```

**Storage requirements:**
- Minimum: 116 MB (Small model only)
- Recommended: 504 MB (Small + Base)
- Complete: ~4 GB (all models)

## Troubleshooting

### "Model checkpoint not found"

```bash
# Check checkpoint directory exists
ls -la checkpoints/

# Re-download missing model
cd checkpoints
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
```

### "depth_anything_3 not found"

**This is expected!** DA3 is not yet publicly released. The package works fine without it:
- X key capture will use VDA model instead
- All other features work normally
- No action needed

### "metric model not found"

Only needed for `--metric` flag. Either:
1. Download metric model (see above)
2. Don't use `--metric` flag (use relative depth instead)

### Model loads but crashes

```bash
# Check model file integrity
md5sum checkpoints/video_depth_anything_vits.pth

# Re-download if corrupted
rm checkpoints/video_depth_anything_vits.pth
# Download again (see above)
```

### Out of memory (OOM) errors

```bash
# Use smaller model
uv run da3d webcam3d --encoder vits

# Reduce resolution
uv run da3d webcam3d --max-res 320

# Increase subsampling
uv run da3d webcam3d --subsample 4
```

## Model Licenses

- **Video-Depth-Anything Small:** Apache 2.0 (commercial use OK)
- **Video-Depth-Anything Base/Large:** CC-BY-NC-4.0 (non-commercial only)
- **Depth-Anything-3:** Check model card when released

**For commercial applications:** Use Small model only, or check licensing.

## References

- [Video-Depth-Anything Paper](https://arxiv.org/abs/2XXX.XXXXX)
- [Video-Depth-Anything Repository](https://github.com/DepthAnything/Video-Depth-Anything)
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (when available)
