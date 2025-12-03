# Testing Quickstart Guide

Quick guide to test each feature with the available assets.

## ✅ Ready to Test Now

### 1. Static 3D Viewing
```bash
uv run da3d view3d tests/data/test_image.jpg tests/data/test_depth.png
```
**Assets:** ✅ test_image.jpg, test_depth.png
**Model:** ✅ VDA Small (vits)

### 2. Projector Test Pattern
```bash
uv run da3d projector-preview --config config/projection_example.yaml --show test_pattern_show
```
**Assets:** ✅ test_pattern.png, projection_example.yaml
**Model:** None required

### 3. Projector Lobby Scene
```bash
uv run da3d projector-preview --config config/projection_example.yaml --show lobby_show
```
**Assets:** ✅ lobby_cube.obj, projection_example.yaml
**Model:** None required

### 4. Real-time Webcam 3D (if you have a camera)
```bash
uv run da3d webcam3d
```
**Assets:** None (uses live camera)
**Model:** ✅ VDA Small (vits)

### 5. Screen Capture 3D
```bash
uv run da3d screen3d-viewer
```
**Assets:** None (captures live screen)
**Model:** ✅ VDA Small (vits)
**Requires:** `mss` library (`uv sync --all-extras`)

## ⚠️ Needs Optional Models

### 6. Metric Depth Mode
```bash
uv run da3d webcam3d --metric
```
**Model:** ❌ Needs metric_video_depth_anything_vits.pth
**Download:** See MODELS.md

### 7. High-Quality X Key Capture
Press X during `webcam3d` or `screen3d-viewer`
**Model:** ❌ Needs Depth-Anything-3 (not yet released)
**Fallback:** Uses VDA model

### 8. Large Model for Best Quality
```bash
uv run da3d webcam3d --encoder vitl
```
**Model:** ✅ VDA Large (vitl) - Already downloaded!

## 📝 Feature Status Summary

| Feature | Assets | Models | Dependencies | Status |
|---------|--------|--------|--------------|--------|
| view3d | ✅ | ✅ VDA Small | ✅ | **Ready** |
| webcam3d | ✅ | ✅ VDA Small | ✅ | **Ready** |
| screen3d-viewer | ✅ | ✅ VDA Small | ⚠️ mss | **Almost Ready** |
| screen3d | ✅ | ✅ VDA Small | ⚠️ mss | **Almost Ready** |
| projector commands | ✅ | ✅ None | ✅ | **Ready** |
| video processing | ⚠️ | ✅ VDA Small | ✅ | **Needs test video** |
| Metric depth | ✅ | ❌ Metric | ✅ | **Needs model** |
| X key (DA3) | ✅ | ❌ DA3 | ❌ | **Not available yet** |
| GUI mode | ✅ | ✅ VDA Small | ✅ | **Ready (experimental)** |

## Quick Tests

### Test 1: Verify Installation
```bash
uv run da3d --help
```
Should show all commands without errors.

### Test 2: Check Models
```bash
ls -lh checkpoints/
```
Should show:
- ✅ video_depth_anything_vits.pth (116 MB)
- ✅ video_depth_anything_vitl.pth (1.5 GB)

### Test 3: View Static 3D
```bash
uv run da3d view3d tests/data/test_image.jpg tests/data/test_depth.png
```
Should open 3D viewer. Controls:
- Mouse drag: Rotate
- Mouse wheel: Zoom
- Q/ESC: Exit

### Test 4: GUI Mode (Experimental)
```bash
uv run da3d view3d tests/data/test_image.jpg tests/data/test_depth.png --gui
```
Should open viewer + parameter adjustment window.

## Install Optional Dependencies

```bash
# For screen capture
uv sync --all-extras

# Or individually
uv pip install mss
uv pip install pyvirtualcam  # For virtual camera
uv pip install gradio        # For web demo
```

## Troubleshooting

### "Model not found"
```bash
# Check checkpoints directory
ls checkpoints/

# If empty, download models (see MODELS.md)
cd checkpoints
curl -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
```

### "mss not found" (screen capture)
```bash
uv pip install mss
```

### "depth_anything_3 not found" warning
**This is normal!** DA3 not yet released. Package works fine without it.

### 3D Viewer doesn't open
- Check if window is behind other windows
- Try Alt+Tab to find it
- Window title: "3D Depth Viewer" or "Real-Time 3D Depth Viewer"

## Next Steps

1. ✅ Test static viewing: `uv run da3d view3d tests/data/test_image.jpg tests/data/test_depth.png`
2. ✅ Test projector: `uv run da3d projector-preview --config config/projection_example.yaml --show test_pattern_show`
3. 📸 Test with your own images
4. 🎥 Try webcam if available: `uv run da3d webcam3d`
5. 📥 Download optional models (see MODELS.md)

## Full Documentation

- **MODELS.md** - Complete model download guide
- **README.md** - Full feature documentation
- **FEATURE_ASSET_AUDIT.md** - Detailed asset inventory
- **CLAUDE.md** - Developer documentation
