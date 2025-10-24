# Real-Time 3D Viewing Guide

Real-time 3D mesh viewing from webcam or screen capture, creating true 3D geometry that updates dynamically as you move or capture new content.

## Overview

Unlike the static `view3d` command, the real-time viewers continuously update a 3D mesh as new frames arrive from your webcam or screen capture. This creates a live 3D scene you can orbit around while the depth and texture update in real-time.

## Commands

### Webcam 3D Viewer

View yourself or your environment in real-time 3D:

```bash
# Basic webcam 3D
uv run vda webcam3d

# Higher quality (slower)
uv run vda webcam3d --subsample 2 --max-res 640

# Different camera
uv run vda webcam3d --camera-id 1
```

### Screen Capture 3D Viewer

Turn your screen into a live 3D mesh:

```bash
# Capture primary monitor
uv run vda screen3d-viewer

# Specific region
uv run vda screen3d-viewer --region 0,0,1920,1080

# Higher FPS
uv run vda screen3d-viewer --fps 15
```

## Quick Start Examples

### Portrait Mode (Webcam)
```bash
# Best settings for viewing yourself in 3D
uv run vda webcam3d --depth-scale 150 --subsample 3 --max-res 480
```

### Gaming Scene (Screen)
```bash
# Capture game window and view in 3D
uv run vda screen3d-viewer --region 0,0,1920,1080 --depth-scale 120 --fps 10
```

### High Quality (Slower)
```bash
# Maximum quality with smoothing
uv run vda webcam3d --subsample 2 --smooth --max-res 640
```

### Performance Mode
```bash
# Fast preview with lower resolution
uv run vda webcam3d --subsample 4 --max-res 320
```

## Command Line Options

### Common Options (Both Commands)

**Model Settings:**
```bash
--encoder vits|vitb|vitl   # Model size (default: vits for real-time)
--metric                   # Use metric depth model
--input-size INT           # Model input size (default: 518)
--fp32                     # Use FP32 precision (slower, more accurate)
--checkpoints-dir PATH     # Checkpoint directory (default: ./checkpoints)
```

**3D Visualization:**
```bash
--depth-scale FLOAT        # Z-displacement scale (default: 100.0)
                          # Higher = more dramatic depth
                          # Try 80-150 for most scenes

--subsample INT           # Mesh downsample factor (default: 3)
                          # 2 = high quality (slower)
                          # 3 = balanced (recommended)
                          # 4 = fast preview

--smooth                  # Enable mesh smoothing (slower but cleaner)
                          # Recommended for noisy depth maps

--invert-depth            # Reverse depth direction
                          # Makes near objects appear far

--background R,G,B        # Background color (0-1 range)
                          # Default: 0.1,0.1,0.1 (dark gray)
                          # Example: --background 1,1,1 (white)
```

**Performance:**
```bash
--max-res INT             # Maximum frame resolution (default: 480)
                          # Lower = faster but less detailed
                          # 320 = very fast
                          # 480 = balanced
                          # 640 = high quality
```

### Webcam-Specific Options

```bash
--camera-id INT           # Camera device ID (default: 0)
                          # Try 1, 2, etc. for other cameras

--camera-width INT        # Set camera width (-1 for default)
--camera-height INT       # Set camera height (-1 for default)
```

### Screen Capture-Specific Options

```bash
--monitor INT             # Monitor number (1 = primary, default: 1)

--region X,Y,W,H          # Capture specific region
                          # Format: "x,y,width,height"
                          # Example: --region 0,0,1920,1080

--fps INT                 # Target frames per second (default: 10)
                          # Higher = smoother but more demanding
```

## Interactive Controls

Once the 3D viewer opens:

| Control | Action |
|---------|--------|
| **Mouse drag** | Rotate camera around mesh |
| **Mouse wheel** | Zoom in/out |
| **Shift + mouse drag** | Pan camera |
| **Close window** | Exit viewer |

The mesh updates automatically as new frames arrive from your webcam or screen.

## Performance Tuning

### For Smooth Real-Time Performance

1. **Lower resolution:**
   ```bash
   --max-res 320
   ```

2. **Increase subsampling:**
   ```bash
   --subsample 4
   ```

3. **Use smaller model:**
   ```bash
   --encoder vits
   ```

4. **Reduce FPS (screen only):**
   ```bash
   --fps 5
   ```

### For Maximum Quality

1. **Higher resolution:**
   ```bash
   --max-res 640
   ```

2. **Lower subsampling:**
   ```bash
   --subsample 2
   ```

3. **Enable smoothing:**
   ```bash
   --smooth
   ```

4. **Use larger model (slower):**
   ```bash
   --encoder vitb
   ```

## Use Cases

### 1. Live Portrait 3D Scanning

```bash
uv run vda webcam3d --depth-scale 150 --subsample 3
```

View yourself in 3D! Move your head slowly to see the depth update. The mesh stays centered on the camera's view.

### 2. Gaming Scene Analysis

```bash
# Capture game window
uv run vda screen3d-viewer --region 0,0,1920,1080 --depth-scale 100
```

Turn your game into a 3D scene. Orbit around to see how the depth estimation works on different game graphics.

### 3. Video Conferencing Background

```bash
uv run vda webcam3d --invert-depth --depth-scale 120
```

See yourself with inverted depth for interesting visual effects.

### 4. Content Creation

```bash
# Record screen capture while viewing 3D
uv run vda screen3d-viewer --monitor 2 --fps 15 --subsample 2
```

Capture creative content or tutorials showing 3D depth in real-time.

## Technical Details

### How It Works

1. **Frame Capture:**
   - Webcam: Reads frames from cv2.VideoCapture
   - Screen: Captures using mss library

2. **Depth Estimation:**
   - Streaming model processes each frame
   - Uses temporal attention for consistency
   - FP16 inference by default for speed

3. **Mesh Generation:**
   - Each pixel becomes a 3D vertex
   - Z-position from depth value
   - Triangles connect adjacent pixels
   - RGB texture from input frame

4. **Visualization:**
   - Open3D non-blocking visualizer
   - Mesh geometry updates each frame
   - Camera stays in same position
   - Interactive controls always available

### Performance Characteristics

| Setting | FPS (GPU) | Quality | Use Case |
|---------|-----------|---------|----------|
| `--subsample 4 --max-res 320` | 15-20 | Low | Fast preview |
| `--subsample 3 --max-res 480` | 8-12 | Medium | Balanced (default) |
| `--subsample 2 --max-res 640` | 4-6 | High | Quality viewing |
| `--subsample 2 --smooth` | 2-4 | Highest | Best quality |

*Performance varies by GPU. CPU mode is significantly slower.*

### Mesh Update Strategy

The viewer uses Open3D's `update_geometry()` for efficient mesh updates:

- **First frame:** Creates initial mesh, adds to scene
- **Subsequent frames:** Updates vertices, colors, and triangles
- **Normals:** Recomputed each frame for proper lighting
- **No allocation:** Reuses mesh object for speed

## Troubleshooting

### "open3d not installed"
```bash
# Install Open3D (included in metric dependencies)
uv sync --extra metric
```

### Low FPS / Laggy

1. **Reduce resolution:**
   ```bash
   --max-res 320
   ```

2. **Increase subsampling:**
   ```bash
   --subsample 4
   ```

3. **Disable smoothing** (if enabled)

4. **Check GPU usage:**
   - Ensure CUDA is available
   - Close other GPU applications

### Mesh Looks Flat

- Increase `--depth-scale`:
  ```bash
  --depth-scale 150
  ```

### Mesh Too Distorted

- Decrease `--depth-scale`:
  ```bash
  --depth-scale 75
  ```

### Camera Not Opening

- Try different camera ID:
  ```bash
  --camera-id 1
  ```

- Check if camera is in use by another app

### Viewer Window Closes Immediately

- This is normal when depth estimation fails
- Check console for error messages
- Ensure checkpoints are downloaded

## Comparison: Real-Time 3D vs Other Modes

| Feature | `webcam3d` / `screen3d-viewer` | `view3d` | `screen3d` |
|---------|-------------------------------|----------|------------|
| **Input** | Live stream | Static files | Live stream |
| **Output** | True 3D mesh | True 3D mesh | 2.5D parallax |
| **Update** | Real-time | Static | Real-time |
| **Viewing** | Full 360° orbit | Full 360° orbit | Limited angles |
| **Performance** | 5-15 FPS | N/A (static) | 10-20 FPS |
| **Use case** | Live 3D view | Inspection | Effects |

## Advanced Usage

### Custom Python Integration

```python
from utils.viewer_3d import RealTime3DViewer
import cv2

# Create viewer
viewer = RealTime3DViewer(
    depth_scale=120,
    subsample=3,
    smooth_mesh=False,
    background_color=(0.1, 0.1, 0.1)
)

# Initialize (call once)
viewer.initialize(width=1280, height=720)

# Your custom frame source
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Your depth estimation code here
    depth = estimate_depth(frame_rgb)

    # Update mesh
    viewer.update_mesh(frame_rgb, depth, invert_depth=False)

    if viewer.should_close():
        break

viewer.close()
cap.release()
```

### Multiple Camera Setup

```python
# Switch between cameras dynamically
cameras = [0, 1, 2]
for cam_id in cameras:
    print(f"Viewing camera {cam_id}")
    # Run viewer with specific camera...
```

## Tips & Best Practices

### For Best Results

1. **Good Lighting:** Depth estimation works better with well-lit scenes
2. **Stable Position:** Mount camera or use tripod for steady view
3. **Gradual Movement:** Move slowly for temporal consistency
4. **Close Range:** Works best within 1-3 meters of camera

### Performance Tips

1. **Start with defaults** then adjust based on performance
2. **Monitor GPU usage** to ensure you're not bottlenecked
3. **Close other applications** using GPU/camera
4. **Lower resolution first** before increasing subsampling

### Visual Quality Tips

1. **Adjust depth scale** based on scene depth range
2. **Use smoothing** for indoor/close-up scenes
3. **Disable smoothing** for outdoor/fast-moving scenes
4. **Try inverted depth** for creative effects

## Example Workflows

### Workflow 1: Live 3D Self-Portrait

```bash
# Step 1: Test your webcam
uv run vda webcam  # Verify webcam works

# Step 2: Start 3D viewer with good defaults
uv run vda webcam3d --depth-scale 140 --subsample 3

# Step 3: Adjust depth scale while viewing
# (Restart with different --depth-scale values)

# Step 4: Enable smoothing if needed
uv run vda webcam3d --depth-scale 140 --subsample 3 --smooth
```

### Workflow 2: Screen Game Analysis

```bash
# Step 1: Identify game window region
# (Use Windows Snipping Tool or similar to get coordinates)

# Step 2: Start viewer with region
uv run vda screen3d-viewer --region 100,100,1920,1080 --depth-scale 100

# Step 3: Tune for performance
uv run vda screen3d-viewer --region 100,100,1920,1080 --depth-scale 100 --max-res 400 --subsample 4

# Step 4: Increase quality once performance is acceptable
uv run vda screen3d-viewer --region 100,100,1920,1080 --depth-scale 100 --max-res 480 --subsample 3
```

## See Also

- [VIEW3D_GUIDE.md](VIEW3D_GUIDE.md) - Static 3D mesh viewing
- [SCREEN3D_GUIDE.md](SCREEN3D_GUIDE.md) - 2.5D parallax effects
- [RESOLUTION_GUIDE.md](RESOLUTION_GUIDE.md) - Resolution optimization
