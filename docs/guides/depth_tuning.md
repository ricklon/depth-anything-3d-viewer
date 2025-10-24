# Depth Range Tuning Guide

Guide to controlling depth extremes and optimizing performance in the 3D viewers.

## Problem: Extreme Depth Values

By default, depth maps use the full range of values (0-1), which can create:
- **Very stretched meshes** with extreme depth differences
- **Noisy far regions** that distort the scene
- **Unnatural depth perception** that doesn't match the scene

## Solution: Percentile Clamping

The new percentile clamping feature lets you focus on the meaningful depth range by filtering out extreme values.

### How It Works

Instead of using the full depth range, you can clamp to specific percentiles:

```
Original depth:     [0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 0.99]
                     ^^^^^^^                                         ^^^^^^^
                     outliers                                         outliers

Clamped (5-95%):    [0.05, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 0.95]
                     ^^^^^                                           ^^^^^
                     clamped to 5%                                   clamped to 95%

Normalized:         [0.00, 0.00, 0.11, 0.28, 0.50, 0.72, 0.89, 1.00, 1.00]
```

This creates a more uniform depth distribution focused on the main scene.

## Usage

### Basic Commands

**Static Viewer (view3d):**
```bash
# Reduce extremes with 5-95% range (recommended)
uv run vda view3d image.jpg depth.png --depth-min-percentile 5 --depth-max-percentile 95

# More aggressive clamping (10-90%)
uv run vda view3d image.jpg depth.png --depth-min-percentile 10 --depth-max-percentile 90

# Focus on close objects (0-80%)
uv run vda view3d image.jpg depth.png --depth-max-percentile 80
```

**Real-Time Webcam:**
```bash
# Default: 5-95% range (already applied!)
uv run vda webcam3d

# More aggressive for cluttered scenes
uv run vda webcam3d --depth-min-percentile 10 --depth-max-percentile 90

# Full range (no clamping)
uv run vda webcam3d --depth-min-percentile 0 --depth-max-percentile 100
```

**Real-Time Screen:**
```bash
# Default: 5-95% range
uv run vda screen3d-viewer

# Tune for specific content
uv run vda screen3d-viewer --depth-min-percentile 10 --depth-max-percentile 85
```

## Performance Optimization

For better real-time performance, combine depth clamping with these settings:

```bash
# Fast performance with lower resolution
uv run vda webcam3d --subsample 4 --max-res 320

# Balanced performance (default)
uv run vda webcam3d --subsample 3 --max-res 480

# Maximum quality (slower)
uv run vda webcam3d --subsample 2 --max-res 640
```

## Recommended Settings by Scenario

### Portrait / Webcam (Close Range)

```bash
# Balanced (default)
uv run vda webcam3d --depth-min-percentile 5 --depth-max-percentile 95

# High quality
uv run vda webcam3d --depth-min-percentile 10 --depth-max-percentile 90 --subsample 2

# Fast performance
uv run vda webcam3d --subsample 4 --max-res 320
```

**Why:** Close-range scenes have less depth variation, so moderate clamping works well.

### Indoor Scenes

```bash
# Reduce background clutter
uv run vda screen3d-viewer --depth-min-percentile 5 --depth-max-percentile 85

# Focus on foreground objects
uv run vda screen3d-viewer --depth-min-percentile 10 --depth-max-percentile 80
```

**Why:** Indoor scenes often have cluttered backgrounds that create noise in depth estimation.

### Outdoor / Landscape

```bash
# Keep more depth range for distant objects
uv run vda screen3d-viewer --depth-min-percentile 2 --depth-max-percentile 98

# Or use full range
uv run vda screen3d-viewer --depth-min-percentile 0 --depth-max-percentile 100
```

**Why:** Outdoor scenes have more meaningful depth variation across the full range.

### Gaming Content

```bash
# Aggressive clamping for stylized graphics
uv run vda screen3d-viewer --depth-min-percentile 10 --depth-max-percentile 90

# Fast performance
uv run vda screen3d-viewer --subsample 4 --depth-min-percentile 10 --depth-max-percentile 90
```

**Why:** Game graphics can have extreme depth jumps (skybox, fog) that don't represent real geometry.

## Parameter Reference

### Depth Percentile Clamping

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `--depth-min-percentile` | 0-100 | 0 (static)<br>5 (real-time) | Clamps near depth values<br>Higher = removes more foreground |
| `--depth-max-percentile` | 0-100 | 100 (static)<br>95 (real-time) | Clamps far depth values<br>Lower = removes more background |

**Common Presets:**
- **No clamping:** `--depth-min-percentile 0 --depth-max-percentile 100`
- **Light clamping (default):** `--depth-min-percentile 5 --depth-max-percentile 95`
- **Moderate clamping:** `--depth-min-percentile 10 --depth-max-percentile 90`
- **Aggressive clamping:** `--depth-min-percentile 15 --depth-max-percentile 85`
- **Foreground focus:** `--depth-min-percentile 0 --depth-max-percentile 70`

### Performance Settings

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--subsample` | 3 | Mesh resolution (2=high quality, 3=balanced, 4=fast) |
| `--max-res` | 480 | Maximum frame resolution (lower = faster depth estimation) |
| `--smooth` | Off | Enable mesh smoothing (slower but cleaner) |

## Visual Examples

### Before Clamping (0-100%)
```
Depth distribution: [||||                                        ||||||||||]
                     ^very close                          very far^

Result: Extreme stretching, noisy background dominates
```

### After Clamping (5-95%)
```
Depth distribution: [||||||||||||||||||||||||||||||||||||||||||||]
                     ^smooth distribution across main scene^

Result: Balanced depth, focused on main subject
```

## Troubleshooting

### Mesh Still Looks Too Extreme

**Try:**
1. Increase minimum percentile: `--depth-min-percentile 10`
2. Decrease maximum percentile: `--depth-max-percentile 85`
3. Reduce depth scale: `--depth-scale 75`

### Mesh Looks Too Flat

**Try:**
1. Decrease clamping (closer to 0-100%)
2. Increase depth scale: `--depth-scale 150`
3. Check if depth map has enough variation

### Foreground Missing

**Try:**
- Lower minimum percentile: `--depth-min-percentile 0` or `2`

### Background Too Noisy

**Try:**
- Lower maximum percentile: `--depth-max-percentile 80` or `85`
- Increase depth threshold: `--depth-threshold 0.90`

### Performance Issues

**Try:**
1. Increase subsample: `--subsample 4`
2. Lower resolution: `--max-res 320`
3. Use smaller model: `--encoder vits` (already default)
4. Increase clamping (less depth variation = simpler mesh)

## Advanced: Combining All Options

### Maximum Quality (Slow)
```bash
uv run vda view3d image.jpg depth.png \
  --subsample 1 \
  --depth-min-percentile 5 \
  --depth-max-percentile 95 \
  --depth-scale 120
```

### Maximum Performance (Fast)
```bash
uv run vda webcam3d \
  --subsample 4 \
  --max-res 320 \
  --depth-min-percentile 10 \
  --depth-max-percentile 90
```

### Balanced (Recommended)
```bash
uv run vda webcam3d \
  --subsample 3 \
  --max-res 480 \
  --depth-min-percentile 5 \
  --depth-max-percentile 95 \
  --depth-scale 100
```

### Analysis Mode with Wireframe
```bash
# Focus on depth structure
uv run vda view3d image.jpg depth.png \
  --wireframe \
  --depth-min-percentile 10 \
  --depth-max-percentile 90
```

## Technical Details

### Percentile Calculation

```python
# For each frame:
depth_min = np.percentile(depth, depth_min_percentile)  # e.g., 5th percentile
depth_max = np.percentile(depth, depth_max_percentile)  # e.g., 95th percentile

# Clamp values
depth_clamped = np.clip(depth, depth_min, depth_max)

# Normalize to 0-1
depth_normalized = (depth_clamped - depth_min) / (depth_max - depth_min)

# Apply depth scale
z_coordinates = depth_normalized * depth_scale
```

### Why This Works

1. **Removes outliers**: Extreme values (sensor noise, far background) are clamped
2. **Improves dynamic range**: More depth values concentrated in meaningful range
3. **Reduces stretching**: Z-coordinates stay within reasonable bounds
4. **Frame-by-frame**: Each frame uses its own percentiles (adapts to scene changes)

### Performance Impact

| Setting | Memory | Speed | Quality |
|---------|---------|--------|---------|
| Percentile clamping | negligible | negligible | ↑↑ Better |
| Subsample 4 vs 2 | ↓ -75% | ↑↑ +4x faster | ↓ Lower resolution |
| Max-res 320 vs 640 | ↓ -75% | ↑↑ +4x faster | ↓ Lower detail |

## Performance Expectations

| Configuration | FPS (GPU) | Quality | Use Case |
|---------------|-----------|---------|----------|
| `--subsample 4 --max-res 320` | 15-20 | Medium | Fast preview |
| `--subsample 3 --max-res 480` | 8-12 | Good | Balanced (default) |
| `--subsample 2 --max-res 640` | 4-6 | High | Quality viewing |
| `--subsample 2 --smooth` | 2-4 | Highest | Maximum quality |

*Performance varies by GPU. CPU mode is significantly slower.*

## See Also

- [VIEW3D_GUIDE.md](VIEW3D_GUIDE.md) - Static 3D mesh viewing
- [REALTIME_3D_GUIDE.md](REALTIME_3D_GUIDE.md) - Real-time 3D viewing
- [RESOLUTION_GUIDE.md](RESOLUTION_GUIDE.md) - Resolution optimization
