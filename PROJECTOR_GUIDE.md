# Projector System Guide

## Understanding the Black Screen Issue

The black screen happens because the configuration has **empty calibration data**:

```yaml
surfaces:
  wall_main:
    dst_quad_pixels: []  # ❌ Empty = black screen!
```

The `dst_quad_pixels` defines the 4 corner points where content appears on the projector. Without these coordinates, the engine skips rendering.

## Quick Fix: Use Pre-Calibrated Config

We've created a ready-to-test config with fullscreen coordinates:

```bash
# Test pattern (should show immediately!)
uv run da3d projector-preview --config config/projection_ready.yaml --show test_pattern_show

# Lobby 3D scene
uv run da3d projector-preview --config config/projection_ready.yaml --show lobby_show

# Press 'q' to quit
```

**What this does:**
- Maps content to fullscreen: `[[0, 0], [1920, 0], [1920, 1080], [0, 1080]]`
- Content fills the entire projector output
- Perfect for quick testing!

## Understanding Calibration Coordinates

The `dst_quad_pixels` field expects 4 points in this order:

```
[0] Top-Left -------- [1] Top-Right
    |                      |
    |    Your Content      |
    |                      |
[3] Bottom-Left ---- [2] Bottom-Right
```

**Example coordinates:**
```yaml
dst_quad_pixels: [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]
#                 TL       TR          BR            BL
```

## Proper Calibration Workflow

For real projection mapping (warping content to physical surfaces), use calibration:

### Step 1: Prepare Your Config

```yaml
projectors:
  main:
    display_index: 1  # Which monitor/projector
    resolution: [1920, 1080]

surfaces:
  my_wall:
    projector: main
    type: flat
    dst_quad_pixels: []  # Will be filled by calibration
```

### Step 2: Run Calibration

```bash
uv run da3d projector-calibrate --config config/projection_example.yaml --projector main
```

**What happens:**
1. Calibration UI opens
2. Test pattern displays on projector
3. Click the 4 corners of your physical surface in order:
   - Top-left corner
   - Top-right corner
   - Bottom-right corner
   - Bottom-left corner
4. Press 'c' to confirm, or 'r' to reset
5. Coordinates automatically saved to config file

### Step 3: Preview Your Calibrated Show

```bash
uv run da3d projector-preview --config config/projection_example.yaml --show test_pattern_show
```

Now the content will be warped to match your physical surface!

## Common Configurations

### Fullscreen (No Warping)
```yaml
dst_quad_pixels: [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]
```

### Smaller Rectangle (Centered)
```yaml
# 960x540 centered in 1920x1080
dst_quad_pixels: [[480, 270], [1440, 270], [1440, 810], [480, 810]]
```

### Keystone Correction (Trapezoid)
```yaml
# Wider at bottom (projector tilted down)
dst_quad_pixels: [[400, 0], [1520, 0], [1920, 1080], [0, 1080]]
```

### Custom Surface Mapping
```yaml
# Warped to match physical wall shape
dst_quad_pixels: [[100, 50], [1800, 30], [1850, 1000], [80, 1050]]
```

## Available Content Sources

### 1. Static Image
```yaml
content_sources:
  my_image:
    type: image
    file: "assets/my_image.png"
    light_vignette: true  # Optional: soften edges
```

### 2. Lobby Scene (3D)
```yaml
content_sources:
  my_scene:
    type: lobby_scene
    scene_asset: "assets/my_model.obj"  # 3D model file
```

### 3. Depth Image (Future)
```yaml
content_sources:
  depth_art:
    type: depth_image
    rgb: "assets/art_rgb.jpg"
    depth: "assets/art_depth.png"
```

## Creating Shows

A "show" defines what content appears and when:

```yaml
shows:
  my_demo:
    loop: true  # Repeat infinitely
    scenes:
      - at: 0    # Time in seconds
        surface: wall_main
        content_layers:
          - source: test_pattern

      - at: 10   # Switch content at 10 seconds
        surface: wall_main
        content_layers:
          - source: lobby_test
```

## Troubleshooting

### Black Screen
**Cause:** Empty `dst_quad_pixels: []`
**Fix:** Use `config/projection_ready.yaml` or run calibration

### "Config file not found"
```bash
# Check file exists
ls config/projection_ready.yaml

# Use absolute path
uv run da3d projector-preview --config /full/path/to/config.yaml --show test_pattern_show
```

### "Show not found"
**Cause:** Typo in show name
**Fix:** Check your config's `shows:` section for exact name

### "Image file not found"
```bash
# Check asset exists
ls assets/test_pattern.png

# Use absolute path in config
file: "/full/path/to/assets/test_pattern.png"
```

### Content appears distorted
**Cause:** Wrong aspect ratio in calibration points
**Fix:** Re-run calibration, ensure corners match physical surface

### Nothing renders / OpenCV window is black
**Cause:** Content source failed to load
**Fix:** Check console for errors, verify asset files exist

## Multiple Projectors Example

```yaml
projectors:
  left_wall:
    display_index: 1
    resolution: [1920, 1080]

  right_wall:
    display_index: 2
    resolution: [1920, 1080]

surfaces:
  surface_left:
    projector: left_wall
    type: flat
    dst_quad_pixels: [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]

  surface_right:
    projector: right_wall
    type: flat
    dst_quad_pixels: [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]

shows:
  dual_wall:
    loop: true
    scenes:
      - at: 0
        surface: surface_left
        content_layers:
          - source: test_pattern
      - at: 0
        surface: surface_right
        content_layers:
          - source: lobby_test
```

## Testing Checklist

- [x] `config/projection_ready.yaml` - Ready to test with fullscreen
- [x] `assets/test_pattern.png` - Test image present
- [x] `assets/lobby_cube.obj` - 3D model present
- [ ] Custom images added to `assets/`
- [ ] Calibration completed for real projection mapping
- [ ] Multiple projector setup configured

## Quick Commands Reference

```bash
# Test with fullscreen (works immediately)
uv run da3d projector-preview --config config/projection_ready.yaml --show test_pattern_show
uv run da3d projector-preview --config config/projection_ready.yaml --show lobby_show

# Calibrate for real projection mapping
uv run da3d projector-calibrate --config config/projection_example.yaml --projector main

# Preview after calibration
uv run da3d projector-preview --config config/projection_example.yaml --show test_pattern_show

# Press 'q' to quit any preview
```

## Next Steps

1. ✅ Test with `projection_ready.yaml` to verify system works
2. Create your own projection surfaces (walls, floors, objects)
3. Run calibration to map content to physical surfaces
4. Add custom images/3D models to `assets/`
5. Create multi-scene shows with timed transitions
6. Set up multiple projectors for immersive installations

## See Also

- **TESTING_QUICKSTART.md** - Quick testing guide for all features
- **MODELS.md** - Depth model setup (for depth_image sources)
- **README.md** - Complete feature documentation
