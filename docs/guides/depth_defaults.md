# Depth Range Defaults Explained

Different commands have different default depth percentile settings optimized for their typical use cases.

## Default Settings by Command

| Command | Min % | Max % | Reasoning |
|---------|-------|-------|-----------|
| `webcam3d` | **0%** | **90%** | Preserves close subjects (you!), removes far background |
| `screen3d-viewer` | **5%** | **95%** | Balanced for mid-range screen content |
| `view3d` | **0%** | **100%** | No clamping, full control for static analysis |

## Why Different Defaults?

### Webcam (0-90%)

**Problem:** Webcam typically shows **you** at close range (0.5-2 meters). Using 5-95% would cut off your face/body!

**Solution:**
- **0% minimum** = Keep ALL foreground (preserves closest parts like face, hands)
- **90% maximum** = Remove far background (wall behind you, clutter)

```bash
# Default webcam settings (optimized for portraits)
uv run vda webcam3d
# Uses: --depth-min-percentile 0 --depth-max-percentile 90
```

**Result:** Your face and body are fully visible in 3D, background is simplified.

### Screen Capture (5-95%)

**Problem:** Screen content is typically mid-range (games, videos, UI) with some extreme depth values.

**Solution:**
- **5% minimum** = Remove very close anomalies
- **95% maximum** = Remove very far anomalies

```bash
# Default screen capture settings
uv run vda screen3d-viewer
# Uses: --depth-min-percentile 5 --depth-max-percentile 95
```

**Result:** Balanced depth range for most screen content.

### Static Viewer (0-100%)

**Problem:** You're analyzing a static file and want full control.

**Solution:**
- **0% minimum** = Keep everything
- **100% maximum** = Keep everything
- **You decide** what to clamp via command line

```bash
# Default static viewer (no auto-clamping)
uv run vda view3d image.jpg depth.png
# Uses: --depth-min-percentile 0 --depth-max-percentile 100

# But you can specify any range:
uv run vda view3d image.jpg depth.png --depth-min-percentile 5 --depth-max-percentile 95
```

## Visual Comparison

### Webcam with 5-95% (OLD - BAD for close subjects)
```
Depth map:  [||||FACE||||||||||||BODY||||||||||||WALL|||||||||||||||]
Clamped:    [----FACE(cut)-------BODY------------WALL(cut)----------]
             ^^^^CUT OFF!                        ^^^^CUT OFF

Result: Face is clipped/missing! ❌
```

### Webcam with 0-90% (NEW - GOOD for close subjects)
```
Depth map:  [||||FACE||||||||||||BODY||||||||||||WALL|||||||||||||||]
Clamped:    [||||FACE||||||||||||BODY||||||||||||WALL(cut)----------]
             ^^^^PRESERVED!                      ^^^^simplified

Result: Face fully visible, background simplified! ✓
```

## When to Override Defaults

### Webcam Scenarios

**Sitting far from camera (2+ meters):**
```bash
# Use more clamping since you're not as close
uv run vda webcam3d --depth-min-percentile 5 --depth-max-percentile 90
```

**Very close to camera (< 0.5 meters):**
```bash
# Keep even more background range
uv run vda webcam3d --depth-min-percentile 0 --depth-max-percentile 95
```

**Clean background (minimal clutter):**
```bash
# Don't clamp background much
uv run vda webcam3d --depth-min-percentile 0 --depth-max-percentile 95
```

**Cluttered background (messy room):**
```bash
# Clamp more background
uv run vda webcam3d --depth-min-percentile 0 --depth-max-percentile 85
```

### Screen Capture Scenarios

**Gaming (extreme depth jumps):**
```bash
# More aggressive clamping
uv run vda screen3d-viewer --depth-min-percentile 10 --depth-max-percentile 90
```

**Video content (natural scenes):**
```bash
# Less clamping
uv run vda screen3d-viewer --depth-min-percentile 2 --depth-max-percentile 98
```

**UI/Desktop (mostly flat):**
```bash
# Focus on what little depth exists
uv run vda screen3d-viewer --depth-min-percentile 5 --depth-max-percentile 80
```

## Quick Reference Commands

### Webcam

```bash
# Default (best for most users)
uv run vda webcam3d

# Far from camera
uv run vda webcam3d --depth-min-percentile 5

# Very close to camera
uv run vda webcam3d --depth-max-percentile 95

# Clean up messy background
uv run vda webcam3d --depth-max-percentile 80
```

### Screen Capture

```bash
# Default (best for most content)
uv run vda screen3d-viewer

# Gaming/stylized content
uv run vda screen3d-viewer --depth-min-percentile 10 --depth-max-percentile 90

# Natural video content
uv run vda screen3d-viewer --depth-min-percentile 2 --depth-max-percentile 98
```

### Static Files

```bash
# Start with no clamping, analyze the result
uv run vda view3d image.jpg depth.png

# If extremes are a problem, add clamping
uv run vda view3d image.jpg depth.png --depth-min-percentile 5 --depth-max-percentile 95
```

## Understanding the Numbers

**Percentiles work like this:**
- **0%** = Keep the CLOSEST depth value in the scene
- **5%** = Remove the closest 5% of depth values (likely noise/outliers)
- **90%** = Remove the farthest 10% of depth values (distant background)
- **95%** = Remove the farthest 5% of depth values
- **100%** = Keep the FARTHEST depth value in the scene

**For webcam portraits:**
- You are typically at 0-20% of the depth range (very close)
- Background is typically at 60-100% of the depth range (far)
- **Clamping at 0-90%** keeps you, simplifies background

**For screen content:**
- Content is typically spread across 10-90% of depth range
- Extreme outliers exist at both ends
- **Clamping at 5-95%** removes outliers, keeps content

## Troubleshooting

### "My face/body is cut off in webcam mode"

This should now be fixed with the new defaults (0-90%)!

If still cut off:
```bash
# Increase max percentile
uv run vda webcam3d --depth-max-percentile 95
```

### "Too much noisy background in webcam mode"

```bash
# Decrease max percentile more aggressively
uv run vda webcam3d --depth-max-percentile 85

# Or even more
uv run vda webcam3d --depth-max-percentile 80
```

### "Screen content looks flat"

```bash
# Reduce clamping to preserve more depth range
uv run vda screen3d-viewer --depth-min-percentile 2 --depth-max-percentile 98
```

### "Screen content has extreme stretching"

```bash
# Increase clamping
uv run vda screen3d-viewer --depth-min-percentile 10 --depth-max-percentile 85
```

## Summary

- **Webcam (0-90%)**: Optimized for close-range portraits, preserves you fully visible
- **Screen (5-95%)**: Balanced for typical screen content
- **Static (0-100%)**: Full control, you decide

Try the defaults first, then adjust based on your specific scene!
