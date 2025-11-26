# 2.5D Screen Capture with OBS Virtual Camera

This guide explains how to use the new `screen3d` feature to capture your screen with depth-based parallax effects and stream it to OBS.

## What is 2.5D Parallax?

The `screen3d` command uses depth maps to create a 3D parallax effect from 2D screen captures. This creates the illusion of depth and allows you to:
- Add dynamic camera movement to static screens
- Create engaging livestream overlays
- Add depth effects to presentations
- Make screen recordings more visually interesting

## Requirements

### For Virtual Camera Output (OBS Integration)

**Windows:**
1. Install [OBS Studio](https://obsproject.com/)
2. Install [OBS VirtualCam Plugin](https://github.com/Fenrirthviti/obs-virtual-cam/releases) (if not built into OBS)

**Note:** On Windows, `pyvirtualcam` requires OBS VirtualCam to be installed.

**Linux:**
```bash
sudo apt-get install v4l2loopback-dkms v4l2loopback-utils
sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="VirtualCam" exclusive_caps=1
```

**macOS:**
Install [OBS Virtual Camera](https://obsproject.com/forum/resources/obs-virtualcam.949/)

## Quick Start

### Basic Usage (Preview Only)

```bash
# Capture primary monitor with auto-rotate
da3d screen3d --auto-rotate

# Capture with mouse control
da3d screen3d --mouse-control

# Capture specific region
da3d screen3d --region 0,0,1920,1080
```

### Stream to OBS Virtual Camera

```bash
# Enable virtual camera output
da3d screen3d --virtual-cam --auto-rotate

# With mouse control for interactive parallax
da3d screen3d --virtual-cam --mouse-control

# Adjust 3D effect strength
da3d screen3d --virtual-cam --depth-scale 0.5
```

## Controls

### Keyboard Controls

- **q** - Quit
- **s** - Save current frame (both original and 3D projection)
- **r** - Toggle recording
- **t** - Toggle auto-rotate mode
- **w/a/s/d** - Manual rotation (up/left/down/right)
- **z/x** - Decrease/increase 3D effect strength
- **r** (hold) - Reset rotation to center

### Mouse Control

When `--mouse-control` is enabled:
- Move mouse left/right to rotate scene horizontally
- Move mouse up/down to rotate scene vertically
- The effect creates a parallax view based on mouse position

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--monitor N` | Monitor number to capture (1 = primary) | 1 |
| `--region X,Y,W,H` | Capture specific region | Full monitor |
| `--encoder {vits,vitb,vitl}` | Model size (vits fastest) | vits |
| `--max-res N` | Maximum resolution | 640 |
| `--fps N` | Target frames per second | 10 |
| `--depth-scale N` | 3D effect strength (0.1-1.0) | 0.3 |
| `--virtual-cam` | Enable OBS Virtual Camera output | Off |
| `--auto-rotate` | Enable automatic rotation | Off |
| `--mouse-control` | Enable mouse-based parallax | Off |
| `--fp32` | Use FP32 precision (slower, more accurate) | FP16 |

## Using with OBS Studio

### Setup

1. Start the screen3d command with virtual camera:
   ```bash
   da3d screen3d --virtual-cam --auto-rotate
   ```

2. Open OBS Studio

3. Add a new source:
   - Click "+" in Sources panel
   - Select "Video Capture Device"
   - Choose "OBS Virtual Camera" or "VirtualCam" from the device list

4. The 3D parallax effect will now appear in OBS!

### Streaming Tips

**For Best Performance:**
- Use `--encoder vits` (fastest)
- Lower `--max-res` (e.g., 480 or 640)
- Reduce `--fps` if needed (5-10 works well)

**For Best Quality:**
- Use `--encoder vitl` (most accurate depth)
- Increase `--max-res` to 1280
- Use `--fp32` for maximum precision

**For Most Engaging Effect:**
- Enable `--auto-rotate` for constant motion
- Or use `--mouse-control` for interactive parallax
- Adjust `--depth-scale` to control how "3D" it looks (0.2-0.4 recommended)

## Example Workflows

### Live Coding Stream with Depth

Capture your IDE with parallax effect:

```bash
# Find your IDE window position (use --region)
da3d screen3d --virtual-cam --region 100,100,1600,900 --auto-rotate --depth-scale 0.25
```

### Gaming Stream with 3D Effect

Capture game window with mouse-controlled parallax:

```bash
da3d screen3d --virtual-cam --mouse-control --max-res 1280 --fps 15
```

### Presentation with Dynamic Camera

Add movement to static slides:

```bash
da3d screen3d --virtual-cam --auto-rotate --depth-scale 0.2 --fps 8
```

### Screen Recording with Depth

Record your screen with 3D effect:

```bash
# Start recording with 'r' key
da3d screen3d --auto-rotate --max-res 1280
# Press 'r' to start/stop recording
# Saved to ./screen3d_outputs/
```

## Troubleshooting

### Virtual Camera Not Working

**Windows:**
- Make sure OBS Studio is installed
- Install OBS VirtualCam plugin
- Try running OBS Studio once before starting screen3d

**Linux:**
```bash
# Check if v4l2loopback is loaded
lsmod | grep v4l2loopback

# If not, load it:
sudo modprobe v4l2loopback
```

**macOS:**
- Install OBS Virtual Camera plugin
- Grant camera permissions in System Preferences

### Performance Issues

1. **Lower resolution:** `--max-res 480`
2. **Reduce FPS:** `--fps 5`
3. **Use smaller model:** `--encoder vits`
4. **Check GPU:** Make sure CUDA is available

### Depth Map Quality

If depth looks incorrect:
- Try different `--depth-scale` values (0.1 to 0.5)
- Some content works better than others (scenes with clear foreground/background)
- 3D rendered content may have less depth variation

## Advanced: Custom Capture Regions

To capture specific windows or regions, use the `--region` option:

```bash
# Capture top-left 1920x1080 region
da3d screen3d --region 0,0,1920,1080 --virtual-cam

# Capture centered 1280x720 region on 1920x1080 screen
da3d screen3d --region 320,180,1280,720 --virtual-cam
```

Use Windows' built-in Snipping Tool or similar to find exact coordinates.

## Technical Details

### How It Works

1. **Screen Capture:** Uses `mss` to capture screen at specified FPS
2. **Depth Estimation:** Runs Video Depth Anything model to estimate depth
3. **3D Projection:** Projects pixels in 3D space using depth values
4. **Parallax Effect:** Applies rotation to create camera movement
5. **Virtual Camera:** Sends output to OBS Virtual Camera device

### Performance Characteristics

| Resolution | Model | FPS | GPU Usage |
|------------|-------|-----|-----------|
| 640x360 | vits | 10-15 | ~30% |
| 1280x720 | vits | 5-8 | ~50% |
| 1280x720 | vitl | 3-5 | ~80% |

*Tested on NVIDIA RTX 3070*

## Contributing

Found a bug or have a feature request? Please open an issue on GitHub!

## License

Same as Video-Depth-Anything (Apache-2.0 for vits, CC-BY-NC-4.0 for vitb/vitl)
