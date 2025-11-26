# 3D Mesh Viewer Guide

The `view3d` command creates an interactive 3D mesh from depth maps, allowing you to view your depth estimations as true 3D geometry from any angle.

## Overview

Unlike the 2.5D parallax effect (`screen3d`), which displaces pixels in 2D space, the 3D viewer converts your depth map into actual 3D geometry. Each pixel becomes a vertex positioned at its corresponding depth value, creating a textured mesh you can orbit around.

## Quick Start

```bash
# View a depth map in 3D
da3d view3d image.jpg depth.png

# Adjust depth scale for more dramatic effect
da3d view3d image.jpg depth.png --depth-scale 150

# Full resolution (may be slow)
da3d view3d image.jpg depth.png --subsample 1

# Invert depth (for reversed perspective)
da3d view3d image.jpg depth.png --invert-depth
```

## How It Works

1. **Mesh Generation**: The viewer creates a triangle mesh by:
   - Converting each pixel to a 3D vertex (X, Y, Z)
   - X and Y come from pixel position
   - Z comes from the depth value (scaled by `--depth-scale`)
   - Connecting adjacent pixels into triangles

2. **3D Rendering**: Uses Open3D to render the mesh with:
   - Texture mapping from the RGB image
   - Normal-based lighting
   - Interactive camera controls

## Command Line Options

### Required Arguments

- `image`: Path to RGB image
- `depth`: Path to depth map (PNG, JPG, or .npy file)

### Optional Arguments

**Depth Configuration:**
```bash
--depth-scale FLOAT        # Z-displacement scale (default: 100.0)
                          # Higher values = more exaggerated depth
                          # Try 50-200 depending on your scene

--depth-threshold FLOAT    # Filter background pixels (default: 0.95)
                          # Removes pixels with depth > this percentile
                          # Range: 0.0 to 1.0

--invert-depth            # Reverse depth direction
                          # Makes near pixels far and vice versa
```

**Performance:**
```bash
--subsample INT           # Downsample factor (default: 2)
                          # 1 = full resolution (slow, detailed)
                          # 2 = half resolution (good balance)
                          # 4 = quarter resolution (fast)
```

**Rendering:**
```bash
--wireframe               # Start in wireframe mode
                          # Shows mesh structure

--no-smooth               # Disable Laplacian smoothing
                          # Faster but noisier mesh

--background R,G,B        # Background color (default: 0.1,0.1,0.1)
                          # Example: --background 1,1,1 for white
```

## Interactive Controls

Once the 3D viewer window opens, you can interact with the mesh:

| Control | Action |
|---------|--------|
| **Mouse drag** | Rotate camera around mesh |
| **Mouse wheel** | Zoom in/out |
| **Shift + mouse drag** | Pan camera (move left/right/up/down) |
| **R key** | Reset camera view to default |
| **W key** | Toggle wireframe mode |
| **Q or ESC** | Close viewer window |

## Examples

### Basic viewing
```bash
# Process a video frame first
da3d video input.mp4 -o outputs/

# View the first frame in 3D
da3d view3d outputs/input_src.mp4_frame0001.jpg outputs/input_depth.mp4_frame0001.png
```

### Dramatic depth effect
```bash
# Use high depth scale for exaggerated 3D
da3d view3d portrait.jpg portrait_depth.png --depth-scale 200
```

### High quality (slower)
```bash
# Full resolution with no subsampling
da3d view3d scene.jpg scene_depth.png --subsample 1
```

### Fast preview
```bash
# Quarter resolution for quick viewing
da3d view3d large_image.jpg large_depth.png --subsample 4
```

### Inverted depth
```bash
# Reverse the depth direction
da3d view3d image.jpg depth.png --invert-depth
```

### Custom background
```bash
# White background for clean presentation
da3d view3d image.jpg depth.png --background 1,1,1
```

## Tips & Best Practices

### Performance Optimization

1. **Use subsampling**: Start with `--subsample 2` (default) for good performance
2. **Filter background**: Adjust `--depth-threshold` to remove noisy far regions
3. **Disable smoothing**: Use `--no-smooth` if speed is critical

### Visual Quality

1. **Adjust depth scale**: Too low = flat, too high = distorted
   - Try 50-100 for close-up scenes
   - Try 100-200 for wide scenes

2. **Smooth noisy depth**: Enable smoothing (default) for cleaner meshes
3. **Use full resolution**: `--subsample 1` for final high-quality renders

### Workflow Integration

```bash
# 1. Generate depth from video
da3d video input.mp4 --save-npz -o outputs/

# 2. Extract a frame pair
# (manually save a frame from the video, or use video frame extraction)

# 3. View in 3D
da3d view3d outputs/frame.jpg outputs/frame_depth.npy
```

## Technical Details

### Coordinate System

- **X axis**: Horizontal (left to right), centered at image center
- **Y axis**: Vertical (top to bottom, flipped), centered at image center
- **Z axis**: Depth (into the scene), scaled by `--depth-scale`

### Mesh Structure

- **Vertices**: One per pixel (after subsampling)
- **Triangles**: Two per pixel quad (grid topology)
- **Colors**: RGB values from input image
- **Normals**: Computed automatically for proper lighting

### Depth Map Format

The viewer accepts:
- **PNG/JPG**: Grayscale images (0-255 maps to 0-1 depth)
- **NPY files**: Numpy arrays with normalized depth values (0-1)

Depth convention:
- **0.0 = near** (front of scene)
- **1.0 = far** (back of scene)

Use `--invert-depth` to reverse this.

## Comparison: 3D Viewer vs Parallax (screen3d)

| Feature | 3D Viewer (`view3d`) | Parallax (`screen3d`) |
|---------|----------------------|----------------------|
| **Type** | True 3D geometry | 2.5D displacement |
| **Camera** | Free orbit | Limited tilt angles |
| **Use case** | Static inspection | Real-time effects |
| **Performance** | Offline rendering | Real-time capable |
| **Output** | Interactive viewer | Video/virtual cam |

## Troubleshooting

### "Mesh has no vertices"
- Check that your depth map has valid values (not all zeros)
- Try lowering `--depth-threshold`

### "open3d not installed"
```bash
# Install Open3D (included in metric dependencies)
uv sync --extra metric
```

### Mesh looks too flat
- Increase `--depth-scale` (try 150 or 200)

### Mesh looks distorted
- Decrease `--depth-scale` (try 50 or 75)
- Check if depth map is correctly normalized

### Slow performance
- Increase `--subsample` (try 4 for fast preview)
- Use `--no-smooth` to skip mesh smoothing

### Noisy mesh
- Enable smoothing (default, remove `--no-smooth` if set)
- Increase `--subsample` to reduce vertex count

## Advanced Usage

### Batch Processing with Python

```python
from utils.viewer_3d import view_depth_3d

# View multiple frames programmatically
for i in range(10):
    view_depth_3d(
        f'outputs/frame_{i:04d}.jpg',
        f'outputs/depth_{i:04d}.png',
        depth_scale=120,
        subsample=2
    )
```

### Custom Mesh Creation

```python
from utils.viewer_3d import DepthMeshViewer
import numpy as np
import cv2

# Load data
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
depth = cv2.imread('depth.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

# Create mesh
viewer = DepthMeshViewer(depth_scale=100)
mesh = viewer.create_mesh_from_depth(
    image, depth,
    subsample=2,
    invert_depth=False,
    smooth_mesh=True
)

# Export mesh for use in other tools
import open3d as o3d
o3d.io.write_triangle_mesh("output.ply", mesh)
o3d.io.write_triangle_mesh("output.obj", mesh)

# Or view it
viewer.view_mesh(mesh)
```

### Export for 3D Software

The viewer uses Open3D meshes, which can be exported to standard 3D formats:

```python
import open3d as o3d
from utils.viewer_3d import DepthMeshViewer

viewer = DepthMeshViewer(depth_scale=100)
mesh = viewer.create_mesh_from_depth(image, depth, subsample=1)

# Export to various formats
o3d.io.write_triangle_mesh("mesh.ply", mesh)      # PLY (recommended)
o3d.io.write_triangle_mesh("mesh.obj", mesh)      # OBJ (Blender, Maya)
o3d.io.write_triangle_mesh("mesh.stl", mesh)      # STL (3D printing)
```

Then import the mesh into:
- **Blender**: File → Import → PLY/OBJ
- **MeshLab**: File → Import Mesh
- **Unity/Unreal**: Standard mesh import

## See Also

- [SCREEN3D_GUIDE.md](SCREEN3D_GUIDE.md) - 2.5D parallax effects
- [DISPLACEMENT_GUIDE.md](DISPLACEMENT_GUIDE.md) - Understanding displacement
- [RESOLUTION_GUIDE.md](RESOLUTION_GUIDE.md) - Resolution settings
