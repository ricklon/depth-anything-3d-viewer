# Changelog

All notable changes to Depth-Anything-3D will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Point cloud visualization mode alongside mesh mode
- `--display-mode` CLI argument to choose between 'mesh' and 'pointcloud'
- Point cloud support in all 3D viewer commands (view3d, webcam3d, screen3d-viewer)
- Proportional Z-depth scaling that matches X/Y coordinate space
- Better default depth scaling (0.5 instead of 100.0) with new scaling system

### Changed
- **Breaking**: Default `depth_scale` changed from 100.0 to 0.5
- **Breaking**: Z-depth now scales proportionally to image dimensions
- Depth scale now represents fraction of image width (1.0 = half width)
- Updated all CLI help text to reflect new depth scaling
- Improved depth scale documentation and parameter descriptions

### Fixed
- Z-depth appearing too extreme/exaggerated in mesh visualizations
- Z coordinates not matching X/Y coordinate space scale
- Depth visualization looking unnatural in 3D space

## [0.1.0] - 2025-01-XX

### Added
- Initial release of Depth-Anything-3D viewer
- Real-time 3D webcam visualization (`webcam3d` command)
- Real-time 3D screen capture (`screen3d-viewer` command)
- Static 3D depth map viewing (`view3d` command)
- 2.5D parallax effects for screen capture (`screen3d` command)
- Percentile-based depth clamping for better visualization
- Performance tuning options (subsample, max-res)
- Virtual camera output support for OBS
- Mouse and keyboard controls for 3D viewing
- Automatic rotation and interactive controls
- Displacement visualization for debugging
- Support for Video-Depth-Anything models (vits, vitb, vitl)
- CLI interface with comprehensive options
- Python API for custom integrations
- Documentation and usage guides
- Apache-2.0 license

### Features

#### 3D Viewing
- Interactive 3D mesh generation from depth maps
- Real-time mesh updates for webcam/screen capture
- Laplacian mesh smoothing option
- Wireframe visualization mode
- Configurable background colors
- Camera controls (rotate, zoom, pan)

#### Depth Control
- Percentile-based depth clamping
- Separate min/max percentile controls
- Depth inversion option
- Adjustable depth scale

#### Performance
- GPU-accelerated inference
- Subsample options (1-4x)
- Resolution limiting
- FP16/FP32 precision modes
- Multiple model sizes (vits for speed, vitl for quality)

#### 2.5D Parallax
- Real-time parallax effects
- Auto-rotation mode
- Mouse-controlled parallax
- 3D lighting simulation
- Displacement visualization
- Virtual camera output

### Documentation
- Comprehensive README with examples
- Setup and installation guide
- Real-time 3D viewing guide
- Static viewing guide
- Screen capture guide
- Depth tuning guide
- Default settings explained
- CLI reference

### Dependencies
- PyTorch 2.0+
- Open3D 0.18+
- OpenCV 4.8+
- NumPy, SciPy, Matplotlib
- Optional: mss, pyvirtualcam, gradio

## Release Notes

### Upgrading from 0.0.x to 0.1.0

If you used custom `depth_scale` values:
- Old default: 100.0
- New default: 0.5
- **To maintain similar appearance:** Divide your old value by ~200
- Example: `--depth-scale 150` → `--depth-scale 0.75`

The new scaling is proportional to image dimensions, making depth scales more consistent across different resolutions.

### Known Issues
- Virtual camera output requires OBS Virtual Camera installed
- GPU required for real-time performance (CPU mode is very slow)
- First frame in real-time modes may be slow (model warmup)
- Mesh smoothing can significantly reduce FPS

### Future Plans
- [ ] Add test suite
- [ ] Python examples directory
- [ ] Video file 3D playback mode
- [ ] Stereo 3D / anaglyph export
- [ ] Depth map smoothing options
- [ ] Custom colormap support
- [ ] Recording/export functionality improvements
- [ ] Better error messages and validation

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and contribution guidelines.

## License

Apache-2.0 License - see [LICENSE](LICENSE) for details.
