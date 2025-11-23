# Project Next Steps & Roadmap

## 1. Metric Depth Refinement
The metric depth visualization is working but can be improved based on Vision Agent feedback (currently rated 6/10).

- [ ] **Tune Outlier Removal (SOR):**
    - Experiment with `sor_neighbors` and `sor_std_ratio` to find the best balance between noise removal and detail preservation.
    - [x] Expose these parameters to the CLI (currently only in Python API).
- [ ] **Address Sparsity:**
    - The Vision Agent noted the point cloud was "sparse".
    - [x] Increased point size in visualization script.
    - Investigate mesh smoothing or surface reconstruction techniques (e.g., Poisson reconstruction) for metric depth.
    - Consider "hole filling" for invalid pixels in the depth map.
- [x] **Verify Scale Accuracy:**
    - [x] Use a known object of known size to verify the real-world metric scale (meters) is accurate.
    - **Result 1:** Measured 3.3cm for a 2.54cm object (Ratio: ~0.77).
    - **Result 2:** With 0.77 scale, measured 4.86cm. This indicates inconsistent depth estimation or measurement error.
    - **Action:** Revert to 1.0 scale for verification tool to establish a stable baseline. The depth estimation might be highly sensitive to the specific frame or object distance.

## 2. Performance Optimization
- [x] **Optimize Mesh Generation:**
    - The current numpy-based mesh generation can be slow at high resolutions (1280p+).
    - [x] Replaced Python loops with vectorized NumPy operations in `_create_grid_mesh`.
    - Investigate moving vertex generation to GPU or using optimized Open3D functions.
- [x] **Resolve Dependencies:**
    - [x] Suppressed `xformers` / `triton` warnings on Windows to clean up output.
    - Ensure `pyvirtualcam` and `mss` work seamlessly across all platforms.

## 3. User Experience & Features
- [x] **CLI Improvements:**
    - [x] Add CLI arguments for SOR parameters (`--sor-neighbors`, `--sor-ratio`).
    - [x] Add a preset flag for "High Quality" vs "Performance" modes (wrapping the resolution/subsample settings).
- [x] **Interactive Controls:**
    - [x] Add keyboard shortcuts to adjust SOR parameters in real-time.
    - [x] Add a "Reset Camera" button or key (now that we have a good default view).
- [ ] **Recording:**
    - Improve recording capability to capture the 3D view directly (currently captures the window, which might have UI overlays).

## 4. Testing & Validation
- [ ] **Automated Visual Testing:**
    - Create a workflow to automatically run the Vision Agent on new commits to track quality regression.
- [ ] **Multi-Camera Support:**
    - Ensure robust handling of multiple cameras (as seen with Camera 0 vs Camera 1 issues).
