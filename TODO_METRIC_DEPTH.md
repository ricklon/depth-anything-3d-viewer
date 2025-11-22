# Metric Depth Mode - Follow-up TODO

## Current Status
The relative depth mode (orthographic projection) is now working correctly:
- Depth direction is correct (face forward, background back)
- RAW mode preserves disparity values for better depth variation
- Normalized mode compresses to 0-1 range

Metric depth mode (perspective projection) has been improved:
- [x] Rotation fixed (Y-up, -Z forward)
- [x] Outlier removal implemented (Statistical Outlier Removal)
- [x] Verified with Vision Agent (7/10 rating)

## Known Issues

### 1. Rotated 90 Degrees (FIXED)
The metric depth visualization appeared rotated 90 degrees.
**Fix:** Aligned coordinate system to Open3D standards (Y-up, -Z forward).

### 2. Scale/Distance Issues (PARTIALLY ADDRESSED)
Objects appear at incorrect distances in the 3D viewer.
**Status:** 
- Metric depth values are now in reasonable range (0.1m - 3.2m).
- Further calibration might be needed for exact measurements.

### 3. Video-Depth-Anything Output Format
Need to verify exactly what the model outputs:
- Is it disparity (1/depth)? **YES**
- Is it inverse depth with some scaling? **YES**
- What are the units? **Unitless disparity, converted to meters via scale factor**

## Proposed Fixes

### Fix 1: Coordinate System Alignment (DONE)
Implemented proper axis swapping:
```python
points = np.stack([
    x_3d.flatten(),
    -y_3d.flatten(),     # Flip Y so image appears right-side up (Y-up)
    -z_scaled.flatten()  # Negative Z = into the screen (standard OpenGL/Open3D camera)
], axis=1)
```

### Fix 2: True Perspective Projection (DONE)
Using standard pinhole camera model.

### Fix 3: Proper Depth Scaling (PENDING)
Still using a default scale factor. Could be improved with auto-calibration.

## Next Steps
- [ ] Optimize Statistical Outlier Removal for real-time performance (currently might be too slow)
- [ ] Verify real-time viewer works with new metric depth settings
- [ ] Implement multi-camera alignment using the now-correct metric depth
