# Metric Depth Mode - Follow-up TODO

## Current Status
The relative depth mode (orthographic projection) is now working correctly:
- Depth direction is correct (face forward, background back)
- RAW mode preserves disparity values for better depth variation
- Normalized mode compresses to 0-1 range

Metric depth mode (perspective projection) still has issues that need investigation.

## Known Issues

### 1. Rotated 90 Degrees
The metric depth visualization appears rotated 90 degrees compared to relative depth mode.

**Possible causes:**
- Coordinate axis mismatch between perspective projection and Open3D viewer
- X/Y swap in the perspective projection formula
- Different camera up-vector convention

**Investigation steps:**
- [ ] Compare the coordinate values between relative and metric modes
- [ ] Check if swapping X and Y in metric mode fixes the rotation
- [ ] Verify the Open3D viewer's default camera orientation

### 2. Scale/Distance Issues
Objects appear at incorrect distances in the 3D viewer.

**Possible causes:**
- The 1/disparity conversion may not give true metric depth
- Scale factor needs calibration
- Focal length assumption (470.4px) may not match actual webcam

**Investigation steps:**
- [ ] Test with different metric_depth_scale values
- [ ] Calibrate actual webcam focal length
- [ ] Consider if Video-Depth-Anything can output true metric depth

### 3. Video-Depth-Anything Output Format
Need to verify exactly what the model outputs:
- Is it disparity (1/depth)?
- Is it inverse depth with some scaling?
- What are the units?

**Investigation steps:**
- [ ] Check Video-Depth-Anything paper/documentation for output format
- [ ] Compare output range with known depth values
- [ ] Test with metric depth model checkpoint if available

## Proposed Fixes

### Fix 1: Coordinate System Alignment
```python
# Current metric depth code
points = np.stack([
    x_3d.flatten(),
    -y_3d.flatten(),
    -z_scaled.flatten()
], axis=1)

# Try swapping or negating axes to fix rotation
# Option A: Swap X and Y
# Option B: Different negation pattern
```

### Fix 2: True Perspective Projection
The perspective projection formula is correct mathematically:
```
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
```

But the viewer may expect a different convention. Consider:
- Using positive Z (away from camera) instead of negative
- Matching the relative depth coordinate frame exactly

### Fix 3: Proper Depth Scaling
Instead of arbitrary scaling, compute the scale to match expected scene dimensions:
```python
# Scale so that the scene fits in similar coordinate range as relative depth
# This ensures consistent viewing experience
```

## Test Plan

1. Create a simple synthetic test (checkerboard at known depth)
2. Compare metric vs relative depth output for the same input
3. Verify face appears at correct distance in meters
4. Check that perspective distortion looks natural (farther objects spread more)

## References

- Video-Depth-Anything paper: Check for depth output format details
- Open3D coordinate system: Right-handed, Y-up by default
- Camera intrinsics: fx=fy=470.4 is typical for 60-degree FOV webcam

## Priority
Medium - Relative depth mode works well for most use cases. Metric depth is only needed for:
- True 3D reconstruction
- Combining with other sensors
- Measuring real-world distances
