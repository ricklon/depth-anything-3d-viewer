# Visualization Analysis Report

## 1. Data Understanding (Numpy/Matplotlib)
**Script:** `tools/check_visualization.py`
**Output Directory:** `test_outputs_multicam/camera_0_logi_cam_c920e`

### Statistics
- **Depth Range:** 0.3582 to 8.9392 (Inverse Depth / Disparity)
- **Mean Depth:** 4.8006
- **Median Depth:** 4.1122
- **Standard Deviation:** 2.7044
- **Outliers (>3 std dev):** 0.15% (Very low, suggesting clean data processing)

### Interpretation
- The depth values are in the range of ~0.3 to ~9.0. Since the model outputs "inverse depth" (disparity), higher values correspond to closer objects.
- The low percentage of statistical outliers suggests that the "flying pixels" might be due to local noise rather than extreme global outliers.
- **Histogram:** A histogram was generated at `test_outputs_multicam/camera_0_logi_cam_c920e/depth_histogram.png` to visualize the distribution.

## 2. Visual Interpretation (Vision Agent)
**Script:** `tools/evaluate_visualizations.py`
**Input Images:** `test_outputs/comparison_*.png` (Generated with Open3D for high resolution)

### Agent Feedback Summary

#### `comparison_metric.png` (Metric Mode)
- **Coherence:** The agent detected a coherent 3D structure.
- **Noise:** "Significant noise" was reported. The agent suggested better filtering methods.
- **Depth Profile:** The side view shows a reasonable profile, though noisy.

#### `comparison_metric_inverted.png` (Inverted Mode)
- **Geometry:** The agent explicitly noted that the geometry looks "inside out" with background elements appearing closer than foreground elements.
- **Conclusion:** This confirms that our standard "Metric Mode" (non-inverted) is the correct orientation for this model's output.

#### `comparison_relative.png` (Relative Mode)
- **Scaling:** The agent noted the depth scaling might look flattened or distorted compared to a typical scene, which is expected for "Relative" mode as it normalizes everything to a fixed range.

## 3. Recommendations
1.  **Noise Reduction:** The "significant noise" feedback suggests we should tune the Statistical Outlier Removal (SOR) parameters.
    -   Current: `neighbors=50`, `std_ratio=1.0`
    -   Proposed: Increase `neighbors` to 100 or decrease `std_ratio` to 0.5 for more aggressive filtering.
2.  **Orientation:** Stick with the non-inverted Metric Mode as the default.
3.  **Visualization:** Continue using Open3D for generating evaluation images as it provides the clarity needed for the Vision Agent to detect these details.
