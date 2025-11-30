import numpy as np
import cv2
from typing import Dict, Any

class DataValidator:
    """Computes statistical metrics for agent verification."""

    def compute_metrics(self, depth: np.ndarray, confidence: np.ndarray = None) -> Dict[str, Any]:
        """
        Compute quality metrics from depth and confidence maps.
        
        Args:
            depth: Depth map (numpy array).
            confidence: Confidence map (numpy array) or None.
            
        Returns:
            Dict containing metrics like 'depth_variance', 'mean_confidence', etc.
        """
        metrics = {}
        
        # Basic depth stats
        metrics['depth_min'] = float(depth.min())
        metrics['depth_max'] = float(depth.max())
        metrics['depth_mean'] = float(depth.mean())
        metrics['depth_std'] = float(depth.std())
        
        # Edge sharpness (gradient magnitude)
        # Strong gradients suggest sharp edges (good), weak gradients might mean blur
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        metrics['edge_sharpness_mean'] = float(grad_mag.mean())
        
        # Confidence stats
        if confidence is not None:
            metrics['confidence_mean'] = float(confidence.mean())
            metrics['confidence_min'] = float(confidence.min())
            # Percentage of high confidence pixels (> 0.8)
            high_conf_mask = confidence > 0.8
            metrics['high_confidence_ratio'] = float(np.sum(high_conf_mask) / confidence.size)
        else:
            metrics['confidence_mean'] = None
            
        # Sky heuristic (top 10% of image)
        # In outdoor scenes, this should be far (high value if metric is distance, low if disparity)
        # Assuming metric depth (distance) -> high values
        h, w = depth.shape
        sky_region = depth[:int(h*0.1), :]
        metrics['sky_region_mean'] = float(sky_region.mean())
        metrics['sky_region_std'] = float(sky_region.std()) # Should be low if it's a flat sky wall (bad) or consistent sky (good?)
        # Actually, if sky is properly handled as infinite, it might be max value or 0 depending on representation.
        # If it's a "sky wall" artifact, it often has some arbitrary finite depth with variance.
        
        return metrics
