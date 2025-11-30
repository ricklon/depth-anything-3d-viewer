import torch
import numpy as np
import cv2
from .base_estimator import BaseEstimator

# Import DA3 from the installed package
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    # Fallback or placeholder if not yet installed in environment during dev
    print("Warning: depth_anything_3 not found. DA3Estimator will fail to load.")
    DepthAnything3 = None

class DA3Estimator(BaseEstimator):
    """Estimator for Depth-Anything-3."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.model = None

    def load_model(self, model_config: dict) -> None:
        """
        Load DA3 model.
        
        Args:
            model_config: Dict containing 'encoder' (e.g. 'da3-large').
        """
        if DepthAnything3 is None:
            raise ImportError("depth_anything_3 package is not installed.")

        encoder = model_config.get('encoder', 'da3-large')
        
        # Instantiate directly
        self.model = DepthAnything3(model_name=encoder)
        self.model = self.model.to(self.device).eval()

    def infer_depth(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Infer depth using DA3.
        
        Args:
            image: Input image (RGB).
            
        Returns:
            depth: Depth map.
            confidence: Confidence map.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        h, w = image.shape[:2]
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        # Add batch and sequence dimensions: (B, S, C, H, W) -> (1, 1, 3, H, W)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Run inference
            # We must pass export_feat_layers=[] because the underlying model expects an iterable
            result = self.model(image_tensor, export_feat_layers=[])
            
            if isinstance(result, tuple):
                depth = result[0]
                confidence = result[1] if len(result) > 1 else None
            elif isinstance(result, dict):
                depth = result.get('depth')
                confidence = result.get('confidence')
            else:
                depth = result
                confidence = None
                
            # Post-process depth
            if isinstance(depth, torch.Tensor):
                depth = depth.squeeze().cpu().numpy()
            
            # Post-process confidence
            if isinstance(confidence, torch.Tensor):
                confidence = confidence.squeeze().cpu().numpy()
                
            # Resize back to original if needed
            if depth.shape[:2] != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            
            if confidence is not None and confidence.shape[:2] != (h, w):
                confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth, confidence
