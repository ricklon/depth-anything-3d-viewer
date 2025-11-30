import torch
import numpy as np
import cv2
from .base_estimator import BaseEstimator
from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream
from da3d.config import MODEL_CONFIGS

class VDAEstimator(BaseEstimator):
    """Estimator for Video-Depth-Anything."""

    def __init__(self, device: str = "cuda", streaming: bool = False):
        super().__init__(device)
        self.model = None
        self.streaming = streaming
        self.input_size = 518

    def load_model(self, model_config: dict) -> None:
        """
        Load VDA model.
        
        Args:
            model_config: Dict containing 'encoder', 'checkpoint_path', etc.
        """
        encoder = model_config.get('encoder', 'vitl')
        checkpoint_path = model_config.get('checkpoint_path')
        metric = model_config.get('metric', True)
        self.input_size = model_config.get('input_size', 518)

        # Get model specific configuration (features, out_channels)
        vda_config = MODEL_CONFIGS.get(encoder, {}).copy() # Copy to avoid modifying global config
        
        # Remove encoder from config if we are passing it explicitly, or just use config
        if 'encoder' in vda_config:
             del vda_config['encoder']

        if self.streaming:
            self.model = VideoDepthAnythingStream(encoder=encoder, **vda_config)
        else:
            self.model = VideoDepthAnything(encoder=encoder, metric=metric, **vda_config)

        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
        
        self.model = self.model.to(self.device).eval()

    def infer_depth(self, image: np.ndarray) -> tuple[np.ndarray, None]:
        """
        Infer depth using VDA.
        
        Args:
            image: Input image (RGB).
            
        Returns:
            depth: Depth map.
            confidence: None (VDA doesn't output confidence).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # VDA expects specific preprocessing handled by its infer_video_depth_one method
        # We assume image is already RGB
        
        with torch.no_grad():
            if self.streaming:
                depth = self.model.infer_video_depth_one(
                    image, 
                    input_size=self.input_size, 
                    device=self.device
                )
            else:
                # For non-streaming, we use infer_video_depth which takes a list of frames
                # Wrap single image in list
                depths = self.model.infer_video_depth(
                    [image], 
                    input_size=self.input_size, 
                    device=self.device
                )
                depth = depths[0]
                
        return depth, None
