from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
import numpy as np
import torch

class BaseEstimator(ABC):
    """Abstract base class for depth estimators."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    @abstractmethod
    def load_model(self, model_config: dict) -> None:
        """Load the model with the given configuration."""
        pass

    @abstractmethod
    def infer_depth(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Infer depth from an image.
        
        Args:
            image: Input image (RGB, HxWxC).
            
        Returns:
            Tuple containing:
                - Depth map (numpy array)
                - Confidence map (numpy array or None)
        """
        pass
