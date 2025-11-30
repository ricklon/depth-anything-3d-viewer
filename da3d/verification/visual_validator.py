import cv2
import numpy as np
from pathlib import Path
import matplotlib.cm as cm

class VisualValidator:
    """Generates visual artifacts for agent inspection."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_composite(self, 
                          image: np.ndarray, 
                          depth: np.ndarray, 
                          confidence: np.ndarray = None, 
                          render_3d: np.ndarray = None,
                          filename: str = "validation_composite.png") -> Path:
        """
        Create a composite image containing RGB, Depth, Confidence (if avail), and 3D Render (if avail).
        
        Args:
            image: RGB image (H, W, 3)
            depth: Depth map (H, W)
            confidence: Confidence map (H, W) or None
            render_3d: Snapshot of 3D view (H, W, 3) or None
            filename: Output filename
            
        Returns:
            Path to the saved composite image.
        """
        h, w = image.shape[:2]
        
        # Normalize depth for visualization
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 1e-8:
            depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
            
        # Colorize depth
        colormap = cm.get_cmap("inferno")
        depth_vis = (colormap(depth_norm / 255.0)[:, :, :3] * 255).astype(np.uint8)
        depth_vis = cv2.resize(depth_vis, (w, h))
        
        # Visualize confidence if available
        if confidence is not None:
            conf_norm = (confidence * 255).astype(np.uint8)
            conf_vis = cv2.applyColorMap(conf_norm, cv2.COLORMAP_JET)
            conf_vis = cv2.resize(conf_vis, (w, h))
            # Add label
            cv2.putText(conf_vis, "Confidence", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            conf_vis = np.zeros_like(image)
            cv2.putText(conf_vis, "No Confidence", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 3D Render placeholder if not provided
        if render_3d is None:
            render_3d = np.zeros_like(image)
            cv2.putText(render_3d, "No 3D Render", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            render_3d = cv2.resize(render_3d, (w, h))
            cv2.putText(render_3d, "3D Render", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Add labels to others
        img_labeled = image.copy()
        cv2.putText(img_labeled, "Original RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        depth_labeled = depth_vis.copy()
        cv2.putText(depth_labeled, "Depth Map", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Arrange in 2x2 grid
        top_row = np.hstack([img_labeled, depth_labeled])
        bottom_row = np.hstack([conf_vis, render_3d])
        composite = np.vstack([top_row, bottom_row])
        
        output_path = self.output_dir / filename
        # Convert RGB to BGR for OpenCV saving if input was RGB
        # Assuming input image is RGB (standard in this project pipeline usually)
        # But cv2.imwrite expects BGR.
        composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), composite_bgr)
        
        return output_path
