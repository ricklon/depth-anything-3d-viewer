
import cv2
import numpy as np
import time
from typing import Dict, List
from da3d.projection.config import ProjectionConfig
from da3d.projection.sources import create_content_source

class ProjectionEngine:
    def __init__(self, config_path: str):
        self.config = ProjectionConfig.load(config_path)
        self.sources = {}
        self._init_sources()
        self.start_time = time.time()

    def _init_sources(self):
        for name, cfg in self.config.content_sources.items():
            source = create_content_source(name, cfg)
            if source:
                self.sources[name] = source

    def render_show(self, show_name: str) -> Dict[str, np.ndarray]:
        """
        Render one frame of the show.
        Returns a dict mapping surface_name -> image.
        """
        if show_name not in self.config.shows:
            print(f"Error: Show {show_name} not found")
            return {}

        show = self.config.shows[show_name]
        t = time.time() - self.start_time
        
        # Determine active scenes (simplified: just take first matching scene for now)
        # In a real engine, we'd handle timeline blending
        
        surface_outputs = {}
        
        for scene in show.scenes:
            # Simple logic: if multiple scenes target same surface, last one wins
            # TODO: Implement proper timeline logic
            
            # Composite layers
            final_img = None
            
            for layer in scene.content_layers:
                if layer.source not in self.sources:
                    continue
                    
                src_img = self.sources[layer.source].render(t)
                if src_img is None:
                    continue
                
                if final_img is None:
                    final_img = src_img.astype(np.float32)
                else:
                    # Resize if needed
                    if src_img.shape != final_img.shape:
                        src_img = cv2.resize(src_img, (final_img.shape[1], final_img.shape[0]))
                    
                    # Blend
                    if layer.blend == "additive":
                        final_img += src_img.astype(np.float32)
                    else: # normal
                        # Simple overwrite for now, should handle alpha
                        final_img = src_img.astype(np.float32)
            
            if final_img is not None:
                surface_outputs[scene.surface] = np.clip(final_img, 0, 255).astype(np.uint8)
                
        return surface_outputs

    def run_preview(self, show_name: str):
        """Run a simple OpenCV preview window."""
        print(f"Starting preview for show: {show_name}")
        print("Press 'q' to quit.")
        
        while True:
            outputs = self.render_show(show_name)
            
            if not outputs:
                time.sleep(0.1)
                continue
                
            # Stitch all surfaces into one debug view
            # For now, just show them in separate windows
            for surface_name, img in outputs.items():
                cv2.imshow(f"Surface: {surface_name}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(16) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
