
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
        Returns a dict mapping projector_name -> image.
        """
        if show_name not in self.config.shows:
            print(f"Error: Show {show_name} not found")
            return {}

        show = self.config.shows[show_name]
        t = time.time() - self.start_time
        
        # Initialize projector buffers
        projector_buffers = {}
        for p_name, p_conf in self.config.projectors.items():
            w, h = p_conf.resolution
            projector_buffers[p_name] = np.zeros((h, w, 3), dtype=np.uint8)
            
        # Render scenes
        for scene in show.scenes:
            surface_name = scene.surface
            if surface_name not in self.config.surfaces:
                continue
                
            surface = self.config.surfaces[surface_name]
            proj_name = surface.projector
            
            if proj_name not in projector_buffers:
                continue
                
            # Composite content for this surface
            surface_img = None
            
            for layer in scene.content_layers:
                if layer.source not in self.sources:
                    continue
                    
                src_img = self.sources[layer.source].render(t)
                if src_img is None:
                    continue
                
                if surface_img is None:
                    surface_img = src_img.astype(np.float32)
                else:
                    if src_img.shape != surface_img.shape:
                        src_img = cv2.resize(src_img, (surface_img.shape[1], surface_img.shape[0]))
                    
                    if layer.blend == "additive":
                        surface_img += src_img.astype(np.float32)
                    else:
                        surface_img = src_img.astype(np.float32)
            
            if surface_img is None:
                continue
                
            # Warp to projector space
            if not surface.dst_quad_pixels or len(surface.dst_quad_pixels) != 4:
                continue
                
            dst_pts = np.array(surface.dst_quad_pixels, dtype=np.float32)
            h, w = surface_img.shape[:2]
            src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            
            H_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            proj_h, proj_w = projector_buffers[proj_name].shape[:2]
            warped = cv2.warpPerspective(surface_img, H_matrix, (proj_w, proj_h))
            
            # Composite onto projector buffer (simple max blending for now to handle overlaps)
            # In reality, we might want masking
            mask = (warped > 0).astype(np.uint8)
            
            # Simple alpha blend or max
            current = projector_buffers[proj_name]
            projector_buffers[proj_name] = np.maximum(current, warped.astype(np.uint8))
                
        return projector_buffers

    def run_preview(self, show_name: str):
        """Run a simple OpenCV preview window."""
        print(f"Starting preview for show: {show_name}")
        print("Press 'q' to quit.")
        
        while True:
            outputs = self.render_show(show_name)
            
            if not outputs:
                time.sleep(0.1)
                continue
                
            for proj_name, img in outputs.items():
                # Scale down for preview if huge
                view_img = img
                if img.shape[1] > 1280:
                    scale = 1280 / img.shape[1]
                    view_img = cv2.resize(img, None, fx=scale, fy=scale)
                    
                cv2.imshow(f"Projector: {proj_name}", cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(16) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
