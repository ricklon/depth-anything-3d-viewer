
import cv2
import numpy as np
import yaml
from pathlib import Path
from da3d.projection.config import ProjectionConfig

class CalibrationApp:
    def __init__(self, config_path: str, projector_name: str):
        self.config_path = config_path
        self.projector_name = projector_name
        self.config = ProjectionConfig.load(config_path)
        
        if projector_name not in self.config.projectors:
            raise ValueError(f"Projector {projector_name} not found in config")
            
        self.proj_config = self.config.projectors[projector_name]
        self.width, self.height = self.proj_config.resolution
        
        # Find surfaces for this projector
        self.surfaces = [s for s in self.config.surfaces.values() if s.projector == projector_name]
        if not self.surfaces:
            print(f"Warning: No surfaces found for projector {projector_name}")
            
        # UI State
        self.selected_surface_idx = 0
        self.selected_corner_idx = -1
        self.dragging = False
        self.window_name = f"Calibration: {projector_name}"
        
        # Initialize missing quads
        for s in self.surfaces:
            if not s.dst_quad_pixels or len(s.dst_quad_pixels) != 4:
                # Default to center square
                margin = 100
                s.dst_quad_pixels = [
                    [margin, margin],
                    [self.width - margin, margin],
                    [self.width - margin, self.height - margin],
                    [margin, self.height - margin]
                ]

    def _draw(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw all surfaces
        for i, s in enumerate(self.surfaces):
            color = (100, 100, 100) if i != self.selected_surface_idx else (0, 255, 0)
            pts = np.array(s.dst_quad_pixels, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw polygon
            cv2.polylines(img, [pts], True, color, 2)
            
            # Draw corners
            for j, pt in enumerate(s.dst_quad_pixels):
                c_color = (0, 0, 255) if (i == self.selected_surface_idx and j == self.selected_corner_idx) else color
                cv2.circle(img, tuple(pt), 10, c_color, -1)
                
            # Draw label
            center = np.mean(s.dst_quad_pixels, axis=0).astype(int)
            cv2.putText(img, s.name, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
        # Draw UI info
        info = [
            f"Projector: {self.projector_name} ({self.width}x{self.height})",
            f"Selected Surface: {self.surfaces[self.selected_surface_idx].name}",
            "Controls:",
            "  TAB: Switch Surface",
            "  Mouse: Drag Corners",
            "  S: Save Config",
            "  Q: Quit"
        ]
        
        y = 30
        for line in info:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            y += 30
            
        return img

    def _mouse_callback(self, event, x, y, flags, param):
        surface = self.surfaces[self.selected_surface_idx]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check for corner click
            min_dist = 20.0
            best_idx = -1
            
            for i, pt in enumerate(surface.dst_quad_pixels):
                dist = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            if best_idx != -1:
                self.selected_corner_idx = best_idx
                self.dragging = True
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_corner_idx != -1:
                surface.dst_quad_pixels[self.selected_corner_idx] = [x, y]
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def save_config(self):
        # Load raw yaml to preserve comments/structure if possible, 
        # but for now we'll just update the specific fields and dump.
        # Ideally we'd use a round-trip loader, but standard yaml is fine for prototype.
        
        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)
            
        # Update surfaces
        if 'surfaces' not in data:
            data['surfaces'] = {}
            
        for s in self.surfaces:
            if s.name not in data['surfaces']:
                data['surfaces'][s.name] = {}
            data['surfaces'][s.name]['dst_quad_pixels'] = s.dst_quad_pixels
            
        with open(self.config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=None)
            
        print(f"Saved config to {self.config_path}")

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("Starting calibration...")
        print("Press 's' to save, 'q' to quit, 'tab' to switch surface.")
        
        while True:
            img = self._draw()
            cv2.imshow(self.window_name, img)
            
            key = cv2.waitKey(16) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_config()
            elif key == 9: # TAB
                self.selected_surface_idx = (self.selected_surface_idx + 1) % len(self.surfaces)
                self.selected_corner_idx = -1
                
        cv2.destroyAllWindows()
