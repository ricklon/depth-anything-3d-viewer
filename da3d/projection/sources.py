
import cv2
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from pathlib import Path

class ContentSource(ABC):
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    @abstractmethod
    def render(self, t: float) -> Optional[np.ndarray]:
        """Render frame at time t. Returns RGB image (H, W, 3)."""
        pass

class ImageSource(ContentSource):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.image = None
        self._load()

    def _load(self):
        path = self.config.file
        if not Path(path).exists():
            print(f"Error: Image not found: {path}")
            return
        
        # Load image
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Failed to load image: {path}")
            return
            
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply vignette if requested
        if self.config.light_vignette:
            self._apply_vignette()

    def _apply_vignette(self):
        rows, cols = self.image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/2)
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        
        # Normalize mask to 0.5-1.0 range to keep center bright
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = 0.5 + 0.5 * mask
        
        self.image = (self.image * mask[:, :, np.newaxis]).astype(np.uint8)

    def render(self, t: float) -> Optional[np.ndarray]:
        return self.image

class DepthImageSource(ContentSource):
    """
    Simulates 2.5D parallax from RGB + Depth image.
    """
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.rgb_image = None
        self.depth_map = None
        self._load()

    def _load(self):
        # Load RGB
        if self.config.rgb and Path(self.config.rgb).exists():
            self.rgb_image = cv2.cvtColor(cv2.imread(self.config.rgb), cv2.COLOR_BGR2RGB)
        
        # Load Depth
        if self.config.depth and Path(self.config.depth).exists():
            # Support EXR or PNG depth
            if self.config.depth.endswith('.exr'):
                self.depth_map = cv2.imread(self.config.depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            else:
                self.depth_map = cv2.imread(self.config.depth, cv2.IMREAD_GRAYSCALE)
                
        if self.rgb_image is not None and self.depth_map is not None:
             # Resize depth to match RGB
             if self.depth_map.shape[:2] != self.rgb_image.shape[:2]:
                 self.depth_map = cv2.resize(self.depth_map, (self.rgb_image.shape[1], self.rgb_image.shape[0]))
                 
             # Normalize depth 0-1
             self.depth_map = (self.depth_map - self.depth_map.min()) / (self.depth_map.max() - self.depth_map.min() + 1e-6)

    def render(self, t: float) -> Optional[np.ndarray]:
        if self.rgb_image is None:
            return None
            
        # Simple parallax effect based on time
        # Orbit camera position
        amount = self.config.parallax_amount
        offset_x = int(np.sin(t * 0.5) * amount * 50)
        offset_y = int(np.cos(t * 0.5) * amount * 30)
        
        # Warp image based on depth
        h, w = self.rgb_image.shape[:2]
        
        # Create mesh grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Displace pixels
        # Closer pixels (higher depth) move more
        shift_x = offset_x * self.depth_map
        shift_y = offset_y * self.depth_map
        
        map_x = (x - shift_x).astype(np.float32)
        map_y = (y - shift_y).astype(np.float32)
        
        return cv2.remap(self.rgb_image, map_x, map_y, cv2.INTER_LINEAR)

class LobbySceneSource(ContentSource):
    """
    Renders a 3D scene (mesh or point cloud) from a specific viewpoint.
    Uses legacy Visualizer for better cross-platform support on Windows/Mac.
    """
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.vis = None
        self.width = 1920 
        self.height = 1080
        self._init_renderer()

    def _init_renderer(self):
        import open3d as o3d
        
        if not self.config.scene_asset or not Path(self.config.scene_asset).exists():
            print(f"Error: Scene asset not found: {self.config.scene_asset}")
            return

        # Initialize Visualizer
        # Note: This might open a window. For a projection system, this is often acceptable
        # as we want to output to a display anyway.
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.width, height=self.height, visible=False)
        
        # Load geometry
        ext = Path(self.config.scene_asset).suffix.lower()
        geometry = None
        
        try:
            if ext in ['.ply', '.pcd', '.xyz']:
                geometry = o3d.io.read_point_cloud(self.config.scene_asset)
            elif ext in ['.obj', '.stl', '.gltf', '.glb']:
                geometry = o3d.io.read_triangle_mesh(self.config.scene_asset)
                if not geometry.has_vertex_normals():
                    geometry.compute_vertex_normals()
            
            if geometry:
                self.vis.add_geometry(geometry)
                
                # Setup render options
                opt = self.vis.get_render_option()
                opt.background_color = np.asarray([0, 0, 0])
                opt.light_on = True
                
        except Exception as e:
            print(f"Error loading scene {self.name}: {e}")

    def render(self, t: float) -> Optional[np.ndarray]:
        if not self.vis:
            return None
            
        # Update camera
        import numpy as np
        
        radius = 2.0
        x = np.sin(t * 0.5) * radius
        z = np.cos(t * 0.5) * radius
        
        ctr = self.vis.get_view_control()
        
        # Legacy look_at is a bit different, but works
        eye = np.array([x, 1.0, z])
        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        
        ctr.set_lookat(center)
        ctr.set_front(eye - center)
        ctr.set_up(up)
        ctr.set_zoom(0.8)
        
        self.vis.poll_events()
        self.vis.update_renderer()
        
        img = self.vis.capture_screen_float_buffer(do_render=True)
        img = (np.asarray(img) * 255).astype(np.uint8)
        return img

    def __del__(self):
        if self.vis:
            self.vis.destroy_window()

def create_content_source(name: str, config) -> ContentSource:
    if config.type == "image":
        return ImageSource(name, config)
    elif config.type == "depth_image":
        return DepthImageSource(name, config)
    elif config.type == "lobby_scene":
        return LobbySceneSource(name, config)
    else:
        print(f"Warning: Unknown content type {config.type} for {name}")
        return None
