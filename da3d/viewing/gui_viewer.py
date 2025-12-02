
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import threading
import time
from typing import Callable, Optional, Tuple, Any

from da3d.viewing.mesh import DepthMeshViewer

class GUI3DViewer:
    """
    Real-time 3D viewer with GUI controls using Open3D's new visualization API.
    """
    def __init__(self, window_name="Depth Anything 3D", width=1280, height=720):
        self.window_name = window_name
        self.width = width
        self.height = height
        
        # State
        self.is_running = True
        self.image = None
        self.depth = None
        self.geometry = None
        self.geometry_name = "geometry"
        self.camera_initialized = False
        
        # Settings
        self.depth_scale = 0.5
        self.subsample = 2
        self.display_mode = "mesh" # "mesh" or "pointcloud"
        self.use_metric = False
        self.sor_neighbors = 50
        self.sor_std_ratio = 1.0
        self.smooth_mesh = False
        self.focal_length_x = 470.4
        self.focal_length_y = 470.4
        self.depth_min = 0.0
        self.depth_max = 100.0
        
        # Mesh generator
        self.mesh_generator = DepthMeshViewer(
            depth_scale=self.depth_scale
        )
        
        # Threading
        self.update_thread = None
        self.data_provider = None
        
        # Initialize GUI
        self._init_gui()

    def _init_gui(self):
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(self.window_name, self.width, self.height)
        
        # 3D Scene
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([0.1, 0.1, 0.1, 1.0])
        
        # Lighting
        self.widget3d.scene.scene.enable_sun_light(True)
        self.widget3d.scene.scene.set_sun_light(
            [0, 0, 1],     # direction (from camera to scene)
            [1, 1, 1],     # color
            100000         # intensity
        )
        self.widget3d.scene.scene.enable_indirect_light(True)
        
        # Sidebar
        em = self.window.theme.font_size
        self.panel = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        
        # Controls
        self._add_control_group()
        
        # Layout
        layout = gui.Horiz(0)
        layout.add_child(self.panel)
        layout.add_child(self.widget3d)
        
        self.window.add_child(layout)
        
        # Camera defaults
        self.widget3d.setup_camera(60, self.widget3d.scene.bounding_box, [0, 0, 0])
        self.widget3d.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _add_control_group(self):
        em = self.window.theme.font_size
        
        # Depth Scale
        self.panel.add_child(gui.Label("Depth Scale"))
        slider_scale = gui.Slider(gui.Slider.DOUBLE)
        slider_scale.set_limits(0.1, 2.0)
        slider_scale.double_value = self.depth_scale
        slider_scale.set_on_value_changed(self._on_depth_scale)
        self.panel.add_child(slider_scale)
        
        # Subsample
        self.panel.add_child(gui.Label("Subsample (Quality)"))
        slider_sub = gui.Slider(gui.Slider.INT)
        slider_sub.set_limits(1, 4)
        slider_sub.int_value = self.subsample
        slider_sub.set_on_value_changed(self._on_subsample)
        self.panel.add_child(slider_sub)
        
        # Display Mode
        self.panel.add_child(gui.Label("Display Mode"))
        combo_mode = gui.Combobox()
        combo_mode.add_item("Mesh")
        combo_mode.add_item("Point Cloud")
        combo_mode.selected_text = "Mesh"
        combo_mode.set_on_selection_changed(self._on_mode_changed)
        self.panel.add_child(combo_mode)
        
        # Metric Depth
        self.chk_metric = gui.Checkbox("Metric Depth (SOR)")
        self.chk_metric.checked = self.use_metric
        self.chk_metric.set_on_checked(self._on_metric_checked)
        self.panel.add_child(self.chk_metric)
        
        # SOR Controls (Collapsable?)
        self.panel.add_child(gui.Label("SOR Neighbors"))
        slider_sor_n = gui.Slider(gui.Slider.INT)
        slider_sor_n.set_limits(10, 100)
        slider_sor_n.int_value = self.sor_neighbors
        slider_sor_n.set_on_value_changed(self._on_sor_neighbors)
        self.panel.add_child(slider_sor_n)
        
        self.panel.add_child(gui.Label("SOR Std Ratio"))
        slider_sor_s = gui.Slider(gui.Slider.DOUBLE)
        slider_sor_s.set_limits(0.1, 3.0)
        slider_sor_s.double_value = self.sor_std_ratio
        slider_sor_s.set_on_value_changed(self._on_sor_std)
        self.panel.add_child(slider_sor_s)
        
        # Reset View
        btn_reset = gui.Button("Reset View")
        btn_reset.set_on_clicked(self._on_reset_view)
        self.panel.add_child(btn_reset)
        
        # Debug
        btn_debug = gui.Button("Debug Info")
        btn_debug.set_on_clicked(self._on_debug)
        self.panel.add_child(btn_debug)

        # Quit
        btn_quit = gui.Button("Quit")
        btn_quit.set_on_clicked(self._on_quit)
        self.panel.add_child(btn_quit)
        
        # Controls Help
        self.panel.add_child(gui.Label("")) # Spacer
        self.panel.add_child(gui.Label("Controls:"))
        self.panel.add_child(gui.Label("  Left Drag: Rotate"))
        self.panel.add_child(gui.Label("  Ctrl + Left: Pan"))
        self.panel.add_child(gui.Label("  Wheel: Zoom"))

    # Callbacks
    def _on_debug(self):
        print("\n--- Debug Info ---")
        camera = self.widget3d.scene.camera
        print(f"Camera Position: {camera.get_position()}")
        print(f"Camera Forward: {camera.get_forward_vector()}")
        print(f"Camera Up: {camera.get_up_vector()}")
        print(f"Field of View: {camera.get_field_of_view()}")
        
        if self.widget3d.scene.has_geometry(self.geometry_name):
            bbox = self.widget3d.scene.get_geometry_bounding_box(self.geometry_name)
            print(f"Geometry Bounds: Min={bbox.get_min_bound()}, Max={bbox.get_max_bound()}")
            print(f"Geometry Center: {bbox.get_center()}")
        else:
            print("No geometry in scene")
            
        print("------------------\n")
    def _on_depth_scale(self, val):
        self.depth_scale = val
        self.mesh_generator.depth_scale = val
        
    def _on_subsample(self, val):
        self.subsample = int(val)
        
    def _on_mode_changed(self, text, idx):
        self.display_mode = text.lower().replace(" ", "")
        self.mesh_generator.display_mode = self.display_mode
        
    def _on_metric_checked(self, checked):
        self.use_metric = checked
        self.mesh_generator.use_metric_depth = checked
        # Ensure focal lengths are set if switching to metric
        if checked and (self.mesh_generator.focal_length_x is None or self.mesh_generator.focal_length_y is None):
             self.mesh_generator.focal_length_x = self.focal_length_x
             self.mesh_generator.focal_length_y = self.focal_length_y
        
    def _on_sor_neighbors(self, val):
        self.sor_neighbors = int(val)
        
    def _on_sor_std(self, val):
        self.sor_std_ratio = val
        
    def _on_reset_view(self):
        # Reset camera on next frame update
        self.camera_initialized = False
        
    def _on_quit(self):
        self.is_running = False
        gui.Application.instance.quit()

    def update_geometry(self):
        """Called on main thread to update geometry."""
        if self.image is None or self.depth is None:
            return
            
        try:
            # Generate geometry
            geometry = self.mesh_generator.create_mesh_from_depth(
                self.image, 
                self.depth,
                subsample=self.subsample,
                use_sor=self.use_metric, 
                sor_neighbors=self.sor_neighbors,
                sor_std_ratio=self.sor_std_ratio,
                smooth_mesh=self.smooth_mesh
            )
            
            # Ensure normals are computed
            if isinstance(geometry, o3d.geometry.TriangleMesh):
                if not geometry.has_vertex_normals():
                    geometry.compute_vertex_normals()
            elif isinstance(geometry, o3d.geometry.PointCloud):
                if not geometry.has_normals():
                    geometry.estimate_normals()
            
            # Update scene safely
            if self.widget3d.scene.has_geometry(self.geometry_name):
                self.widget3d.scene.remove_geometry(self.geometry_name)
            
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            if self.display_mode == "pointcloud":
                mat.point_size = 3.0
                
            self.widget3d.scene.add_geometry(self.geometry_name, geometry, mat)
            
            # Setup camera only once
            if not self.camera_initialized:
                bbox = geometry.get_axis_aligned_bounding_box()
                self.widget3d.setup_camera(60, bbox, [0, 0, 0])
                self.camera_initialized = True
                
            self.window.post_redraw()
            
        except Exception as e:
            print(f"Error updating geometry: {e}")

    def _background_loop(self):
        while self.is_running:
            if self.data_provider:
                try:
                    res = self.data_provider()
                    if res is None:
                        break
                    
                    # Store data
                    self.image, self.depth = res
                    
                    # Schedule update on main thread
                    gui.Application.instance.post_to_main_thread(self.window, self.update_geometry)
                    
                except Exception as e:
                    print(f"Error in data provider: {e}")
                    break
            else:
                time.sleep(0.1)

    def start(self, data_provider: Callable[[], Tuple[np.ndarray, np.ndarray]]):
        """
        Start the viewer.
        
        Args:
            data_provider: Function that returns (image, depth) tuples. 
                          Should block until new frame is available.
                          Return None to stop.
        """
        self.data_provider = data_provider
        
        # Start background thread
        self.update_thread = threading.Thread(target=self._background_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Run GUI (blocking)
        gui.Application.instance.run()
