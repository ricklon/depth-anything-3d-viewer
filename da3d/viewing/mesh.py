#!/usr/bin/env python3
# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
3D Mesh Viewer for depth maps using Open3D.
Converts 2D images with depth into true 3D geometry viewable from any angle.
"""

import numpy as np
import cv2
import open3d as o3d
from typing import Optional, Tuple


class DepthMeshViewer:
    """
    Creates an interactive 3D mesh from a depth map and RGB image.
    Allows viewing the scene from any angle with mouse/keyboard controls.
    """

    def __init__(
        self,
        depth_scale: float = 0.5,
        max_depth_threshold: float = 0.95,
        depth_min_percentile: float = 0.0,
        depth_max_percentile: float = 100.0,
        display_mode: str = 'mesh',
        use_raw_depth: bool = False,
        use_metric_depth: bool = False,
        focal_length_x: Optional[float] = None,
        focal_length_y: Optional[float] = None,
        principal_point_x: Optional[float] = None,
        principal_point_y: Optional[float] = None,
        metric_depth_scale: float = 1.0
    ) -> None:
        """
        Initialize the 3D mesh viewer.

        Args:
            depth_scale: Scale factor for Z-displacement (0.1-2.0, where 1.0 = depth spans half the image width)
                        Ignored when use_metric_depth=True
            max_depth_threshold: Filter out pixels with depth > this percentile (removes background)
            depth_min_percentile: Clamp depth values below this percentile (0-100)
            depth_max_percentile: Clamp depth values above this percentile (0-100)
            display_mode: Display mode - 'mesh' for triangle mesh, 'pointcloud' for point cloud
            use_raw_depth: If True, use raw depth values without normalizing to 0-1 (preserves more detail)
            use_metric_depth: If True, depth values are in meters and camera intrinsics are used
                             for accurate 3D reconstruction (perspective projection)
            focal_length_x: Camera focal length in X direction (pixels). Required if use_metric_depth=True
            focal_length_y: Camera focal length in Y direction (pixels). Required if use_metric_depth=True
            principal_point_x: Camera principal point X coordinate (pixels). If None, uses image center
            principal_point_y: Camera principal point Y coordinate (pixels). If None, uses image center
            metric_depth_scale: Scale factor to convert raw metric depth values to meters (default: 1.0)
                               Adjust this if the depth range is too large/small for visualization
        """
        self.depth_scale = depth_scale
        self.max_depth_threshold = max_depth_threshold
        self.depth_min_percentile = depth_min_percentile
        self.depth_max_percentile = depth_max_percentile
        self.display_mode = display_mode
        self.use_raw_depth = use_raw_depth
        self.use_metric_depth = use_metric_depth
        self.focal_length_x = focal_length_x
        self.focal_length_y = focal_length_y
        self.principal_point_x = principal_point_x
        self.principal_point_y = principal_point_y
        self.metric_depth_scale = metric_depth_scale

        if use_metric_depth and (focal_length_x is None or focal_length_y is None):
            raise ValueError("focal_length_x and focal_length_y are required when use_metric_depth=True")

    def create_mesh_from_depth(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        subsample: int = 1,
        invert_depth: bool = False,
        smooth_mesh: bool = True,
        use_sor: bool = True,
        sor_neighbors: int = 50,
        sor_std_ratio: float = 1.0
    ) -> o3d.geometry.Geometry:
        """
        Create a 3D geometry (mesh or point cloud) from an image and its depth map.

        Args:
            image: RGB image (H, W, 3), values 0-255
            depth: Depth map (H, W) - for Video-Depth-Anything, higher values = closer (inverse depth)
            subsample: Downsample factor (1=full res, 2=half res, etc.) for performance
            invert_depth: If True, invert depth values
            smooth_mesh: Apply Laplacian smoothing to reduce noise (mesh mode only)
            use_sor: Apply Statistical Outlier Removal (metric depth only)
            sor_neighbors: Number of neighbors for SOR (default: 50)
            sor_std_ratio: Standard deviation ratio for SOR (default: 1.0, lower = more aggressive)

        Returns:
            Open3D geometry (TriangleMesh or PointCloud)
        """
        # Subsample for performance
        if subsample > 1:
            h, w = depth.shape
            new_h, new_w = h // subsample, w // subsample
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Apply bilateral filter to smooth depth quantization (layering) while preserving edges
        # Only apply if not subsampled heavily, as it can be slow
        if smooth_mesh:
             # Convert to float32 for filtering if not already
             depth_float = depth.astype(np.float32)
             # Sigma values: spatial=5, range=0.1 (adjust range based on depth scale)
             depth = cv2.bilateralFilter(depth_float, d=5, sigmaColor=0.1, sigmaSpace=5)

        h, w = depth.shape

        # Create coordinate grids
        x = np.arange(w)
        y = np.arange(h)
        x_grid, y_grid = np.meshgrid(x, y)

        if self.use_metric_depth:
            # Metric depth mode: Use perspective projection with camera intrinsics
            # Video-Depth-Anything outputs values where higher = closer (like disparity)
            # For perspective projection we need actual distance where higher = farther

            # Convert to actual depth (reciprocal)
            # Add small epsilon to avoid division by zero
            # Convert to actual depth
            # CHALLENGING ASSUMPTION: The "layering" suggests we are treating disparity as linear depth.
            # If the model outputs inverse depth (disparity), we MUST invert it to get linear depth (Z).
            # Default behavior: Assume output is inverse depth (disparity) -> Invert to get Z.
            if not invert_depth:
                # Default: Convert inverse depth to real depth (Z = 1/d)
                actual_depth = 1.0 / (depth + 1e-6)
            else:
                # User requested to use values as-is (linear depth)
                actual_depth = depth.copy()

            # Apply scale factor
            z = actual_depth * self.metric_depth_scale

            # Debug output
            print(f"[DEBUG] Image size: {w}x{h}, Focal length: {self.focal_length_x:.1f}px")
            print(f"[DEBUG] Input depth (inverse) - min: {depth.min():.3f}, max: {depth.max():.3f}")
            print(f"[DEBUG] Actual depth - min: {z.min():.3f}m, max: {z.max():.3f}m, mean: {z.mean():.3f}m")

            # Calculate field of view for reference
            fov_degrees = 2 * np.arctan(w / (2 * self.focal_length_x)) * 180 / np.pi
            print(f"[DEBUG] Horizontal FOV: {fov_degrees:.1f} degrees")

            # Use provided principal point or default to image center
            cx = self.principal_point_x if self.principal_point_x is not None else w / 2.0
            cy = self.principal_point_y if self.principal_point_y is not None else h / 2.0

            # Perspective projection: X = (x - cx) * Z / fx, Y = (y - cy) * Z / fy
            x_3d = (x_grid - cx) * z / self.focal_length_x
            y_3d = (y_grid - cy) * z / self.focal_length_y

            # Stack into (H*W, 3) array
            # Open3D coordinate system: Right-handed, Y-up, -Z forward
            # Our data: +Y is down (image coordinates), +Z is forward (depth)
            # To align:
            # - Invert Y to make +Y up
            # - Invert Z to make -Z forward (standard camera view)
            points = np.stack([
                x_3d.flatten(),
                -y_3d.flatten(),     # Flip Y so image appears right-side up (Y-up)
                -z.flatten()         # Negative Z = into the screen (standard OpenGL/Open3D camera)
            ], axis=1)

            # ---- Outlier removal (metric depth) ----
            # 1. Percentile clipping to remove extreme depth values
            z_vals = points[:, 2]
            # Use configured percentiles (default 0-100 means no clipping)
            lower = np.percentile(z_vals, self.depth_min_percentile)
            upper = np.percentile(z_vals, self.depth_max_percentile)
            mask_valid = (z_vals >= lower) & (z_vals <= upper)
            
            # 2. Statistical Outlier Removal (SOR)
            # This is more expensive but effectively removes "flying pixels"
            # We perform this on a temporary point cloud
            if use_sor and np.sum(mask_valid) > 0:
                try:
                    pcd_temp = o3d.geometry.PointCloud()
                    pcd_temp.points = o3d.utility.Vector3dVector(points[mask_valid])
                    
                    # nb_neighbors: number of neighbors to analyze for each point
                    # std_ratio: threshold based on standard deviation of mean distance
                    # Lower std_ratio = more aggressive filtering
                    cl, ind = pcd_temp.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio)
                    
                    # 'ind' contains indices of inliers within the 'mask_valid' subset
                    # We need to map this back to the original full mask
                    
                    # Create a full-size boolean mask initialized to False
                    final_mask = np.zeros(len(points), dtype=bool)
                    
                    # Get indices of the valid points in the original array
                    valid_indices = np.where(mask_valid)[0]
                    
                    # Mark the statistically valid points as True
                    final_mask[valid_indices[ind]] = True
                    
                    mask_valid = final_mask
                except Exception as e:
                    print(f"[ERROR] SOR failed: {e}")
            
            # points = points[mask_valid]  <-- REMOVED: Keep full points for grid mesh generation
            
            # No thresholding - use all pixels
            mask = mask_valid

        else:
            # Relative depth mode: Orthographic projection
            # Video-Depth-Anything outputs DISPARITY (inverse depth): higher value = closer to camera

            # Apply percentile clamping to reduce outliers
            depth_min = np.percentile(depth, self.depth_min_percentile)
            depth_max = np.percentile(depth, self.depth_max_percentile)
            depth_clamped = np.clip(depth, depth_min, depth_max)

            if self.use_raw_depth:
                # Use disparity values directly without 0-1 normalization
                # This preserves the actual depth variation in the scene
                # High disparity = close, low disparity = far
                # We need: close = small Z, far = large Z
                # So invert: Z = max - disparity
                depth_processed = depth_max - depth_clamped
                # Now: face (high disparity ~9) -> small Z (~0)
                #      background (low disparity ~0) -> large Z (~9)

                if invert_depth:
                    depth_processed = depth_clamped - depth_min
            else:
                # Traditional approach: use disparity directly
                # Normalize to 0-1 range
                if depth_max - depth_min > 1e-8:
                    depth_processed = (depth_clamped - depth_min) / (depth_max - depth_min)
                else:
                    depth_processed = np.zeros_like(depth_clamped)
                # Now: face (high disparity) = 1.0, background (low disparity) = 0.0
                # We need to invert: face should have small Z, background large Z
                depth_processed = 1.0 - depth_processed

                if invert_depth:
                    depth_processed = 1.0 - depth_processed

            # Center the mesh coordinates
            # We want to preserve the aspect ratio, so we don't scale x/y differently
            x_centered = x_grid - w / 2
            y_centered = y_grid - h / 2
            
            max_dim = max(w, h)

            # Scale Z to match X/Y coordinate space
            z_scale_factor = max_dim * 0.5
            z = depth_processed * self.depth_scale * z_scale_factor

            # Stack into (H*W, 3) array
            # In Open3D: +Z points toward camera, so closer = larger Z
            # depth_processed: small = close, large = far
            # We need to invert: close should be large Z
            points = np.stack([
                x_centered.flatten(),
                -y_centered.flatten(),  # Flip Y so image appears right-side up
                -z.flatten()            # Negate: close (small depth) -> large Z (toward camera)
            ], axis=1)

            # No thresholding - use all pixels
            mask = np.ones(h * w, dtype=bool)

        # Get colors from image (normalize to 0-1)
        if image.dtype == np.uint8:
            colors = image.astype(np.float32) / 255.0
        else:
            colors = image
        colors = colors.reshape(-1, 3)
        
        # REMOVED: Do not filter colors here, as we need full arrays for grid mesh
        # if 'mask_valid' in locals():
        #     colors = colors[mask_valid]


        # Return point cloud if in pointcloud mode
        if self.display_mode == 'pointcloud':
            # Create Open3D point cloud with FILTERED points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[mask])
            pcd.colors = o3d.utility.Vector3dVector(colors[mask])
            
            # Estimate normals for better lighting
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=20)
            )
            return pcd

        # Create mesh using grid triangulation
        # We pass FULL points and colors, and the mask to skip invalid triangles
        mesh = self._create_grid_mesh(points, colors, w, h, mask)

        # Remove vertices that are not part of any triangle (i.e. masked out points)
        mesh.remove_unreferenced_vertices()

        if smooth_mesh and mesh.has_vertices():
            # Apply Laplacian smoothing to reduce depth map noise
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=2)

        # Compute normals for proper lighting
        mesh.compute_vertex_normals()

        return mesh

    def _create_grid_mesh(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        width: int,
        height: int,
        mask: np.ndarray
    ) -> o3d.geometry.TriangleMesh:
        """
        Create a triangle mesh by connecting adjacent pixels in a grid pattern.
        Optimized using vectorized numpy operations.

        Args:
            points: (N, 3) array of 3D vertices
            colors: (N, 3) array of RGB colors
            width: Original grid width
            height: Original grid height
            mask: (N,) boolean array indicating valid vertices

        Returns:
            TriangleMesh with grid connectivity
        """
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        # Vectorized grid generation
        # Create a grid of indices (H-1, W-1)
        # We stop at H-1 and W-1 because we look at the pixel to the right/bottom
        x_idxs = np.arange(width - 1)
        y_idxs = np.arange(height - 1)
        x_grid, y_grid = np.meshgrid(x_idxs, y_idxs)
        
        # Calculate indices for the top-left corner of each quad
        # shape: (H-1, W-1)
        tl_indices = y_grid * width + x_grid
        
        # Calculate indices for other corners relative to top-left
        tr_indices = tl_indices + 1
        bl_indices = tl_indices + width
        br_indices = tl_indices + width + 1
        
        # Flatten indices to 1D arrays
        tl = tl_indices.flatten()
        tr = tr_indices.flatten()
        bl = bl_indices.flatten()
        br = br_indices.flatten()
        
        # Check validity of all 4 corners for each quad
        # mask is (N,) boolean array
        valid_quads = mask[tl] & mask[tr] & mask[bl] & mask[br]
        
        # Filter indices to keep only valid quads
        tl = tl[valid_quads]
        tr = tr[valid_quads]
        bl = bl[valid_quads]
        br = br[valid_quads]
        
        if len(tl) > 0:
            # Create two triangles per valid quad
            # Triangle 1: top-left, bottom-left, top-right
            t1 = np.stack([tl, bl, tr], axis=1)
            
            # Triangle 2: top-right, bottom-left, bottom-right
            t2 = np.stack([tr, bl, br], axis=1)
            
            # Combine all triangles
            triangles = np.concatenate([t1, t2], axis=0)
            
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
        else:
            # No valid triangles
            mesh.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))

        return mesh

    def view_mesh(
        self,
        geometry: o3d.geometry.Geometry,
        window_name: str = "3D Depth Viewer",
        width: int = 1280,
        height: int = 720,
        show_wireframe: bool = False,
        background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    ) -> None:
        """
        Display the 3D geometry (mesh or point cloud) in an interactive viewer window.

        Controls:
            - Mouse drag: Rotate camera
            - Mouse wheel: Zoom in/out
            - Shift + mouse drag: Pan camera
            - R: Reset view
            - W: Toggle wireframe mode (mesh only)
            - Q/ESC: Close window

        Args:
            geometry: Open3D TriangleMesh or PointCloud to display
            window_name: Window title
            width: Window width
            height: Window height
            show_wireframe: Start in wireframe mode (mesh only)
            background_color: RGB background color (0-1 range)
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=width, height=height)

        vis.add_geometry(geometry)

        # Configure rendering options
        render_option = vis.get_render_option()
        render_option.background_color = np.array(background_color)

        # Check if geometry is a mesh or point cloud
        is_mesh = isinstance(geometry, o3d.geometry.TriangleMesh)
        if is_mesh:
            render_option.mesh_show_back_face = True
            if show_wireframe:
                render_option.mesh_show_wireframe = True
        else:
            # Point cloud rendering options
            render_option.point_size = 3.0

        # Set up camera view
        view_control = vis.get_view_control()
        # Standard front view: looking down -Z axis, Y is up
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 1, 0])
        view_control.set_zoom(0.7)

        mode_str = "Mesh" if is_mesh else "Point Cloud"
        print(f"\n=== 3D Depth Viewer ({mode_str}) ===")
        print("Mouse drag: Rotate camera")
        print("Mouse wheel: Zoom in/out")
        print("Shift + mouse drag: Pan camera")
        print("R: Reset view")
        if is_mesh:
            print("W: Toggle wireframe mode")
        print("Q or ESC: Close window")
        print("=" * 32 + "\n")

        vis.run()
        vis.destroy_window()

    def process_and_view(
        self,
        image_path: str,
        depth_path: str,
        subsample: int = 2,
        invert_depth: bool = False,
        **view_kwargs
    ) -> None:
        """
        Load image and depth map from files, create mesh, and display.

        Args:
            image_path: Path to RGB image
            depth_path: Path to depth map (grayscale image or .npy file)
            subsample: Downsample factor for performance
            invert_depth: Whether to invert depth values
            **view_kwargs: Additional arguments for view_mesh()
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load depth
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            if depth is None:
                raise ValueError(f"Could not load depth map: {depth_path}")
            depth = depth.astype(np.float32) / 255.0

        # Ensure depth is 2D
        if depth.ndim > 2:
            depth = depth[:, :, 0]

        # Resize if needed
        if image.shape[:2] != depth.shape:
            depth = cv2.resize(depth, (image.shape[1], image.shape[0]))

        print(f"Creating 3D mesh from {image.shape[1]}x{image.shape[0]} image...")
        mesh = self.create_mesh_from_depth(
            image, depth, subsample=subsample, invert_depth=invert_depth
        )

        print(f"Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        self.view_mesh(mesh, **view_kwargs)


def view_depth_3d(
    image_path: str,
    depth_path: str,
    depth_scale: float = 0.5,
    subsample: int = 2,
    invert_depth: bool = False,
    smooth: bool = True,
    display_mode: str = 'mesh'
) -> None:
    """
    Convenience function to quickly view a depth map in 3D.

    Args:
        image_path: Path to RGB image
        depth_path: Path to depth map
        depth_scale: Z-displacement scale (0.1-2.0, where 1.0 = depth spans half the image width)
        subsample: Downsample factor (1=full res, 2=half, 4=quarter)
        invert_depth: Invert depth values
        smooth: Apply mesh smoothing (mesh mode only)
        display_mode: Display mode - 'mesh' or 'pointcloud'
    """
    viewer = DepthMeshViewer(depth_scale=depth_scale, display_mode=display_mode)
    viewer.process_and_view(
        image_path,
        depth_path,
        subsample=subsample,
        invert_depth=invert_depth,
        smooth_mesh=smooth
    )


class RealTime3DViewer:
    """
    Real-time 3D mesh viewer that updates dynamically from video/webcam/screen.
    Uses non-blocking Open3D visualization for continuous mesh updates.
    """
    def __init__(
        self,
        depth_scale: float = 0.5,
        subsample: int = 2,
        smooth_mesh: bool = False,
        max_depth_threshold: float = 0.95,
        depth_min_percentile: float = 0.0,
        depth_max_percentile: float = 95.0,
        background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        display_mode: str = 'mesh',
        use_raw_depth: bool = False,
        use_metric_depth: bool = False,
        focal_length_x: Optional[float] = None,
        focal_length_y: Optional[float] = None,
        principal_point_x: Optional[float] = None,
        principal_point_y: Optional[float] = None,
        metric_depth_scale: float = 1.0,
        use_sor: bool = True,
        sor_neighbors: int = 50,
        sor_std_ratio: float = 1.0
    ) -> None:
        """
        Initialize real-time 3D viewer.

        Args:
            depth_scale: Scale factor for depth displacement
            subsample: Downsample factor for mesh geometry
            smooth_mesh: Enable Laplacian smoothing
            max_depth_threshold: Threshold for filtering far points
            depth_min_percentile: Clamp near depth percentile
            depth_max_percentile: Clamp far depth percentile
            background_color: Background color (R, G, B)
            display_mode: 'mesh' or 'pointcloud'
            use_raw_depth: Use raw depth values without normalization
            use_metric_depth: Use metric depth projection
            focal_length_x: Camera focal length X (pixels)
            focal_length_y: Camera focal length Y (pixels)
            principal_point_x: Principal point X (pixels)
            principal_point_y: Principal point Y (pixels)
            metric_depth_scale: Scale factor for metric depth values
            use_sor: Apply Statistical Outlier Removal (metric depth only)
            sor_neighbors: Number of neighbors for SOR
            sor_std_ratio: Standard deviation ratio for SOR
        """
        self.depth_scale = depth_scale
        self.subsample = subsample
        self.smooth_mesh = smooth_mesh
        self.max_depth_threshold = max_depth_threshold
        self.depth_min_percentile = depth_min_percentile
        self.depth_max_percentile = depth_max_percentile
        self.background_color = background_color
        self.display_mode = display_mode
        self.use_sor = use_sor
        self.sor_neighbors = sor_neighbors
        self.sor_std_ratio = sor_std_ratio

        self.vis = None
        self.geometry = None  # Can be mesh or point cloud
        self.mesh_viewer = DepthMeshViewer(
            depth_scale=depth_scale,
            max_depth_threshold=max_depth_threshold,
            depth_min_percentile=depth_min_percentile,
            depth_max_percentile=depth_max_percentile,
            display_mode=display_mode,
            use_raw_depth=use_raw_depth,
            use_metric_depth=use_metric_depth,
            focal_length_x=focal_length_x,
            focal_length_y=focal_length_y,
            principal_point_x=principal_point_x,
            principal_point_y=principal_point_y,
            metric_depth_scale=metric_depth_scale
        )
        self.frame_count = 0

    def initialize(self, width: int, height: int) -> None:
        """
        Initialize the Open3D visualizer window.

        Args:
            width: Window width
            height: Window height
        """
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name='Real-Time 3D Depth Viewer',
            width=width,
            height=height
        )

        # Configure rendering
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(self.background_color)
        if self.display_mode == 'mesh':
            render_option.mesh_show_back_face = True
        else:
            render_option.point_size = 3.0

        # Set initial view to a standard front-facing position
        view_control = self.vis.get_view_control()
        
        # Standard front view: looking down -Z axis, Y is up
        # This ensures the scene is centered and upright
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 1, 0])
        view_control.set_zoom(0.7)  # 0.7 frames the scene well without being too far

        print("\n=== Real-Time 3D Viewer ===")
        print("Controls:")
        print("  Mouse drag: Rotate")
        print("  Mouse wheel: Zoom")
        print("  Shift + drag: Pan")
        print("  [ / ]: Decrease/Increase SOR Neighbors")
        print("  - / =: Decrease/Increase SOR Std Ratio")
        print("  R: Reset View")
        print("  Q: Quit")
        print("=" * 28 + "\n")

        # Register callbacks
        self.vis.register_key_callback(ord('R'), self._reset_view)
        self.vis.register_key_callback(ord('['), self._decrease_sor_neighbors)
        self.vis.register_key_callback(ord(']'), self._increase_sor_neighbors)
        self.vis.register_key_callback(ord('-'), self._decrease_sor_std)
        self.vis.register_key_callback(ord('='), self._increase_sor_std)

    def _reset_view(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
        view_control = vis.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 1, 0])
        view_control.set_zoom(0.7)
        print("[INFO] View reset")
        return False

    def _decrease_sor_neighbors(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
        self.sor_neighbors = max(10, self.sor_neighbors - 10)
        print(f"[INFO] SOR Neighbors: {self.sor_neighbors}")
        return False

    def _increase_sor_neighbors(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
        self.sor_neighbors += 10
        print(f"[INFO] SOR Neighbors: {self.sor_neighbors}")
        return False

    def _decrease_sor_std(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
        self.sor_std_ratio = max(0.1, self.sor_std_ratio - 0.1)
        print(f"[INFO] SOR Std Ratio: {self.sor_std_ratio:.1f}")
        return False

    def _increase_sor_std(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
        self.sor_std_ratio += 0.1
        print(f"[INFO] SOR Std Ratio: {self.sor_std_ratio:.1f}")
        return False

    def update_mesh(self, image: np.ndarray, depth: np.ndarray, invert_depth: bool = False) -> None:
        """
        Update the 3D geometry (mesh or point cloud) with new frame data.

        Args:
            image: RGB image (H, W, 3)
            depth: Depth map (H, W)
            invert_depth: Invert depth values
        """
        # Create new geometry (mesh or point cloud)
        new_geometry = self.mesh_viewer.create_mesh_from_depth(
            image,
            depth,
            subsample=self.subsample,
            invert_depth=invert_depth,
            smooth_mesh=self.smooth_mesh,
            use_sor=self.use_sor,
            sor_neighbors=self.sor_neighbors,
            sor_std_ratio=self.sor_std_ratio
        )

        # Update visualization
        if self.geometry is None:
            # First frame: add geometry
            self.geometry = new_geometry
            self.vis.add_geometry(self.geometry)
        else:
            # Subsequent frames: update geometry
            if isinstance(new_geometry, o3d.geometry.TriangleMesh):
                # Triangle mesh uses vertices, not points
                self.geometry.vertices = new_geometry.vertices
                self.geometry.vertex_colors = new_geometry.vertex_colors
                self.geometry.triangles = new_geometry.triangles
                self.geometry.compute_vertex_normals()
            else:
                # Point cloud uses points
                self.geometry.points = new_geometry.points
                self.geometry.colors = new_geometry.colors
                # Copy normals if available
                if new_geometry.has_normals():
                    self.geometry.normals = new_geometry.normals

            self.vis.update_geometry(self.geometry)

        # Render frame
        self.vis.poll_events()
        self.vis.update_renderer()

        self.frame_count += 1

    def should_close(self) -> bool:
        """Check if the viewer window should close."""
        if self.vis is None:
            return True
        # Check if window is still open
        return not self.vis.poll_events()

    def close(self) -> None:
        """Close the viewer window."""
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None
        print(f"\nProcessed {self.frame_count} frames")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python viewer_3d.py <image_path> <depth_path> [depth_scale] [subsample]")
        print("\nExample:")
        print("  python viewer_3d.py output.jpg output_depth.png 150 2")
        sys.exit(1)

    image_path = sys.argv[1]
    depth_path = sys.argv[2]
    depth_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    subsample = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    view_depth_3d(image_path, depth_path, depth_scale=depth_scale, subsample=subsample)
