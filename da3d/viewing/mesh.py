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
        display_mode: str = 'mesh'
    ):
        """
        Initialize the 3D mesh viewer.

        Args:
            depth_scale: Scale factor for Z-displacement (0.1-2.0, where 1.0 = depth spans half the image width)
            max_depth_threshold: Filter out pixels with depth > this percentile (removes background)
            depth_min_percentile: Clamp depth values below this percentile (0-100)
            depth_max_percentile: Clamp depth values above this percentile (0-100)
            display_mode: Display mode - 'mesh' for triangle mesh, 'pointcloud' for point cloud
        """
        self.depth_scale = depth_scale
        self.max_depth_threshold = max_depth_threshold
        self.depth_min_percentile = depth_min_percentile
        self.depth_max_percentile = depth_max_percentile
        self.display_mode = display_mode

    def create_mesh_from_depth(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        subsample: int = 1,
        invert_depth: bool = False,
        smooth_mesh: bool = True
    ):
        """
        Create a 3D geometry (mesh or point cloud) from an image and its depth map.

        Args:
            image: RGB image (H, W, 3), values 0-255
            depth: Depth map (H, W), normalized 0-1 (0=near, 1=far)
            subsample: Downsample factor (1=full res, 2=half res, etc.) for performance
            invert_depth: If True, invert depth values (1=near, 0=far)
            smooth_mesh: Apply Laplacian smoothing to reduce noise (mesh mode only)

        Returns:
            Open3D TriangleMesh or PointCloud object depending on display_mode
        """
        # Subsample for performance
        if subsample > 1:
            h, w = depth.shape
            new_h, new_w = h // subsample, w // subsample
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        h, w = depth.shape

        # Invert depth if requested
        if invert_depth:
            depth = 1.0 - depth

        # Normalize depth to reasonable Z range with percentile clamping
        # This reduces extreme depth values for better visualization
        depth_min = np.percentile(depth, self.depth_min_percentile)
        depth_max = np.percentile(depth, self.depth_max_percentile)

        # Clamp depth to the percentile range
        depth_clamped = np.clip(depth, depth_min, depth_max)

        # Normalize to 0-1 range
        if depth_max - depth_min > 1e-8:
            depth_normalized = (depth_clamped - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_clamped)

        # Optional: Filter out far background (often noisy)
        depth_threshold = np.percentile(depth, self.max_depth_threshold * 100)
        mask = depth < depth_threshold

        # Create coordinate grids
        x = np.arange(w)
        y = np.arange(h)
        x_grid, y_grid = np.meshgrid(x, y)

        # Center the mesh
        x_centered = x_grid - w / 2
        y_centered = y_grid - h / 2

        # Create 3D points: (X, Y, Z) where Z comes from depth
        # Scale Z proportionally to image dimensions so it matches X/Y coordinate space
        # Use width as reference since X coordinates span [-w/2, w/2]
        z_scale_factor = w * 0.5  # Z will span half the width when depth_scale=1.0
        z = depth_normalized * self.depth_scale * z_scale_factor

        # Stack into (H*W, 3) array
        points = np.stack([
            x_centered.flatten(),
            -y_centered.flatten(),  # Flip Y so image appears right-side up
            z.flatten()
        ], axis=1)

        # Get colors from image (normalize to 0-1)
        # Always use full RGB for mesh texture
        if image.dtype == np.uint8:
            colors = image.astype(np.float32) / 255.0
        else:
            colors = image
        colors = colors.reshape(-1, 3)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Return point cloud if in pointcloud mode
        if self.display_mode == 'pointcloud':
            # Estimate normals for better lighting
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30)
            )
            return pcd

        # Create mesh using grid triangulation
        mesh = self._create_grid_mesh(points, colors, w, h, mask.flatten())

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

        # Create triangles by connecting grid neighbors
        triangles = []
        for y in range(height - 1):
            for x in range(width - 1):
                # Get indices of 2x2 quad
                idx_tl = y * width + x           # top-left
                idx_tr = y * width + (x + 1)     # top-right
                idx_bl = (y + 1) * width + x     # bottom-left
                idx_br = (y + 1) * width + (x + 1)  # bottom-right

                # Only create triangles if all 4 vertices are valid
                if mask[idx_tl] and mask[idx_tr] and mask[idx_bl] and mask[idx_br]:
                    # Create two triangles per quad
                    # Triangle 1: top-left, bottom-left, top-right
                    triangles.append([idx_tl, idx_bl, idx_tr])
                    # Triangle 2: top-right, bottom-left, bottom-right
                    triangles.append([idx_tr, idx_bl, idx_br])

        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

        return mesh

    def view_mesh(
        self,
        geometry,
        window_name: str = "3D Depth Viewer",
        width: int = 1280,
        height: int = 720,
        show_wireframe: bool = False,
        background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    ):
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
        view_control.set_zoom(0.8)

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
    ):
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
):
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
        smooth_mesh: bool = False,  # Disable smoothing for speed
        max_depth_threshold: float = 0.95,
        depth_min_percentile: float = 0.0,
        depth_max_percentile: float = 90.0,
        background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        display_mode: str = 'mesh'
    ):
        """
        Initialize real-time 3D viewer.

        Args:
            depth_scale: Z-displacement scale (0.1-2.0, where 1.0 = depth spans half the image width)
            subsample: Downsample factor (higher = faster)
            smooth_mesh: Apply smoothing (slower but cleaner, mesh mode only)
            max_depth_threshold: Filter background pixels
            depth_min_percentile: Clamp near depth (reduces extremes)
            depth_max_percentile: Clamp far depth (reduces extremes)
            background_color: RGB background (0-1 range)
            display_mode: Display mode - 'mesh' or 'pointcloud'
        """
        self.depth_scale = depth_scale
        self.subsample = subsample
        self.smooth_mesh = smooth_mesh
        self.max_depth_threshold = max_depth_threshold
        self.depth_min_percentile = depth_min_percentile
        self.depth_max_percentile = depth_max_percentile
        self.background_color = background_color
        self.display_mode = display_mode

        self.vis = None
        self.geometry = None  # Can be mesh or point cloud
        self.mesh_viewer = DepthMeshViewer(
            depth_scale,
            max_depth_threshold,
            depth_min_percentile,
            depth_max_percentile,
            display_mode
        )
        self.frame_count = 0

    def initialize(self, width: int, height: int):
        """
        Initialize the Open3D visualizer window.

        Args:
            width: Window width
            height: Window height
        """
        self.vis = o3d.visualization.Visualizer()
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

        # Set initial view
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.6)

        print(f"\n=== Real-Time 3D Viewer ===")
        print("Controls:")
        print("  Mouse drag: Rotate")
        print("  Mouse wheel: Zoom")
        print("  Shift + drag: Pan")
        print("  Q: Quit")
        print("=" * 28 + "\n")

    def update_mesh(self, image: np.ndarray, depth: np.ndarray, invert_depth: bool = False):
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
            smooth_mesh=self.smooth_mesh
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

    def close(self):
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
