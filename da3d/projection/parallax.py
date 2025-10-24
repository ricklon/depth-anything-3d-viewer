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
3D Projection utilities for creating 2.5D parallax effects from depth maps.
"""

import numpy as np
import cv2
from scipy.ndimage import map_coordinates


class DepthProjector:
    """Projects 2D images into 3D space using depth maps for parallax effects."""

    def __init__(self, width, height, focal_length=500, enable_lighting=True):
        """
        Initialize the depth projector.

        Args:
            width: Image width
            height: Image height
            focal_length: Virtual camera focal length (affects perspective strength)
            enable_lighting: Apply 3D lighting effects based on depth normals
        """
        self.width = width
        self.height = height
        self.focal_length = focal_length
        self.enable_lighting = enable_lighting

        # Create pixel coordinate grids
        self.x_coords, self.y_coords = np.meshgrid(
            np.arange(width),
            np.arange(height)
        )

        # Center coordinates
        self.cx = width / 2
        self.cy = height / 2

        # Store last displacement for visualization
        self.last_displacement_x = None
        self.last_displacement_y = None

    def project_with_parallax(self, image, depth, rotation_x=0, rotation_y=0, scale_z=1.0, lighting_intensity=0.4, zoom=1.0, invert_depth=False):
        """
        Project image with depth into 3D and apply parallax effect.

        Args:
            image: RGB image (H, W, 3)
            depth: Depth map (H, W), normalized 0-1
            rotation_x: Rotation around X axis in degrees (tilt up/down)
            rotation_y: Rotation around Y axis in degrees (tilt left/right)
            scale_z: Depth scale factor (how much 3D effect, 0.1-2.0)
            lighting_intensity: Strength of simulated 3D lighting (0.0-1.0)
            zoom: Zoom level (1.0 = normal, >1.0 = zoomed in, <1.0 = zoomed out)
            invert_depth: If True, reverse depth (near becomes far, creates "pop out" effect)

        Returns:
            Projected image with parallax effect
        """
        h, w = depth.shape[:2]

        # Normalize depth to reasonable range with enhanced contrast
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Invert depth if requested (makes close objects "pop out")
        if invert_depth:
            depth_norm = 1.0 - depth_norm

        # Apply gamma correction to enhance depth perception
        depth_norm = np.power(depth_norm, 0.7)  # Makes depth differences more pronounced

        # Scale depth for 3D effect (MUCH larger range now)
        z_values = depth_norm * scale_z * self.focal_length * 2.0  # 2x multiplier for stronger effect

        # Create 3D point cloud from 2D + depth
        x_3d = (self.x_coords - self.cx) * z_values / self.focal_length
        y_3d = (self.y_coords - self.cy) * z_values / self.focal_length
        z_3d = z_values

        # Apply rotation (convert degrees to radians)
        rot_x = np.deg2rad(rotation_x)
        rot_y = np.deg2rad(rotation_y)

        # Rotation matrices
        # Rotate around X axis (tilt up/down)
        if abs(rot_x) > 0.001:
            y_rot = y_3d * np.cos(rot_x) - z_3d * np.sin(rot_x)
            z_rot_x = y_3d * np.sin(rot_x) + z_3d * np.cos(rot_x)
            y_3d = y_rot
            z_3d = z_rot_x

        # Rotate around Y axis (tilt left/right)
        if abs(rot_y) > 0.001:
            x_rot = x_3d * np.cos(rot_y) + z_3d * np.sin(rot_y)
            z_rot_y = -x_3d * np.sin(rot_y) + z_3d * np.cos(rot_y)
            x_3d = x_rot
            z_3d = z_rot_y

        # Project back to 2D (perspective projection)
        z_3d = np.maximum(z_3d, 0.1)  # Avoid division by zero

        # Apply zoom by scaling the projected coordinates relative to center
        x_proj = (x_3d * self.focal_length / z_3d) * zoom + self.cx
        y_proj = (y_3d * self.focal_length / z_3d) * zoom + self.cy

        # Store displacement for visualization (how much each pixel moved)
        self.last_displacement_x = x_proj - self.x_coords
        self.last_displacement_y = y_proj - self.y_coords

        # Debug: Store displacement stats for first call
        if not hasattr(self, '_debug_printed'):
            disp_mag = np.sqrt(self.last_displacement_x**2 + self.last_displacement_y**2)
            print(f"\n=== Displacement Debug ===")
            print(f"Displacement X range: [{self.last_displacement_x.min():.2f}, {self.last_displacement_x.max():.2f}]")
            print(f"Displacement Y range: [{self.last_displacement_y.min():.2f}, {self.last_displacement_y.max():.2f}]")
            print(f"Displacement magnitude range: [{disp_mag.min():.2f}, {disp_mag.max():.2f}]")
            print(f"Displacement magnitude mean: {disp_mag.mean():.2f}")
            print(f"Non-zero displacements: {(disp_mag > 1.0).sum()} / {disp_mag.size}")
            print(f"Rotation: x={rotation_x:.1f}, y={rotation_y:.1f}")
            print(f"Scale_z: {scale_z:.2f}")
            self._debug_printed = True

        # Remap the image using the projected coordinates
        output = self._remap_image(image, x_proj, y_proj)

        # Apply simulated 3D lighting if enabled
        if self.enable_lighting and lighting_intensity > 0:
            output = self._apply_lighting(output, depth_norm, rotation_x, rotation_y, lighting_intensity)

        return output

    def _remap_image(self, image, x_coords, y_coords):
        """
        Remap image using coordinate transformation with interpolation.

        Args:
            image: Source image (H, W, 3)
            x_coords: X coordinates for each pixel (H, W)
            y_coords: Y coordinates for each pixel (H, W)

        Returns:
            Remapped image
        """
        h, w = image.shape[:2]

        # Clip coordinates to valid range
        x_coords = np.clip(x_coords, 0, w - 1)
        y_coords = np.clip(y_coords, 0, h - 1)

        # Use OpenCV remap for better performance
        map_x = x_coords.astype(np.float32)
        map_y = y_coords.astype(np.float32)

        output = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return output

    def get_displacement_visualization(self, grayscale=False, show_arrows=True):
        """
        Create a visualization of pixel displacement.

        Args:
            grayscale: If True, return grayscale displacement map (faster)
            show_arrows: If True, overlay direction arrows (slower)

        Returns:
            RGB or grayscale image showing displacement
        """
        if self.last_displacement_x is None or self.last_displacement_y is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Calculate displacement magnitude
        displacement_mag = np.sqrt(self.last_displacement_x**2 + self.last_displacement_y**2)

        # Normalize displacement for visualization
        max_disp = np.percentile(displacement_mag, 99)  # Use 99th percentile to avoid outliers
        if max_disp > 0:
            disp_norm = np.clip(displacement_mag / max_disp, 0, 1)
        else:
            disp_norm = np.zeros_like(displacement_mag)

        if grayscale:
            # Simple grayscale: white = max displacement, black = no displacement
            displacement_vis = (disp_norm * 255).astype(np.uint8)
            displacement_vis = cv2.cvtColor(displacement_vis, cv2.COLOR_GRAY2BGR)
        else:
            # Create colormap visualization (hot colors = more displacement)
            import matplotlib.cm as cm
            colormap = cm.get_cmap("jet")
            displacement_vis = (colormap(disp_norm)[:, :, :3] * 255).astype(np.uint8)

        # Optionally overlay displacement vectors
        if show_arrows:
            # Draw arrows on a grid to show direction
            step = 40  # Grid spacing
            for y in range(0, self.height, step):
                for x in range(0, self.width, step):
                    if y < self.height and x < self.width:
                        dx = int(self.last_displacement_x[y, x] * 2)  # Scale for visibility
                        dy = int(self.last_displacement_y[y, x] * 2)
                        if abs(dx) > 2 or abs(dy) > 2:  # Only draw significant displacements
                            cv2.arrowedLine(
                                displacement_vis,
                                (x, y),
                                (x + dx, y + dy),
                                (255, 255, 0) if grayscale else (255, 255, 255),
                                1,
                                tipLength=0.3
                            )

        return displacement_vis

    def _apply_lighting(self, image, depth, rotation_x, rotation_y, intensity):
        """
        Apply simulated 3D lighting based on depth surface normals.

        Args:
            image: RGB image to light
            depth: Normalized depth map (0-1)
            rotation_x: Camera rotation X (for light direction)
            rotation_y: Camera rotation Y (for light direction)
            intensity: Lighting effect strength (0-1)

        Returns:
            Lit image
        """
        # Calculate surface normals from depth gradients
        gy, gx = np.gradient(depth)

        # Compute normal vectors (dx, dy, dz)
        # Larger gradients = steeper surfaces
        normal_x = -gx * 10.0  # Scale up for visibility
        normal_y = -gy * 10.0
        normal_z = np.ones_like(depth)

        # Normalize the normal vectors
        norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x /= norm + 1e-8
        normal_y /= norm + 1e-8
        normal_z /= norm + 1e-8

        # Light direction based on camera rotation (opposite of view)
        light_x = np.sin(np.deg2rad(rotation_y))
        light_y = -np.sin(np.deg2rad(rotation_x))
        light_z = 0.7  # Pointing mostly at the screen

        # Normalize light direction
        light_norm = np.sqrt(light_x**2 + light_y**2 + light_z**2)
        light_x /= light_norm
        light_y /= light_norm
        light_z /= light_norm

        # Compute dot product (diffuse lighting)
        diffuse = (normal_x * light_x + normal_y * light_y + normal_z * light_z)
        diffuse = np.clip(diffuse, -1, 1)

        # Add ambient + diffuse lighting
        ambient = 0.6
        lighting = ambient + (1.0 - ambient) * (diffuse * 0.5 + 0.5)

        # Apply lighting with intensity control
        lighting = 1.0 + (lighting - 1.0) * intensity

        # Expand to RGB and apply
        lighting_rgb = np.stack([lighting, lighting, lighting], axis=-1)
        lit_image = (image * lighting_rgb).clip(0, 255).astype(np.uint8)

        return lit_image

    def create_anaglyph(self, image, depth, eye_separation=0.05):
        """
        Create red-cyan anaglyph 3D image.

        Args:
            image: RGB image (H, W, 3)
            depth: Depth map (H, W)
            eye_separation: Distance between virtual eyes (affects 3D strength)

        Returns:
            Anaglyph 3D image
        """
        # Create left and right eye views
        left_view = self.project_with_parallax(image, depth, rotation_y=-eye_separation)
        right_view = self.project_with_parallax(image, depth, rotation_y=eye_separation)

        # Combine: red channel from left, cyan (green+blue) from right
        anaglyph = np.zeros_like(image)
        anaglyph[:, :, 0] = left_view[:, :, 0]  # Red from left
        anaglyph[:, :, 1] = right_view[:, :, 1]  # Green from right
        anaglyph[:, :, 2] = right_view[:, :, 2]  # Blue from right

        return anaglyph


class InteractiveParallaxController:
    """Controls interactive parallax effect with mouse/keyboard."""

    def __init__(self, auto_rotate=False, auto_speed=0.5):
        """
        Initialize the controller.

        Args:
            auto_rotate: Enable automatic rotation
            auto_speed: Speed of automatic rotation
        """
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.scale_z = 0.5  # Depth scale (increased default)
        self.lighting_intensity = 0.5  # Lighting effect strength
        self.zoom = 1.0  # Zoom level (1.0 = normal, >1.0 = zoomed in, <1.0 = zoomed out)
        self.invert_depth = False  # Invert parallax (near becomes far)
        self.auto_rotate = auto_rotate
        self.auto_speed = auto_speed
        self.auto_angle = 0.0

        # Mouse control
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        self.max_rotation = 25.0  # Maximum rotation in degrees (increased from 15)

    def update_from_mouse(self, x, y, width, height):
        """Update rotation based on mouse position."""
        self.mouse_x = x / width
        self.mouse_y = y / height

        # Map mouse position to rotation (-max to +max)
        self.rotation_y = (self.mouse_x - 0.5) * 2 * self.max_rotation
        self.rotation_x = (self.mouse_y - 0.5) * 2 * self.max_rotation

    def update_auto_rotate(self, dt=0.016):
        """Update automatic rotation (call each frame)."""
        if self.auto_rotate:
            self.auto_angle += self.auto_speed * dt * 60  # ~60 FPS normalized
            self.rotation_y = np.sin(self.auto_angle * 0.05) * self.max_rotation
            self.rotation_x = np.cos(self.auto_angle * 0.03) * self.max_rotation * 0.5

    def handle_key(self, key):
        """
        Handle keyboard input for manual control.

        Args:
            key: OpenCV key code

        Returns:
            True if key was handled, False otherwise
        """
        if key == ord('w'):
            self.rotation_x -= 2.0  # Increased step
        elif key == ord('s'):
            self.rotation_x += 2.0
        elif key == ord('a'):
            self.rotation_y -= 2.0
        elif key == ord('d'):
            self.rotation_y += 2.0
        elif key == ord('z'):
            self.scale_z = max(0.1, self.scale_z - 0.1)  # Larger steps
        elif key == ord('x'):
            self.scale_z = min(2.0, self.scale_z + 0.1)  # Increased max to 2.0
        elif key == ord('c'):
            self.lighting_intensity = max(0.0, self.lighting_intensity - 0.1)
        elif key == ord('v'):
            self.lighting_intensity = min(1.0, self.lighting_intensity + 0.1)
        elif key == ord('[') or key == ord('-'):
            # Zoom out
            self.zoom = max(0.5, self.zoom - 0.1)
        elif key == ord(']') or key == ord('='):
            # Zoom in
            self.zoom = min(3.0, self.zoom + 0.1)
        elif key == ord('i'):
            # Toggle invert depth
            self.invert_depth = not self.invert_depth
            return True
        elif key == ord('r'):
            # Reset
            self.rotation_x = 0.0
            self.rotation_y = 0.0
            self.scale_z = 0.5
            self.lighting_intensity = 0.5
            self.zoom = 1.0
            self.invert_depth = False
        elif key == ord('t'):
            # Toggle auto-rotate
            self.auto_rotate = not self.auto_rotate
            return True
        else:
            return False

        # Clamp rotations
        self.rotation_x = np.clip(self.rotation_x, -self.max_rotation, self.max_rotation)
        self.rotation_y = np.clip(self.rotation_y, -self.max_rotation, self.max_rotation)

        return True
