import numpy as np
import open3d as o3d
import cv2
from typing import Tuple, Optional
from ..camera import CameraIntrinsics

class PointCloudCreator:
    """Handle point cloud creation from depth and color images."""
    
    def __init__(self, camera_intrinsics: Optional[np.ndarray] = None):
        """
        Initialize point cloud creator.
        
        Args:
            camera_intrinsics: Optional 3x3 camera intrinsics matrix
        """
        self.camera_intrinsics = camera_intrinsics
        
    def depth_to_points(self, depth_image: np.ndarray, 
                       camera_intrinsics: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert depth image to 3D points.
        
        Args:
            depth_image: Input depth image
            camera_intrinsics: Optional camera intrinsics matrix
            
        Returns:
            Tuple of (x, y, z) coordinate arrays
        """
        height, width = depth_image.shape
        if camera_intrinsics is None:
            camera_intrinsics = self.camera_intrinsics
            if camera_intrinsics is None:
                camera_intrinsics = CameraIntrinsics(height, width).intrinsics_matrix
        
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        u, v = np.meshgrid(x, y)
        
        x_over_z = (u - cx) / fx
        y_over_z = (v - cy) / fy
        
        z = depth_image / np.sqrt(1.0 + x_over_z**2 + y_over_z**2)
        x = x_over_z * z
        y = y_over_z * z
        
        return x, y, z
        
    def create_point_cloud(self, depth_image: np.ndarray, 
                          color_image: np.ndarray,
                          scale_ratio: float = 25.0,
                          shift: float = 20) -> o3d.geometry.PointCloud:
        """
        Create colored point cloud from depth and color images.
        
        Args:
            depth_image: Input depth image
            color_image: Input color image
            scale_ratio: Depth scaling factor
            shift: Depth shift value
            
        Returns:
            Colored point cloud
        """
        height, width = depth_image.shape
        
        # Convert and resize color image
        if isinstance(color_image, np.ndarray):
            color_image = cv2.resize(color_image, (width, height))
        else:
            color_image = np.array(color_image)
            color_image = cv2.resize(color_image, (width, height))
        
        # Process depth
        depth_image = np.maximum(depth_image, 1e-5)
        depth_image = scale_ratio / depth_image + shift
        
        # Create point cloud
        x, y, z = self.depth_to_points(depth_image)
        point_image = np.stack((x, y, z), axis=-1)
        
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(point_image.reshape(-1, 3))
        cloud.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)
        
        return cloud