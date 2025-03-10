import open3d as o3d
from typing import Union
import numpy as np

class PointCloudVisualizer:
    """Visualize and manipulate point clouds."""
    
    @staticmethod
    def visualize(point_cloud: Union[str, o3d.geometry.PointCloud], 
                 rotation: tuple = (0.0, -1000.0)) -> None:
        """
        Visualize point cloud with optional camera rotation.
        
        Args:
            point_cloud: Point cloud object or path to .ply file
            rotation: Tuple of rotation angles (horizontal, vertical)
        """
        # Load point cloud if path provided
        if isinstance(point_cloud, str):
            pcd = o3d.io.read_point_cloud(point_cloud)
        else:
            pcd = point_cloud
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        
        # Set camera view
        ctr = vis.get_view_control()
        ctr.rotate(*rotation)
        
        # Run visualizer
        vis.run()
        vis.destroy_window()
    
    @staticmethod
    def save_point_cloud(point_cloud: o3d.geometry.PointCloud,
                        output_path: str) -> None:
        """
        Save point cloud to file.
        
        Args:
            point_cloud: Point cloud to save
            output_path: Output file path (.ply format)
        """
        o3d.io.write_point_cloud(output_path, point_cloud)