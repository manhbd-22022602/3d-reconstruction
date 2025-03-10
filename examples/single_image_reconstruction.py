import argparse
import os
from pathlib import Path
import logging
from typing import Optional

import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.camera import CameraIntrinsics
from src.depth import DepthEstimator
from src.pointcloud import PointCloudCreator, PointCloudVisualizer
from src.utils.image_processing import load_image, resize_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Reconstruction3DPipeline:
    """Pipeline for 3D reconstruction from single image."""
    
    def __init__(self, 
                 model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 image_size: tuple = (480, 640),
                 camera_params: Optional[dict] = None):
        """
        Initialize reconstruction pipeline.
        
        Args:
            model_name: Name of depth estimation model
            image_size: Target image size (height, width)
            camera_params: Optional camera parameters
        """
        self.image_size = image_size
        
        # Initialize depth estimator
        logger.info("Initializing depth estimator...")
        self.depth_estimator = DepthEstimator(model_name=model_name)
        
        # Set up camera intrinsics
        if camera_params is None:
            self.camera = CameraIntrinsics(
                height=image_size[0],
                width=image_size[1],
                fov=55.0
            )
        else:
            # Use custom camera matrix if provided
            self.camera_matrix = np.array(camera_params['matrix'])
        
        # Initialize point cloud creator
        self.pc_creator = PointCloudCreator(
            camera_intrinsics=self.camera.intrinsics_matrix
        )
        
        # Initialize visualizer
        self.visualizer = PointCloudVisualizer()
        
    def process_image(self, 
                     image_path: str,
                     output_dir: str = "output",
                     visualize: bool = False,
                     scale_ratio: float = 25.0,
                     shift: float = 20.0) -> str:
        """
        Process single image through reconstruction pipeline.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            visualize: Whether to visualize results
            scale_ratio: Depth scaling factor
            shift: Depth shift value
            
        Returns:
            Path to saved point cloud file
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess image
        logger.info("Loading image...")
        cv_image, pil_image = load_image(image_path)
        
        # Resize images
        cv_image = resize_image(cv_image, self.image_size)
        pil_image = resize_image(pil_image, self.image_size)
        
        # Estimate depth
        logger.info("Estimating depth...")
        depth_map = self.depth_estimator.estimate_depth(pil_image)
        
        # Create point cloud
        logger.info("Creating point cloud...")
        point_cloud = self.pc_creator.create_point_cloud(
            depth_image=depth_map,
            color_image=cv_image,
            scale_ratio=scale_ratio,
            shift=shift
        )
        
        # Save point cloud
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_pointcloud.ply")
        logger.info(f"Saving point cloud to {output_path}")
        self.visualizer.save_point_cloud(point_cloud, output_path)
        
        # Visualize if requested
        if visualize:
            logger.info("Visualizing point cloud...")
            self.visualizer.visualize(point_cloud)
            
        return output_path

def main():
    parser = argparse.ArgumentParser(description='3D Reconstruction from single image')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize the point cloud')
    parser.add_argument('--image_size', type=int, nargs=2, default=[480, 640],
                      help='Target image size (height width)')
    parser.add_argument('--scale_ratio', type=float, default=25.0,
                      help='Depth scaling factor')
    parser.add_argument('--shift', type=float, default=20.0,
                      help='Depth shift value')
    parser.add_argument('--custom_camera', action='store_true',
                      help='Use custom camera parameters')
    
    args = parser.parse_args()
    
    # Example custom camera parameters (Hamlyn rec 4)
    custom_camera_params = {
        'matrix': [
            [579.05693, 0, 139.9316005706787],
            [0, 579.05693, 159.0189905166626],
            [0, 0, 1]
        ]
    } if args.custom_camera else None
    
    try:
        # Initialize pipeline
        pipeline = Reconstruction3DPipeline(
            image_size=tuple(args.image_size),
            camera_params=custom_camera_params
        )
        
        # Process image
        output_path = pipeline.process_image(
            image_path=args.image_path,
            output_dir=args.output_dir,
            visualize=args.visualize,
            scale_ratio=args.scale_ratio,
            shift=args.shift
        )
        
        logger.info(f"Processing complete. Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()