import torch
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from typing import Union, Tuple
import numpy as np

class DepthEstimator:
    """Depth estimation using Depth Anything model."""
    
    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        """
        Initialize depth estimator.
        
        Args:
            model_name: Name of the pretrained model to use
        """
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.eval()
        
    def estimate_depth(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Estimate depth from single image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            Depth map as numpy array
        """
        # Handle different input types
        if isinstance(image, str):
            pil_image = Image.open(image)
            cv_image = cv2.imread(image)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            cv_image = image
            pil_image = Image.fromarray(cv_image)
        elif isinstance(image, Image.Image):
            pil_image = image
            cv_image = np.array(pil_image)
        else:
            raise ValueError("Unsupported image type")

        # Prepare image for depth estimation
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        # Get depth map
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
            # Normalize depth
            depth_map = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(cv_image.shape[0], cv_image.shape[1]),
                mode="bicubic",
                align_corners=False,
            )
            depth_map = depth_map.squeeze().numpy()
            
        return depth_map