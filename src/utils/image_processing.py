import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple

def load_image(image_path: str, 
               convert_rgb: bool = True) -> Tuple[np.ndarray, Image.Image]:
    """
    Load image and convert to both CV2 and PIL formats.
    
    Args:
        image_path: Path to image file
        convert_rgb: Whether to convert BGR to RGB for CV2 image
        
    Returns:
        Tuple of (cv2_image, pil_image)
    """
    cv_image = cv2.imread(image_path)
    if convert_rgb:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.open(image_path)
    
    return cv_image, pil_image

def resize_image(image: Union[np.ndarray, Image.Image],
                size: Tuple[int, int],
                keep_aspect: bool = True) -> Union[np.ndarray, Image.Image]:
    """
    Resize image while optionally maintaining aspect ratio.
    
    Args:
        image: Input image
        size: Target size (width, height)
        keep_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if isinstance(image, np.ndarray):
        if keep_aspect:
            h, w = image.shape[:2]
            aspect = w / h
            if size[0] / size[1] > aspect:
                new_w = int(size[1] * aspect)
                new_h = size[1]
            else:
                new_w = size[0]
                new_h = int(size[0] / aspect)
            return cv2.resize(image, (new_w, new_h))
        return cv2.resize(image, size)
    else:
        if keep_aspect:
            image.thumbnail(size, Image.LANCZOS)
            return image
        return image.resize(size, Image.LANCZOS)