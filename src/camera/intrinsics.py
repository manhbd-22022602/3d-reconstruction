import numpy as np

class CameraIntrinsics:
    """Camera intrinsics parameters and calculations."""
    
    def __init__(self, height, width, fov=55.0):
        self.height = height
        self.width = width
        self.fov = fov
        self.intrinsics_matrix = self._compute_intrinsics()
    
    def _compute_intrinsics(self):
        """Compute intrinsics matrix for pinhole camera model."""
        f = 0.5 * self.width / np.tan(0.5 * self.fov * np.pi / 180.0)
        cx = 0.5 * self.width
        cy = 0.5 * self.height
        return np.array([[f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]])