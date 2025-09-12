from pathlib import Path
import numpy as np

class CameraModel:
    # hardcoded LUT paths
    _cam1_lut_path = Path(__file__).parent.parent.parent / "datasets" / "manitou" / "lut" / "camera1_lut_xyz.npy"
    _cam2_lut_path = Path(__file__).parent.parent.parent / "datasets" / "manitou" / "lut" / "camera2_lut_xyz.npy"
    _cam3_lut_path = Path(__file__).parent.parent.parent / "datasets" / "manitou" / "lut" / "camera3_lut_xyz.npy"
    _cam4_lut_path = Path(__file__).parent.parent.parent / "datasets" / "manitou" / "lut" / "camera4_lut_xyz.npy"
    LUT = {
        "camera1": np.load(_cam1_lut_path),
        "camera2": np.load(_cam2_lut_path),
        "camera3": np.load(_cam3_lut_path),
        "camera4": np.load(_cam4_lut_path)
    }
    
    @staticmethod
    def forward_project(cam_K, cam_D, points_3d):
        if points_3d.shape[0] == 0:
            return np.empty((0, 2))

        Xc = points_3d[:, 0]
        Yc = points_3d[:, 1]
        Zc = points_3d[:, 2]

        r = np.hypot(Xc, Yc)
        theta = np.arctan2(r, Zc)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        k1, k2, k3, k4 = cam_D.flatten()[:4]
        theta_d = theta + k1 * theta * theta2 + k2 * theta * theta4 + k3 * theta * theta6 + k4 * theta * theta8

        scale = np.where(r > 1e-8, theta_d / r, 0.0)
        xd = Xc * scale
        yd = Yc * scale

        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        u = fx * xd + cx
        v = fy * yd + cy

        return np.vstack((u, v)).T
    
    @staticmethod
    def inverse_project_lut_xyz(pixels, lut, constant_z=0.0):
        """
        Map 2D pixels to 3D ground plane coordinates using a lookup table.

        Args:
            pixels (np.ndarray): An array of shape (N, 2) representing N 2D pixel coordinates.
            lut (np.ndarray): Precomputed lookup table for angles of view (theta) and angles in the uv plane (phi).
            constant_z (float): The z-coordinate to use, defaults to 0.0.
        Returns:
            np.ndarray: An array of shape (N, 3) representing the ground plane coordinates (x, y, 0).
        """
        if pixels.shape[0] == 0:
            return np.empty((0, 3))
        
        u = pixels[:, 0]
        v = pixels[:, 1]

        if lut.shape[-1] != 3:
            raise ValueError("Lookup table must have shape (H, W, 3).")
        
        lut_h, lut_w, _ = lut.shape
        if u.min() >= 0 and u.max() < lut_w and v.min() >= 0 and v.max() < lut_h:
            xyz = lut[v, u, :]
            xyz[:, 2] = constant_z
            return xyz
        
        raise ValueError("Pixels are out of bounds of the provided lookup table for ground plane coordinates.")