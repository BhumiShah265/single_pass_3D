import numpy as np
import open3d as o3d
import pyproj
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from app.config import logger

@dataclass
class GeoTransform:
    """Stores the 7-parameter similarity transformation parameters."""
    scale: float
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector

class Georeferencer:
    """
    Computes and applies Umeyama 7-parameter similarity transformation 
    between local SfM coords and UTM/GPS coords.
    """
    def __init__(self):
        self.transform: GeoTransform = None
        self.crs_source = None
        self.crs_target = None
        
    def _umeyama(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Estimates the Umeyama transform between two sets of points.
        X (source): Nx3 array of local SfM points.
        Y (target): Nx3 array of UTM points.
        Returns scale, rotation (3x3), translation (3,).
        """
        assert X.shape == Y.shape
        assert X.shape[1] == 3
        
        n = X.shape[0]
        mu_x = X.mean(axis=0)
        mu_y = Y.mean(axis=0)
        
        X_c = X - mu_x
        Y_c = Y - mu_y
        
        sigma_x = np.mean(np.sum(X_c ** 2, axis=1))
        
        cov = (Y_c.T @ X_c) / n
        
        U, D, V_t = np.linalg.svd(cov)
        S = np.eye(3)
        if np.linalg.det(cov) < 0:
            S[2, 2] = -1
            
        R = U @ S @ V_t
        c = np.trace(np.diag(D) @ S) / sigma_x
        t = mu_y - c * (R @ mu_x)
        
        return c, R, t

    def compute_transform(self, local_pts: np.ndarray, global_pts: np.ndarray) -> GeoTransform:
        """
        Computes the transformation from local SfM to Global coords.
        """
        logger.info(f"Computing Umeyama transform with {len(local_pts)} points")
        c, R, t = self._umeyama(local_pts, global_pts)
        self.transform = GeoTransform(scale=c, rotation=R, translation=t)
        return self.transform

    def apply_transform_to_points(self, points: np.ndarray) -> np.ndarray:
        """
        Applies the computed transform to a Nx3 array of points.
        """
        if self.transform is None:
            raise ValueError("Transform has not been computed yet.")
        
        transformed = self.transform.scale * (points @ self.transform.rotation.T) + self.transform.translation
        return transformed

    def transform_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Transforms an Open3D point cloud."""
        if self.transform is None:
            raise ValueError("Transform has not been computed yet.")
        
        points = np.asarray(pcd.points)
        transformed_points = self.apply_transform_to_points(points)
        pcd.points = o3d.utility.Vector3dVector(transformed_points)
        return pcd

    def transform_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Transforms an Open3D mesh."""
        if self.transform is None:
            raise ValueError("Transform has not been computed yet.")
        
        vertices = np.asarray(mesh.vertices)
        transformed_vertices = self.apply_transform_to_points(vertices)
        mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
        return mesh
