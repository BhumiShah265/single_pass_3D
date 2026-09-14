import numpy as np
import open3d as o3d
import pyproj
from typing import Tuple, Dict, Any, List, Optional
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
    between local SfM coords and UTM/GPS coords when real telemetry exists.
    If no telemetry is provided, operates strictly in local metric coordinates.
    """
    def __init__(self):
        self.transform: Optional[GeoTransform] = None
        self.crs_name: str = "Local (Non-georeferenced)"
        self.is_georeferenced: bool = False
        
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
        c = np.trace(np.diag(D) @ S) / max(1e-8, sigma_x)
        t = mu_y - c * (R @ mu_x)
        
        return float(c), R, t

    def georeference_scene(
        self,
        local_camera_centers: np.ndarray,
        gps_coords: List[Tuple[float, float, float]]
    ) -> Optional[GeoTransform]:
        """
        Computes the transformation from local SfM coordinates to UTM coordinates
        using true GPS telemetry points.
        """
        if len(gps_coords) < 3 or len(local_camera_centers) < 3:
            logger.info("Fewer than 3 telemetry points provided. Keeping local coordinates.")
            self.is_georeferenced = False
            self.crs_name = "Local (Non-georeferenced)"
            return None

        # Determine UTM zone from first GPS coordinate
        lat0, lon0 = gps_coords[0][0], gps_coords[0][1]
        utm_zone = int((lon0 + 180) / 6) + 1
        epsg_code = 32600 + utm_zone if lat0 >= 0 else 32700 + utm_zone
        self.crs_name = f"EPSG:{epsg_code} (WGS 84 / UTM Zone {utm_zone}{'N' if lat0 >= 0 else 'S'})"

        transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
        
        utm_points = []
        for lat, lon, alt in gps_coords:
            easting, northing = transformer.transform(lon, lat)
            utm_points.append([easting, northing, alt])
        utm_arr = np.array(utm_points, dtype=np.float64)

        try:
            c, R, t = self._umeyama(local_camera_centers, utm_arr)
            self.transform = GeoTransform(scale=c, rotation=R, translation=t)
            self.is_georeferenced = True
            logger.info(f"Georeferencing computed successfully via Umeyama transform: scale={c:.4f}, CRS={self.crs_name}")
            return self.transform
        except Exception as e:
            logger.warning(f"Umeyama georeferencing notice: {e}. Keeping local coordinates.")
            self.is_georeferenced = False
            self.crs_name = "Local (Non-georeferenced)"
            return None

    def apply_transform_to_points(self, points: np.ndarray) -> np.ndarray:
        """Applies the computed transform to an Nx3 array of points."""
        if not self.is_georeferenced or self.transform is None:
            return points
        return self.transform.scale * (points @ self.transform.rotation.T) + self.transform.translation

    def transform_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Transforms an Open3D point cloud if georeferenced."""
        if not self.is_georeferenced or self.transform is None:
            return pcd
        pts = np.asarray(pcd.points)
        transformed_pts = self.apply_transform_to_points(pts)
        pcd.points = o3d.utility.Vector3dVector(transformed_pts)
        return pcd

    def transform_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Transforms an Open3D mesh if georeferenced."""
        if not self.is_georeferenced or self.transform is None:
            return mesh
        verts = np.asarray(mesh.vertices)
        transformed_verts = self.apply_transform_to_points(verts)
        mesh.vertices = o3d.utility.Vector3dVector(transformed_verts)
        return mesh
