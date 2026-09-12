import numpy as np
import open3d as o3d
from app.config import logger

class MeasurementTool:
    """
    Computes 3D distances, bounding boxes, surface areas, volumes, 
    and height profiles on reconstructed meshes/clouds.
    """
    
    @staticmethod
    def compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute Euclidean distance between two 3D points."""
        return float(np.linalg.norm(p1 - p2))
        
    @staticmethod
    def compute_bounding_box(points: np.ndarray) -> dict:
        """Compute axis-aligned bounding box for a set of points."""
        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)
        extents = max_pt - min_pt
        return {
            "min": min_pt.tolist(),
            "max": max_pt.tolist(),
            "dimensions": extents.tolist(),
            "volume_approx": float(np.prod(extents))
        }
        
    @staticmethod
    def compute_surface_area(mesh: o3d.geometry.TriangleMesh) -> float:
        """Compute total surface area of a mesh."""
        logger.info("Computing mesh surface area")
        return mesh.get_surface_area()
        
    @staticmethod
    def compute_volume(mesh: o3d.geometry.TriangleMesh) -> float:
        """Compute volume of a watertight mesh."""
        logger.info("Computing mesh volume")
        if not mesh.is_watertight():
            logger.warning("Mesh is not watertight, volume calculation may be inaccurate")
        return mesh.get_volume()
        
    @staticmethod
    def height_profile(points: np.ndarray, start_pt: np.ndarray, end_pt: np.ndarray, num_samples: int = 100) -> list:
        """
        Compute height profile along a line segment between start_pt and end_pt.
        """
        logger.info("Computing height profile")
        direction = end_pt - start_pt
        length = np.linalg.norm(direction[:2]) # 2D distance
        if length == 0:
            return []
            
        dir_norm = direction[:2] / length
        
        # Build 2D KD-Tree for quick nearest neighbor in XY plane
        pcd = o3d.geometry.PointCloud()
        pts_3d = np.zeros((len(points), 3))
        pts_3d[:, :2] = points[:, :2] 
        pcd.points = o3d.utility.Vector3dVector(pts_3d)
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        profile = []
        for i in range(num_samples):
            t = i / max(1, num_samples - 1)
            sample_pt_2d = start_pt[:2] + t * direction[:2]
            sample_pt_3d = np.array([sample_pt_2d[0], sample_pt_2d[1], 0.0])
            
            [k, idx, dists] = tree.search_knn_vector_3d(sample_pt_3d, 1)
            if k > 0:
                z_val = points[idx[0], 2]
                profile.append({"distance": float(t * length), "elevation": float(z_val)})
                
        return profile
