import numpy as np
import open3d as o3d
from app.config import logger

class ConfidenceAnalyzer:
    """
    Computes point cloud / mesh confidence using multi-view observation count, 
    reprojection error, and density.
    """
    def __init__(self):
        pass

    def compute_point_cloud_confidence(self, pcd: o3d.geometry.PointCloud, 
                                     obs_counts: np.ndarray = None, 
                                     reproj_errors: np.ndarray = None) -> np.ndarray:
        """
        Compute confidence for each point in the point cloud.
        Returns an array of confidence scores [0, 1].
        """
        logger.info("Computing point cloud confidence")
        points = np.asarray(pcd.points)
        n_points = len(points)
        
        confidence = np.ones(n_points, dtype=np.float32)
        
        if obs_counts is not None:
            # Normalize observation counts (heuristic: 2 is bad, 10+ is good)
            obs_conf = np.clip((obs_counts - 2) / 8.0, 0, 1)
            confidence *= obs_conf
            
        if reproj_errors is not None:
            # Normalize reprojection error (heuristic: 0 is best, >2px is bad)
            err_conf = np.clip(1.0 - (reproj_errors / 2.0), 0, 1)
            confidence *= err_conf
            
        # Density based confidence
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        densities = np.zeros(n_points)
        radius = 0.5  # 50cm radius for density calculation
        for i in range(n_points):
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
            densities[i] = k
        
        # Normalize densities (heuristic depends on scale, using percentile here)
        if np.max(densities) > 0:
            density_p95 = np.percentile(densities, 95)
            density_conf = np.clip(densities / (density_p95 + 1e-6), 0, 1)
            confidence *= density_conf
            
        return confidence

    def compute_mesh_confidence(self, mesh: o3d.geometry.TriangleMesh, 
                              point_confidence: np.ndarray, 
                              point_coords: np.ndarray) -> np.ndarray:
        """
        Compute confidence for each vertex in the mesh by querying nearest point cloud points.
        """
        logger.info("Computing mesh confidence")
        vertices = np.asarray(mesh.vertices)
        n_vertices = len(vertices)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_coords)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        vertex_confidence = np.zeros(n_vertices, dtype=np.float32)
        
        for i in range(n_vertices):
            [k, idx, dists] = pcd_tree.search_knn_vector_3d(mesh.vertices[i], 3)
            if k > 0:
                # Inverse distance weighted average of confidence
                weights = 1.0 / (np.asarray(dists) + 1e-6)
                weights /= np.sum(weights)
                vertex_confidence[i] = np.sum(point_confidence[idx] * weights)
                
        return vertex_confidence
