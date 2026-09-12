import numpy as np
import open3d as o3d
from app.config import logger

class SemanticLabeler:
    """
    Heuristic classification of point clouds based on height and color.
    Labels: 0: unclassified, 1: ground, 2: vegetation, 3: structure/road
    """
    
    def __init__(self):
        pass
        
    def label_point_cloud(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Assign semantic labels to points.
        Assumes Z is up.
        """
        logger.info("Running heuristic semantic labeling")
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)
        
        labels = np.zeros(len(points), dtype=np.int32)
        
        # Simple heuristic:
        # 1. Ground: Lowest points in local neighborhoods
        # 2. Vegetation: Greenish colors
        # 3. Structure: Non-ground, non-vegetation
        
        z_vals = points[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        
        # Compute "greenness"
        if pcd.has_colors():
            r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
            # Excess Green Index
            exg = 2 * g - r - b
            vegetation_mask = exg > 0.1
        else:
            vegetation_mask = np.zeros(len(points), dtype=bool)
            
        # Ground estimation (simple height thresholding in global sense for heuristic)
        # Better approach would be CSF (Cloth Simulation Filter) but keeping it simple here
        ground_threshold = z_min + 0.1 * (z_max - z_min)
        ground_mask = z_vals < ground_threshold
        
        # Apply labels
        labels[ground_mask] = 1
        labels[vegetation_mask & ~ground_mask] = 2
        labels[~ground_mask & ~vegetation_mask] = 3
        
        return labels
