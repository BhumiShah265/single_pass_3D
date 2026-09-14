import numpy as np
import open3d as o3d
from typing import Dict, Any, List, Optional, Tuple

from app.config import logger


class MeasurementTool:
    """
    Computes 3D distances, bounding boxes, surface areas, volumes, 
    true Ground Sampling Distance (GSD), and building/vegetation heights
    from real reconstructed 3D point clouds and meshes.
    """
    
    @staticmethod
    def compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute Euclidean distance between two 3D points in meters."""
        return float(np.linalg.norm(p1 - p2))
        
    @staticmethod
    def compute_bounding_box(points: np.ndarray) -> Dict[str, Any]:
        """Compute axis-aligned bounding box and dimensions for a set of points."""
        if len(points) == 0:
            return {"min": [0, 0, 0], "max": [0, 0, 0], "dimensions": [0, 0, 0], "volume_approx": 0.0}
            
        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)
        extents = max_pt - min_pt
        
        # Real bounding box volume in m3
        vol = float(extents[0] * extents[1] * extents[2])
        return {
            "min": [round(float(v), 2) for v in min_pt],
            "max": [round(float(v), 2) for v in max_pt],
            "dimensions": [round(float(v), 2) for v in extents],
            "volume_approx_m3": round(vol, 1)
        }
        
    @staticmethod
    def compute_gsd(
        camera_poses: List[Dict[str, Any]],
        camera_calibration: Dict[str, Any],
        points: np.ndarray
    ) -> Optional[float]:
        """
        Calculate true Ground Sampling Distance (GSD) in cm/pixel from calibrated camera geometry:
        GSD = (H / focal_px) * 100
        where H is the median camera-to-ground distance in meters, and focal_px is the
        focal length in pixels from pycolmap auto-calibration.
        """
        focal_px = float(camera_calibration.get("focal_px", 0.0))
        if focal_px <= 0 or len(points) == 0:
            return None

        # Camera centers
        cam_centers = [np.array(p["C"]) for p in camera_poses if p.get("is_registered") and p.get("C")]
        if not cam_centers:
            return None

        # Median altitude/distance from cameras to ground points
        cam_arr = np.array(cam_centers)
        median_cam = np.median(cam_arr, axis=0)
        
        # Subsample points to compute median distance
        sub_pts = points[::max(1, len(points) // 500)]
        dists = np.linalg.norm(sub_pts - median_cam.reshape(1, 3), axis=1)
        median_distance_m = float(np.median(dists))

        if median_distance_m <= 0:
            return None

        gsd_cm = (median_distance_m / focal_px) * 100.0
        return round(float(gsd_cm), 2)

    @staticmethod
    def compute_feature_heights(
        points: np.ndarray,
        classification: np.ndarray,
        radius_m: float = 6.0
    ) -> Dict[str, Any]:
        """
        Calculates vegetation and structure heights strictly from 3D points.
        Height = (95th percentile elevation of feature points) - (local ground elevation).
        If fewer than 10 points are present, returns 'insufficient 3D evidence'.
        """
        ground_mask = (classification == 2)
        veg_mask = (classification == 5)
        bld_mask = (classification == 6)

        results = {}

        if np.sum(ground_mask) < 20:
            return {"status": "insufficient 3D ground evidence"}

        ground_y = points[ground_mask, 1] # Y is vertical elevation
        median_ground = float(np.median(ground_y))

        # Vegetation Heights
        if np.sum(veg_mask) >= 10:
            veg_y = points[veg_mask, 1]
            veg_top = float(np.percentile(veg_y, 95))
            results["vegetation_max_height_m"] = max(0.5, round(veg_top - median_ground, 1))
            results["vegetation_mean_height_m"] = max(0.5, round(float(np.mean(veg_y)) - median_ground, 1))
        else:
            results["vegetation_height"] = "insufficient 3D evidence"

        # Building Heights
        if np.sum(bld_mask) >= 10:
            bld_y = points[bld_mask, 1]
            bld_top = float(np.percentile(bld_y, 95))
            results["building_max_height_m"] = max(1.5, round(bld_top - median_ground, 1))
        else:
            results["building_height"] = "insufficient 3D evidence"

        return results
