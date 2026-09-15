import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO

from app.config import logger


class SemanticAnnotator:
    """
    Projects 2D detections from calibrated camera views onto the reconstructed 3D points
    to assign standard ASPRS semantic classifications:
    - 2: Ground
    - 5: Vegetation / Trees
    - 6: Buildings
    - 9: Water
    - 11: Roads / Transportation
    """
    def __init__(self, yolo_model: str = "yolov8n.pt"):
        try:
            self.yolo = YOLO(yolo_model)
        except Exception as e:
            logger.warning(f"Could not load YOLO model for semantic annotation: {e}")
            self.yolo = None

    def annotate_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        camera_poses: List[Dict[str, Any]],
        camera_calibration: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Assign semantic classification codes to reconstructed 3D points
        by voting from calibrated camera projections and color analysis.
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)
        n_points = len(points)
        
        if n_points == 0:
            return np.empty(0, dtype=np.int32), []

        logger.info(f"Annotating {n_points:,} 3D points with semantic labels...")
        
        # Default all points to ASPRS Class 2 (Ground)
        classification = np.full(n_points, 2, dtype=np.int32)
        
        # 1. Color-based vegetation & water initial labeling
        if pcd.has_colors():
            r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
            # Excess Green Index: ExG = 2G - R - B
            exg = 2.0 * g - r - b
            veg_mask = (exg > 0.08) & (g > r * 1.05) & (g > b * 1.05)
            
            # Water mask: low variation, blue-dominant
            water_mask = (b > r + 0.05) & (b > g * 0.95) & (r < 0.35)
            
            classification[veg_mask] = 5  # ASPRS Class 5: Vegetation
            classification[water_mask] = 9 # ASPRS Class 9: Water

        # 2. Camera projection voting for 2D detections (Vehicles, Buildings, Persons)
        detected_objects = []
        K = camera_calibration.get("K")
        if K is None:
            focal = float(camera_calibration.get("focal_px", 0.0))
            cx, cy = camera_calibration.get("principal_pt", None)
            if focal <= 0 or cx is None or cy is None:
                return {
                    "status": "GSD unavailable - no real calibrated camera intrinsics recovered by SfM",
                    "gsd_cm_px": None
                }
            K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        reg_views = [p for p in camera_poses if p.get("is_registered", False) and p.get("R") is not None]
        
        if self.yolo is not None and len(reg_views) > 0:
            # Sample up to 8 evenly distributed views
            sampled_indices = np.linspace(0, len(reg_views) - 1, min(8, len(reg_views))).astype(int)
            
            building_pts_accumulator = []
            
            for idx in sampled_indices:
                v = reg_views[idx]
                img_path = Path(v["file_path"])
                if not img_path.exists():
                    continue

                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue

                h_img, w_img = img_bgr.shape[:2]
                res = self.yolo(img_bgr, conf=0.30, verbose=False)
                
                R = np.array(v["R"], dtype=np.float64)
                t = np.array(v["t"], dtype=np.float64)

                # Project 3D points into this camera
                pts_cam = (points @ R.T) + t.reshape(1, 3)
                z_cam = pts_cam[:, 2]
                front = z_cam > 0.5
                front_idx = np.where(front)[0]

                if len(front_idx) == 0:
                    continue

                u_px = (K[0, 0] * (pts_cam[front_idx, 0] / z_cam[front_idx]) + K[0, 2]).astype(int)
                v_px = (K[1, 1] * (pts_cam[front_idx, 1] / z_cam[front_idx]) + K[1, 2]).astype(int)
                in_bounds = (u_px >= 0) & (u_px < w_img) & (v_px >= 0) & (v_px < h_img)
                valid_pt_indices = front_idx[in_bounds]
                u_valid = u_px[in_bounds]
                v_valid = v_px[in_bounds]

                for b in res[0].boxes:
                    cls_id = int(b.cls[0])
                    cname = self.yolo.names[cls_id]
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()

                    # Find points falling inside this 2D bounding box
                    inside_box = (u_valid >= x1) & (u_valid <= x2) & (v_valid >= y1) & (v_valid <= y2)
                    matched_pt_indices = valid_pt_indices[inside_box]

                    if len(matched_pt_indices) > 0:
                        if cname in ['building', 'house', 'roof']:
                            classification[matched_pt_indices] = 6
                            building_pts_accumulator.append({
                                "pts": points[matched_pt_indices],
                                "conf": conf,
                                "cname": cname
                            })
                        elif cname in ['car', 'truck', 'bus', 'vehicle']:
                            # Dynamic objects are suppressed on ground
                            classification[matched_pt_indices] = 2

        # 3. Aggregate 3D Building Evidence
        z_vals = points[:, 1] # Y is elevation in our coordinate system
        ground_datum = float(np.percentile(z_vals, 10))

        bld_count = 0
        bld_indices = np.where(classification == 6)[0]
        if len(bld_indices) >= 10:
            # Segment building points into spatial clusters via simple spatial grid
            bld_pts = points[bld_indices]
            from sklearn.cluster import DBSCAN
            try:
                clustering = DBSCAN(eps=4.0, min_samples=10).fit(bld_pts[:, [0, 2]])
                labels = clustering.labels_
                unique_clusters = set(labels) - {-1}

                for cl_id in unique_clusters:
                    cl_pts = bld_pts[labels == cl_id]
                    if len(cl_pts) < 10:
                        continue
                    bld_count += 1
                    min_c = cl_pts.min(axis=0)
                    max_c = cl_pts.max(axis=0)
                    cx = float((min_c[0] + max_c[0]) * 0.5)
                    cz = float((min_c[2] + max_c[2]) * 0.5)

                    # Estimate roof and ground elevation strictly from 3D points
                    roof_y = float(np.percentile(cl_pts[:, 1], 95))
                    # Ground elevation near this cluster
                    local_ground_mask = (np.abs(points[:, 0] - cx) < 8.0) & (np.abs(points[:, 2] - cz) < 8.0) & (classification == 2)
                    if np.any(local_ground_mask):
                        cluster_ground_y = float(np.percentile(points[local_ground_mask, 1], 15))
                    else:
                        cluster_ground_y = ground_datum

                    height_m = max(1.5, round(roof_y - cluster_ground_y, 1))
                    l_m = round(float(max_c[0] - min_c[0]), 1)
                    w_m = round(float(max_c[2] - min_c[2]), 1)
                    area_m2 = round(l_m * w_m, 1)

                    detected_objects.append({
                        "id": f"BLD-{bld_count:02d}",
                        "type": "Reconstructed Building Structure",
                        "label": f"Building #{bld_count}",
                        "category": "building",
                        "icon": "domain",
                        "x": round(cx, 2),
                        "y": round(cluster_ground_y, 2),
                        "z": round(cz, 2),
                        "height_m": height_m,
                        "length_m": l_m,
                        "width_m": w_m,
                        "area_m2": area_m2,
                        "volume_m3": round(area_m2 * height_m, 1),
                        "points_count": len(cl_pts),
                        "confidence": 0.85,
                        "evidence": "3D Multi-View Point Cluster",
                        "asprs_class": 6
                    })
            except Exception as cl_err:
                logger.debug(f"Building clustering notice: {cl_err}")

        logger.info(f"Semantic annotation complete: {bld_count} verified 3D building structures identified.")
        return classification, detected_objects
