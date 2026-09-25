import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from app.config import KeyframeConfig, logger


class KeyframeSelector:
    """
    Selects keyframes based on optical flow and/or GPS displacement
    to ensure good baseline separation for structure-from-motion.
    """
    def __init__(self, config: KeyframeConfig):
        self.config = config

    def compute_optical_flow(self, prev_img: np.ndarray, curr_img: np.ndarray) -> float:
        """Return robust median static-feature displacement in source pixels."""
        prev_small, curr_small = prev_img, curr_img
        h, w = prev_img.shape[:2]
        scale = 1.0
        if w > 800:
            scale = 800.0 / w
            size = (800, max(1, int(h * scale)))
            prev_small = cv2.resize(prev_img, size, interpolation=cv2.INTER_AREA)
            curr_small = cv2.resize(curr_img, size, interpolation=cv2.INTER_AREA)
        pts0 = cv2.goodFeaturesToTrack(prev_small, maxCorners=800, qualityLevel=0.01, minDistance=8, blockSize=7)
        if pts0 is None or len(pts0) < 30:
            return 0.0
        pts1, status, _ = cv2.calcOpticalFlowPyrLK(prev_small, curr_small, pts0, None, winSize=(21,21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if pts1 is None or status is None:
            return 0.0
        good = status.reshape(-1).astype(bool)
        if good.sum() < 20:
            return 0.0
        p0 = pts0.reshape(-1,2)[good]
        p1 = pts1.reshape(-1,2)[good]
        displacement = np.linalg.norm(p1-p0, axis=1)
        displacement = displacement[np.isfinite(displacement)]
        if len(displacement) < 20:
            return 0.0
        return float(np.median(displacement) / max(scale, 1e-6))

    def select_keyframes(
        self, 
        image_paths: List[Path], 
        gps_data: Optional[Dict[Path, Tuple[float, float, float]]] = None
    ) -> List[Path]:
        """
        Select keyframes from a list of sequential image paths.
        Ensure sufficient baseline between consecutive keyframes.
        
        Args:
            image_paths: Sorted list of image paths to select from.
            gps_data: Optional mapping of image path to (lat, lon, alt) or (x, y, z) 
                      for displacement checking.
                      
        Returns:
            List of selected keyframe paths.
        """
        if not image_paths:
            return []

        selected = [image_paths[0]]
        
        prev_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
        if prev_img is None:
            logger.error(f"Failed to read image: {image_paths[0]}")
            
        prev_path = image_paths[0]

        logger.info(f"Selecting keyframes from {len(image_paths)} frames...")
        
        for curr_path in image_paths[1:]:
            baseline_sufficient = False
            curr_img = None
            
            # 1. Check GPS displacement if available
            if gps_data and prev_path in gps_data and curr_path in gps_data:
                p1 = gps_data[prev_path]
                p2 = gps_data[curr_path]
                if abs(p1[0]) <= 90.0 and abs(p1[1]) <= 180.0:
                    # Convert lat/lon degrees to metric distance in meters
                    m_lat = (p2[0] - p1[0]) * 111139.0
                    m_lon = (p2[1] - p1[1]) * 111139.0 * np.cos(np.radians(p1[0]))
                    m_alt = p2[2] - p1[2]
                    dist = float(np.sqrt(m_lat**2 + m_lon**2 + m_alt**2))
                else:
                    dist = float(np.linalg.norm(np.array(p1) - np.array(p2)))
                    
                if dist >= self.config.min_gps_distance:
                    baseline_sufficient = True
            
            # 2. Robust static-feature displacement when GPS is unavailable. Median
            # tracked motion is less sensitive to moving objects and camera rotation
            # than mean dense optical flow.
            if not baseline_sufficient:
                curr_img = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
                if curr_img is not None and prev_img is not None:
                    flow_mag = self.compute_optical_flow(prev_img, curr_img)
                    if flow_mag >= self.config.min_optical_flow:
                        baseline_sufficient = True

            if baseline_sufficient:
                selected.append(curr_path)
                if curr_img is None:
                    curr_img = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
                prev_img = curr_img
                prev_path = curr_path

        # If keyframe count exceeds budget, uniformly subsample to span the entire flight
        if len(selected) > self.config.max_frames:
            step = len(selected) / float(self.config.max_frames)
            subsampled = [selected[int(i * step)] for i in range(self.config.max_frames)]
            if selected[-1] not in subsampled:
                subsampled[-1] = selected[-1]
            selected = subsampled

        # If optical flow selected too few frames (e.g. ultra smooth drone pass), sample evenly
        min_desired = min(20, len(image_paths))
        if len(selected) < min_desired:
            step = max(1, len(image_paths) // min_desired)
            selected = image_paths[::step]

        logger.info(f"Selected {len(selected)} keyframes from {len(image_paths)} total frames.")
        return selected