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
        """
        Compute the average optical flow magnitude between two grayscale images.
        """
        # Downscale for high-speed optical flow calculation
        h, w = prev_img.shape[:2]
        if w > 480:
            scale = 480.0 / w
            prev_small = cv2.resize(prev_img, (480, int(h * scale)))
            curr_small = cv2.resize(curr_img, (480, int(h * scale)))
        else:
            prev_small, curr_small = prev_img, curr_img

        flow = cv2.calcOpticalFlowFarneback(
            prev_small, curr_small, None,
            pyr_scale=0.5, levels=2, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))

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
            if len(selected) >= self.config.max_frames:
                logger.info(f"Reached max keyframes ({self.config.max_frames}). Stopping selection.")
                break
                
            baseline_sufficient = False
            curr_img = None
            
            # 1. Check GPS displacement if available
            if gps_data and prev_path in gps_data and curr_path in gps_data:
                p1 = np.array(gps_data[prev_path])
                p2 = np.array(gps_data[curr_path])
                # Simple Euclidean distance assuming local Cartesian or small lat/lon displacement
                dist = np.linalg.norm(p1 - p2)
                if dist >= self.config.min_gps_distance:
                    baseline_sufficient = True
            
            # 2. Fallback to optical flow if GPS not available or distance is small
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

        # If optical flow selected too few frames (e.g. ultra smooth drone pass), sample evenly
        min_desired = min(20, len(image_paths))
        if len(selected) < min_desired:
            step = max(1, len(image_paths) // min_desired)
            selected = image_paths[::step]

        logger.info(f"Selected {len(selected)} keyframes from {len(image_paths)} total frames.")
        return selected
