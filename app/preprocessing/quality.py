import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

from app.config import QualityConfig, logger


class QualityFilter:
    """
    Filters out low-quality frames based on blur and brightness.
    """
    def __init__(self, config: QualityConfig):
        self.config = config
        self.stats: Dict[str, Any] = {}

    def assess_frame(self, image_path: str | Path) -> Dict[str, Any]:
        """
        Assess the quality of a single frame.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            A dictionary containing quality metrics and a boolean indicating
            whether the frame passes the quality thresholds.
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error(f"Failed to read image for quality assessment: {image_path}")
            # Keep the returned schema identical for successful and failed
            # decodes.  ``filter_frames`` records the rejection reason for
            # every rejected frame, including a corrupt or incomplete image.
            return {
                "is_good": False,
                "score": 0.0,
                "blur": 0.0,
                "brightness": 0.0,
                "rejection_reasons": ["image could not be decoded"],
            }

        # Laplacian variance is a standard measure of focus/sharpness
        blur = float(cv2.Laplacian(image, cv2.CV_64F).var())
        brightness = float(image.mean())

        is_good = True
        reasons = []
        if blur < self.config.blur_threshold:
            is_good = False
            reasons.append(f"blur ({blur:.1f} < {self.config.blur_threshold})")
        if brightness < self.config.min_brightness:
            is_good = False
            reasons.append(f"underexposed ({brightness:.1f} < {self.config.min_brightness})")
        elif brightness > self.config.max_brightness:
            is_good = False
            reasons.append(f"overexposed ({brightness:.1f} > {self.config.max_brightness})")

        return {
            "is_good": is_good,
            "score": blur,
            "blur": blur,
            "brightness": brightness,
            "rejection_reasons": reasons
        }
        
    def filter_frames(self, image_paths: List[Path]) -> List[Path]:
        """
        Filter a list of frames, returning only those that meet quality standards.
        Stores true quality statistics for downstream reporting.
        """
        good_frames = []
        blur_scores = []
        brightness_scores = []
        
        logger.info(f"Running quality assessment on {len(image_paths)} frames (blur_thresh={self.config.blur_threshold})...")
        for path in image_paths:
            res = self.assess_frame(path)
            blur_scores.append(res["blur"])
            brightness_scores.append(res["brightness"])
            if res["is_good"]:
                good_frames.append(path)
            else:
                logger.debug(f"Filtered out {path.name}: {', '.join(res['rejection_reasons'])}")
                
        # Prevent zero-frame starvation: if threshold was too strict, retain the top 50% sharpest frames
        if len(good_frames) < min(10, len(image_paths)):
            logger.warning(f"Quality filter threshold yielded too few frames ({len(good_frames)}). Selecting top sharpest frames.")
            sorted_by_sharpness = sorted(zip(image_paths, blur_scores), key=lambda x: x[1], reverse=True)
            keep_count = max(len(good_frames), len(image_paths) // 2)
            good_frames = [p for p, s in sorted_by_sharpness[:keep_count]]

        self.stats = {
            "total_frames_evaluated": len(image_paths),
            "frames_passed": len(good_frames),
            "frames_rejected": len(image_paths) - len(good_frames),
            "mean_blur_score": round(float(np.mean(blur_scores)), 2) if blur_scores else 0.0,
            "mean_brightness": round(float(np.mean(brightness_scores)), 2) if brightness_scores else 0.0,
            "pass_rate_pct": round(len(good_frames) / max(1, len(image_paths)) * 100.0, 1)
        }
        logger.info(f"Quality filter complete: kept {len(good_frames)}/{len(image_paths)} frames ({self.stats['pass_rate_pct']}%)")
        return good_frames
