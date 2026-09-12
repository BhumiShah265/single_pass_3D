import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from app.config import QualityConfig, logger


class QualityFilter:
    """
    Filters out low-quality frames based on blur and brightness.
    """
    def __init__(self, config: QualityConfig):
        self.config = config

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
            return {"is_good": False, "score": 0.0, "blur": 0.0, "brightness": 0.0}

        # Laplacian variance is a common measure of focus/blur
        blur = cv2.Laplacian(image, cv2.CV_64F).var()
        
        # Mean brightness
        brightness = image.mean()

        is_good = True
        if blur < self.config.blur_threshold:
            is_good = False
        if brightness < self.config.min_brightness or brightness > self.config.max_brightness:
            is_good = False

        score = blur  # Higher laplacian variance -> sharper image -> higher score

        return {
            "is_good": is_good,
            "score": score,
            "blur": blur,
            "brightness": brightness
        }
        
    def filter_frames(self, image_paths: List[Path]) -> List[Path]:
        """
        Filter a list of frames, returning only those that meet quality standards.
        
        Args:
            image_paths: List of paths to the extracted frames.
            
        Returns:
            A list of paths for frames that passed the quality check.
        """
        good_frames = []
        
        logger.info(f"Running quality assessment on {len(image_paths)} frames...")
        for path in image_paths:
            res = self.assess_frame(path)
            if res["is_good"]:
                good_frames.append(path)
            else:
                logger.debug(
                    f"Filtered out {path.name} "
                    f"(Blur: {res['blur']:.1f}, Brightness: {res['brightness']:.1f})"
                )
                
        logger.info(f"Quality filter: kept {len(good_frames)}/{len(image_paths)} frames.")
        return good_frames
