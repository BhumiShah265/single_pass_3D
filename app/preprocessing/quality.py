import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

from app.config import QualityConfig, logger


try:
    import torch
    _HAS_TORCH_CUDA = torch.cuda.is_available()
    if _HAS_TORCH_CUDA:
        _LAPLACIAN_KERNEL = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32, device="cuda").view(1, 1, 3, 3)
    else:
        _LAPLACIAN_KERNEL = None
except Exception:
    _HAS_TORCH_CUDA = False
    _LAPLACIAN_KERNEL = None


class QualityFilter:
    """
    Filters out low-quality frames based on blur and brightness.
    Leverages GPU CUDA acceleration for instant batch frame quality estimation when available.
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
        # Computes on CUDA GPU when available for maximum speed and utilization
        if _HAS_TORCH_CUDA and _LAPLACIAN_KERNEL is not None:
            try:
                t = torch.from_numpy(image).to("cuda", non_blocking=True).float().unsqueeze(0).unsqueeze(0)
                filtered = torch.nn.functional.conv2d(t, _LAPLACIAN_KERNEL)
                blur = float(filtered.var().item())
                brightness = float(t.mean().item())
            except Exception:
                blur = float(cv2.Laplacian(image, cv2.CV_64F).var())
                brightness = float(image.mean())
        else:
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
        
        logger.info(f"Running quality assessment on {len(image_paths)} frames via multi-threaded GPU batch processing...")
        from concurrent.futures import ThreadPoolExecutor

        def _read_gray(p: Path):
            return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

        batch_size = 16
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=min(len(chunk), 8)) as ex:
                raw_imgs = list(ex.map(_read_gray, chunk))

            valid = [(p, img) for p, img in zip(chunk, raw_imgs) if img is not None]
            if not valid:
                continue

            curr_paths = [v[0] for v in valid]
            curr_imgs = [v[1] for v in valid]

            if _HAS_TORCH_CUDA and _LAPLACIAN_KERNEL is not None:
                try:
                    # Parallel CUDA batch convolution across all frames in chunk
                    batch_t = torch.stack([torch.from_numpy(m).float() for m in curr_imgs]).unsqueeze(1).to("cuda", non_blocking=True)
                    filtered = torch.nn.functional.conv2d(batch_t, _LAPLACIAN_KERNEL)
                    vars_gpu = filtered.var(dim=[2, 3]).squeeze(1).cpu().tolist()
                    means_gpu = batch_t.mean(dim=[2, 3]).squeeze(1).cpu().tolist()
                    del batch_t, filtered
                except Exception:
                    vars_gpu = [float(cv2.Laplacian(m, cv2.CV_64F).var()) for m in curr_imgs]
                    means_gpu = [float(m.mean()) for m in curr_imgs]
            else:
                vars_gpu = [float(cv2.Laplacian(m, cv2.CV_64F).var()) for m in curr_imgs]
                means_gpu = [float(m.mean()) for m in curr_imgs]

            for p, blur, brightness in zip(curr_paths, vars_gpu, means_gpu):
                blur_scores.append(blur)
                brightness_scores.append(brightness)
                if (blur >= self.config.blur_threshold and
                    self.config.min_brightness <= brightness <= self.config.max_brightness):
                    good_frames.append(p)
                else:
                    logger.debug(f"Filtered out {p.name}: blur={blur:.1f}, brightness={brightness:.1f}")
                
        # Prevent zero-frame starvation: if threshold was too strict, retain the top 50% sharpest frames
        if len(good_frames) < min(10, len(image_paths)):
            sorted_by_sharpness = sorted(zip(image_paths, blur_scores), key=lambda x: x[1], reverse=True)
            keep_count = max(len(good_frames), len(image_paths) // 2)
            good_frames = [p for p, s in sorted_by_sharpness[:keep_count]]
            good_frames.sort(key=lambda p: p.name)  # Restore chronological flight order

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
