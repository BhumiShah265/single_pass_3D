import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from ultralytics import YOLO

from app.config import DynamicMaskConfig, DEVICE, logger


class DynamicObjectMasker:
    """Detect and mask transient objects with a device-aware YOLO/SAM path."""
    def __init__(self, config: DynamicMaskConfig, device=None):
        self.config = config
        self.device = device or DEVICE
        self.device_name = str(self.device)

        logger.info("Loading YOLO model %s on %s", config.yolo_model, self.device_name)
        self.yolo = YOLO(config.yolo_model)
        self.yolo.to(self.device)

        self.use_sam = False
        self.sam_processor = None
        self.sam_model = None

        use_sam_flag = getattr(config, "use_sam", False) or getattr(config, "use_sam2", False)
        if use_sam_flag:
            sam_model_name = getattr(config, "sam_model", "facebook/sam-vit-base")
            try:
                from transformers import SamModel, SamProcessor
                logger.info("Loading SAM model %s on %s", sam_model_name, self.device_name)
                self.sam_processor = SamProcessor.from_pretrained(sam_model_name)
                self.sam_model = SamModel.from_pretrained(sam_model_name).to(self.device)
                self.use_sam = True
            except ImportError:
                logger.info("transformers not installed; using YOLO segmentation/bounding masks")
            except Exception as exc:
                logger.warning("Could not load SAM model %s: %s; using YOLO masks", sam_model_name, exc)

    def process_frame(self, image_path: Path, output_dir: Path) -> Optional[Path]:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error("Failed to load %s for dynamic masking", image_path)
            return None

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        is_cuda = "cuda" in self.device_name.lower()
        results = self.yolo(
            img,
            verbose=False,
            conf=self.config.confidence,
            classes=self.config.target_classes,
            device=self.device_name,
            half=is_cuda,
        )

        boxes = []
        has_seg_masks = False
        for result in results:
            if getattr(result, "masks", None) is not None and len(result.masks) > 0:
                has_seg_masks = True
                seg_masks = result.masks.data.cpu().numpy()
                for sm in seg_masks:
                    resized = cv2.resize((sm > 0.5).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_or(mask, resized)

            if getattr(result, "boxes", None) is not None:
                boxes_data = result.boxes.xyxy.cpu().numpy()
                boxes.extend(boxes_data.tolist())

        if boxes and not has_seg_masks:
            if self.use_sam:
                try:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    inputs = self.sam_processor(img_rgb, input_boxes=[boxes], return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.sam_model(**inputs)
                    pred_masks = outputs.pred_masks.squeeze(1).cpu().numpy()
                    for box_idx in range(pred_masks.shape[1]):
                        refined = pred_masks[0, box_idx, 0, :, :]
                        mask[refined > 0.0] = 255
                except Exception as exc:
                    logger.debug("SAM refinement failed; using boxes: %s", exc)
                    self._apply_boxes_to_mask(mask, boxes)
            else:
                self._apply_boxes_to_mask(mask, boxes)

        colmap_mask = cv2.bitwise_not(mask)
        mask_path = output_dir / f"{image_path.name}.png"
        cv2.imwrite(str(mask_path), colmap_mask)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_mask.png"), mask)
        return mask_path

    @staticmethod
    def _apply_boxes_to_mask(mask: np.ndarray, boxes: List[List[float]]) -> None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    def process_frames(self, image_paths: List[Path], output_dir: Path, batch_size: int = 8) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_paths: List[Path] = []
        logger.info("Generating dynamic object masks for %d frames on %s (batch_size=%d, multi-threaded)", len(image_paths), self.device_name, batch_size)
        from concurrent.futures import ThreadPoolExecutor

        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=min(len(chunk), 8)) as ex:
                imgs = list(ex.map(cv2.imread, [str(p) for p in chunk]))

            valid_pairs = [(p, img) for p, img in zip(chunk, imgs) if img is not None]
            if not valid_pairs:
                continue

            batch_imgs = [pair[1] for pair in valid_pairs]
            batch_paths = [pair[0] for pair in valid_pairs]

            try:
                # Batch inference saturates the GPU Tensor Cores
                results = self.yolo(
                    batch_imgs,
                    batch=len(batch_imgs),
                    verbose=False,
                    conf=self.config.confidence,
                    classes=self.config.target_classes,
                    device=self.device_name,
                )
            except Exception as exc:
                logger.warning("Batch YOLO failed (%s); falling back to single frame", exc)
                for p in chunk:
                    res = self.process_frame(p, output_dir)
                    if res:
                        mask_paths.append(res)
                continue

            def save_mask(item):
                p, img, result = item
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                if getattr(result, "masks", None) is not None and len(result.masks) > 0:
                    for sm in result.masks.data.cpu().numpy():
                        resized = cv2.resize((sm > 0.5).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask = cv2.bitwise_or(mask, resized)
                elif getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
                    for box in result.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)

                colmap_mask = cv2.bitwise_not(mask)
                m_path = output_dir / f"{p.name}.png"
                cv2.imwrite(str(m_path), colmap_mask)
                cv2.imwrite(str(output_dir / f"{p.stem}_mask.png"), mask)
                return m_path

            items = list(zip(batch_paths, batch_imgs, results))
            with ThreadPoolExecutor(max_workers=min(len(items), 8)) as ex:
                saved = list(ex.map(save_mask, items))
                mask_paths.extend([m for m in saved if m])

        logger.info("Saved %d masks to %s", len(mask_paths), output_dir)
        return mask_paths
