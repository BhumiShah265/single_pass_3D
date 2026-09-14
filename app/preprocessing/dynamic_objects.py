import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from ultralytics import YOLO

from app.config import DynamicMaskConfig, DEVICE, logger


class DynamicObjectMasker:
    """
    Detects and masks out dynamic objects (people, cars, buses, etc.)
    using YOLOv8 and optionally refines masks with SAM2/transformers.
    Outputs binary masks (0=static, 255=dynamic).
    """
    def __init__(self, config: DynamicMaskConfig):
        self.config = config
        self.device = DEVICE
        
        logger.info(f"Loading YOLO model {config.yolo_model} on {self.device}")
        self.yolo = YOLO(config.yolo_model)
        self.yolo.to(self.device)
        
        self.use_sam = False
        self.sam_processor = None
        self.sam_model = None
        
        # Check use_sam or backward-compatibility use_sam2
        use_sam_flag = getattr(config, "use_sam", False) or getattr(config, "use_sam2", False)
        if use_sam_flag:
            sam_model_name = getattr(config, "sam_model", "facebook/sam-vit-base")
            try:
                from transformers import SamModel, SamProcessor
                logger.info(f"Loading SAM model ({sam_model_name}) for fine mask boundary refinement...")
                self.sam_processor = SamProcessor.from_pretrained(sam_model_name)
                self.sam_model = SamModel.from_pretrained(sam_model_name).to(self.device)
                self.use_sam = True
            except ImportError:
                logger.info("transformers not installed. Using YOLOv8 native instance segmentation / bounding masks.")
            except Exception as e:
                logger.warning(f"Could not load SAM model '{sam_model_name}': {e}. Using YOLOv8 masks.")

    def process_frame(self, image_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Process a single frame to generate a dynamic object mask.
        Supports native YOLOv8 instance segmentation masks, SAM refinement, or bounding boxes.
        
        Args:
            image_path: Path to the input frame.
            output_dir: Directory to save the resulting mask.
            
        Returns:
            Path to the saved mask file, or None on failure.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load {image_path} for dynamic masking.")
            return None
            
        # Initialize binary mask (0 = static background, 255 = dynamic object)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Run YOLO inference
        results = self.yolo(img, verbose=False, conf=self.config.confidence, classes=self.config.target_classes)
        
        boxes = []
        has_seg_masks = False

        for r in results:
            # 1. Native YOLOv8 instance segmentation masks if available
            if hasattr(r, 'masks') and r.masks is not None and len(r.masks) > 0:
                has_seg_masks = True
                seg_masks = r.masks.data.cpu().numpy() # (N, H_mask, W_mask)
                for sm in seg_masks:
                    sm_resized = cv2.resize((sm > 0.5).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_or(mask, sm_resized)

            if hasattr(r, 'boxes') and r.boxes is not None:
                boxes_data = r.boxes.xyxy.cpu().numpy()
                if len(boxes_data) > 0:
                    boxes.extend(boxes_data.tolist())
                
        if len(boxes) > 0 and not has_seg_masks:
            if self.use_sam:
                try:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    inputs = self.sam_processor(img_rgb, input_boxes=[boxes], return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.sam_model(**inputs)
                    pred_masks = outputs.pred_masks.squeeze(1).cpu().numpy()
                    for box_idx in range(pred_masks.shape[1]):
                        m = pred_masks[0, box_idx, 0, :, :]
                        mask[m > 0.0] = 255
                except Exception as e:
                    logger.debug(f"SAM refinement fallback to bounding boxes: {e}")
                    self._apply_boxes_to_mask(mask, boxes)
            else:
                self._apply_boxes_to_mask(mask, boxes)
            
        # Save masks matching COLMAP naming conventions
        # In COLMAP, pixel > 0 is the VALID region to extract features from, and 0 is IGNORED/masked.
        colmap_mask = cv2.bitwise_not(mask)
        mask_path = output_dir / f"{image_path.name}.png"
        cv2.imwrite(str(mask_path), colmap_mask)
        # Also save legacy stem mask where 255 = dynamic object (for MVS depth zeroing)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_mask.png"), mask)
        return mask_path

    def _apply_boxes_to_mask(self, mask: np.ndarray, boxes: List[List[float]]):
        """Helper to draw filled bounding boxes onto the mask."""
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    def process_frames(self, image_paths: List[Path], output_dir: Path) -> List[Path]:
        """
        Process a list of frames to generate dynamic object masks for each.
        
        Args:
            image_paths: List of paths to the frames.
            output_dir: Directory where masks should be saved.
            
        Returns:
            List of paths to the successfully generated masks.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_paths = []
        
        logger.info(f"Generating dynamic object masks for {len(image_paths)} frames...")
        for p in image_paths:
            m = self.process_frame(p, output_dir)
            if m:
                mask_paths.append(m)
                
        logger.info(f"Saved {len(mask_paths)} masks to {output_dir}")
        return mask_paths
