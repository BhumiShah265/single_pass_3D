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
        
        if config.use_sam2:
            try:
                from transformers import SamModel, SamProcessor
                logger.info("Loading SAM model for mask refinement via transformers...")
                self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
                self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
                self.use_sam = True
            except ImportError:
                logger.warning("transformers not installed. Falling back to bounding box masks.")
            except Exception as e:
                logger.warning(f"Could not load SAM model: {e}. Falling back to bounding box masks.")

    def process_frame(self, image_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Process a single frame to generate a dynamic object mask.
        
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
            
        # Initialize binary mask (0 = static, 255 = dynamic)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Run YOLO inference
        results = self.yolo(img, verbose=False, conf=self.config.confidence, classes=self.config.target_classes)
        
        boxes = []
        for r in results:
            boxes_data = r.boxes.xyxy.cpu().numpy()
            if len(boxes_data) > 0:
                boxes.extend(boxes_data.tolist())
                
        if len(boxes) > 0:
            if self.use_sam:
                try:
                    # Convert BGR to RGB for SAM
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    inputs = self.sam_processor(img_rgb, input_boxes=[boxes], return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.sam_model(**inputs)
                    
                    # Extract masks from SAM output (batch, num_boxes, 3, H, W) usually
                    # Taking the highest score mask (index 0 for each box usually)
                    pred_masks = outputs.pred_masks.squeeze(1).cpu().numpy()
                    
                    # pred_masks is usually (1, num_boxes, 3, H, W). We take the best mask per box.
                    for box_idx in range(pred_masks.shape[1]):
                        # Best mask is usually index 0 in the 3 returned by SAM
                        m = pred_masks[0, box_idx, 0, :, :]
                        mask[m > 0.0] = 255
                        
                except Exception as e:
                    logger.error(f"SAM refinement failed: {e}. Using bounding boxes.")
                    self._apply_boxes_to_mask(mask, boxes)
            else:
                self._apply_boxes_to_mask(mask, boxes)
            
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
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
