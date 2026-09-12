import os
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
import torch
import pycolmap

try:
    import glomap
    HAS_GLOMAP = True
except ImportError:
    HAS_GLOMAP = False

from app.config import DEVICE, logger

class SfMPipeline:
    """
    Structure from Motion pipeline using COLMAP/GLOMAP and optional custom feature extractors like LightGlue.
    """
    def __init__(self, workspace_dir: Union[str, Path], use_glomap: bool = True):
        self.workspace_dir = Path(workspace_dir)
        self.database_path = self.workspace_dir / "database.db"
        self.image_dir = self.workspace_dir / "images"
        self.mask_dir = self.workspace_dir / "masks"
        self.sparse_dir = self.workspace_dir / "sparse"
        self.use_glomap = use_glomap and HAS_GLOMAP
        
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_features(self, camera_model: str = "SIMPLE_RADIAL", use_lightglue: bool = False) -> None:
        """
        Extract features from images. Supports default pycolmap or custom LightGlue + ALIKED.
        Dynamic masks are respected if present in self.mask_dir.
        """
        logger.info("Extracting features...")
        
        if use_lightglue:
            self._extract_features_lightglue()
        else:
            logger.info(f"Running pycolmap extract_features on {self.image_dir}")
            
            image_options = pycolmap.ImageReaderOptions()
            image_options.camera_model = camera_model
            if self.mask_dir.exists() and any(self.mask_dir.iterdir()):
                logger.info(f"Using dynamic masks from {self.mask_dir}")
                image_options.mask_path = str(self.mask_dir)
                
            pycolmap.extract_features(
                database_path=self.database_path,
                image_path=self.image_dir,
                image_options=image_options
            )
            
    def _extract_features_lightglue(self) -> None:
        """
        Placeholder for LightGlue + ALIKED extraction logic.
        Requires writing keypoints and descriptors directly to the COLMAP database.
        """
        logger.info(f"Extracting features using LightGlue/ALIKED on device: {DEVICE}")
        logger.warning("LightGlue extraction requires direct database manipulation. Fallback to pycolmap for now.")
        # E.g. filtering out keypoints in masked regions before inserting them into SQLite db.
        self.extract_features(use_lightglue=False)

    def match_features(self, method: str = "sequential") -> None:
        """
        Match extracted features.
        """
        logger.info(f"Matching features using {method} method...")
        if method == "exhaustive":
            pycolmap.match_exhaustive(self.database_path)
        elif method == "sequential":
            pycolmap.match_sequential(self.database_path)
        else:
            raise ValueError(f"Unknown matching method: {method}")
            
    def map(self) -> Path:
        """
        Run sparse reconstruction (mapping) using GLOMAP or COLMAP.
        """
        logger.info(f"Running mapping. Using GLOMAP: {self.use_glomap}")
        if self.use_glomap:
            logger.info("Running GLOMAP mapper...")
            glomap.mapper(
                database_path=str(self.database_path),
                image_path=str(self.image_dir),
                output_path=str(self.sparse_dir)
            )
        else:
            logger.info("Running pycolmap incremental mapping...")
            maps = pycolmap.incremental_mapping(
                database_path=self.database_path,
                image_path=self.image_dir,
                output_path=self.sparse_dir
            )
            if maps:
                logger.info(f"Reconstructed {len(maps)} models.")
                # Save the largest model
                best_model = max(maps.values(), key=lambda m: m.num_images())
                best_model.write(str(self.sparse_dir))
            else:
                logger.error("Mapping failed to reconstruct any models.")
        
        return self.sparse_dir
        
    def run_pipeline(self, use_lightglue: bool = False, match_method: str = "sequential") -> Path:
        """
        Run the full SfM pipeline.
        """
        self.extract_features(use_lightglue=use_lightglue)
        self.match_features(method=match_method)
        return self.map()
