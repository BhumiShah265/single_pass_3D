"""
Shared configuration and device management for the reconstruction pipeline.
"""

import torch
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the pipeline logger."""
    logger = logging.getLogger("vid-img")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# Device Detection (CUDA > MPS > CPU)
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


DEVICE = get_device()


# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------
@dataclass
class VideoConfig:
    """Video extraction settings."""
    target_fps: float = 2.0            # Extract frames at this FPS
    max_frames: int = 500              # Maximum frames to extract
    output_format: str = "png"         # Output image format


@dataclass
class QualityConfig:
    """Frame quality filtering settings."""
    blur_threshold: float = 100.0      # Laplacian variance threshold
    min_brightness: float = 30.0       # Minimum mean brightness (0-255)
    max_brightness: float = 240.0      # Maximum mean brightness (0-255)


@dataclass
class KeyframeConfig:
    """Keyframe selection settings."""
    min_gps_distance: float = 2.0      # Minimum GPS displacement (meters)
    min_optical_flow: float = 15.0     # Minimum mean optical flow magnitude
    max_frames: int = 300              # Maximum keyframes to select


@dataclass
class DynamicMaskConfig:
    """Dynamic object masking settings."""
    yolo_model: str = "yolov8n.pt"     # YOLO model variant
    confidence: float = 0.25           # Detection confidence threshold
    target_classes: list = field(default_factory=lambda: [0, 2, 5, 7, 8])
    # COCO IDs: 0=person, 2=car, 5=bus, 7=truck, 8=boat
    use_sam2: bool = True              # Use SAM2 for precise masks


@dataclass
class SfMConfig:
    """Structure-from-Motion settings."""
    feature_type: str = "aliked"       # "aliked" or "superpoint"
    max_keypoints: int = 4096          # Max features per image
    match_window: int = 8              # Sequential matching window size
    mapper: str = "glomap"             # "glomap" or "colmap"
    use_gpu: bool = True               # Use GPU for matching
    gps_prior_weight: float = 1.0      # GPS prior strength in BA


@dataclass
class ReconstructionConfig:
    """Dense reconstruction settings."""
    method: str = "depth_fusion"       # "depth_fusion" or "openmvs"
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    voxel_size: float = 0.02           # TSDF voxel size (meters)
    depth_trunc: float = 50.0          # Maximum depth (meters)
    sdf_trunc: float = 0.1             # SDF truncation distance (meters)


@dataclass
class MeshConfig:
    """Meshing settings."""
    method: str = "poisson"            # "poisson" or "tsdf"
    poisson_depth: int = 11            # Poisson octree depth
    target_faces: int = 500_000        # Target face count for decimation
    smooth_iterations: int = 3         # Laplacian smoothing iterations
    texture_resolution: int = 4096     # Texture atlas resolution


@dataclass
class GeoConfig:
    """Georeferencing settings."""
    crs: str = "auto"                  # "auto" (detect from GPS) or EPSG code
    use_gps_priors: bool = True        # Use GPS for alignment
    altitude_mode: str = "relative"    # "relative" or "absolute"


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    # I/O paths
    input_video: Optional[str] = None
    input_dir: Optional[str] = None
    output_dir: str = "data/output"
    workspace_dir: str = "data/workspace"

    # Stage configs
    video: VideoConfig = field(default_factory=VideoConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    dynamic_mask: DynamicMaskConfig = field(default_factory=DynamicMaskConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)

    # Stage toggles
    skip_dynamic_masking: bool = False
    skip_depth_estimation: bool = False
    skip_georeferencing: bool = False
    skip_analysis: bool = False

    def get_workspace(self) -> Path:
        """Get and create workspace directory."""
        ws = Path(self.workspace_dir)
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def get_output(self) -> Path:
        """Get and create output directory."""
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out
