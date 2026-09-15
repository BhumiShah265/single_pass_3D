"""
Shared configuration and device management for the reconstruction pipeline.
"""

import torch
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


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
    capture_profile: str = "aerial_drone"  # Landscape UAV capture is the supported profile


@dataclass
class QualityConfig:
    """Frame quality filtering settings."""
    blur_threshold: float = 80.0       # Laplacian variance threshold
    min_brightness: float = 30.0       # Minimum mean brightness (0-255)
    max_brightness: float = 240.0      # Maximum mean brightness (0-255)


@dataclass
class KeyframeConfig:
    """Keyframe selection settings."""
    min_gps_distance: float = 1.5      # Minimum GPS displacement in meters (if telemetry available)
    min_optical_flow: float = 12.0     # Minimum mean optical flow magnitude for motion change
    max_frames: int = 150              # Maximum keyframes to select for SfM


import shutil

def find_glomap_binary() -> Optional[str]:
    """Auto-detect GLOMAP binary location on PATH or local build directory."""
    local_glomap = Path(__file__).resolve().parent.parent / ".local" / "bin" / "glomap"
    if local_glomap.exists() and os.access(local_glomap, os.X_OK):
        return str(local_glomap)
    sys_glomap = shutil.which("glomap")
    if sys_glomap:
        return sys_glomap
    return None

import os


@dataclass
class DynamicMaskConfig:
    """Dynamic object masking settings."""
    yolo_model: str = "yolov8n-seg.pt"     # YOLO instance segmentation model
    confidence: float = 0.25               # Detection confidence threshold
    target_classes: list = field(default_factory=lambda: [0, 2, 5, 7, 8])
    # COCO IDs: 0=person, 2=car, 5=bus, 7=truck, 8=boat
    use_sam: bool = False                  # Optional SAM refinement flag


@dataclass
class SfMConfig:
    """Structure-from-Motion settings."""
    mapper_backend: str = "glomap"         # "glomap" or "pycolmap"
    feature_type: str = "sift"             # SIFT feature extraction
    camera_model: str = "SIMPLE_RADIAL"    # pycolmap auto-calibrated camera model
    camera_mode: str = "SINGLE"            # Single camera shared across all drone video frames
    max_keypoints: int = 4096              # Max SIFT features per image
    match_window: int = 8                  # Sequential matching overlap window size


@dataclass
class ReconstructionConfig:
    """Dense multi-view stereo reconstruction settings."""
    method: str = "openmvs"                # "openmvs"
    stereo_num_disparities: int = 64       # SGBM disparity range (must be multiple of 16)
    stereo_block_size: int = 7             # SGBM block size
    voxel_size: float = 0.12               # Open3D voxel downsampling size (meters)
    outlier_nb_neighbors: int = 20         # Statistical outlier removal neighbor count
    outlier_std_ratio: float = 2.0         # Statistical outlier removal std ratio


@dataclass
class MeshConfig:
    """Meshing and texturing settings."""
    method: str = "poisson"            # Poisson surface reconstruction
    poisson_depth: int = 9             # Poisson octree depth
    density_trim_quantile: float = 0.05# Trim unobserved low-density vertices
    texture_resolution: int = 2048     # Texture resolution for UV map


@dataclass
class GeoConfig:
    """Georeferencing settings."""
    crs: str = "auto"                  # Detect UTM zone from GPS coordinates if present
    use_gps_priors: bool = True        # Use GPS for alignment if telemetry available
    telemetry_path: Optional[str] = None


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    # I/O paths
    input_video: Optional[str] = None
    input_dir: Optional[str] = None
    telemetry_path: Optional[str] = None
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
