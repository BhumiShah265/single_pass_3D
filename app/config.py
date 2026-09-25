"""Runtime configuration, dependency probing, and pipeline settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import os
import platform
import shutil

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("aerosynth3d")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = setup_logging()


def get_device(requested: str | None = None):
    """Return a real torch.device using CUDA -> MPS -> CPU selection."""
    try:
        import torch
    except Exception:
        torch = None
    requested = (requested or os.getenv("AEROSYNTH_COMPUTE_BACKEND", "auto")).lower()
    
    if torch is not None and torch.cuda.is_available():
        requested = "cuda"

    if platform.system() == "Darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if torch is None:
        if requested not in {"auto", "cpu"}:
            raise RuntimeError(f"{requested} requested but PyTorch is not installed")
        logger.info("PyTorch unavailable; using portable CPU mode")
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return device

    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
        device = torch.device("mps")
        try:
            logger.info("Using Apple MPS device: %s", torch.backends.mps.get_name())
        except Exception:
            logger.info("Using Apple MPS device")
        return device

    if requested not in {"auto", "cpu"}:
        raise ValueError("compute backend must be one of: auto, cuda, mps, cpu")

    if requested == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return device
    if requested == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        try:
            logger.info("Using Apple MPS device: %s", torch.backends.mps.get_name())
        except Exception:
            logger.info("Using Apple MPS device")
        return device

    device = torch.device("cpu")
    logger.info("Using CPU device")
    return device


def _detect_default_device() -> str:
    env = os.getenv("AEROSYNTH_COMPUTE_BACKEND")
    if env:
        return env.lower()
    try:
        import torch
        if torch.cuda.is_available():
            try:
                # Lock to RTX (device 0 after CUDA_VISIBLE_DEVICES=0 is set in main.py)
                torch.cuda.set_device(0)
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True    # auto-tune kernels for this GPU
                torch.backends.cudnn.allow_tf32 = True   # faster matmuls on Ampere+
                torch.backends.cuda.matmul.allow_tf32 = True
                cores = max(1, os.cpu_count() or 12)
                torch.set_num_threads(cores)             # maximum intra-op CPU-GPU worker threads
                torch.set_num_interop_threads(max(1, cores // 2))
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            except Exception:
                pass
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE = _detect_default_device()


def find_binary(name: str, extra_roots: tuple[Path, ...] = ()) -> Optional[str]:
    """Find an executable on PATH or in common repo-local tool directories."""
    on_path = shutil.which(name)
    if on_path:
        return on_path
    repo_root = Path(__file__).resolve().parent.parent
    roots = (
        repo_root / ".local" / "bin",
        repo_root / "bin",
        repo_root / "openMVS_build" / "bin",
        repo_root / "third_party" / "bin",
        *extra_roots,
    )
    names = [name]
    if os.name == "nt" and not name.lower().endswith(".exe"):
        names.append(f"{name}.exe")
    for root in roots:
        for candidate_name in names:
            candidate = root / candidate_name
            if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
                return str(candidate)
    return None


def find_glomap_binary() -> Optional[str]:
    return find_binary("glomap")


def find_colmap_binary() -> Optional[str]:
    return find_binary("colmap")


@dataclass
class VideoConfig:
    target_fps: float = 3.0
    max_frames: int = 450
    output_format: str = "png"
    capture_profile: str = "aerial_drone"


@dataclass
class QualityConfig:
    blur_threshold: float = 80.0
    min_brightness: float = 30.0
    max_brightness: float = 240.0


@dataclass
class KeyframeConfig:
    min_gps_distance: float = 1.5
    min_optical_flow: float = 8.0
    max_frames: int = 240
    min_time_gap_sec: float = 0.20


@dataclass
class DynamicMaskConfig:
    yolo_model: str = "yolov8n-seg.pt"
    confidence: float = 0.25
    target_classes: list[int] = field(default_factory=lambda: [0, 2, 5, 7, 8])
    use_sam: bool = False


@dataclass
class SfMConfig:
    mapper_backend: str = "glomap"
    feature_type: str = "sift"
    camera_model: str = "SIMPLE_RADIAL"
    camera_mode: str = "SINGLE"
    max_keypoints: int = 16384          # 4× more keypoints → better oblique matching
    match_window: int = 30              # match 30 neighbours → full-orbit coverage
    min_registered_views: int = 3
    min_sparse_points: int = 100


SYSTEM_CPU_CORES = max(1, os.cpu_count() or 12)


@dataclass
class ReconstructionConfig:
    method: str = "openmvs"
    voxel_size: float = 0.04            # finer voxel grid → more surface detail
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 1.5      # tighter outlier removal
    openmvs_resolution_level: int = 0   # full resolution depth maps
    openmvs_max_resolution: int = 3200  # cap per-image resolution (VRAM safe)
    openmvs_number_views: int = 8       # more views considered per depth sample
    openmvs_number_views_fuse: int = 3  # min views needed to accept a fused point
    openmvs_max_threads: int = 0
    allow_colmap_fallback: bool = True


@dataclass
class MeshConfig:
    method: str = "openmvs"
    poisson_depth: int = 10
    density_trim_quantile: float = 0.05
    texture_resolution: int = 4096
    target_face_count: int = 500_000


@dataclass
class GeoConfig:
    crs: str = "auto"
    use_gps_priors: bool = True
    telemetry_path: Optional[str] = None


@dataclass
class PipelineConfig:
    input_video: Optional[str] = None
    input_dir: Optional[str] = None
    telemetry_path: Optional[str] = None
    output_dir: str = "data/output"
    workspace_dir: str = "data/workspace"
    compute_backend: str = field(default_factory=lambda: "cuda" if _detect_default_device() == "cuda" else "auto")
    cpu_threads: int = SYSTEM_CPU_CORES
    video: VideoConfig = field(default_factory=VideoConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    dynamic_mask: DynamicMaskConfig = field(default_factory=DynamicMaskConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)
    skip_dynamic_masking: bool = False
    skip_depth_estimation: bool = False
    skip_georeferencing: bool = False
    skip_analysis: bool = False

    def validate(self) -> None:
        if not self.input_video and not self.input_dir:
            raise ValueError("An input video or input directory is required.")
        if self.input_video and not Path(self.input_video).is_file():
            raise FileNotFoundError(f"Input video not found: {self.input_video}")
        if self.video.target_fps <= 0:
            raise ValueError("video.target_fps must be > 0")
        if self.video.max_frames == 0 or self.video.max_frames < -1:
            raise ValueError("video.max_frames must be -1 or a positive integer")
        if self.keyframe.max_frames < 3:
            raise ValueError("keyframe.max_frames must be >= 3")
        if self.reconstruction.voxel_size <= 0:
            raise ValueError("reconstruction.voxel_size must be > 0")
        if self.mesh.texture_resolution < 256:
            raise ValueError("mesh.texture_resolution is unrealistically small")
        if self.compute_backend not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("compute_backend must be one of: auto, cuda, mps, cpu")
        if not 1 <= self.cpu_threads <= 64:
            raise ValueError("cpu_threads must be between 1 and 64")
        if self.sfm.mapper_backend not in {"glomap", "pycolmap"}:
            raise ValueError("sfm.mapper_backend must be glomap or pycolmap")

    def resolved_device(self):
        return get_device(self.compute_backend)

    def get_workspace(self) -> Path:
        ws = Path(self.workspace_dir)
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def get_output(self) -> Path:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out