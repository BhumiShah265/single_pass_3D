"""FastAPI routes for upload, reconstruction jobs, status, and deliverables."""
from __future__ import annotations

import json
import mimetypes
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import DEVICE, PipelineConfig, find_binary, find_colmap_binary, find_glomap_binary, logger

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = ROOT / "data" / "uploads"
OUTPUT_ROOT = ROOT / "data" / "output"
STATE_ROOT = ROOT / "data" / "job_state"
for directory in (UPLOAD_ROOT, OUTPUT_ROOT, STATE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api/v1", tags=["reconstruction"])
STAGES = [("video_extraction", "Video ingestion & frame extraction"), ("quality_filtering", "Frame quality filtering"), ("keyframe_selection", "Keyframe selection"), ("dynamic_masking", "Dynamic-object masking"), ("sfm", "Structure from Motion"), ("dense_reconstruction", "Dense multi-view stereo"), ("meshing", "Surface meshing & texturing"), ("georeferencing", "Georeferencing"), ("analysis", "Metrology & semantic analysis"), ("export_deliverables", "Deliverable export")]
STAGE_INDEX = {name: i + 1 for i, (name, _) in enumerate(STAGES)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(filename: str | None, fallback: str) -> str:
    name = Path(filename or fallback).name.strip()
    return name if name and name not in {".", ".."} else fallback


def _state_path(job_id: str) -> Path:
    return STATE_ROOT / f"{job_id}.json"


def _save_job(job_id: str) -> None:
    path = _state_path(job_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs[job_id], indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _load_jobs() -> dict[str, dict[str, Any]]:
    result = {}
    for path in STATE_ROOT.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(item, dict) and item.get("job_id"):
                result[str(item["job_id"])] = item
        except Exception as exc:
            logger.warning("Ignoring unreadable job state %s: %s", path, exc)
    return result

jobs = _load_jobs()
_running_jobs: set[str] = set()
_running_lock = threading.Lock()
_pipeline_slots = threading.Semaphore(max(1, int(os.getenv("AEROSYNTH_MAX_CONCURRENT_JOBS", "1"))))


class JobStatus(BaseModel):
    job_id: str
    status: str
    stage: str = "queued"
    stage_index: int = 0
    stage_total: int = len(STAGES)
    progress: float = 0.0
    message: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    mapper_backend: Optional[str] = None


SYSTEM_CPU_CORES = max(1, os.cpu_count() or 12)


class ReconstructRequest(BaseModel):
    skip_dynamic_masking: bool = False
    skip_depth_estimation: bool = False
    skip_georeferencing: bool = False
    skip_analysis: bool = False
    target_fps: float = Field(default=2.5, gt=0, le=10)
    mapper_backend: str = Field(default="pycolmap", pattern="^(glomap|pycolmap)$")
    processing_profile: str = Field(default="balanced", pattern="^(fast|balanced|quality)$")
    cpu_threads: int = Field(default=SYSTEM_CPU_CORES, ge=1, le=64)
    telemetry_filename: Optional[str] = None

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_TELEMETRY_EXTENSIONS = {".srt", ".gpx", ".csv"}
_DELIVERABLES = {"obj": "model.obj", "mtl": "model.mtl", "texture": "texture.jpg", "ply": "cloud.ply", "las": "cloud.las", "glb": "model.glb", "gltf": "model.glb", "complete": "model_complete.glb", "fbx": "model.fbx", "orthophoto": "ortho.tif", "ortho": "ortho.tif", "geotiff": "ortho.tif", "dsm": "dsm.tif", "report": "report.pdf", "pdf": "report.pdf", "manifest": "manifest.json"}
_MIME = {"glb": "model/gltf-binary", "obj": "text/plain", "mtl": "text/plain", "ply": "application/octet-stream", "las": "application/octet-stream", "fbx": "application/octet-stream", "tif": "image/tiff", "pdf": "application/pdf", "json": "application/json", "jpg": "image/jpeg", "png": "image/png"}


def _set_job(job_id: str, **updates: Any) -> None:
    jobs[job_id].update(updates)
    jobs[job_id]["updated_at"] = _now()
    _save_job(job_id)


def _run_pipeline_task(job_id: str, config: PipelineConfig) -> None:
    # Keep the large Open3D/CV pipeline out of the web server's startup path.
    # It is only needed after a user has explicitly started a reconstruction.
    with _running_lock:
        if job_id in _running_jobs:
            return
        _running_jobs.add(job_id)
    try:
        _set_job(job_id, status="queued", stage="queued", stage_index=0, progress=0.0, message="Waiting for reconstruction worker slot")
        with _pipeline_slots:
            thread_count = str(config.cpu_threads)
            for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[variable] = thread_count
            os.environ["OMP_DYNAMIC"] = "FALSE"
            import torch
            if torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                torch.cuda.set_device(0)
                gpu_name = torch.cuda.get_device_name(0)
                device_desc = f"Compulsory GPU ({gpu_name}) + {thread_count} CPU threads"
            else:
                device_desc = f"{thread_count} CPU threads"
            from app.pipeline_runtime import ReconstructionPipeline
            _set_job(job_id, status="running", stage="initializing", stage_index=0, progress=0.0, message=f"Initializing reconstruction engine on {device_desc}", started_at=_now(), mapper_backend=config.sfm.mapper_backend)
            pipeline = ReconstructionPipeline(config)
            for stage_name, _ in STAGES:
                method = getattr(pipeline, f"_stage_{stage_name}", None)
                if method is None:
                    continue
                index = STAGE_INDEX[stage_name]
                def wrapped(*args: Any, _method=method, _name=stage_name, _index=index, **kwargs: Any):
                    _set_job(job_id, status="running", stage=_name, stage_index=_index, progress=round((_index - 1) * 100 / len(STAGES), 1), message=dict(STAGES)[_name])
                    result = _method(*args, **kwargs)
                    _set_job(job_id, status="running", stage=_name, stage_index=_index, progress=round(_index * 100 / len(STAGES), 1), message=f"Completed: {dict(STAGES)[_name]}")
                    return result
                setattr(pipeline, f"_stage_{stage_name}", wrapped)
            result = pipeline.run()
            if result.get("status") == "success":
                _set_job(job_id, status="completed", stage="completed", stage_index=len(STAGES), progress=100.0, message="Reconstruction completed", finished_at=_now(), output_dir=str(config.get_output().resolve()))
            else:
                _set_job(job_id, status="failed", stage="failed", message=result.get("message", "Pipeline failed"), finished_at=_now())
    except Exception as exc:
        logger.exception("Reconstruction failed for %s", job_id)
        _set_job(job_id, status="failed", stage="failed", message=str(exc), finished_at=_now())
    finally:
        with _running_lock:
            _running_jobs.discard(job_id)


@router.get("/health")
async def health():
    return {"status": "ok", "service": "aerosynth3d", "time": _now()}


@router.get("/preflight")
async def preflight():
    import importlib.util
    packages = {p: bool(importlib.util.find_spec(p)) for p in ("cv2", "numpy", "PIL", "open3d", "pycolmap", "ultralytics", "pyproj", "rasterio", "laspy", "trimesh", "pygltflib")}
    openmvs = {name: bool(find_binary(name)) for name in ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh")}
    ready = all(packages[p] for p in ("cv2", "numpy", "PIL", "open3d", "pycolmap")) and all(openmvs.values())
    return {"status": "ready" if ready else "incomplete", "device": str(DEVICE), "glomap": find_glomap_binary(), "colmap": find_colmap_binary(), "python_packages": packages, "openmvs": openmvs, "workspace_writable": os.access(str(ROOT / "data"), os.W_OK)}


@router.get("/system-info")
async def system_info():
    import platform, subprocess, sys
    gpu_name = None
    vram_gb = None
    try:
        nvidia_smi = find_binary("nvidia-smi")
        if nvidia_smi:
            probe = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1.5, check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                gpu_name, memory_mb = (part.strip() for part in probe.stdout.splitlines()[0].split(",", 1))
                vram_gb = round(float(memory_mb) / 1024, 2)
    except Exception:
        pass
    return {"device": str(DEVICE), "device_name": gpu_name or platform.processor() or "Host CPU", "vram_gb": vram_gb, "python": sys.version.split()[0], "platform": platform.platform(), "pipeline_version": "5.0.0"}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = _safe_filename(file.filename, "input.mp4")
    if Path(filename).suffix.lower() not in _VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported video format: {Path(filename).suffix}")
    job_id = str(uuid.uuid4())
    upload_dir = UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / filename
    size = 0
    limit = int(os.getenv("AEROSYNTH_MAX_UPLOAD_BYTES", str(20 * 1024**3)))
    try:
        with path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, "Video exceeds configured upload limit")
                target.write(chunk)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    jobs[job_id] = {"job_id": job_id, "status": "uploaded", "stage": "uploaded", "stage_index": 0, "stage_total": len(STAGES), "progress": 0.0, "message": "Video uploaded successfully", "filename": filename, "file_path": str(path.resolve()), "file_size": size, "output_dir": str((OUTPUT_ROOT / job_id).resolve()), "created_at": _now(), "updated_at": _now()}
    _save_job(job_id)
    return {"job_id": job_id, "filename": filename, "size_bytes": size}


@router.post("/upload-telemetry/{job_id}")
async def upload_telemetry(job_id: str, file: UploadFile = File(...)):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    filename = _safe_filename(file.filename, "telemetry.srt")
    if Path(filename).suffix.lower() not in _TELEMETRY_EXTENSIONS:
        raise HTTPException(400, "Telemetry must be .srt, .gpx, or .csv")
    path = UPLOAD_ROOT / job_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as target:
        while chunk := await file.read(1024 * 1024):
            target.write(chunk)
    await file.close()
    _set_job(job_id, telemetry_path=str(path.resolve()), telemetry_filename=filename)
    return {"job_id": job_id, "telemetry_filename": filename, "status": "telemetry_uploaded"}


@router.post("/reconstruct/{job_id}")
async def start_reconstruction(job_id: str, background_tasks: BackgroundTasks, options: Optional[ReconstructRequest] = None):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id].get("status") == "running" or job_id in _running_jobs:
        raise HTTPException(409, "Reconstruction is already running")
    video = Path(jobs[job_id].get("file_path", ""))
    if not video.is_file():
        raise HTTPException(404, "Uploaded video not found")
    opts = options or ReconstructRequest()
    telemetry = jobs[job_id].get("telemetry_path")
    if opts.telemetry_filename:
        candidate = video.parent / _safe_filename(opts.telemetry_filename, "telemetry.srt")
        if candidate.is_file():
            telemetry = str(candidate)
    config = PipelineConfig(input_video=str(video.resolve()), telemetry_path=telemetry, workspace_dir=str((ROOT / "data" / "workspace" / job_id).resolve()), output_dir=str((OUTPUT_ROOT / job_id).resolve()), skip_dynamic_masking=opts.skip_dynamic_masking, skip_depth_estimation=opts.skip_depth_estimation, skip_georeferencing=opts.skip_georeferencing, skip_analysis=opts.skip_analysis)
    config.video.target_fps = opts.target_fps
    config.sfm.mapper_backend = opts.mapper_backend
    config.cpu_threads = opts.cpu_threads
    config.reconstruction.openmvs_max_threads = opts.cpu_threads
    profiles = {
        "fast": {
            "max_frames": 200, "max_keyframes": 60, "resolution_level": 2,
            "max_resolution": 1920, "number_views": 4, "poisson_depth": 9,
            "texture_resolution": 4096,
        },
        "balanced": {
            "max_frames": 350, "max_keyframes": 120, "resolution_level": 1,
            "max_resolution": 2560, "number_views": 5, "poisson_depth": 10,
            "texture_resolution": 4096,
        },
        "quality": {
            "max_frames": 500, "max_keyframes": 180, "resolution_level": 1,
            "max_resolution": 2560, "number_views": 6, "poisson_depth": 10,
            "texture_resolution": 8192,
        },
    }
    profile = profiles[opts.processing_profile]
    config.video.max_frames = profile["max_frames"]
    config.keyframe.max_frames = profile["max_keyframes"]
    config.reconstruction.openmvs_resolution_level = profile["resolution_level"]
    config.reconstruction.openmvs_max_resolution = profile["max_resolution"]
    config.reconstruction.openmvs_number_views = profile["number_views"]
    config.reconstruction.openmvs_number_views_fuse = max(2, profile["number_views"] // 3)
    config.mesh.poisson_depth = profile["poisson_depth"]
    config.mesh.texture_resolution = profile["texture_resolution"]
    # Optimal keypoints and matching window for high-speed, high-density SfM
    config.sfm.max_keypoints = 8192
    if opts.processing_profile == "quality":
        config.sfm.match_window = 24
    elif opts.processing_profile == "balanced":
        config.sfm.match_window = 16
    else:
        config.sfm.match_window = 10
    config.validate()
    _set_job(job_id, status="queued", stage="queued", progress=0.0, mapper_backend=opts.mapper_backend, processing_profile=opts.processing_profile, cpu_threads=opts.cpu_threads)
    background_tasks.add_task(_run_pipeline_task, job_id, config)
    return {"job_id": job_id, "status": "started", "mapper_backend": opts.mapper_backend, "processing_profile": opts.processing_profile}


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    data = jobs[job_id]
    return JobStatus(job_id=job_id, status=data.get("status", "unknown"), stage=data.get("stage", "queued"), stage_index=data.get("stage_index", 0), progress=data.get("progress", 0.0), message=data.get("message", ""), started_at=data.get("started_at"), finished_at=data.get("finished_at"), mapper_backend=data.get("mapper_backend"))


@router.get("/jobs")
async def list_jobs():
    """Return safe job summaries for the web application's history view."""
    public_fields = (
        "job_id", "filename", "status", "stage", "stage_index", "stage_total",
        "progress", "message", "mapper_backend", "processing_profile", "cpu_threads", "file_size", "created_at",
        "updated_at", "started_at", "finished_at",
    )
    summaries = [
        {field: item.get(field) for field in public_fields if field in item}
        for item in jobs.values()
    ]
    summaries.sort(
        key=lambda item: item.get("updated_at") or item.get("created_at") or "",
        reverse=True,
    )
    return {"jobs": summaries, "count": len(summaries)}


@router.get("/job-details/{job_id}")
async def job_details(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    data = dict(jobs[job_id])
    video = Path(data.get("file_path", ""))
    if video.is_file():
        data["video_url"] = f"/data/uploads/{job_id}/{video.name}"
        data["file_size"] = video.stat().st_size
    return data


def _output_dir(job_id: str) -> Path:
    return Path(jobs.get(job_id, {}).get("output_dir", OUTPUT_ROOT / job_id)).resolve()


@router.get("/available/{job_id}")
async def available(job_id: str):
    directory = _output_dir(job_id)
    if not directory.exists():
        raise HTTPException(404, "Output directory not found")
    result = {}
    for key, filename in _DELIVERABLES.items():
        path = directory / filename
        ok = path.is_file() and path.stat().st_size > 0
        result[key] = {"filename": filename, "exists": ok, "size_bytes": path.stat().st_size if ok else 0}
    return result


@router.get("/download/{job_id}/{deliverable_type}")
async def download(job_id: str, deliverable_type: str):
    directory = _output_dir(job_id)
    filename = _DELIVERABLES.get(deliverable_type.lower())
    if not filename:
        raise HTTPException(400, f"Unsupported deliverable: {deliverable_type}")
    path = directory / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise HTTPException(404, f"Deliverable not available: {filename}")
    media_type = _MIME.get(path.suffix.lower().lstrip("."), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    return FileResponse(str(path), filename=filename, media_type=media_type)


@router.get("/points/{job_id}")
async def points(job_id: str):
    path = _output_dir(job_id) / "points.json"
    if not path.is_file():
        raise HTTPException(404, "Point data not available yet")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/measurements/{job_id}")
async def measurements(job_id: str):
    path = _output_dir(job_id) / "measurements.json"
    if not path.is_file():
        raise HTTPException(404, "Measurements not available yet")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/latest")
async def latest():
    candidates = []
    for directory in OUTPUT_ROOT.iterdir():
        if directory.is_dir() and ((directory / "manifest.json").exists() or (directory / "points.json").exists()):
            candidates.append((directory.stat().st_mtime, directory.name))
    return {"job_id": max(candidates)[1] if candidates else None}
