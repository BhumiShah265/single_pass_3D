from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
from pathlib import Path
import uuid
import os
import json
import torch
import mimetypes
import shutil
import numpy as np
from PIL import Image, ImageOps

from app.config import PipelineConfig, DEVICE, logger
from app.pipeline import ReconstructionPipeline

router = APIRouter(prefix="/api/v1", tags=["reconstruction"])

# In-memory job store
jobs: Dict[str, dict] = {}

class JobStatus(BaseModel):
    job_id: str
    status: str
    stage: Optional[str] = "queued"
    message: str = ""

class ReconstructRequest(BaseModel):
    skip_dynamic_masking: bool = False
    skip_depth_estimation: bool = False
    skip_georeferencing: bool = False
    skip_analysis: bool = False
    target_fps: Optional[float] = 4.0
    mapper_backend: Optional[str] = "glomap"
    telemetry_filename: Optional[str] = None

MIME_MAP = {
    "glb": "model/gltf-binary",
    "gltf": "model/gltf+json",
    "obj": "text/plain",
    "mtl": "text/plain",
    "ply": "application/octet-stream",
    "las": "application/octet-stream",
    "fbx": "application/octet-stream",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "pdf": "application/pdf",
    "json": "application/json",
    "jpg": "image/jpeg",
    "png": "image/png"
}

def get_mime_type(ext: str) -> str:
    ext_clean = ext.lower().lstrip(".")
    return MIME_MAP.get(ext_clean, mimetypes.guess_type(f"file.{ext_clean}")[0] or "application/octet-stream")

@router.get("/system-info")
async def get_system_info():
    """Return hardware and compute device information."""
    if torch.cuda.is_available():
        device_name = f"NVIDIA {torch.cuda.get_device_name(0)}"
        vram = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "Apple Silicon Neural Engine (MPS)"
        vram = "Unified Memory"
    else:
        device_name = "Host CPU (Multi-core)"
        vram = "System RAM"

    return {
        "device": str(DEVICE),
        "device_name": device_name,
        "vram": vram,
        "pipeline_version": "4.0.0"
    }

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video for reconstruction."""
    job_id = str(uuid.uuid4())
    upload_dir = Path("data/uploads") / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    size_bytes = 0
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB buffer
            f.write(chunk)
            size_bytes += len(chunk)
        
    jobs[job_id] = {
        "status": "uploaded", 
        "stage": "uploaded",
        "file_path": str(file_path),
        "filename": file.filename,
        "file_size": size_bytes,
        "output_dir": f"data/output/{job_id}"
    }
    logger.info(f"Video uploaded successfully for job {job_id}: {file.filename} ({size_bytes} bytes)")
    return {
        "job_id": job_id, 
        "filename": file.filename,
        "size_bytes": size_bytes,
        "message": "Video uploaded successfully"
    }

@router.post("/upload-telemetry/{job_id}")
async def upload_telemetry(job_id: str, file: UploadFile = File(...)):
    """Upload an explicit telemetry log (.srt, .gpx, .csv) for a job."""
    upload_dir = Path("data/uploads") / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    telemetry_path = upload_dir / file.filename
    with open(telemetry_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
            
    if job_id not in jobs:
        jobs[job_id] = {"status": "uploaded", "output_dir": f"data/output/{job_id}"}
        
    jobs[job_id]["telemetry_path"] = str(telemetry_path)
    logger.info(f"Telemetry uploaded for job {job_id}: {file.filename}")
    return {"job_id": job_id, "telemetry_filename": file.filename, "status": "telemetry_uploaded"}

def run_pipeline_task(job_id: str, config: PipelineConfig):
    """Background task to run the reconstruction pipeline with progress tracking."""
    if job_id not in jobs:
        jobs[job_id] = {
            "status": "running",
            "stage": "video_extraction",
            "output_dir": config.output_dir
        }
    else:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["stage"] = "video_extraction"
        jobs[job_id]["output_dir"] = config.output_dir
        
    logger.info(f"Starting pipeline execution for job {job_id}")
    try:
        pipeline = ReconstructionPipeline(config)
        result = pipeline.run()
        
        if result.get("status") == "success":
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["stage"] = "completed"
            jobs[job_id]["output_dir"] = result.get("output", config.output_dir)
            logger.info(f"Job {job_id} completed successfully")
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["stage"] = "failed"
            jobs[job_id]["message"] = result.get("message", "Pipeline execution failed")
            logger.error(f"Job {job_id} failed: {jobs[job_id]['message']}")
    except Exception as e:
        logger.exception(f"Unhandled exception in pipeline for job {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["stage"] = "failed"
        jobs[job_id]["message"] = str(e)

@router.post("/reconstruct/{job_id}")
async def start_reconstruction(
    job_id: str, 
    background_tasks: BackgroundTasks,
    options: Optional[ReconstructRequest] = None
):
    """Start the 3D reconstruction process."""
    upload_dir = Path("data/uploads") / job_id
    file_path = None
    telemetry_path = None
    
    if job_id in jobs and "file_path" in jobs[job_id]:
        file_path = jobs[job_id]["file_path"]
        telemetry_path = jobs[job_id].get("telemetry_path")
    elif upload_dir.exists():
        files = list(upload_dir.glob("*.mp4")) + list(upload_dir.glob("*.mov")) + list(upload_dir.glob("*.avi"))
        if files:
            file_path = str(files[0])
            jobs[job_id] = {
                "status": "uploaded",
                "file_path": file_path,
                "filename": files[0].name,
                "output_dir": f"data/output/{job_id}"
            }
        tfiles = list(upload_dir.glob("*.srt")) + list(upload_dir.glob("*.gpx")) + list(upload_dir.glob("*.csv"))
        if tfiles:
            telemetry_path = str(tfiles[0])
            jobs[job_id]["telemetry_path"] = telemetry_path
            
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Uploaded video not found for this job")
        
    opts = options or ReconstructRequest()
    config = PipelineConfig(
        input_video=file_path,
        telemetry_path=telemetry_path,
        workspace_dir=f"data/workspace/{job_id}",
        output_dir=f"data/output/{job_id}",
        skip_dynamic_masking=opts.skip_dynamic_masking,
        skip_depth_estimation=opts.skip_depth_estimation,
        skip_georeferencing=opts.skip_georeferencing,
        skip_analysis=opts.skip_analysis
    )
    if opts.mapper_backend:
        config.sfm.mapper_backend = opts.mapper_backend
        
    if opts.target_fps and opts.target_fps > 0:
        config.video.target_fps = float(opts.target_fps)
    
    background_tasks.add_task(run_pipeline_task, job_id, config)
    logger.info(f"Queued reconstruction background task for job {job_id} using {config.sfm.mapper_backend}")
    return {"job_id": job_id, "status": "started", "mapper_backend": config.sfm.mapper_backend}

@router.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Poll job status."""
    if job_id not in jobs:
        # Check if output already exists on disk
        out_dir = Path("data/output") / job_id
        if out_dir.exists() and (out_dir / "points.json").exists():
            return JobStatus(job_id=job_id, status="completed", message="Pipeline finished")
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    return JobStatus(job_id=job_id, status=job.get("status", "unknown"), message=job.get("message", ""))

@router.get("/download/{job_id}/{deliverable_type}")
async def download_deliverable(job_id: str, deliverable_type: str):
    """Download deliverables (OBJ, PLY, LAS, GLB, FBX, GeoTIFF, DSM, Report)."""
    output_dir = Path(jobs.get(job_id, {}).get("output_dir", f"data/output/{job_id}"))
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Output directory not found for job '{job_id}'")
        
    file_map = {
        "mesh": "model.obj",
        "obj": "model.obj",
        "point_cloud": "cloud.ply",
        "ply": "cloud.ply",
        "las": "cloud.las",
        "glb": "model.glb",
        "gltf": "model.glb",
        "fbx": "model.fbx",
        "orthophoto": "ortho.tif",
        "ortho": "ortho.tif",
        "geotiff": "ortho.tif",
        "dsm": "dsm.tif",
        "report": "report.pdf",
        "pdf": "report.pdf"
    }
    
    dt_key = deliverable_type.lower()
    if dt_key not in file_map:
        raise HTTPException(status_code=400, detail=f"Invalid deliverable type '{deliverable_type}'. Supported: {list(file_map.keys())}")
        
    filename = file_map[dt_key]
    file_path = output_dir / filename
    
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise HTTPException(status_code=404, detail=f"Deliverable '{deliverable_type}' ({filename}) not found or empty")

    mime_type = get_mime_type(file_path.suffix)
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=mime_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@router.get("/available/{job_id}")
async def list_available_deliverables(job_id: str):
    """Return a JSON mapping of each deliverable type to its existence and file size."""
    output_dir = Path(jobs.get(job_id, {}).get("output_dir", f"data/output/{job_id}"))
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' output not found")
        
    file_map = {
        "obj": "model.obj",
        "ply": "cloud.ply",
        "las": "cloud.las",
        "glb": "model.glb",
        "gltf": "model.glb",
        "fbx": "model.fbx",
        "orthophoto": "ortho.tif",
        "ortho": "ortho.tif",
        "geotiff": "ortho.tif",
        "dsm": "dsm.tif",
        "report": "report.pdf",
        "pdf": "report.pdf"
    }
    result = {}
    for key, fname in file_map.items():
        fpath = output_dir / fname
        exists = fpath.is_file() and fpath.stat().st_size > 0
        result[key] = {
            "filename": fname,
            "exists": exists,
            "size_bytes": fpath.stat().st_size if exists else 0
        }
    return result

@router.get("/points/{job_id}")
async def get_reconstructed_points(job_id: str):
    """Get 3D points, colors, and camera trajectory for a completed reconstruction."""
    output_dir = Path(jobs.get(job_id, {}).get("output_dir", f"data/output/{job_id}"))
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Output directory not found for job '{job_id}'")
        
    points_file = output_dir / "points.json"
    if points_file.exists():
        with open(points_file, "r") as f:
            return json.load(f)
            
    return {"points": [], "trajectory": [], "measurements": {}}

@router.get("/measurements/{job_id}")
async def get_measurements(job_id: str):
    """Get calculated measurements for a completed reconstruction."""
    output_dir = Path(jobs.get(job_id, {}).get("output_dir", f"data/output/{job_id}"))
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Output directory not found for job '{job_id}'")
        
    meas_file = output_dir / "measurements.json"
    if meas_file.exists():
        with open(meas_file, "r") as f:
            return json.load(f)

    raise HTTPException(status_code=404, detail="Measurements not found for this job. Awaiting reconstruction.")

@router.get("/latest")
async def get_latest_job():
    """Return the most recently completed job id from data/output."""
    output_base = Path("data/output")
    if not output_base.exists():
        return {"job_id": None}
    
    candidates = []
    for p in output_base.iterdir():
        if p.is_dir() and ((p / "model.glb").exists() or (p / "points.json").exists()):
            candidates.append((p.stat().st_mtime, p.name))
            
    if not candidates:
        return {"job_id": None}
        
    candidates.sort(reverse=True)
    return {"job_id": candidates[0][1]}

@router.get("/job-details/{job_id}")
async def get_job_details(job_id: str):
    """Return video metadata and reconstruction status for a specific job."""
    upload_dir = Path("data/uploads") / job_id
    output_dir = Path("data/output") / job_id
    
    filename = "Unknown Video"
    video_url = None
    file_size = 0
    width = 1920
    height = 1080
    duration_s = 20.0
    fps = 30.0
    
    if upload_dir.exists():
        vfiles = list(upload_dir.glob("*.mp4")) + list(upload_dir.glob("*.mov")) + list(upload_dir.glob("*.avi")) + list(upload_dir.glob("*.mkv"))
        if vfiles:
            target_vid = vfiles[0]
            filename = target_vid.name
            file_size = target_vid.stat().st_size
            video_url = f"/data/uploads/{job_id}/{filename}"
            try:
                import cv2
                cap = cv2.VideoCapture(str(target_vid))
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration_s = round(frames / fps, 1) if fps > 0 else 20.0
                    cap.release()
            except Exception:
                pass

    return {
        "job_id": job_id,
        "filename": filename,
        "video_url": video_url,
        "file_size": file_size,
        "width": width,
        "height": height,
        "duration_s": duration_s,
        "fps": round(fps, 1),
        "has_model": (output_dir / "model.glb").exists(),
        "has_points": (output_dir / "points.json").exists()
    }

@router.get("/diagnostics/{job_id}")
async def get_formation_diagnostics(job_id: str):
    """Expose real previews from the completed reconstruction workspace."""
    out_dir = Path("data/output") / job_id
    workspace_dir = Path("data/workspace") / job_id
    info_file = out_dir / "diagnostic_info.json"
    info = {}
    if info_file.exists():
        with open(info_file, "r") as f:
            try:
                info = json.load(f)
            except Exception:
                pass

    source_candidates = sorted((workspace_dir / "images").glob("*.png"))
    source_path = source_candidates[0] if source_candidates else None
    source_preview = out_dir / "diagnostic_source_keyframe.jpg"
    mask_preview = out_dir / "diagnostic_mask_overlay.jpg"
    texture_preview = out_dir / "diagnostic_texture_atlas.jpg"
    dsm_preview = out_dir / "diagnostic_dsm.jpg"

    if source_path and not source_preview.exists():
        source = Image.open(source_path).convert("RGB")
        source.save(source_preview, quality=92)

        mask_path = workspace_dir / "masks" / f"{source_path.stem}_mask.png"
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L").resize(source.size)
            red = Image.new("RGB", source.size, (255, 80, 45))
            highlighted = Image.composite(red, source, mask)
            Image.blend(source, highlighted, 0.45).save(mask_preview, quality=92)

    texture_path = out_dir / "texture.jpg"
    if texture_path.exists() and not texture_preview.exists():
        shutil.copy2(texture_path, texture_preview)

    dsm_path = out_dir / "dsm.tif"
    if dsm_path.exists() and not dsm_preview.exists():
        try:
            import rasterio
            with rasterio.open(dsm_path) as raster:
                values = raster.read(1).astype(np.float32)
            finite = np.isfinite(values)
            if finite.any():
                lo, hi = np.percentile(values[finite], [2, 98])
                normalized = np.clip((values - lo) / max(1e-6, hi - lo), 0, 1)
                gray = Image.fromarray((normalized * 255).astype(np.uint8), mode="L")
                ImageOps.colorize(gray, black="#123b66", mid="#36a87a", white="#f4d35e").save(
                    dsm_preview, quality=92
                )
        except Exception as exc:
            logger.warning(f"DSM diagnostic preview notice: {exc}")

    return {
        "job_id": job_id,
        "source_keyframe_url": f"/data/output/{job_id}/diagnostic_source_keyframe.jpg" if source_preview.exists() else None,
        "mask_overlay_url": f"/data/output/{job_id}/diagnostic_mask_overlay.jpg" if mask_preview.exists() else None,
        "texture_atlas_url": f"/data/output/{job_id}/diagnostic_texture_atlas.jpg" if texture_preview.exists() else None,
        "dsm_url": f"/data/output/{job_id}/diagnostic_dsm.jpg" if dsm_preview.exists() else None,
        "info": info
    }
