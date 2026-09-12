from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path
import uuid
import os

from app.config import PipelineConfig, logger
from app.pipeline import ReconstructionPipeline

router = APIRouter(prefix="/api/v1", tags=["reconstruction"])

# In-memory job store for demo purposes
jobs: Dict[str, dict] = {}

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str = ""

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video for reconstruction."""
    job_id = str(uuid.uuid4())
    upload_dir = Path("data/uploads") / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    jobs[job_id] = {"status": "uploaded", "file_path": str(file_path)}
    return {"job_id": job_id, "message": "Video uploaded successfully"}

def run_pipeline_task(job_id: str, config: PipelineConfig):
    """Background task to run the pipeline."""
    jobs[job_id]["status"] = "running"
    pipeline = ReconstructionPipeline(config)
    result = pipeline.run()
    
    if result["status"] == "success":
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_dir"] = result["output"]
    else:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = result.get("message", "Unknown error")

@router.post("/reconstruct/{job_id}")
async def start_reconstruction(job_id: str, background_tasks: BackgroundTasks):
    """Start the 3D reconstruction process."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    config = PipelineConfig(
        input_video=jobs[job_id]["file_path"],
        workspace_dir=f"data/workspace/{job_id}",
        output_dir=f"data/output/{job_id}"
    )
    
    background_tasks.add_task(run_pipeline_task, job_id, config)
    return {"job_id": job_id, "status": "started"}

@router.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Poll job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return JobStatus(job_id=job_id, status=job["status"], message=job.get("message", ""))

@router.get("/download/{job_id}/{deliverable_type}")
async def download_deliverable(job_id: str, deliverable_type: str):
    """Download deliverables (mesh, point_cloud, orthophoto, dsm, report)."""
    if job_id not in jobs or jobs[job_id].get("status") != "completed":
        raise HTTPException(status_code=404, detail="Deliverable not ready or job not found")
        
    output_dir = Path(jobs[job_id]["output_dir"])
    
    file_map = {
        "mesh": "model.obj",
        "point_cloud": "cloud.ply",
        "orthophoto": "ortho.tif",
        "dsm": "dsm.tif",
        "report": "report.pdf"
    }
    
    if deliverable_type not in file_map:
        raise HTTPException(status_code=400, detail="Invalid deliverable type")
        
    file_path = output_dir / file_map[deliverable_type]
    
    # Check if we should fake the file for development/testing if it doesn't exist
    if not file_path.exists():
        # In a real app we'd throw 404, but to prevent errors before tools exist
        # we could mock it. We'll raise 404 here properly.
        raise HTTPException(status_code=404, detail="File not found on disk")
        
    return FileResponse(path=file_path, filename=file_map[deliverable_type])

@router.get("/measurements/{job_id}")
async def get_measurements(job_id: str):
    """Get calculated measurements for a completed reconstruction."""
    if job_id not in jobs or jobs[job_id].get("status") != "completed":
        raise HTTPException(status_code=404, detail="Measurements not ready or job not found")
        
    # Return placeholder measurements
    return {
        "volume_m3": 123.45, 
        "surface_area_m2": 67.89,
        "bounding_box": {
            "min": [0.0, 0.0, 0.0],
            "max": [10.0, 10.0, 5.0]
        }
    }
