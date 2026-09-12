import argparse
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.api.routes import router
from app.pipeline import ReconstructionPipeline
from app.config import PipelineConfig, logger

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AeroSynth 3D Reconstruction API",
        description="Engineering Neural Photogrammetry Studio API",
        version="1.0.0"
    )
    app.include_router(router)
    
    # Mount frontend and static assets
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        assets_dir = frontend_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
        
        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(str(frontend_dir / "index.html"))

    return app

def run_server(host: str, port: int):
    """Run the FastAPI server using uvicorn."""
    logger.info(f"Starting server on {host}:{port}")
    app = create_app()
    uvicorn.run(app, host=host, port=port)

def run_cli(args):
    """Run the reconstruction pipeline via CLI."""
    logger.info(f"Starting CLI pipeline with video: {args.input_video}")
    config = PipelineConfig(
        input_video=args.input_video,
        output_dir=args.output_dir,
        skip_dynamic_masking=args.skip_dynamic_masking,
        skip_depth_estimation=args.skip_depth_estimation,
        skip_georeferencing=args.skip_georeferencing,
        skip_analysis=args.skip_analysis
    )
    pipeline = ReconstructionPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Video to 3D Model Reconstruction")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser("serve", help="Run the FastAPI server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    server_parser.add_argument("--port", type=int, default=8000, help="Port")
    
    # CLI command
    cli_parser = subparsers.add_parser("run", help="Run the pipeline via CLI")
    cli_parser.add_argument("input_video", type=str, help="Path to input video file")
    cli_parser.add_argument("--output_dir", type=str, default="data/output", help="Output directory")
    cli_parser.add_argument("--skip_dynamic_masking", action="store_true", help="Skip dynamic object masking")
    cli_parser.add_argument("--skip_depth_estimation", action="store_true", help="Skip dense depth reconstruction")
    cli_parser.add_argument("--skip_georeferencing", action="store_true", help="Skip georeferencing")
    cli_parser.add_argument("--skip_analysis", action="store_true", help="Skip analysis and measurements")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        run_server(args.host, args.port)
    elif args.command == "run":
        run_cli(args)
    else:
        parser.print_help()
