"""CLI and FastAPI application entry point."""
from __future__ import annotations

# ── Force NVIDIA RTX GPU before any CUDA library loads ────────────────────────
# On laptops with NVIDIA Optimus (Intel iGPU + RTX), Windows reports the Intel
# GPU as GPU 1 in Task Manager. We must pin CUDA to PCI bus order and make only
# device 0 (RTX) visible so every library (torch, pycolmap, cv2 CUDA, OpenMVS)
# exclusively uses the dedicated GPU.
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")   # async GPU kernel launch
# OpenMVS / OpenCV CUDA thread hint
os.environ.setdefault("OMP_NUM_THREADS", "4")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
from pathlib import Path

import mimetypes
import uvicorn

mimetypes.add_type("image/webp", ".webp")
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import PipelineConfig, logger, SYSTEM_CPU_CORES

ROOT = Path(__file__).resolve().parent.parent


def create_app() -> FastAPI:
    app = FastAPI(title="AeroSynth 3D Reconstruction API", description="Single-pass aerial photogrammetry and metrology engine", version="5.1.0")
    app.include_router(router)
    frontend_dir = ROOT / "frontend"
    assets_dir = frontend_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    if frontend_dir.is_dir():
        app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(frontend_dir / "index.html"))

        page_files = {
            "reconstruct": "reconstruct.html",
            "jobs": "jobs.html",
            "outputs": "outputs.html",
            "system": "system.html",
            "studio": "studio.html",
        }
        def make_page_handler(filename: str):
            async def page() -> FileResponse:
                return FileResponse(str(frontend_dir / filename), headers={"Cache-Control": "no-store"})

            return page

        for route, filename in page_files.items():
            app.add_api_route(f"/{route}", make_page_handler(filename), include_in_schema=False)

    data_dir = ROOT / "data"
    if data_dir.is_dir():
        app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")
    return app


app = create_app()


def run_server(host: str, port: int, reload: bool = False) -> None:
    logger.info("Starting AeroSynth 3D server on %s:%s", host, port)
    options = {"host": host, "port": port, "reload": reload}
    if reload:
        options["reload_excludes"] = ["data/*", ".local/*", "openMVS_build/*", "*.ply", "*.obj", "*.bin"]
    uvicorn.run("app.main:app" if reload else app, **options)


def run_cli(args: argparse.Namespace) -> int:
    from app.pipeline_runtime import ReconstructionPipeline

    if args.command == "doctor":
        from app.doctor import run_doctor
        code, report = run_doctor(args.video, args.strict)
        if args.json:
            import json
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            summary = report["summary"]
            print(f"Host: {report['host']['os']} / {report['host']['machine']} / Python {report['host']['python']}")
            print(f"Full reconstruction ready: {'YES' if summary['full_reconstruction_ready'] else 'NO'}")
            print(f"Portable CPU/SfM path: {'YES' if summary['portable_cpu_path'] else 'NO'}")
            torch_info = report.get("torch", {})
            print(f"PyTorch: {torch_info.get('version', 'not installed')} | CUDA={torch_info.get('cuda_available', False)} | MPS={torch_info.get('mps_available', False)}")
            for name, item in report["native_backends"]["openmvs"].items():
                print(f"OpenMVS {name:18} {'OK' if item['found'] and item['executable'] else 'MISSING'}")
        return code

    config = PipelineConfig(
        input_video=str(Path(args.input_video).expanduser().resolve()),
        telemetry_path=args.telemetry_path,
        workspace_dir=args.workspace_dir,
        output_dir=args.output_dir,
        compute_backend=args.compute_backend,
        cpu_threads=args.cpu_threads,
        skip_dynamic_masking=args.skip_dynamic_masking,
        skip_depth_estimation=args.skip_depth_estimation,
        skip_georeferencing=args.skip_georeferencing,
        skip_analysis=args.skip_analysis,
    )
    config.reconstruction.openmvs_max_threads = args.cpu_threads
    config.video.target_fps = args.target_fps
    config.video.max_frames = args.max_frames
    config.sfm.mapper_backend = args.mapper_backend
    config.validate()
    # Resolve once so an explicit request fails before frame extraction starts.
    config.resolved_device()
    return 0 if ReconstructionPipeline(config).run().get("status") == "success" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-pass drone video to 3D reconstruction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the FastAPI engineering studio")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Check cross-platform runtime and native reconstruction prerequisites")
    doctor.add_argument("video", nargs="?", help="Optional drone video to validate")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--strict", action="store_true")

    run = subparsers.add_parser("run", help="Run a reconstruction directly")
    run.add_argument("input_video", help="Path to an aerial drone video")
    run.add_argument("--telemetry-path", default=None)
    run.add_argument("--mapper-backend", choices=["glomap", "pycolmap"], default="pycolmap")
    try:
        import torch
        default_backend = "cuda" if torch.cuda.is_available() else "auto"
    except Exception:
        default_backend = "auto"
    run.add_argument("--compute-backend", choices=["auto", "cuda", "mps", "cpu"], default=default_backend)
    run.add_argument("--cpu-threads", type=int, default=SYSTEM_CPU_CORES)
    run.add_argument("--target-fps", type=float, default=3.0)
    run.add_argument("--max-frames", type=int, default=450)
    run.add_argument("--workspace-dir", default="data/workspace")
    run.add_argument("--output-dir", default="data/output")
    run.add_argument("--skip-dynamic-masking", action="store_true")
    run.add_argument("--skip-depth-estimation", action="store_true")
    run.add_argument("--skip-georeferencing", action="store_true")
    run.add_argument("--skip-analysis", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.command == "serve":
        run_server(args.host, args.port, args.reload)
    else:
        raise SystemExit(run_cli(args))
