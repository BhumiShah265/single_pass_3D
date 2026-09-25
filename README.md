# AeroSynth 3D — Single-Pass Drone Video Reconstruction

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/viditshah5656/single_pass_3D?quickstart=1)

AeroSynth 3D is a **real photogrammetry pipeline** for turning an aerial drone video into a metric 3D reconstruction. The system is designed around a single continuous flight pass and uses real Structure-from-Motion and Multi-View Stereo backends rather than synthetic depth or fabricated accuracy values.

> **Important:** a single video pass cannot guarantee survey accuracy by software alone. Reconstruction quality depends on camera motion, overlap, texture, exposure, scene geometry, telemetry/GCP quality, and the native SfM/MVS toolchain.

## Pipeline

1. **Video ingestion** — metadata validation and controlled frame extraction.
2. **Quality filtering** — blur and exposure rejection.
3. **Keyframe selection** — motion/GPS-aware selection for useful baseline.
4. **Dynamic masking** — optional YOLO segmentation of transient objects.
5. **Structure from Motion** — SIFT + geometric matching with GLOMAP or pycolmap, camera calibration, poses and sparse points.
6. **Dense MVS** — native OpenMVS `InterfaceCOLMAP → DensifyPointCloud`.
7. **Surface reconstruction** — OpenMVS mesh/texturing with a Poisson fallback for explicitly supported workflows.
8. **Georeferencing** — real telemetry only; otherwise coordinates remain local metric coordinates.
9. **Analysis** — 3D measurements, GSD calculation and semantic annotations where available.
10. **Deliverable gate** — every declared artifact is checked for existence and non-zero size before a run can be reported as successful.

### Complete 3D view from a single pass

The Studio offers two separate geometry products. `model.glb` retains the photographed OpenMVS mesh. `model_complete.glb` is a **watertight inferred model** made from the same colored dense points: it aligns the cloud to the textured mesh before fitting a robust upper height field, interpolates small gaps, and closes the surface with a flat, earth-toned base. The Complete 3D view opens by default when this optional export exists; Textured remains available for inspection of the original reconstruction.

The inferred model is continuous and can be viewed from either side. The base and closed edges are display geometry; hidden walls, undercuts, and backside texture were not measured. Camera reprojection/GSD metrics do not describe inferred regions. `model_complete.json` records the grid spacing, interpolated cell count, base depth, and geometry checks.

The repository includes the [V5 Complete 3D sample](frontend/assets/v5_complete_2026-09-20_1811.glb), recovered from the September 20, 2026 18:11 local-time browser cache. It has 297,718 vertices, 595,432 triangles, and point-derived vertex colors (SHA-256: `42a3d413e0c731c4b5854d5247f295a0981b7f94be1af416443c084b8d8df93a`). Run the server and open `/studio?sample=v5` to inspect this exact model without the original V5 job data. The sample is an inferred surface; its unseen side is not a measured reconstruction. New reconstructions use the current generator and may produce different geometry and colors.

## Architecture

```text
Drone Video + optional SRT/GPX/CSV
              │
              ▼
       Frame Extraction
              │
              ▼
     Quality + Keyframes
              │
              ▼
     Dynamic Object Masks
              │
              ▼
   COLMAP Database / SIFT
              │
       ┌──────┴──────┐
       ▼             ▼
    GLOMAP       pycolmap
       └──────┬──────┘
              ▼
       Sparse Cameras
       + Tie Points
              │
              ▼
       InterfaceCOLMAP
              │
              ▼
     DensifyPointCloud
              │
              ▼
       Dense Point Cloud
              │
              ▼
       ReconstructMesh
              │
              ▼
         TextureMesh
              │
              ▼
     Analysis / Georeference
              │
              ▼
       Artifact Validation
              │
              ▼
   OBJ / GLB / FBX / PLY / LAS
   texture / DSM / raster / PDF
   points.json / measurements.json
   manifest.json
```

## GitHub Codespaces — start from zero

The repository has a reproducible `.devcontainer` and dedicated Codespaces scripts. Use this when you want a clean Linux development environment without configuring your Mac or Windows machine first.

### Launch

Click the Codespaces badge at the top of this README or use:

```text
https://codespaces.new/viditshah5656/single_pass_3D?quickstart=1
```

Choose **main**. Codespaces reads `.devcontainer/devcontainer.json` and runs the post-create installer automatically.

### First verification

```bash
python --version
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python -m compileall -q app tests
pytest -q
python -m app.main doctor --json
```

The Codespace is deliberately a **CPU development/validation target**. Use a native M4 machine to validate Apple MPS and a native NVIDIA machine to validate CUDA acceleration.

### Full Codespace smoke test

Start the API and run the bounded integration check:

```bash
python -m app.main serve --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
bash scripts/codespace-smoke.sh
```

Or, when the server is already running:

```bash
python scripts/smoke_api.py
```

Codespaces forwards port `8000`, so the same FastAPI process serves the local control room at `/` and the API at `/api/v1/*`.

### Native OpenMVS build

A fresh Codespace now builds the pinned OpenMVS native backend during post-create, so the dense pipeline is ready instead of silently stopping at sparse SfM. The four required tools are installed under `.local/bin`:

```text
InterfaceCOLMAP
DensifyPointCloud
ReconstructMesh
TextureMesh
```

The build is also safe to repeat:

```bash
bash scripts/build_openmvs_linux.sh
python -m app.main doctor --strict --json
bash scripts/verify_full_runtime.sh
```

If you intentionally need a Python/UI-only Codespace, set `OPENMVS_SKIP_BUILD=1` before running the post-create setup. That mode is not a full dense-reconstruction environment.

The repository also has a GitHub Actions `Full reconstruction runtime` workflow that builds the same pinned OpenMVS toolchain and runs the full runtime checks on Linux.

See [docs/CODESPACES.md](docs/CODESPACES.md) for the complete sequence and the distinction between Codespaces, M4/MPS, and NVIDIA/CUDA validation.

## Requirements

- Python 3.10 or 3.11
- Linux/Ubuntu, Windows 11 + WSL2, or macOS
- 8 GB RAM minimum; more is strongly recommended for dense MVS
- GPU is optional for the Python stages, but useful where supported
- **OpenMVS native binaries are required for dense reconstruction**
- GLOMAP is optional because pycolmap is available as a real SfM mapper fallback

Install Python dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\\Scripts\\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install/build OpenMVS so these executables are available on `PATH` or under `.local/bin` / `openMVS_build/bin`:

```text
InterfaceCOLMAP
DensifyPointCloud
ReconstructMesh
TextureMesh
```

If GLOMAP is used, its executable should be available as `glomap` on `PATH` or under `.local/bin`.

## Preflight

Before uploading a large video, query:

```bash
curl http://localhost:8000/api/v1/preflight
```

The response reports Python packages, OpenMVS binaries, GLOMAP/COLMAP discovery, compute device and writable workspace state.

## Web application

```bash
python -m app.main serve --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000
```

The public GitHub Pages control room is:

```text
https://viditshah5656.github.io/single_pass_3D/
```

GitHub Pages hosts the browser UI only. The heavy Python/COLMAP/OpenMVS reconstruction remains on the machine running the FastAPI backend. The web UI's **API base** field can be pointed at that backend, including a GitHub Codespaces forwarded port.

## CLI

```bash
python -m app.main run data/uploads/flight.mp4 \
  --mapper-backend glomap \
  --compute-backend auto \
  --target-fps 3 \
  --max-frames 450 \
  --output-dir data/output/run_01
```

Use pycolmap directly when GLOMAP is not available:

```bash
python -m app.main run data/uploads/flight.mp4 \
  --mapper-backend pycolmap \
  --compute-backend auto \
  --output-dir data/output/run_01
```

Optional telemetry:

```bash
python -m app.main run data/uploads/flight.mp4 \
  --telemetry-path data/uploads/flight.srt
```

## API

| Endpoint | Method | Purpose |
|---|---:|---|
| `/api/v1/health` | GET | Service health |
| `/api/v1/preflight` | GET | Runtime/backend readiness |
| `/api/v1/system-info` | GET | CPU/GPU/runtime information |
| `/api/v1/upload` | POST | Upload a drone video |
| `/api/v1/upload-telemetry/{job_id}` | POST | Attach SRT/GPX/CSV telemetry |
| `/api/v1/reconstruct/{job_id}` | POST | Start reconstruction |
| `/api/v1/status/{job_id}` | GET | Persistent stage/progress state |
| `/api/v1/job-details/{job_id}` | GET | Job metadata |
| `/api/v1/available/{job_id}` | GET | Artifact availability |
| `/api/v1/download/{job_id}/{type}` | GET | Download an artifact |
| `/api/v1/points/{job_id}` | GET | WebGL point/trajectory payload |
| `/api/v1/measurements/{job_id}` | GET | Computed measurements |
| `/api/v1/latest` | GET | Latest completed output |

Swagger UI is available at `/docs`.

## Output contract

A successful reconstruction must pass the artifact gate and produce non-empty files for:

- `model.obj`, `model.mtl`, `texture.jpg`
- `model.glb`
- `model.fbx`
- `cloud.ply`
- `cloud.las`
- `ortho.tif`
- `dsm.tif`
- `points.json`
- `measurements.json`
- `report.pdf`
- `final_report.json`
- `manifest.json`

`manifest.json` records the mapper/backend, registered views, point/mesh counts, reprojection error, georeferencing state and any missing artifact. A run with missing artifacts is **failed**, not successful.

### Coordinate and accuracy policy

Without real telemetry or ground-control points, the reconstruction remains in the SfM coordinate frame. The application does not invent GPS coordinates or claim an externally validated spatial accuracy.

A reported reprojection error is an internal SfM diagnostic; it is **not** equivalent to absolute survey accuracy.

`ortho.tif` is currently a derived texture raster from the reconstruction. It should not be interpreted as a rigorously rectified orthomosaic unless an actual orthorectification workflow has been supplied and independently validated.

## Testing

Run the unit/integrity tests with:

```bash
python -m pytest -q
```

Codespaces-specific checks:

```bash
bash scripts/codespace-check.sh
bash scripts/codespace-smoke.sh
```

Tests are designed to validate the no-fabrication contract, telemetry handling, model-directory discovery and export/backend behavior without requiring every native binary on every CI runner.

## Project layout

```text
.devcontainer/
  devcontainer.json
  post-create.sh

app/
  main.py
  config.py
  pipeline.py              # core 10-stage reconstruction engine
  pipeline_runtime.py      # artifact validation + manifest gate
  platform.py              # cross-platform host/compute diagnostics
  doctor.py                # dependency/native preflight
  api/routes.py
  video/
  preprocessing/
  reconstruction/
  geospatial/
  analysis/
frontend/
docs/
  CODESPACES.md
  UNIVERSAL_PLATFORM.md
scripts/
  build_openmvs_linux.sh
  codespace-check.sh
  codespace-smoke.sh
  smoke_api.py
data/
tests/
requirements.txt
```

## Design principle

The project favors **real geometry over impressive-looking placeholders**. When a native reconstruction backend is missing or the scene does not contain enough overlap/parallax for dense MVS, the correct result is an explicit failure with diagnostics—not a synthetic mesh presented as a successful reconstruction.
