"""Operational wrapper around the reconstruction engine.

The legacy stage implementation remains responsible for geometry. This wrapper
adds hard validation at the boundary so a run cannot be reported as successful
when one of its declared artifacts is missing or empty.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.pipeline import ReconstructionPipeline as _CoreReconstructionPipeline
from app.config import logger


class ReconstructionPipeline(_CoreReconstructionPipeline):
    REQUIRED_ARTIFACTS = (
        "model.glb",
        "model.obj",
        "model.mtl",
        "texture.jpg",
        "cloud.ply",
    )

    def _validate_artifacts(self) -> list[str]:
        missing = []
        for name in self.REQUIRED_ARTIFACTS:
            path = self.output_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(name)
        return missing

    def _write_manifest(self, status: str, missing: list[str] | None = None) -> Path:
        manifest = {
            "schema_version": "1.0",
            "status": status,
            "fabricated": False,
            "input_video": str(self.config.input_video) if self.config.input_video else None,
            "mapper_backend": getattr(self, "mapper_backend_used", self.config.sfm.mapper_backend),
            "dense_backend": "OpenMVS" if not self.config.skip_depth_estimation else None,
            "registered_views": sum(1 for p in self.camera_trajectory if p.get("is_registered")),
            "sparse_points": len(self.sparse_points),
            "dense_points": len(self.dense_pcd.points) if self.dense_pcd is not None else 0,
            "mesh_vertices": len(self.mesh_vertices),
            "mesh_faces": len(self.mesh_faces),
            "reprojection_error_px": self.sfm_reprojection_error,
            "georeferenced": bool(self.georeferencer.is_georeferenced),
            "crs": self.georeferencer.crs_name,
            "missing_artifacts": missing or [],
            "artifacts": {
                name: {
                    "path": str(self.output_dir / name),
                    "bytes": (self.output_dir / name).stat().st_size if (self.output_dir / name).is_file() else 0,
                }
                for name in self.REQUIRED_ARTIFACTS
            },
        }
        complete_model = self.output_dir / "model_complete.glb"
        if complete_model.is_file() and complete_model.stat().st_size > 0:
            manifest["artifacts"]["model_complete.glb"] = {
                "path": str(complete_model), "bytes": complete_model.stat().st_size,
                "inferred": True,
            }
        path = self.output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    def run(self) -> dict[str, Any]:
        result = super().run()
        if result.get("status") != "success":
            return result
        missing = self._validate_artifacts()
        self._write_manifest("failed" if missing else "success", missing)
        if missing:
            message = "Reconstruction reached the export stage but artifact validation failed: " + ", ".join(missing)
            logger.error(message)
            return {"status": "error", "message": message, "output": str(self.output_dir)}
        logger.info("Artifact gate passed: %d required deliverables are non-empty.", len(self.REQUIRED_ARTIFACTS))
        return {"status": "success", "output": str(self.output_dir), "manifest": str(self.output_dir / "manifest.json")}
