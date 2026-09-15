import os
import shutil
import subprocess
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional, Any
import numpy as np
import open3d as o3d

from app.config import DEVICE, logger


class DenseReconstructor:
    """
    Dense Multi-View Stereo (MVS) reconstruction module.

    Source of truth: the OpenMVS CLI binaries (InterfaceCOLMAP, DensifyPointCloud,
    ReconstructMesh, TextureMesh) installed on the host OS. Their calibrated output
    is returned and validated as the dense deliverable.

    If the binaries are not installed, this module fails honestly with a clear
    message instead of fabricating a point cloud via arbitrary-pair stereo matching.
    """
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir)
        self.dense_dir = self.workspace_dir / "dense"
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        self.dense_ply = self.dense_dir / "scene_dense.ply"

    def is_openmvs_available(self) -> bool:
        """True only if all OpenMVS CLI binaries required for the full pipeline are on PATH."""
        return (
            shutil.which("InterfaceCOLMAP") is not None and
            shutil.which("DensifyPointCloud") is not None and
            shutil.which("ReconstructMesh") is not None and
            shutil.which("TextureMesh") is not None
        )

    def run_openmvs(self, use_cpu: bool = False) -> Tuple[Path, Path]:
        """
        Execute the native OpenMVS CLI pipeline on the calibrated COLMAP workspace.
        Returns (dense_ply_path, textured_mesh_obj_path). All subprocess steps
        use check=True so any CLI failure propagates as a hard error.
        """
        logger.info("Executing native OpenMVS dense reconstruction...")

        # 1. InterfaceCOLMAP : convert the calibrated COLMAP workspace to OpenMVS format
        mvs_scene = self.dense_dir / "scene.mvs"
        subprocess.run([
            "InterfaceCOLMAP",
            "-i", str(self.workspace_dir),
            "-o", str(mvs_scene),
            "--image-folder", str(self.workspace_dir / "images")
        ], check=True)

        # 2. DensifyPointCloud : true multi-view stereo depth fusion
        dense_mvs = self.dense_dir / "scene_dense.mvs"
        cpu_flag = ["--cuda-device", "-1"] if use_cpu else []
        subprocess.run([
            "DensifyPointCloud",
            str(mvs_scene),
            "-o", str(dense_mvs)
        ] + cpu_flag, check=True)

        # 3. ReconstructMesh : Poisson-ish surface from the fused dense cloud
        mesh_mvs = self.dense_dir / "scene_dense_mesh.mvs"
        subprocess.run([
            "ReconstructMesh",
            str(dense_mvs),
            "-o", str(mesh_mvs)
        ] + cpu_flag, check=True)

        # 4. TextureMesh : camera-based texture projection from the input views
        textured_obj = self.dense_dir / "scene_dense_mesh_refine_texture.obj"
        subprocess.run([
            "TextureMesh",
            str(mesh_mvs),
            "-o", str(self.dense_dir / "scene_dense_mesh_refine_texture.mvs"),
            "--export-type", "obj"
        ] + cpu_flag, check=True)

        # 5. Validate the OpenMVS outputs actually exist and are non-trivial
        if not self.dense_ply.exists() or self.dense_ply.stat().st_size == 0:
            raise RuntimeError(
                f"Dense reconstruction produced no usable point cloud: expected {self.dense_ply} "
                "but the file is missing or empty."
            )
        if not textured_obj.exists() or textured_obj.stat().st_size == 0:
            raise RuntimeError(
                f"Dense reconstruction produced no textured mesh: expected {textured_obj} "
                "but the file is missing or empty."
            )
        return self.dense_ply, textured_obj

    def reconstruct(
        self,
        keyframes: List[Path],
        camera_poses: List[Dict[str, Any]],
        camera_calibration: Dict[str, Any],
        sparse_points: Optional[List[List[float]]] = None,
        dynamic_mask_paths: Optional[List[Path]] = None,
        voxel_size: float = 0.12,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0
    ) -> Dict[str, Any]:
        """
        Master entry point for dense multi-view reconstruction.

        Only the native OpenMVS backend is used. If the OpenMVS CLI binaries are not
        installed on this host, this raises RuntimeError — it never falls back to
        arbitrary-pair stereo matching or any other fabricated dense geometry.
        """
        if not self.is_openmvs_available():
            raise RuntimeError(
                "Dense reconstruction backend unavailable: OpenMVS CLI binaries "
                "(InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, TextureMesh) were not found "
                "in the system PATH. Install OpenMVS to run real multi-view stereo reconstruction, "
                "or provide a pre-computed dense point cloud. No fallback dense engine is executed."
            )

        try:
            ply_path, mesh_path = self.run_openmvs(use_cpu=(DEVICE.type == "cpu"))
            pcd = o3d.io.read_point_cloud(str(ply_path))
            if len(pcd.points) == 0:
                raise RuntimeError(f"OpenMVS returned an empty point cloud: {ply_path}")
            return {
                "dense_ply_path": ply_path,
                "mesh_path": mesh_path,
                "pcd": pcd,
                "method": "OpenMVS_Native",
                "num_points": len(pcd.points)
            }
        except Exception as e:
            raise RuntimeError(f"Dense reconstruction backend failed: {e}") from e
