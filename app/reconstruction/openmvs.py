import os
import shutil
import subprocess
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional, Any
import numpy as np
import open3d as o3d

from app.config import DEVICE, logger


def find_openmvs_binary(name: str) -> Optional[str]:
    """Auto-detect OpenMVS CLI binary on PATH, openMVS_build/bin, or .local/bin."""
    sys_path = shutil.which(name)
    if sys_path:
        return sys_path
    
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        repo_root / "openMVS_build" / "bin" / name,
        repo_root / ".local" / "bin" / name,
        Path("/usr/local/bin") / name
    ]
    for cand in candidates:
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


class DenseReconstructor:
    """
    Dense Multi-View Stereo (MVS) reconstruction module.

    Source of truth: the OpenMVS CLI binaries (InterfaceCOLMAP, DensifyPointCloud,
    ReconstructMesh, TextureMesh) installed on the host OS or built in openMVS_build/bin.
    Their calibrated output is returned and validated as the dense deliverable.

    If the binaries are not installed, this module fails honestly with a clear
    message instead of fabricating a point cloud via arbitrary-pair stereo matching.
    """
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.dense_dir = (self.workspace_dir / "dense").resolve()
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        self.dense_ply = self.dense_dir / "scene_dense.ply"

    def is_openmvs_available(self) -> bool:
        """True only if all OpenMVS CLI binaries required for the full pipeline are available."""
        return (
            find_openmvs_binary("InterfaceCOLMAP") is not None and
            find_openmvs_binary("DensifyPointCloud") is not None and
            find_openmvs_binary("ReconstructMesh") is not None and
            find_openmvs_binary("TextureMesh") is not None
        )

    def _prepare_openmvs_inputs(
        self, sparse_dir: Path, image_dir: Path
    ) -> Tuple[Path, Path]:
        """Return a PINHOLE sparse model and matching images for OpenMVS.

        InterfaceCOLMAP does not accept distorted SIMPLE_RADIAL cameras. GLOMAP
        commonly exports that model, so undistort it with pycolmap first.
        """
        import pycolmap

        sparse_dir = Path(sparse_dir).resolve()
        image_dir = Path(image_dir).resolve()

        required_groups = (
            ("cameras.bin", "cameras.txt"),
            ("images.bin", "images.txt"),
            ("points3D.bin", "points3D.txt"),
        )
        if not all(any((sparse_dir / name).exists() for name in group)
                   for group in required_groups):
            raise RuntimeError(
                f"COLMAP sparse model is incomplete or missing: {sparse_dir}"
            )

        reconstruction = pycolmap.Reconstruction(str(sparse_dir))
        camera_models = {camera.model.name for camera in reconstruction.cameras.values()}
        if camera_models and camera_models <= {"PINHOLE"}:
            return sparse_dir, image_dir

        undistorted_root = (self.dense_dir / "undistorted").resolve()
        if undistorted_root.exists():
            shutil.rmtree(undistorted_root)
        logger.info(
            "OpenMVS requires PINHOLE cameras; undistorting the GLOMAP model "
            f"from {sorted(camera_models)}"
        )
        pycolmap.undistort_images(
            output_path=str(undistorted_root),
            input_path=str(sparse_dir),
            image_path=str(image_dir),
            output_type="COLMAP",
        )

        undistorted_sparse = (undistorted_root / "sparse").resolve()
        undistorted_images = (undistorted_root / "images").resolve()
        if not undistorted_sparse.exists() or not undistorted_images.exists():
            raise RuntimeError(
                "pycolmap undistortion did not produce the required COLMAP "
                "sparse and images directories."
            )
        return undistorted_sparse, undistorted_images

    def run_openmvs(
        self, 
        sparse_dir: Optional[Path] = None, 
        image_dir: Optional[Path] = None, 
        use_cpu: bool = False
    ) -> Tuple[Path, Path, Path, Path]:
        """
        Execute the native OpenMVS CLI pipeline on the calibrated COLMAP workspace.
        Returns (dense_ply_path, textured_mesh_obj_path, mtl_path, texture_img_path).
        All subprocess steps use check=True so any CLI failure propagates as a hard error.
        """
        logger.info("Executing native OpenMVS dense reconstruction...")

        interface_bin = find_openmvs_binary("InterfaceCOLMAP")
        densify_bin = find_openmvs_binary("DensifyPointCloud")
        mesh_bin = find_openmvs_binary("ReconstructMesh")
        texture_bin = find_openmvs_binary("TextureMesh")

        if not all([interface_bin, densify_bin, mesh_bin, texture_bin]):
            raise RuntimeError("One or more required OpenMVS binaries (InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, TextureMesh) are missing.")

        if sparse_dir is None:
            sparse_0 = self.workspace_dir / "sparse" / "0"
            sparse_dir = (sparse_0 if sparse_0.exists() else self.workspace_dir / "sparse").resolve()
        else:
            sparse_dir = Path(sparse_dir).resolve()
            
        if image_dir is None:
            image_dir = (self.workspace_dir / "images").resolve()
        else:
            image_dir = Path(image_dir).resolve()

        logger.info(f"OpenMVS reading sparse model from {sparse_dir} and images from {image_dir}")
        sparse_dir, image_dir = self._prepare_openmvs_inputs(
            Path(sparse_dir), Path(image_dir)
        )

        # Do not accept stale dense artifacts from an earlier attempt.
        stale_outputs = [
            self.dense_ply,
            self.dense_dir / "scene_dense.mvs",
            self.dense_dir / "scene_dense_mesh.ply",
            self.dense_dir / "scene_dense_mesh_refine_texture.obj",
            self.dense_dir / "scene_dense_mesh_refine_texture.mtl",
        ]
        stale_outputs.extend(self.dense_dir.glob("scene_dense_mesh_refine_texture*.png"))
        stale_outputs.extend(self.dense_dir.glob("scene_dense_mesh_refine_texture*.jpg"))
        for output_path in stale_outputs:
            if output_path.exists():
                output_path.unlink()

        # InterfaceCOLMAP expects a COLMAP project root containing sparse/.
        # GLOMAP normally writes the model into sparse/0, so stage a small
        # self-contained project rather than passing sparse/0 directly.
        interface_workspace = (self.dense_dir / "colmap_input").resolve()
        interface_sparse = (interface_workspace / "sparse").resolve()
        interface_images = (interface_workspace / "images").resolve()
        
        if (interface_workspace / "images").is_symlink():
            (interface_workspace / "images").unlink()
        if interface_workspace.exists():
            shutil.rmtree(interface_workspace)
            
        interface_sparse.mkdir(parents=True, exist_ok=True)
        for model_file in Path(sparse_dir).iterdir():
            if model_file.is_file() and model_file.suffix.lower() in {".bin", ".txt"}:
                shutil.copy2(model_file, interface_sparse / model_file.name)
        if not any((interface_sparse / name).exists() for name in ("cameras.bin", "cameras.txt")):
            raise RuntimeError(f"COLMAP sparse model has no cameras file: {sparse_dir}")

        if interface_images.is_symlink() or os.path.islink(str(interface_images)):
            interface_images.unlink()
        elif interface_images.exists():
            if interface_images.is_dir():
                shutil.rmtree(interface_images)
            else:
                interface_images.unlink()
                
        if interface_images != Path(image_dir).resolve():
            try:
                interface_images.symlink_to(Path(image_dir).resolve(), target_is_directory=True)
            except Exception:
                shutil.copytree(Path(image_dir).resolve(), interface_images)

        # 1. InterfaceCOLMAP : convert the calibrated COLMAP workspace to OpenMVS format
        mvs_scene = (self.dense_dir / "scene.mvs").resolve()
        cmd_interface = [
            interface_bin,
            "-i", str(interface_workspace),
            "-o", str(mvs_scene),
            "--image-folder", str(interface_images)
        ]
        subprocess.run(cmd_interface, check=True, cwd=str(self.dense_dir))

        # 2. DensifyPointCloud : true multi-view stereo depth fusion
        dense_mvs = self.dense_dir / "scene_dense.mvs"
        # OpenMVS 2.4.0 does not accept the historical --cuda-device option.
        # This Apple Silicon build uses its available CPU path by default.
        cpu_flag: List[str] = []
        cmd_densify = [
            densify_bin,
            str(mvs_scene),
            "-o", str(dense_mvs)
        ] + cpu_flag
        subprocess.run(cmd_densify, check=True, cwd=str(self.dense_dir))

        if not self.dense_ply.exists() or self.dense_ply.stat().st_size == 0:
            # OpenMVS can return success while producing an empty MVS scene.
            # Stop before ReconstructMesh turns the sparse points into a
            # misleading mesh and expose the useful diagnostic from its log.
            log_detail = ""
            densify_logs = list(self.dense_dir.glob("DensifyPointCloud-*.log"))
            if densify_logs:
                latest_log = max(densify_logs, key=lambda path: path.stat().st_mtime)
                try:
                    lines = latest_log.read_text(errors="replace").splitlines()
                    relevant = [
                        line.strip()
                        for line in lines
                        if "Densifying point-cloud completed:" in line
                        or "Depth-maps dense fused" in line
                        or "paired with 0 views" in line
                    ]
                    if relevant:
                        log_detail = " OpenMVS log: " + " | ".join(relevant[-4:])
                except OSError:
                    pass
            raise RuntimeError(
                "OpenMVS DensifyPointCloud produced no dense point cloud. "
                "The sparse views do not contain enough textured overlap and camera "
                "translation/parallax for dense MVS; use footage with more side/forward "
                "motion over textured ground rather than mostly smooth water or rotation."
                + log_detail
            )

        # 3. ReconstructMesh : surface from the fused dense cloud. OpenMVS
        # exports a PLY mesh by default; the output extension is not enough to
        # change that behavior.
        mesh_ply = self.dense_dir / "scene_dense_mesh.ply"
        cmd_mesh = [
            mesh_bin,
            str(dense_mvs),
            "-o", str(mesh_ply),
            "--export-type", "ply"
        ] + cpu_flag
        subprocess.run(cmd_mesh, check=True, cwd=str(self.dense_dir))

        # 4. TextureMesh : camera-based texture projection from the input views
        textured_obj = self.dense_dir / "scene_dense_mesh_refine_texture.obj"
        cmd_texture = [
            texture_bin,
            "-i", str(dense_mvs),
            "-m", str(mesh_ply),
            "-o", str(textured_obj),
            "--export-type", "obj"
        ] + cpu_flag
        subprocess.run(cmd_texture, check=True, cwd=str(self.dense_dir))

        # Output paths
        textured_mtl = self.dense_dir / "scene_dense_mesh_refine_texture.mtl"
        
        # Locate generated texture image (can be .png or .jpg)
        texture_images = list(self.dense_dir.glob("scene_dense_mesh_refine_texture*.png")) + \
                         list(self.dense_dir.glob("scene_dense_mesh_refine_texture*.jpg")) + \
                         list(self.dense_dir.glob("*.png")) + list(self.dense_dir.glob("*.jpg"))
        
        texture_img = texture_images[0] if texture_images else (self.dense_dir / "scene_dense_mesh_refine_texture.png")

        # 5. Validate the OpenMVS outputs actually exist and are non-empty
        if not self.dense_ply.exists() or self.dense_ply.stat().st_size == 0:
            raise RuntimeError(
                f"OpenMVS DensifyPointCloud failed: expected {self.dense_ply} but file is missing or empty."
            )
        if not textured_obj.exists() or textured_obj.stat().st_size == 0:
            raise RuntimeError(
                f"OpenMVS TextureMesh failed: expected {textured_obj} but file is missing or empty."
            )
        if not textured_mtl.exists() or textured_mtl.stat().st_size == 0:
            raise RuntimeError(
                f"OpenMVS TextureMesh failed: expected MTL file {textured_mtl} but file is missing or empty."
            )
        if not texture_img.exists() or texture_img.stat().st_size == 0:
            raise RuntimeError(
                f"OpenMVS TextureMesh failed: expected texture map {texture_img} but file is missing or empty."
            )

        logger.info(
            f"OpenMVS successful: PLY={self.dense_ply.stat().st_size:,}B, "
            f"OBJ={textured_obj.stat().st_size:,}B, MTL={textured_mtl.stat().st_size:,}B"
        )
        return self.dense_ply, textured_obj, textured_mtl, texture_img

    def reconstruct(
        self,
        keyframes: List[Path],
        camera_poses: List[Dict[str, Any]],
        camera_calibration: Dict[str, Any],
        sparse_points: Optional[List[List[float]]] = None,
        sparse_dir: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        dynamic_mask_paths: Optional[List[Path]] = None,
        voxel_size: float = 0.12,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0
    ) -> Dict[str, Any]:
        """
        Master entry point for dense multi-view reconstruction via OpenMVS.

        Only native OpenMVS execution is supported. Fails cleanly with RuntimeError
        if binaries are missing or if reconstruction produces 0 points. Zero fake fallbacks.
        """
        if not self.is_openmvs_available():
            raise RuntimeError(
                "Dense reconstruction backend unavailable: OpenMVS CLI binaries "
                "(InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, TextureMesh) were not found "
                "in PATH. Install OpenMVS to run real multi-view stereo reconstruction. "
                "No fallback dense engine is executed."
            )

        try:
            ply_path, mesh_obj_path, mtl_path, tex_img_path = self.run_openmvs(
                sparse_dir=sparse_dir,
                image_dir=image_dir,
                use_cpu=(DEVICE.type == "cpu")
            )
            pcd = o3d.io.read_point_cloud(str(ply_path))
            if len(pcd.points) == 0:
                raise RuntimeError(f"OpenMVS returned an empty point cloud: {ply_path}")
            return {
                "dense_ply_path": ply_path,
                "mesh_path": mesh_obj_path,
                "mtl_path": mtl_path,
                "texture_img_path": tex_img_path,
                "pcd": pcd,
                "method": "OpenMVS_Native",
                "num_points": len(pcd.points)
            }
        except Exception as e:
            raise RuntimeError(f"Dense reconstruction backend failed: {e}") from e
