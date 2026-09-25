"""Native OpenMVS dense reconstruction backend.

This module deliberately has no synthetic stereo or sparse-point fallback. If the
native MVS toolchain is unavailable or produces an invalid artifact, the job fails
with a diagnostic instead of reporting a fake reconstruction.

OpenMVS CLI binaries are mandatory for dense reconstruction; this backend never
substitutes a synthetic or approximate dense reconstruction implementation.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import open3d as o3d
import numpy as np

from app.config import DEVICE, find_binary, logger


def find_openmvs_binary(name: str) -> Optional[str]:
    return find_binary(name)


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _validate_dense_cloud(ply_path: Path, min_points: int = 1000) -> Dict[str, Any]:
    """Validate dense MVS geometry before allowing surface reconstruction."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) < min_points:
        raise RuntimeError(f"Dense MVS produced only {len(points):,} points; refusing to build a mesh from an insufficient cloud.")
    if not np.isfinite(points).all():
        raise RuntimeError("Dense MVS point cloud contains NaN/Inf coordinates.")
    extent = np.ptp(points, axis=0)
    if not np.all(np.isfinite(extent)) or float(np.max(extent)) <= 1e-6:
        raise RuntimeError("Dense MVS point cloud has no measurable spatial extent.")
    centered = points - np.mean(points, axis=0)
    covariance = np.cov(centered, rowvar=False) if len(points) >= 3 else np.eye(3)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    planarity = float(eigenvalues[2] / max(eigenvalues[0], 1e-12))
    logger.info("Dense MVS validation: %s points, extent=(%.3f, %.3f, %.3f), planarity=%.6f", f"{len(points):,}", extent[0], extent[1], extent[2], planarity)
    return {"points": int(len(points)), "extent": extent.tolist(), "planarity": planarity}


def _sanitize_openmvs_mesh(mesh_path: Path) -> None:
    """Remove only clearly unsupported long-edge bridge triangles from the raw MVS mesh."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) < 10:
        raise RuntimeError("OpenMVS produced an unusably small surface mesh.")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tri = vertices[faces]
    edge_lengths = np.concatenate((
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    ))
    median_edge = float(np.median(edge_lengths)) if len(edge_lengths) else 0.0
    if median_edge > 0:
        edges = np.stack((
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ), axis=1)
        invalid = edges.max(axis=1) > median_edge * 12.0
        if invalid.any() and int((~invalid).sum()) >= 10:
            mesh.remove_triangles_by_mask(invalid)
            mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    if len(mesh.triangles) < 10:
        raise RuntimeError("OpenMVS mesh became empty after bridge-triangle filtering.")
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False, compressed=False)
    logger.info("OpenMVS mesh validation retained %s triangles.", f"{len(mesh.triangles):,}")

    # ── Robust PyMeshLab hole-fill pass ──────────────────────────────────────
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(mesh_path))
        before_faces = ms.current_mesh().face_number()
        # Interior gaps in the observed surface are small compared with the
        # perimeter. 5000 also closes the entire perimeter and folds a cap
        # through the scene (the V5 survey has a 944-edge outer boundary).
        ms.meshing_close_holes(maxholesize=500, newfaceselected=False, selfintersection=False)
        after_faces = ms.current_mesh().face_number()
        if after_faces > before_faces:
            ms.save_current_mesh(str(mesh_path))
            logger.info("PyMeshLab sealed interior surface gaps (+%d triangles, %d total); survey perimeter remains open.", after_faces - before_faces, after_faces)
    except Exception as _hf_err:
        logger.warning("PyMeshLab hole-fill skipped: %s", _hf_err)


def _model_dir(root: Path) -> Path:
    candidates = [root / "0", root]
    for candidate in candidates:
        if all(_first_existing([candidate / f, candidate / f.replace(".bin", ".txt")]) for f in ("cameras.bin", "images.bin", "points3D.bin")):
            return candidate
    raise RuntimeError(f"No complete COLMAP model found below {root}")


class DenseReconstructor:
    def __init__(self, workspace_dir: Union[str, Path], device: Optional[Any] = None):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.dense_dir = self.workspace_dir / "dense"
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        self.dense_ply = self.dense_dir / "scene_dense.ply"
        self.device = device

    def is_openmvs_available(self) -> bool:
        return all(find_openmvs_binary(name) for name in ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh"))

    def _prepare_openmvs_inputs(self, sparse_dir: Path, image_dir: Path) -> Tuple[Path, Path]:
        import pycolmap
        sparse_dir = _model_dir(Path(sparse_dir).resolve())
        image_dir = Path(image_dir).resolve()
        reconstruction = pycolmap.Reconstruction(str(sparse_dir))
        models = {camera.model.name for camera in reconstruction.cameras.values()}
        if models <= {"PINHOLE"}:
            return sparse_dir, image_dir

        root = self.dense_dir / "undistorted"
        if root.exists():
            shutil.rmtree(root)
        logger.info("Undistorting %s camera model(s) for OpenMVS", sorted(models))
        pycolmap.undistort_images(output_path=str(root), input_path=str(sparse_dir), image_path=str(image_dir), output_type="COLMAP")
        sparse_out = _model_dir(root / "sparse")
        images_out = root / "images"
        if not images_out.is_dir():
            raise RuntimeError(f"COLMAP undistortion did not create {images_out}")
        return sparse_out, images_out

    @staticmethod
    def _run(cmd: list[str], cwd: Path, label: str) -> None:
        logger.info("OpenMVS %s: %s", label, " ".join(cmd))
        try:
            completed = subprocess.run(cmd, cwd=str(cwd), check=True, text=True, capture_output=True)
            if completed.stdout:
                logger.info("%s", completed.stdout[-1500:])
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc))[-4000:]
            raise RuntimeError(f"OpenMVS {label} failed (exit {exc.returncode}):\n{detail}") from exc

    def run_openmvs(self, sparse_dir: Optional[Path] = None, image_dir: Optional[Path] = None, use_cpu: bool = False, resolution_level: int = 1, max_resolution: int = 2560, number_views: int = 5, number_views_fuse: int = 2, max_threads: int = 0) -> Tuple[Path, Path, Path, Path]:
        bins = {name: find_openmvs_binary(name) for name in ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh")}
        if not all(bins.values()):
            missing = [name for name, path in bins.items() if not path]
            raise RuntimeError(f"OpenMVS backend unavailable; missing binaries: {', '.join(missing)}")

        sparse_dir = Path(sparse_dir or (self.workspace_dir / "sparse")).resolve()
        image_dir = Path(image_dir or (self.workspace_dir / "images")).resolve()
        sparse_dir, image_dir = self._prepare_openmvs_inputs(sparse_dir, image_dir)

        interface_root = self.dense_dir / "colmap_input"
        if interface_root.exists():
            shutil.rmtree(interface_root)
        (interface_root / "sparse").mkdir(parents=True)
        for source in sparse_dir.iterdir():
            if source.is_file() and source.suffix.lower() in {".bin", ".txt"}:
                shutil.copy2(source, interface_root / "sparse" / source.name)
        staged_model = _model_dir(interface_root / "sparse")
        if staged_model != interface_root / "sparse":
            raise RuntimeError("OpenMVS staging produced a nested sparse model unexpectedly")
        staged_images = interface_root / "images"
        try:
            staged_images.symlink_to(image_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(image_dir, staged_images)

        scene = self.dense_dir / "scene.mvs"
        dense_scene = self.dense_dir / "scene_dense.mvs"
        mesh = self.dense_dir / "scene_dense_mesh.ply"
        textured_obj = self.dense_dir / "scene_dense_mesh_refine_texture.obj"
        for path in (scene, dense_scene, mesh, textured_obj, textured_obj.with_suffix(".mtl"), self.dense_ply):
            path.unlink(missing_ok=True)
        for pattern in ("scene_dense*.png", "scene_dense*.jpg", "scene_dense*.jpeg"):
            for path in self.dense_dir.glob(pattern):
                path.unlink(missing_ok=True)

        # 0 in OpenMVS instructs it to utilize all available CPU cores
        threads_arg = str(max_threads) if max_threads >= 0 else "0"

        mask_folder = self.workspace_dir / "masks"
        interface_cmd = [
            bins["InterfaceCOLMAP"], "-i", str(interface_root), "-o", str(scene),
            "--image-folder", str(staged_images),
            "--max-threads", threads_arg, "--process-priority", "0"
        ]
        if mask_folder.is_dir() and any(mask_folder.iterdir()):
            interface_cmd.extend(["--mask-folder", str(mask_folder)])

        self._run(interface_cmd, self.dense_dir, "InterfaceCOLMAP")
        if not scene.is_file() or scene.stat().st_size == 0:
            raise RuntimeError("InterfaceCOLMAP completed without a non-empty scene.mvs")

        densify_cmd = [
            bins["DensifyPointCloud"], str(scene),
            "-o", str(dense_scene),
            "--resolution-level", str(max(1, resolution_level)),
            "--max-resolution", str(min(2560, max_resolution)),
            "--number-views", str(min(6, max(2, number_views))),
            "--number-views-fuse", str(max(2, number_views_fuse)),
            "--estimate-colors", "1",
            "--estimate-normals", "1",
            "--sub-resolution-levels", "1",   # fast single pyramid pass (10x faster)
            "--geometric-iters", "2",          # standard 2 geometric verification passes
            "--iters", "2",                    # standard 2 depth estimation passes
            "--max-threads", threads_arg,
            "--process-priority", "0",
        ]
        self._run(densify_cmd, self.dense_dir, "DensifyPointCloud")
        if not dense_scene.is_file() or dense_scene.stat().st_size == 0:
            raise RuntimeError("DensifyPointCloud returned successfully but scene_dense.mvs is missing/empty")

        dense_candidates = [self.dense_ply, self.dense_dir / "scene_dense.ply", self.dense_dir / "scene_dense.ply.bin"]
        dense_ply = _first_existing(dense_candidates)
        if dense_ply is None:
            export_cmd = [bins["DensifyPointCloud"], str(dense_scene), "--export-type", "ply", "-o", str(self.dense_ply), "--max-threads", threads_arg, "--process-priority", "0"]
            try:
                self._run(export_cmd, self.dense_dir, "DensifyPointCloud PLY export")
            except RuntimeError:
                pass
            dense_ply = _first_existing([self.dense_ply, self.dense_dir / "scene_dense.ply"])
        if dense_ply is None:
            raise RuntimeError("OpenMVS produced scene_dense.mvs but no dense point-cloud PLY. Check DensifyPointCloud build/export support.")
        if dense_ply != self.dense_ply:
            shutil.copy2(dense_ply, self.dense_ply)

        _validate_dense_cloud(dense_ply, min_points=1000)

        self._run([
            bins["ReconstructMesh"], str(dense_scene),
            "-o", str(mesh), "--export-type", "ply",
            "-d", "1.0",                     # fine detail distance threshold
            "-f", "1",                       # free-space carving support for vertical walls & weak side surfaces
            "--remove-spikes", "1",          # remove jagged spikes
            "--close-holes", "100",          # seal boundary gaps
            "--smooth", "2",
            "--target-face-num", "1000000",
            "--max-threads", threads_arg, "--process-priority", "0"
        ], self.dense_dir, "ReconstructMesh")
        if not mesh.is_file() or mesh.stat().st_size == 0:
            raise RuntimeError("ReconstructMesh did not create a non-empty mesh")
        _sanitize_openmvs_mesh(mesh)

        # Note: RefineMesh is intentionally bypassed because on single-pass drone sequences
        # it aggressively decimates unobserved surfaces (dropping 80%+ of triangles),
        # creating lattice holes and Swiss-cheese artifacts. ReconstructMesh provides a complete continuous surface.

        self._run([
            bins["TextureMesh"], "-i", str(dense_scene), "-m", str(mesh),
            "-o", str(textured_obj), "--export-type", "obj",
            "--global-seam-leveling", "0", "--local-seam-leveling", "0",
            "--virtual-face-images", "3",    # group coplanar triangles on roofs and walls
            "--cost-smoothness-ratio", "0.1",
            "--patch-packing-heuristic", "3",
            "--max-texture-size", "8192",
            "--sharpness-weight", "0.6",     # crisp photographic texture projection
            "--empty-color", "8421504",      # 0x808080 neutral grey instead of black
            "--max-threads", threads_arg, "--process-priority", "0"
        ], self.dense_dir, "TextureMesh")
        if not textured_obj.is_file() or textured_obj.stat().st_size == 0:
            raise RuntimeError("TextureMesh did not create a non-empty OBJ")
        mtl = textured_obj.with_suffix(".mtl")
        if not mtl.is_file() or mtl.stat().st_size == 0:
            raise RuntimeError(f"TextureMesh did not create expected MTL: {mtl}")

        textures = sorted([*self.dense_dir.glob("scene_dense_mesh_refine_texture*.png"), *self.dense_dir.glob("scene_dense_mesh_refine_texture*.jpg"), *self.dense_dir.glob("*.png"), *self.dense_dir.glob("*.jpg")], key=lambda p: p.stat().st_mtime, reverse=True)
        texture = _first_existing(textures)
        if texture is None:
            raise RuntimeError("TextureMesh did not create a texture atlas")
        return self.dense_ply, textured_obj, mtl, texture

    def reconstruct(self, keyframes: List[Path], camera_poses: List[Dict[str, Any]], camera_calibration: Dict[str, Any], sparse_points: Optional[List[List[float]]] = None, sparse_dir: Optional[Path] = None, image_dir: Optional[Path] = None, dynamic_mask_paths: Optional[List[Path]] = None, voxel_size: float = 0.08, outlier_nb_neighbors: int = 20, outlier_std_ratio: float = 2.0, resolution_level: int = 1, max_resolution: int = 2560, number_views: int = 5, number_views_fuse: int = 2, max_threads: int = 0) -> Dict[str, Any]:
        if not self.is_openmvs_available():
            raise RuntimeError("Dense reconstruction backend unavailable: install InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, and TextureMesh. No synthetic dense fallback is executed.")
        device_type = getattr(self.device, "type", "cpu") if self.device is not None else getattr(DEVICE, "type", "cpu")
        ply_path, mesh_path, mtl_path, texture_path = self.run_openmvs(
            sparse_dir=sparse_dir,
            image_dir=image_dir,
            use_cpu=(device_type == "cpu"),
            resolution_level=resolution_level,
            max_resolution=max_resolution,
            number_views=number_views,
            number_views_fuse=number_views_fuse,
            max_threads=max_threads,
        )
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if len(pcd.points) == 0:
            raise RuntimeError(f"OpenMVS returned an empty point cloud: {ply_path}")
        if voxel_size > 0:
            import numpy as np
            points = np.asarray(pcd.points)
            if len(points) > 0:
                extent = np.max(points, axis=0) - np.min(points, axis=0)
                max_extent = np.max(extent)
                effective_voxel_size = min(voxel_size, max_extent / 800.0)
                pcd = pcd.voxel_down_sample(effective_voxel_size)
        if len(pcd.points) >= max(20, outlier_nb_neighbors):
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=outlier_nb_neighbors, std_ratio=outlier_std_ratio)
        if len(pcd.points) == 0:
            raise RuntimeError("OpenMVS point cloud became empty after validation/outlier filtering")
        return {"pcd": pcd, "dense_ply_path": ply_path, "mesh_path": mesh_path, "mtl_path": mtl_path, "texture_img_path": texture_path}