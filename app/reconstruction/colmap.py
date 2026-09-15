import os
import shutil
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
import numpy as np
# Load Open3D before PyTorch/pycolmap to avoid duplicate libomp initialization
# on macOS when the two native extension stacks are imported in one process.
import open3d  # noqa: F401
import pycolmap

from app.config import DEVICE, logger


class SfMPipeline:
    """
    Structure from Motion pipeline using pycolmap with full camera auto-calibration,
    SIFT feature extraction, dynamic mask exclusion, sequential matching,
    and incremental bundle adjustment.
    """
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir)
        self.database_path = self.workspace_dir / "database.db"
        self.image_dir = self.workspace_dir / "images"
        self.mask_dir = self.workspace_dir / "masks"
        self.sparse_dir = self.workspace_dir / "sparse"
        
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.best_model: Optional[pycolmap.Reconstruction] = None

    def prepare_workspace(self, keyframe_paths: List[Path], mask_paths: Optional[List[Path]] = None) -> List[Path]:
        """
        Populate workspace/images and workspace/masks with keyframes and dynamic masks.
        Returns the list of paths within self.image_dir.
        """
        self.image_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Preparing SfM workspace with {len(keyframe_paths)} keyframes...")
        
        target_names = {kf.name for kf in keyframe_paths}
        for existing in list(self.image_dir.glob("*.*")):
            if existing.name not in target_names:
                try:
                    existing.unlink()
                except Exception:
                    pass

        prepared_keyframes = []
        for kf in keyframe_paths:
            dst = self.image_dir / kf.name
            if kf.resolve() != dst.resolve():
                shutil.copy2(kf, dst)
            prepared_keyframes.append(dst)

        if mask_paths and self.mask_dir.exists():
            logger.info(f"Dynamic masks configured in {self.mask_dir} for feature exclusion.")

        return prepared_keyframes

    def extract_features(self, camera_model: str = "SIMPLE_RADIAL") -> None:
        """
        Extract features from images using pycolmap SIFT extractor.
        Uses CameraMode.SINGLE so all drone video frames share the calibrated camera intrinsics.
        Dynamic masks are respected if present in self.mask_dir.
        """
        if self.database_path.exists():
            try:
                self.database_path.unlink()
            except Exception:
                pass

        logger.info(f"Extracting SIFT features on {self.image_dir} (Camera model: {camera_model}, SINGLE camera mode)...")
        
        image_options = pycolmap.ImageReaderOptions()
        image_options.camera_model = camera_model
        
        if self.mask_dir.exists() and any(self.mask_dir.iterdir()):
            logger.info(f"Respecting dynamic masks from {self.mask_dir} during feature extraction")
            image_options.mask_path = str(self.mask_dir)

        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.sift.max_num_features = 4096
        
        pycolmap.extract_features(
            database_path=self.database_path,
            image_path=self.image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=image_options,
            extraction_options=extraction_options
        )
            
    def match_features(self, method: str = "sequential") -> None:
        """
        Match extracted features sequentially with geometric verification.
        """
        logger.info(f"Matching features using {method} matching...")
        if method == "exhaustive":
            pycolmap.match_exhaustive(self.database_path)
        else:
            pairing_options = pycolmap.SequentialPairingOptions()
            pairing_options.overlap = 8
            pycolmap.match_sequential(self.database_path, pairing_options=pairing_options)
            
    def detect_sparse_model_dir(self) -> Path:
        """
        Detect the actual valid sparse-model directory (supporting workspace/sparse or workspace/sparse/0).
        """
        candidates = [
            self.sparse_dir / "0",
            self.sparse_dir
        ]
        for cand in candidates:
            if cand.exists() and (
                (cand / "cameras.bin").exists() or 
                (cand / "cameras.txt").exists() or
                (cand / "images.bin").exists() or
                (cand / "images.txt").exists()
            ):
                return cand
        return self.sparse_dir

    def prepare_glomap_database(self) -> Path:
        """Create a private database copy compatible with this GLOMAP build.

        The bundled COLMAP revision in the archived GLOMAP source expects the
        legacy pose_priors schema. pycolmap 4.2 creates the newer schema. The
        table is empty for normal video SfM, so only the copy is adapted; the
        source database is never modified.
        """
        import sqlite3

        glomap_db = self.workspace_dir / "database_glomap.db"
        if glomap_db.exists():
            glomap_db.unlink()

        with sqlite3.connect(str(self.database_path)) as source:
            with sqlite3.connect(str(glomap_db)) as target:
                source.backup(target)

        with sqlite3.connect(str(glomap_db)) as conn:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(pose_priors)")
            }
            if "image_id" not in columns:
                prior_count = conn.execute(
                    "SELECT COUNT(*) FROM pose_priors"
                ).fetchone()[0]
                if prior_count:
                    raise RuntimeError(
                        "GLOMAP database schema is incompatible and contains "
                        f"{prior_count} pose priors; refusing to discard them."
                    )
                conn.execute("DROP INDEX IF EXISTS pose_prior_data_assignment")
                conn.execute("DROP TABLE IF EXISTS pose_priors")
                conn.execute(
                    "CREATE TABLE pose_priors ("
                    "image_id INTEGER PRIMARY KEY NOT NULL,"
                    "position BLOB,"
                    "coordinate_system INTEGER NOT NULL DEFAULT 0,"
                    "position_covariance BLOB,"
                    "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE"
                    ")"
                )
                conn.commit()
        return glomap_db

    def map(self, mapper_backend: str = "glomap") -> Optional[pycolmap.Reconstruction]:
        """
        Run sparse reconstruction and bundle adjustment using GLOMAP (default) or pycolmap.
        """
        import subprocess
        from app.config import find_glomap_binary

        if mapper_backend == "glomap":
            glomap_bin = find_glomap_binary()
            if not glomap_bin:
                raise RuntimeError(
                    "GLOMAP executable not found. Ensure GLOMAP is built in .local/bin/glomap "
                    "or installed on system PATH."
                )
            
            glomap_database = self.prepare_glomap_database()
            logger.info(f"Using GLOMAP-compatible database copy: {glomap_database}")

            logger.info(f"Running GLOMAP global SfM mapper using: {glomap_bin}")
            cmd = [
                glomap_bin, "mapper",
                "--database_path", str(glomap_database),
                "--image_path", str(self.image_dir),
                "--output_path", str(self.sparse_dir)
            ]
            
            try:
                res = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"GLOMAP completed successfully: {res.stdout[:200]}")
            except subprocess.CalledProcessError as e:
                logger.error(f"GLOMAP mapper failed with returncode {e.returncode}: {e.stderr}")
                raise RuntimeError(f"GLOMAP mapper execution failed: {e.stderr}") from e

            sparse_model_dir = self.detect_sparse_model_dir()
            logger.info(f"Detected GLOMAP sparse model directory: {sparse_model_dir}")
            
            try:
                rec = pycolmap.Reconstruction(str(sparse_model_dir))
                if rec.num_images() > 0:
                    self.best_model = rec
                    self.actual_sparse_dir = sparse_model_dir
                    logger.info(
                        f"GLOMAP SfM successful: {rec.num_images()} registered images, "
                        f"{len(rec.points3D)} 3D sparse points, "
                        f"mean reprojection error: {rec.compute_mean_reprojection_error():.2f} px"
                    )
                    return self.best_model
                else:
                    raise RuntimeError(f"GLOMAP generated 0 registered views in {sparse_model_dir}")
            except Exception as e:
                raise RuntimeError(f"Failed to load GLOMAP reconstruction from {sparse_model_dir}: {e}") from e

        else:
            logger.info("Running pycolmap incremental mapping & bundle adjustment...")
            try:
                pycolmap_output = self.sparse_dir / "pycolmap_models"
                if pycolmap_output.exists():
                    shutil.rmtree(pycolmap_output)
                pycolmap_output.mkdir(parents=True, exist_ok=True)
                maps = pycolmap.incremental_mapping(
                    database_path=self.database_path,
                    image_path=self.image_dir,
                    output_path=pycolmap_output
                )
                if maps and len(maps) > 0:
                    best_key, self.best_model = max(
                        maps.items(), key=lambda item: item[1].num_images()
                    )
                    self.actual_sparse_dir = pycolmap_output / str(best_key)
                    logger.info(
                        f"pycolmap SfM successful: {self.best_model.num_images()} registered images, "
                        f"{len(self.best_model.points3D)} 3D sparse points, "
                        f"mean reprojection error: {self.best_model.compute_mean_reprojection_error():.2f} px"
                    )
                    return self.best_model
                else:
                    logger.error("pycolmap incremental mapping produced 0 registered models.")
            except Exception as e:
                logger.error(f"pycolmap incremental mapping failed: {e}")

            return None

    def get_reconstruction_data(
        self, 
        keyframe_paths: List[Path], 
        gps_data: Optional[Dict[Path, Tuple[float, float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Extract real camera poses, intrinsics, 3D tie points, and reprojection error
        from the pycolmap reconstruction.
        Fails cleanly if SfM did not register sufficient images.
        """
        if self.best_model is None or self.best_model.num_images() < 3:
            num_reg = self.best_model.num_images() if self.best_model else 0
            raise RuntimeError(
                f"Structure from Motion failed: registered only {num_reg} / {len(keyframe_paths)} keyframes. "
                f"Multi-view geometry requires at least 3 overlapping registered views."
            )

        rec = self.best_model
        mean_reproj_err = round(float(rec.compute_mean_reprojection_error()), 2)
        
        # 1. Extract true camera intrinsics from shared camera
        cam = list(rec.cameras.values())[0]
        params = list(cam.params)
        focal_px = float(params[0]) if len(params) > 0 else float(cam.width)
        cx = float(params[1]) if len(params) > 1 else float(cam.width / 2.0)
        cy = float(params[2]) if len(params) > 2 else float(cam.height / 2.0)
        
        K = np.array([
            [focal_px, 0.0, cx],
            [0.0, focal_px, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        camera_calib = {
            "model": cam.model.name,
            "focal_px": round(focal_px, 1),
            "principal_pt": (round(cx, 1), round(cy, 1)),
            "width": cam.width,
            "height": cam.height,
            "params": [round(float(p), 4) for p in params],
            "K": K
        }

        # 2. Extract calibrated camera poses
        img_map = {img.name: img for img in rec.images.values()}
        poses = []
        registered_indices = []

        for idx, kf in enumerate(keyframe_paths):
            img_name = kf.name
            if img_name in img_map and img_map[img_name].has_pose:
                col_img = img_map[img_name]
                center = col_img.projection_center()
                cfw = col_img.cam_from_world() if callable(col_img.cam_from_world) else col_img.cam_from_world
                rot_mat = cfw.rotation.matrix()
                trans_vec = cfw.translation
                
                # Euler angles from rotation matrix
                pitch = float(np.degrees(np.arctan2(-rot_mat[2, 1], rot_mat[2, 2])))
                yaw = float(np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0])))
                roll = float(np.degrees(np.arctan2(rot_mat[2, 0], rot_mat[2, 2])))
                
                poses.append({
                    "frame_idx": idx,
                    "image_name": img_name,
                    "file_path": str(kf),
                    "is_registered": True,
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                    "R": rot_mat.tolist(),
                    "t": trans_vec.tolist(),
                    "C": [float(c) for c in center],
                    "pitch": round(pitch, 1),
                    "yaw": round(yaw, 1),
                    "roll": round(roll, 1)
                })
                registered_indices.append(idx)
            else:
                poses.append({
                    "frame_idx": idx,
                    "image_name": img_name,
                    "file_path": str(kf),
                    "is_registered": False,
                    "x": 0.0, "y": 0.0, "z": 0.0,
                    "R": None, "t": None, "C": None,
                    "pitch": 0.0, "yaw": 0.0, "roll": 0.0
                })

        # Interpolate poses for any un-registered keyframe between registered neighbors
        for i, p in enumerate(poses):
            if not p["is_registered"]:
                prev_idx = max([idx for idx in registered_indices if idx < i], default=None)
                next_idx = min([idx for idx in registered_indices if idx > i], default=None)
                
                if prev_idx is not None and next_idx is not None:
                    alpha = (i - prev_idx) / float(next_idx - prev_idx)
                    p0 = poses[prev_idx]
                    p1 = poses[next_idx]
                    p["x"] = p0["x"] * (1 - alpha) + p1["x"] * alpha
                    p["y"] = p0["y"] * (1 - alpha) + p1["y"] * alpha
                    p["z"] = p0["z"] * (1 - alpha) + p1["z"] * alpha
                    p["pitch"] = p0["pitch"] * (1 - alpha) + p1["pitch"] * alpha
                    p["yaw"] = p0["yaw"] * (1 - alpha) + p1["yaw"] * alpha
                    p["C"] = [p["x"], p["y"], p["z"]]
                    p["R"] = p0["R"]
                    p["t"] = p0["t"]
                elif prev_idx is not None:
                    p["x"] = poses[prev_idx]["x"]
                    p["y"] = poses[prev_idx]["y"]
                    p["z"] = poses[prev_idx]["z"]
                    p["C"] = poses[prev_idx]["C"]
                    p["R"] = poses[prev_idx]["R"]
                    p["t"] = poses[prev_idx]["t"]
                elif next_idx is not None:
                    p["x"] = poses[next_idx]["x"]
                    p["y"] = poses[next_idx]["y"]
                    p["z"] = poses[next_idx]["z"]
                    p["C"] = poses[next_idx]["C"]
                    p["R"] = poses[next_idx]["R"]
                    p["t"] = poses[next_idx]["t"]

        # 3. Extract real 3D sparse tie points
        sparse_points = []
        for p3d in rec.points3D.values():
            xyz = p3d.xyz
            rgb = [float(c) / 255.0 for c in p3d.color]
            error = float(p3d.error)
            sparse_points.append([
                float(xyz[0]), float(xyz[1]), float(xyz[2]),
                rgb[0], rgb[1], rgb[2],
                error
            ])

        return {
            "camera_poses": poses,
            "registered_count": len(registered_indices),
            "total_count": len(keyframe_paths),
            "sparse_points": sparse_points,
            "reprojection_error_px": mean_reproj_err,
            "camera_calibration": camera_calib,
            "reconstruction_model": rec
        }
