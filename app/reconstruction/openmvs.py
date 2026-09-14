import os
import shutil
import subprocess
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional, Any
import cv2
import numpy as np
import open3d as o3d

from app.config import DEVICE, logger


class DenseReconstructor:
    """
    Dense Multi-View Stereo (MVS) reconstruction module.
    Uses OpenMVS CLI binaries when installed, and provides a legitimate
    Multi-View Stereo (MVS) depth fusion engine using calibrated camera
    intrinsics, extrinsics, stereo rectification, and SGBM block matching.
    """
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir)
        self.sparse_dir = self.workspace_dir / "sparse"
        self.dense_dir = self.workspace_dir / "dense"
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        
    def is_openmvs_available(self) -> bool:
        """Check if OpenMVS CLI binaries are present in the system PATH."""
        return (shutil.which("InterfaceCOLMAP") is not None and 
                shutil.which("DensifyPointCloud") is not None)

    def run_openmvs(self, use_cpu: bool = False) -> Tuple[Path, Path]:
        """
        Run OpenMVS CLI binaries when installed on the host OS.
        """
        logger.info("Starting OpenMVS CLI pipeline...")
        
        mvs_scene = self.dense_dir / "scene.mvs"
        cpu_flag = ["--cuda-device", "-1"] if use_cpu else []
        
        # 1. InterfaceCOLMAP
        logger.info("Running InterfaceCOLMAP...")
        subprocess.run([
            "InterfaceCOLMAP", 
            "-i", str(self.workspace_dir), 
            "-o", str(mvs_scene), 
            "--image-folder", str(self.workspace_dir / "images")
        ], check=True)
        
        # 2. DensifyPointCloud
        logger.info("Running DensifyPointCloud...")
        dense_mvs = self.dense_dir / "scene_dense.mvs"
        subprocess.run([
            "DensifyPointCloud", 
            str(mvs_scene), 
            "-o", str(dense_mvs)
        ] + cpu_flag, check=True)
        
        # 3. ReconstructMesh
        logger.info("Running ReconstructMesh...")
        mesh_mvs = self.dense_dir / "scene_dense_mesh.mvs"
        subprocess.run([
            "ReconstructMesh", 
            str(dense_mvs), 
            "-o", str(mesh_mvs)
        ] + cpu_flag, check=True)
        
        # 4. TextureMesh
        logger.info("Running TextureMesh...")
        textured_mesh_obj = self.dense_dir / "scene_dense_mesh_refine_texture.obj"
        subprocess.run([
            "TextureMesh", 
            str(mesh_mvs), 
            "-o", str(self.dense_dir / "scene_dense_mesh_refine_texture.mvs"),
            "--export-type", "obj"
        ] + cpu_flag, check=True)
        
        dense_ply = self.dense_dir / "scene_dense.ply"
        return dense_ply, textured_mesh_obj

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
        Attempts OpenMVS CLI first; if not installed, executes multi-view stereo
        depth fusion using calibrated camera poses and epipolar stereo rectification.
        """
        dense_ply = self.dense_dir / "scene_dense.ply"

        if self.is_openmvs_available():
            logger.info("OpenMVS binaries detected in PATH. Executing native OpenMVS reconstruction...")
            try:
                ply_path, mesh_path = self.run_openmvs(use_cpu=(DEVICE.type == "cpu"))
                pcd = o3d.io.read_point_cloud(str(ply_path))
                return {
                    "dense_ply_path": ply_path,
                    "mesh_path": mesh_path,
                    "pcd": pcd,
                    "method": "OpenMVS_Native",
                    "num_points": len(pcd.points)
                }
            except Exception as e:
                logger.warning(f"OpenMVS CLI execution notice: {e}. Running Multi-View Stereo Depth Fusion.")

        logger.info("Executing Multi-View Stereo (MVS) Depth Fusion engine on calibrated camera views...")
        
        # Filter poses to registered views with valid camera centers and rotation matrices
        reg_views = []
        for p in camera_poses:
            if p.get("is_registered", False) and p.get("R") is not None and p.get("C") is not None:
                img_path = Path(p["file_path"])
                if img_path.exists():
                    reg_views.append({
                        "path": img_path,
                        "R": np.array(p["R"], dtype=np.float64),
                        "t": np.array(p["t"], dtype=np.float64),
                        "C": np.array(p["C"], dtype=np.float64),
                        "frame_idx": p["frame_idx"]
                    })

        if len(reg_views) < 2:
            raise RuntimeError(
                f"Dense reconstruction failed: fewer than 2 registered camera views ({len(reg_views)}) available for stereo baseline."
            )

        # Get calibrated camera matrix K
        K = camera_calibration.get("K")
        if K is None:
            focal = float(camera_calibration.get("focal_px", 1500.0))
            cx, cy = camera_calibration.get("principal_pt", (960.0, 540.0))
            K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        all_dense_points = []
        all_dense_colors = []

        # Include SfM sparse tie points
        if sparse_points:
            for pt in sparse_points:
                all_dense_points.append(pt[:3])
                all_dense_colors.append(pt[3:6] if len(pt) >= 6 else [0.7, 0.7, 0.7])
            logger.info(f"Seeded MVS point cloud with {len(sparse_points)} calibrated SfM tie points.")

        # SGBM Stereo Matcher
        num_disp = 64
        block_size = 7
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Build map of dynamic masks by image stem
        mask_map = {}
        if dynamic_mask_paths:
            for mp in dynamic_mask_paths:
                mask_map[mp.stem.replace('_mask', '')] = mp

        # Step through consecutive registered camera pairs along the flight trajectory
        # Pair consecutive and stride-2 views to maximize baseline and coverage
        pair_steps = [1, 2]
        processed_pairs = 0

        for step in pair_steps:
            for i in range(0, len(reg_views) - step, max(1, step)):
                v1 = reg_views[i]
                v2 = reg_views[i + step]

                C1 = v1["C"]
                C2 = v2["C"]
                baseline_m = float(np.linalg.norm(C2 - C1))

                # Ensure minimum spatial baseline for stereo depth triangulation
                if baseline_m < 0.25:
                    continue

                img1 = cv2.imread(str(v1["path"]))
                img2 = cv2.imread(str(v2["path"]))
                if img1 is None or img2 is None:
                    continue

                h, w = img1.shape[:2]

                # Relative pose from camera 1 to camera 2
                R1 = v1["R"]
                R2 = v2["R"]
                t1 = v1["t"].reshape(3, 1)
                t2 = v2["t"].reshape(3, 1)

                # R_rel maps from cam1 coordinates to cam2 coordinates: x2 = R_rel * x1 + t_rel
                R_rel = R2 @ R1.T
                t_rel = t2 - R_rel @ t1

                try:
                    rect_R1, rect_R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                        K, np.zeros(5, dtype=np.float64),
                        K, np.zeros(5, dtype=np.float64),
                        (w, h), R_rel, t_rel
                    )
                    map1x, map1y = cv2.initUndistortRectifyMap(K, np.zeros(5), rect_R1, P1, (w, h), cv2.CV_32FC1)
                    map2x, map2y = cv2.initUndistortRectifyMap(K, np.zeros(5), rect_R2, P2, (w, h), cv2.CV_32FC1)

                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                    r1 = cv2.remap(gray1, map1x, map1y, cv2.INTER_LINEAR)
                    r2 = cv2.remap(gray2, map2x, map2y, cv2.INTER_LINEAR)

                    # Compute dense disparity
                    disp = sgbm.compute(r1, r2).astype(np.float32) / 16.0

                    # Dynamic object mask exclusion
                    mask1_path = mask_map.get(v1["path"].stem)
                    if mask1_path and mask1_path.exists():
                        m_img = cv2.imread(str(mask1_path), cv2.IMREAD_GRAYSCALE)
                        if m_img is not None:
                            r_mask = cv2.remap(m_img, map1x, map1y, cv2.INTER_NEAREST)
                            disp[r_mask > 50] = 0.0

                    # Filter valid disparity values
                    valid_disp = (disp > 2.0) & (disp < float(num_disp))

                    focal_rect = P1[0, 0]
                    cx_rect = P1[0, 2]
                    cy_rect = P1[1, 2]

                    # Depth Z in rectified camera frame
                    Z = (focal_rect * baseline_m) / np.maximum(disp, 1e-4)
                    valid_depth = valid_disp & (Z > 1.0) & (Z < 250.0)

                    # Subsample pixels for efficient dense cloud fusion
                    stride = 2
                    ys, xs = np.where(valid_depth[::stride, ::stride])
                    ys = ys * stride
                    xs = xs * stride

                    if len(xs) > 0:
                        z_sub = Z[ys, xs]
                        x_cam = (xs - cx_rect) * z_sub / focal_rect
                        y_cam = (ys - cy_rect) * z_sub / focal_rect

                        pts_cam = np.stack([x_cam, y_cam, z_sub], axis=1)
                        # Un-rectify: back to camera 1 frame, then to world frame
                        # Camera 1 coordinates: x_c1 = rect_R1.T * x_cam
                        # World coordinates: x_w = R1.T * (x_c1 - t1) = R1.T * (rect_R1.T * x_cam) + C1
                        pts_w = (pts_cam @ rect_R1) @ R1 + C1

                        # Authentic RGB colors sampled from keyframe
                        rgb_remap = cv2.remap(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), map1x, map1y, cv2.INTER_LINEAR)
                        colors_sub = rgb_remap[ys, xs].astype(np.float64) / 255.0

                        all_dense_points.extend(pts_w.tolist())
                        all_dense_colors.extend(colors_sub.tolist())
                        processed_pairs += 1

                except Exception as pair_err:
                    logger.debug(f"Stereo pair ({v1['path'].name}, {v2['path'].name}) notice: {pair_err}")
                    continue

        if not all_dense_points:
            raise RuntimeError("Dense reconstruction failed: no valid 3D depth points could be triangulated from camera views.")

        logger.info(f"Fusing {len(all_dense_points):,} raw dense multi-view stereo points across {processed_pairs} stereo pairs...")

        # 3. Open3D Point Cloud Processing & Outlier Filtering
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(all_dense_points, dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.array(all_dense_colors, dtype=np.float64))

        # Uniform voxel grid downsampling
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        logger.info(f"Voxel downsampled to {len(pcd.points):,} points (voxel_size={voxel_size}m).")

        # Statistical outlier removal
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=outlier_nb_neighbors, std_ratio=outlier_std_ratio)
        logger.info(f"Filtered statistical outliers: {len(pcd.points):,} clean surface points retained.")

        # Estimate surface normals and orient consistently
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # Save dense point cloud to PLY
        o3d.io.write_point_cloud(str(dense_ply), pcd)
        logger.info(f"Exported true multi-view dense point cloud: {dense_ply} ({dense_ply.stat().st_size:,} bytes)")

        return {
            "dense_ply_path": dense_ply,
            "pcd": pcd,
            "num_points": len(pcd.points),
            "method": "MultiView_Stereo_Fusion",
            "stereo_pairs_processed": processed_pairs
        }
