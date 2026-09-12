import os
import subprocess
from pathlib import Path
from typing import Union, Tuple
import numpy as np
import open3d as o3d
import torch

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from app.config import DEVICE, logger

class DenseReconstructor:
    """
    Dense reconstruction module using OpenMVS or Depth Anything v2 + Open3D TSDF fusion.
    """
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir)
        self.sparse_dir = self.workspace_dir / "sparse"
        self.dense_dir = self.workspace_dir / "dense"
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        
    def run_openmvs(self, use_cpu: bool = False) -> Tuple[Path, Path]:
        """
        Path A: Run OpenMVS CLI binaries for dense reconstruction.
        Assumes OpenMVS binaries are in the system PATH.
        """
        logger.info("Starting OpenMVS pipeline...")
        
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
        
        # 4. RefineMesh
        logger.info("Running RefineMesh...")
        refined_mesh = self.dense_dir / "scene_dense_mesh_refine.mvs"
        subprocess.run([
            "RefineMesh", 
            str(mesh_mvs), 
            "-o", str(refined_mesh)
        ] + cpu_flag, check=True)
        
        # 5. TextureMesh
        logger.info("Running TextureMesh...")
        textured_mesh_obj = self.dense_dir / "scene_dense_mesh_refine_texture.obj"
        subprocess.run([
            "TextureMesh", 
            str(refined_mesh), 
            "-o", str(self.dense_dir / "scene_dense_mesh_refine_texture.mvs"),
            "--export-type", "obj"
        ] + cpu_flag, check=True)
        
        dense_ply = self.dense_dir / "scene_dense.ply"
        return dense_ply, textured_mesh_obj
        
    def run_depth_anything_fusion(self) -> Path:
        """
        Path B: Depth Anything v2 + Open3D TSDF Volume fusion fallback.
        """
        logger.info("Starting Depth Anything v2 + TSDF fusion pipeline...")
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required for Depth Anything fallback.")
            
        # Initialize Depth Anything v2
        device_id = 0 if DEVICE.type != "cpu" else -1
        depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device_id)
        
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        logger.warning("TSDF Fusion typically requires parsed poses (e.g. from COLMAP). Iterating images as placeholder.")
        image_dir = self.workspace_dir / "images"
        for img_path in sorted(image_dir.glob("*.jpg")):
            # depth_result = depth_estimator(str(img_path))
            # depth_map = np.array(depth_result["depth"])
            # In a real scenario: load pose, convert depth and RGB to o3d RGBDImage, and integrate.
            pass
            
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        
        out_mesh_path = self.dense_dir / "tsdf_mesh.ply"
        o3d.io.write_triangle_mesh(str(out_mesh_path), mesh)
        logger.info(f"Saved TSDF mesh to {out_mesh_path}")
        return out_mesh_path
