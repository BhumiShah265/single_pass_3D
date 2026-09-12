import os
from pathlib import Path
from typing import Union, List, Dict
import open3d as o3d
import trimesh
import pymeshlab
import numpy as np

from app.config import logger

class MeshProcessor:
    """
    Processes point clouds to meshes, and performs mesh cleanup and export.
    """
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir)
        self.mesh_dir = self.workspace_dir / "mesh"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        
    def poisson_reconstruction(self, point_cloud_path: Path, depth: int = 9) -> Path:
        """
        Run Poisson surface reconstruction on a dense point cloud using Open3D.
        """
        logger.info(f"Running Poisson reconstruction on {point_cloud_path} with depth={depth}...")
        pcd = o3d.io.read_point_cloud(str(point_cloud_path))
        
        if not pcd.has_normals():
            logger.info("Estimating normals for point cloud...")
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(100)
            
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        
        logger.info("Filtering low density vertices...")
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        out_path = self.mesh_dir / "poisson_mesh.ply"
        o3d.io.write_triangle_mesh(str(out_path), mesh)
        logger.info(f"Saved Poisson mesh to {out_path}")
        return out_path
        
    def cleanup_mesh(self, mesh_path: Path) -> Path:
        """
        Clean up mesh using PyMeshLab (remove non-manifold edges, smoothing, decimation).
        """
        logger.info(f"Cleaning up mesh {mesh_path} using PyMeshLab...")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(mesh_path))
        
        # Remove non-manifold edges/vertices
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        
        # Laplacian smooth
        ms.apply_coord_laplacian_smoothing()
        
        # Decimation placeholder (if needed)
        # ms.meshing_decimation_quadric_edge_collapse(targetfacenum=500000)
        
        clean_path = self.mesh_dir / f"{mesh_path.stem}_cleaned.ply"
        ms.save_current_mesh(str(clean_path))
        logger.info(f"Saved cleaned mesh to {clean_path}")
        return clean_path
        
    def export_mesh(self, mesh_path: Path, formats: List[str] = ["obj", "glb"]) -> Dict[str, Path]:
        """
        Export mesh to various formats using trimesh.
        """
        logger.info(f"Exporting mesh {mesh_path} to formats: {formats}")
        mesh = trimesh.load(str(mesh_path))
        
        exported_paths = {}
        for fmt in formats:
            out_path = self.mesh_dir / f"{mesh_path.stem}_export.{fmt}"
            mesh.export(str(out_path))
            exported_paths[fmt] = out_path
            logger.info(f"Exported to {out_path}")
            
        return exported_paths
