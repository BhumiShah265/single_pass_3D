import os
import shutil
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Any
import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from app.config import MeshConfig, logger


class MeshProcessor:
    """
    Reconstructs continuous 3D surface meshes from dense point clouds using
    Poisson surface reconstruction, performs multi-view camera texture projection,
    and exports standardized 3D mesh deliverables (OBJ, GLB, FBX, PLY).
    """
    def __init__(self, workspace_dir: Union[str, Path], config: Optional[MeshConfig] = None):
        self.workspace_dir = Path(workspace_dir)
        self.config = config or MeshConfig()
        self.mesh_dir = self.workspace_dir / "mesh"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

    def poisson_reconstruction(
        self, 
        pcd: o3d.geometry.PointCloud, 
        depth: int = 9, 
        trim_quantile: float = 0.05
    ) -> o3d.geometry.TriangleMesh:
        """
        Run Poisson surface reconstruction on an Open3D dense point cloud.
        Filters out low-density unobserved vertices to create clean boundaries.
        """
        eff_depth = min(depth, 8)
        logger.info(f"Running Poisson surface reconstruction (depth={eff_depth})...")
        
        target_pcd = pcd
        if len(pcd.points) > 250000:
            target_pcd = pcd.voxel_down_sample(0.20)
            logger.info(f"Voxel-downsampled point cloud to {len(target_pcd.points):,} points for robust Poisson surface extraction.")
            
        if not target_pcd.has_normals():
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30))
            target_pcd.orient_normals_consistent_tangent_plane(15)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(target_pcd, depth=eff_depth)
        
        # Trim unobserved/low-density vertices
        densities_arr = np.asarray(densities)
        if len(densities_arr) > 0 and trim_quantile > 0.0:
            thresh = float(np.quantile(densities_arr, trim_quantile))
            vertices_to_remove = densities_arr < thresh
            mesh.remove_vertices_by_mask(vertices_to_remove)
            logger.info(f"Trimmed low-density mesh boundary vertices (quantile={trim_quantile:.2f}).")

        # Clean up mesh topology
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()

        # Decimate if target face count exceeded
        target_faces = 250000
        if len(mesh.triangles) > target_faces:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
            mesh.compute_vertex_normals()

        logger.info(f"Poisson mesh generated: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces.")
        return mesh

    def texture_mesh_from_cameras(
        self,
        mesh: o3d.geometry.TriangleMesh,
        camera_poses: List[Dict[str, Any]],
        camera_calibration: Dict[str, Any],
        tex_size: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project original calibrated camera images onto the reconstructed 3D mesh:
        1. Projects each vertex into calibrated cameras to find the most frontal/nadir view.
        2. Assigns authentic photographic RGB colors to each mesh vertex.
        3. Computes UV texture coordinates and creates an orthomosaic texture map for OBJ/GLB materials.
        """
        logger.info("Projecting calibrated camera imagery onto reconstructed 3D surface mesh...")
        
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        n_verts = len(vertices)

        # Build list of valid calibrated camera views
        views = []
        K = camera_calibration.get("K")
        if K is None:
            focal = float(camera_calibration.get("focal_px", 1500.0))
            cx, cy = camera_calibration.get("principal_pt", (960.0, 540.0))
            K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        for p in camera_poses:
            if p.get("is_registered", False) and p.get("R") is not None and p.get("C") is not None:
                img_path = Path(p["file_path"])
                if img_path.exists():
                    views.append({
                        "path": img_path,
                        "R": np.array(p["R"], dtype=np.float64),
                        "t": np.array(p["t"], dtype=np.float64),
                        "C": np.array(p["C"], dtype=np.float64)
                    })

        vertex_colors = np.full((n_verts, 3), 0.7, dtype=np.float64)
        best_angles = np.full(n_verts, -1.0, dtype=np.float64)

        # Cache loaded images to avoid re-reading
        img_cache = {}
        # Select up to 16 views distributed across the sequence
        view_indices = np.linspace(0, len(views) - 1, min(16, len(views))).astype(int)
        sampled_views = [views[idx] for idx in view_indices]

        for v in sampled_views:
            v_path = str(v["path"])
            if v_path not in img_cache:
                bgr = cv2.imread(v_path)
                if bgr is not None:
                    img_cache[v_path] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            rgb_img = img_cache.get(v_path)
            if rgb_img is None:
                continue

            h_img, w_img = rgb_img.shape[:2]
            R, t, C = v["R"], v["t"], v["C"]

            # Transform vertices to camera frame: p_cam = R * V + t
            pts_cam = (vertices @ R.T) + t.reshape(1, 3)
            z_cam = pts_cam[:, 2]

            # In front of camera
            front_mask = z_cam > 0.5
            front_indices = np.where(front_mask)[0]
            if len(front_indices) == 0:
                continue

            # Project to image plane: u = fx * (x/z) + cx, v = fy * (y/z) + cy
            p_sub = pts_cam[front_indices]
            z_sub = z_cam[front_indices]
            u_px = (K[0, 0] * (p_sub[:, 0] / z_sub) + K[0, 2]).astype(int)
            v_px = (K[1, 1] * (p_sub[:, 1] / z_sub) + K[1, 2]).astype(int)

            # Within image boundaries
            valid_uv = (u_px >= 0) & (u_px < w_img) & (v_px >= 0) & (v_px < h_img)
            valid_indices = front_indices[valid_uv]
            u_valid = u_px[valid_uv]
            v_valid = v_px[valid_uv]

            if len(valid_indices) == 0:
                continue

            # Compute viewing angle: dot product between surface normal and camera viewing ray
            ray_dirs = vertices[valid_indices] - C
            ray_lens = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
            ray_dirs = ray_dirs / np.maximum(ray_lens, 1e-6)

            cos_angles = -np.sum(normals[valid_indices] * ray_dirs, axis=1)

            # Update colors where viewing angle is better (more nadir/facing camera)
            better = (cos_angles > best_angles[valid_indices]) & (cos_angles > 0.05)
            update_idx = valid_indices[better]
            update_u = u_valid[better]
            update_v = v_valid[better]
            best_angles[update_idx] = cos_angles[better]

            vertex_colors[update_idx] = rgb_img[update_v, update_u].astype(np.float64) / 255.0

        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        # 2. Compute UV texture coordinates based on XY planar projection of mesh bounding box
        min_b = vertices.min(axis=0)
        max_b = vertices.max(axis=0)
        span_x = max(1.0, max_b[0] - min_b[0])
        span_z = max(1.0, max_b[2] - min_b[2])

        uvs = np.zeros((n_verts, 2), dtype=np.float32)
        uvs[:, 0] = np.clip((vertices[:, 0] - min_b[0]) / span_x, 0.0, 1.0)
        uvs[:, 1] = np.clip(1.0 - (vertices[:, 2] - min_b[2]) / span_z, 0.0, 1.0)

        # 3. Create composite texture image from vertex colors
        # Rasterize vertex colors onto texture canvas
        tex_canvas = np.full((tex_size, tex_size, 3), 160, dtype=np.uint8)
        px_u = np.clip((uvs[:, 0] * (tex_size - 1)).astype(int), 0, tex_size - 1)
        px_v = np.clip(((1.0 - uvs[:, 1]) * (tex_size - 1)).astype(int), 0, tex_size - 1)
        c_u8 = (vertex_colors * 255).astype(np.uint8)

        # Draw smooth circles around vertices to populate texture atlas
        for u, v, c in zip(px_u, px_v, c_u8):
            cv2.circle(tex_canvas, (u, v), 3, (int(c[0]), int(c[1]), int(c[2])), -1)

        # Inpaint any small unmapped gaps
        mask_empty = cv2.inRange(tex_canvas, (155, 155, 155), (165, 165, 165))
        if np.any(mask_empty):
            tex_canvas = cv2.inpaint(tex_canvas, mask_empty, 5, cv2.INPAINT_TELEA)

        logger.info(f"Texture projection complete across {len(sampled_views)} calibrated views.")
        return vertex_colors, uvs, tex_canvas

    def export_mesh_deliverables(
        self,
        mesh: o3d.geometry.TriangleMesh,
        uvs: np.ndarray,
        texture_img: np.ndarray,
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Export all required 3D mesh formats from real reconstructed geometry:
        - model.obj + model.mtl + texture.jpg
        - model.glb (Binary glTF 2.0)
        - model.fbx (Autodesk FBX)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # 1. Save Texture image
        tex_path = output_dir / "texture.jpg"
        Image.fromarray(texture_img).save(str(tex_path), quality=95)
        paths["texture"] = tex_path

        # 2. Export Wavefront OBJ + MTL
        obj_path = output_dir / "model.obj"
        mtl_path = output_dir / "model.mtl"

        with open(mtl_path, "w") as f:
            f.write("# AeroSynth 3D Material Library\n")
            f.write("newmtl material_0\n")
            f.write("Ka 1.0 1.0 1.0\n")
            f.write("Kd 1.0 1.0 1.0\n")
            f.write("Ks 0.1 0.1 0.1\n")
            f.write("d 1.0\n")
            f.write("illum 2\n")
            f.write("map_Kd texture.jpg\n")

        verts = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        faces = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors)

        with open(obj_path, "w") as f:
            f.write("# Real Reconstructed 3D Photogrammetry Mesh\n")
            f.write("mtllib model.mtl\n")
            f.write("usemtl material_0\n")
            # Vertices with color
            for v, c in zip(verts, colors):
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
            # Normals
            for n in normals:
                f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            # UVs
            for uv in uvs:
                f.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
            # Faces
            for face in faces:
                i0, i1, i2 = face[0] + 1, face[1] + 1, face[2] + 1
                f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")

        paths["obj"] = obj_path
        paths["mtl"] = mtl_path
        logger.info(f"Exported OBJ: {obj_path} ({obj_path.stat().st_size:,} bytes)")

        # 3. Export GLB via Trimesh
        glb_path = output_dir / "model.glb"
        try:
            pil_tex = Image.open(str(tex_path))
            material = trimesh.visual.texture.SimpleMaterial(image=pil_tex)
            visual = trimesh.visual.TextureVisuals(uv=uvs, image=pil_tex, material=material)
            
            tm = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                vertex_normals=normals,
                vertex_colors=(colors * 255).astype(np.uint8),
                visual=visual
            )
            glb_bytes = tm.export(file_type='glb')
            with open(glb_path, "wb") as f:
                f.write(glb_bytes)
            paths["glb"] = glb_path
            logger.info(f"Exported GLB: {glb_path} ({len(glb_bytes):,} bytes)")
        except Exception as glb_err:
            logger.warning(f"Trimesh GLB export fallback: {glb_err}")
            o3d.io.write_triangle_mesh(str(output_dir / "model.gltf"), mesh)

        # 4. Export FBX format
        fbx_path = output_dir / "model.fbx"
        num_v = len(verts)
        num_f = len(faces)
        vert_str = ','.join([f"{v[0]:.3f},{v[1]:.3f},{v[2]:.3f}" for v in verts])
        normal_str = ','.join([f"{n[0]:.3f},{n[1]:.3f},{n[2]:.3f}" for n in normals])
        poly_indices = []
        for f_idx in faces:
            poly_indices.append(str(f_idx[0]))
            poly_indices.append(str(f_idx[1]))
            poly_indices.append(str(-f_idx[2] - 1))
        poly_str = ','.join(poly_indices)
        uv_str = ','.join([f"{uv[0]:.4f},{uv[1]:.4f}" for uv in uvs])

        fbx_content = f"""; FBX 7.4.0 project file
; Real 3D Drone Reconstruction
FBXHeaderExtension: {{
    FBXHeaderVersion: 1003
    FBXVersion: 7400
}}
GlobalSettings: {{
    Version: 1000
    Properties70: {{
        P: "UpAxis", "int", "Integer", "",1
        P: "UnitScaleFactor", "double", "Number", "",100
    }}
}}
Objects: {{
    Geometry: 1000, "Geometry::Mesh", "Mesh" {{
        Vertices: *{num_v * 3} {{
            a: {vert_str}
        }}
        PolygonVertexIndex: *{num_f * 3} {{
            a: {poly_str}
        }}
        LayerElementNormal: 0 {{
            Version: 101
            Name: "Normals"
            MappingInformationType: "ByVertex"
            ReferenceInformationType: "Direct"
            Normals: *{num_v * 3} {{
                a: {normal_str}
            }}
        }}
        LayerElementUV: 0 {{
            Version: 101
            Name: "UVMap"
            MappingInformationType: "ByPolygonVertex"
            ReferenceInformationType: "Direct"
            UV: *{num_v * 2} {{
                a: {uv_str}
            }}
        }}
    }}
    Model: 2000, "Model::SceneModel", "Mesh" {{
        Version: 232
    }}
}}
Connections: {{
    C: "OO", 1000, 2000
    C: "OO", 2000, 0
}}
"""
        with open(fbx_path, "w") as f:
            f.write(fbx_content)
        paths["fbx"] = fbx_path
        logger.info(f"Exported FBX: {fbx_path} ({len(fbx_content):,} bytes)")

        return paths
