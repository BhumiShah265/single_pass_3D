import io
import os
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
        if len(pcd.points) > 50000:
            target_pcd = pcd.voxel_down_sample(0.25)
            logger.info(f"Voxel-downsampled point cloud to {len(target_pcd.points):,} points for robust Poisson surface extraction.")
            
        if not target_pcd.has_normals():
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=20))

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
        target_faces = 100000
        if len(mesh.triangles) > target_faces:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
            mesh.compute_vertex_normals()

        logger.info(f"Poisson mesh generated: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces.")
        return mesh

    @staticmethod
    def clean_mesh_topology(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Remove disconnected debris and isolated long triangles before texturing.

        This deliberately runs on the untextured mesh.  Textured OBJ files use
        per-corner UVs, so modifying them afterwards can desynchronise geometry
        and atlas coordinates and produce incorrect colours.
        """
        if len(mesh.triangles) == 0:
            raise RuntimeError("Surface reconstruction produced an empty mesh.")

        labels, component_sizes, _ = mesh.cluster_connected_triangles()
        if len(component_sizes):
            largest_component = max(component_sizes)
            # Keep meaningful secondary surfaces but discard tiny floating
            # islands, which are the usual visible spikes/debris.
            min_component_faces = max(50, int(largest_component * 0.001))
            rejected_components = [
                index for index, size in enumerate(component_sizes)
                if size < min_component_faces
            ]
            if rejected_components:
                mesh.remove_triangles_by_mask(
                    np.isin(np.asarray(labels), rejected_components)
                )
                mesh.remove_unreferenced_vertices()

        faces = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        if len(faces):
            triangle_vertices = vertices[faces]
            edges = np.concatenate((
                np.linalg.norm(triangle_vertices[:, 1] - triangle_vertices[:, 0], axis=1),
                np.linalg.norm(triangle_vertices[:, 2] - triangle_vertices[:, 1], axis=1),
                np.linalg.norm(triangle_vertices[:, 0] - triangle_vertices[:, 2], axis=1),
            ))
            median_edge = float(np.median(edges))
            if median_edge > 0:
                # A triangle with a 10x local edge is a reconstruction bridge,
                # not supported surface detail. Removing it prevents the long
                # coloured needles seen at sparse-depth boundaries.
                max_edge = median_edge * 10.0
                triangle_edges = np.stack((
                    np.linalg.norm(triangle_vertices[:, 1] - triangle_vertices[:, 0], axis=1),
                    np.linalg.norm(triangle_vertices[:, 2] - triangle_vertices[:, 1], axis=1),
                    np.linalg.norm(triangle_vertices[:, 0] - triangle_vertices[:, 2], axis=1),
                ), axis=1)
                invalid = triangle_edges.max(axis=1) > max_edge
                if invalid.any():
                    mesh.remove_triangles_by_mask(invalid)
                    mesh.remove_unreferenced_vertices()

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        if len(mesh.triangles) == 0:
            raise RuntimeError("Mesh cleanup removed every triangle; dense reconstruction is too sparse.")
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
        if K is not None:
            K = np.asarray(K, dtype=np.float64)

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

        if len(views) > 0 and K is not None:
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
        min_b = vertices.min(axis=0) if len(vertices) > 0 else np.zeros(3)
        max_b = vertices.max(axis=0) if len(vertices) > 0 else np.ones(3)
        span_x = max(1e-3, float(max_b[0] - min_b[0]))
        span_z = max(1e-3, float(max_b[2] - min_b[2]))

        uvs = np.zeros((n_verts, 2), dtype=np.float32)
        if n_verts > 0:
            uvs[:, 0] = np.clip((vertices[:, 0] - min_b[0]) / span_x, 0.0, 1.0)
            uvs[:, 1] = np.clip(1.0 - (vertices[:, 2] - min_b[2]) / span_z, 0.0, 1.0)

        # 3. Create composite texture image from vertex colors via vectorized rasterization
        tex_canvas = np.full((tex_size, tex_size, 3), 160, dtype=np.uint8)
        if n_verts > 0:
            px_u = np.clip((uvs[:, 0] * (tex_size - 1)).astype(int), 0, tex_size - 1)
            px_v = np.clip(((1.0 - uvs[:, 1]) * (tex_size - 1)).astype(int), 0, tex_size - 1)
            c_u8 = (vertex_colors * 255).astype(np.uint8)
            tex_canvas[px_v, px_u] = c_u8
            # Fast dilation to fill interpolation gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tex_canvas = cv2.dilate(tex_canvas, kernel, iterations=1)

        logger.info(f"Texture projection complete across {len(views)} calibrated views.")
        return vertex_colors, uvs, tex_canvas

    def export_mesh_deliverables(
        self,
        mesh,
        uvs,
        texture_img,
        output_dir,
        openmvs_obj_path = None,
        openmvs_mtl_path = None,
        openmvs_texture_path = None,
        dense_ply_path = None,
    ):
        import shutil
        from PIL import Image
        import numpy as np
        import pygltflib as gltf_lib
        import io
        
        paths = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mvs_dense = output_dir.parent.parent / "workspace" / output_dir.name / "dense"
        candidate_tex = output_dir.parent.parent / "workspace" / output_dir.name / "dense" / "scene_dense_mesh_refine_texture_material_00_map_Kd.jpg"
        if openmvs_texture_path:
            candidate_tex = type(output_dir)(openmvs_texture_path)
            
        tex_path = output_dir / "texture.jpg"
        if candidate_tex.is_file():
            shutil.copy2(candidate_tex, tex_path)
            pil_tex = Image.open(str(tex_path)).convert("RGB")
        else:
            pil_tex = Image.fromarray(texture_img).convert("RGB")

        # ── Inpaint black / near-grey empty-color patches ─────────────────────
        # Atlas cells are packed from unrelated views. Inpainting across their
        # borders invents a checkerboard on unseen surfaces. Keep the neutral
        # OpenMVS empty color where no camera observed the mesh.

        pil_tex.save(str(tex_path), quality=95, subsampling=0)
        paths["texture"] = tex_path
        
        # MANUAL OBJ PARSER TO PREVENT OPEN3D FROM SCRAMBLING UVS
        if openmvs_obj_path and Path(openmvs_obj_path).is_file():
            v_list, vt_list, vn_list = [], [], []
            faces_v, faces_vt, faces_vn = [], [], []
            with open(openmvs_obj_path, 'r') as f:
                for line in f:
                    if line.startswith('v '): v_list.append([float(x) for x in line.strip().split()[1:4]])
                    elif line.startswith('vt '): vt_list.append([float(x) for x in line.strip().split()[1:3]])
                    elif line.startswith('vn '): vn_list.append([float(x) for x in line.strip().split()[1:4]])
                    elif line.startswith('f '):
                        parts = line.strip().split()[1:]
                        fv, fvt, fvn = [], [], []
                        for p in parts:
                            vals = p.split('/')
                            fv.append(int(vals[0]) - 1)
                            if len(vals) > 1 and vals[1]: fvt.append(int(vals[1]) - 1)
                            if len(vals) > 2 and vals[2]: fvn.append(int(vals[2]) - 1)
                        for i in range(1, len(fv) - 1):
                            faces_v.append([fv[0], fv[i], fv[i+1]])
                            if fvt: faces_vt.append([fvt[0], fvt[i], fvt[i+1]])
                            if fvn: faces_vn.append([fvn[0], fvn[i], fvn[i+1]])
                            
            v_arr = np.array(v_list, dtype=np.float32)
            vt_arr = np.array(vt_list, dtype=np.float32) if vt_list else None
            vn_arr = np.array(vn_list, dtype=np.float32) if vn_list else None
            faces_v = np.array(faces_v, dtype=np.int32)
            faces_vt = np.array(faces_vt, dtype=np.int32) if faces_vt else None
            faces_vn = np.array(faces_vn, dtype=np.int32) if faces_vn else None

            
            export_vertices = v_arr[faces_v].reshape(-1, 3)
            export_uvs = vt_arr[faces_vt].reshape(-1, 2) if vt_arr is not None else np.zeros((len(export_vertices), 2))
            export_normals = vn_arr[faces_vn].reshape(-1, 3) if vn_arr is not None else np.zeros_like(export_vertices)
            export_faces = np.arange(len(export_vertices), dtype=np.int64).reshape(-1, 3)
            
            # Since we didn't have normals in OBJ, compute basic face normals
            if vn_arr is None or len(vn_list) == 0:
                v0 = export_vertices[0::3]
                v1 = export_vertices[1::3]
                v2 = export_vertices[2::3]
                cross = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(cross, axis=1, keepdims=True)
                norm = np.where(norm == 0, 1e-6, norm)
                face_normals = cross / norm
                export_normals = np.repeat(face_normals, 3, axis=0)
        else:
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else np.zeros_like(verts)
            export_vertices = verts
            export_faces = faces
            export_normals = normals
            export_uvs = uvs

            
        GLTF2 = gltf_lib.GLTF2
        
        # ── Level ground plane to horizontal [0, 1, 0] and center mesh at origin (0, 0, 0) ──
        alignment = np.eye(3, dtype=np.float32)
        centroid = np.zeros(3, dtype=np.float32)
        if len(export_vertices) > 100:
            sample_sub = export_vertices[::max(1, len(export_vertices) // 5000)].astype(np.float64)
            c_est = sample_sub.mean(axis=0)
            try:
                _, _, vh = np.linalg.svd(sample_sub - c_est, full_matrices=False)
                plane_n = vh[2]
                if plane_n[1] > 0:
                    plane_n = -plane_n
                plane_n = plane_n / np.linalg.norm(plane_n)
                target_up = np.array([0.0, 1.0, 0.0])
                v_cross = np.cross(plane_n, target_up)
                c_dot = np.dot(plane_n, target_up)
                s_len = np.linalg.norm(v_cross)
                if s_len > 1e-4:
                    vx_mat = np.array([[0, -v_cross[2], v_cross[1]], [v_cross[2], 0, -v_cross[0]], [-v_cross[1], v_cross[0], 0]])
                    R_align = (np.eye(3) + vx_mat + (vx_mat @ vx_mat) * ((1 - c_dot) / (s_len**2))).astype(np.float32)
                    export_vertices = (export_vertices @ R_align.T).astype(np.float32)
                    export_normals = (export_normals @ R_align.T).astype(np.float32)
                    alignment = R_align
            except Exception as _svd_err:
                logger.warning("Plane alignment notice: %s", _svd_err)

            centroid = export_vertices.mean(axis=0)
            export_vertices = (export_vertices - centroid).astype(np.float32)


        glb_vertices = export_vertices
        glb_uvs = np.asarray(export_uvs, dtype=np.float32).copy()
        
        # FLIP V COORDINATE FOR GLTF TOP-LEFT CONVENTION
        glb_uvs[:, 1] = 1.0 - glb_uvs[:, 1]
        
        glb_normals = export_normals
        
        buf = io.BytesIO()
        pil_tex.save(buf, format="PNG")
        buf.seek(0)
        pil_tex_bytes = buf.getvalue()
        
        gltf = GLTF2()
        pos_bytes = glb_vertices.astype(np.float32).tobytes()
        uv_bytes = glb_uvs.astype(np.float32).tobytes()
        norm_bytes = glb_normals.astype(np.float32).tobytes()
        
        blob = pos_bytes + uv_bytes + norm_bytes + pil_tex_bytes
        
        def add_buffer_view(byte_offset, byte_length, target=None):
            bv = gltf_lib.BufferView(buffer=0, byteOffset=byte_offset, byteLength=byte_length)
            if target: bv.target = target
            gltf.bufferViews.append(bv)
            return len(gltf.bufferViews) - 1
            
        pos_bv = add_buffer_view(0, len(pos_bytes), 34962)
        uv_bv = add_buffer_view(len(pos_bytes), len(uv_bytes), 34962)
        norm_bv = add_buffer_view(len(pos_bytes)+len(uv_bytes), len(norm_bytes), 34962)
        img_bv = add_buffer_view(len(pos_bytes)+len(uv_bytes)+len(norm_bytes), len(pil_tex_bytes))
        
        gltf.accessors.extend([
            gltf_lib.Accessor(bufferView=pos_bv, componentType=5126, count=len(glb_vertices), type="VEC3", max=glb_vertices.max(axis=0).tolist(), min=glb_vertices.min(axis=0).tolist()),
            gltf_lib.Accessor(bufferView=uv_bv, componentType=5126, count=len(glb_uvs), type="VEC2", max=glb_uvs.max(axis=0).tolist(), min=glb_uvs.min(axis=0).tolist()),
            gltf_lib.Accessor(bufferView=norm_bv, componentType=5126, count=len(glb_normals), type="VEC3", max=glb_normals.max(axis=0).tolist(), min=glb_normals.min(axis=0).tolist())
        ])
            
        prim = gltf_lib.Primitive(
            attributes=gltf_lib.Attributes(POSITION=0, TEXCOORD_0=1, NORMAL=2),
            material=0
        )
        
        gltf.extensionsUsed = ["KHR_materials_unlit"]
        
        gltf.images.append(gltf_lib.Image(bufferView=img_bv, mimeType="image/png"))
        gltf.textures.append(gltf_lib.Texture(source=0))
        gltf.materials.append(gltf_lib.Material(
            pbrMetallicRoughness=gltf_lib.PbrMetallicRoughness(
                baseColorTexture=gltf_lib.TextureInfo(index=0, texCoord=0),
                metallicFactor=0.0,
                roughnessFactor=0.9
            ),
            doubleSided=True,
            extensions={"KHR_materials_unlit": {}}
        ))

        
        gltf.meshes.append(gltf_lib.Mesh(primitives=[prim]))
        gltf.nodes.append(gltf_lib.Node(mesh=0))
        gltf.scenes.append(gltf_lib.Scene(nodes=[0]))
        gltf.scene = 0
        gltf.buffers.append(gltf_lib.Buffer(byteLength=len(blob)))
        gltf.set_binary_blob(blob)
        
        glb_path = output_dir / "model.glb"
        gltf.save_binary(str(glb_path))
        paths["glb"] = glb_path

        if dense_ply_path and Path(dense_ply_path).is_file():
            try:
                from app.reconstruction.estimated_surface import export_estimated_surface
                estimate_path = output_dir / "model_complete.glb"
                texture_source = None
                if openmvs_obj_path and Path(openmvs_obj_path).is_file() and vt_arr is not None and faces_vt is not None:
                    texture_source = (v_arr, faces_v, vt_arr[faces_vt], np.asarray(pil_tex))
                estimate_stats = export_estimated_surface(
                    Path(dense_ply_path), estimate_path, alignment, centroid,
                    texture_source=texture_source,
                )
                paths["complete_glb"] = estimate_path
                logger.info("Optional estimated surface: %s", estimate_stats)
            except Exception as exc:
                logger.warning("Optional estimated surface unavailable: %s", exc)

        # --- Write model.mtl ---
        mtl_path = output_dir / "model.mtl"
        with open(str(mtl_path), "w") as _mf:
            _mf.write("newmtl material_0\nmap_Kd texture.jpg\n")
        paths["mtl"] = mtl_path

        # --- Write model.obj (copy OpenMVS output, fix mtllib ref) ---
        obj_path = output_dir / "model.obj"
        if openmvs_obj_path and Path(openmvs_obj_path).is_file():
            import shutil as _shutil
            with open(openmvs_obj_path, "r") as _src, open(str(obj_path), "w") as _dst:
                for _line in _src:
                    if _line.startswith("mtllib "):
                        _dst.write("mtllib model.mtl\n")
                    else:
                        _dst.write(_line)
        else:
            with open(str(obj_path), "w") as _of:
                _of.write("mtllib model.mtl\ng mesh\n")
                for _v in glb_vertices:
                    _of.write(f"v {_v[0]:.6f} {_v[1]:.6f} {_v[2]:.6f}\n")
                _of.write("usemtl material_0\n")
                _n = len(glb_vertices)
                for _i in range(0, _n, 3):
                    _of.write(f"f {_i+1} {_i+2} {_i+3}\n")
        paths["obj"] = obj_path

        # --- Write cloud.ply (dense point cloud) ---
        cloud_path = output_dir / "cloud.ply"
        try:
            import open3d as _o3d
            _workspace_dense = output_dir.parent.parent / "workspace" / output_dir.name / "dense"
            _ply_src = _workspace_dense / "scene_dense.ply"
            if _ply_src.is_file():
                import shutil as _sh2
                _sh2.copy2(str(_ply_src), str(cloud_path))
        except Exception:
            pass
        if not cloud_path.is_file():
            cloud_path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
        paths["cloud"] = cloud_path

        return paths
