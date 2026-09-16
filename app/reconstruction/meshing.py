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
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.full((len(verts), 3), 0.7)

        # OpenMVS stores UVs per triangle corner. Expand those corners here as
        # well, so OBJ, GLB, and the in-memory cleaned mesh use identical faces.
        triangle_uvs = np.asarray(mesh.triangle_uvs)
        if len(triangle_uvs) == len(faces) * 3:
            export_vertices = verts[faces].reshape(-1, 3)
            export_faces = np.arange(len(export_vertices), dtype=np.int64).reshape(-1, 3)
            export_normals = normals[faces].reshape(-1, 3) if len(normals) == len(verts) else np.zeros_like(export_vertices)
            export_colors = colors[faces].reshape(-1, 3)
            export_uvs = triangle_uvs
        else:
            export_vertices = verts
            export_faces = faces
            export_normals = normals
            export_colors = colors
            export_uvs = uvs

        with open(obj_path, "w") as f:
            f.write("# Real Reconstructed 3D Photogrammetry Mesh\n")
            f.write("mtllib model.mtl\n")
            f.write("usemtl material_0\n")
            for v, c in zip(export_vertices, export_colors):
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
            for n in export_normals:
                f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            for uv in export_uvs:
                f.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
            for face in export_faces:
                i0, i1, i2 = face[0] + 1, face[1] + 1, face[2] + 1
                f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")

        paths["obj"] = obj_path
        paths["mtl"] = mtl_path
        logger.info(f"Exported OBJ: {obj_path} ({obj_path.stat().st_size:,} bytes)")

        # 3. Export GLB via pygltflib (Three.js GLTFLoader-compatible)
        import pygltflib as gltf_lib
        GLTF2 = gltf_lib.GLTF2
        GLTFBuffer = gltf_lib.Buffer
        GLTFBufferView = gltf_lib.BufferView
        GLTFAccessor = gltf_lib.Accessor
        GLTFImage = gltf_lib.Image
        GLTFTexture = gltf_lib.Texture
        GLTFMaterial = gltf_lib.Material
        GLTFMesh = gltf_lib.Mesh
        GLTFNode = gltf_lib.Node
        GLTFScene = gltf_lib.Scene
        GLTFPrimitive = gltf_lib.Primitive
        GLTFPbrMetallicRoughness = gltf_lib.PbrMetallicRoughness
        GLTFTextureInfo = gltf_lib.TextureInfo
        GLTFAttributes = gltf_lib.Attributes
        glb_path = output_dir / "model.glb"
        try:
            pil_tex = Image.open(str(tex_path))
            # OpenMVS stores UVs per triangle corner. Shared Open3D vertices
            # can therefore have multiple valid UVs; expand corners so faces
            # cannot sample unrelated atlas regions.
            glb_vertices = export_vertices
            glb_faces = export_faces
            glb_uvs = np.asarray(export_uvs, dtype=np.float32).copy()
            glb_normals = export_normals if len(export_normals) == len(export_vertices) else None

            # glTF uses the opposite image-space V origin from the OpenMVS
            # atlas/OBJ convention used by the source texture.
            glb_uvs[:, 1] = 1.0 - glb_uvs[:, 1]

            # Convert texture to PNG for maximum Three.js GLTFLoader
            # compatibility; trimesh embeds images as-is and some viewers
            # fail on JPEG-in-GLB.
            pil_tex = pil_tex.convert("RGB")
            buf = io.BytesIO()
            pil_tex.save(buf, format="PNG")
            buf.seek(0)
            pil_tex_bytes = buf.getvalue()

            gltf = GLTF2()

            # Build binary payload: indices, positions, UVs, normals, image
            idx_bytes = glb_faces.astype(np.uint32).tobytes()
            pos_bytes = glb_vertices.astype(np.float32).tobytes()
            uv_bytes = glb_uvs.astype(np.float32).tobytes()
            norm_bytes = glb_normals.astype(np.float32).tobytes()
            all_data = idx_bytes + pos_bytes + uv_bytes + norm_bytes + pil_tex_bytes

            # Buffer (uri=None: data embedded in the GLB binary chunk)
            gltf.buffers.append(GLTFBuffer(uri=None, byteLength=len(all_data)))
            gltf.set_binary_blob(all_data)

            # Buffer views (in order: indices, positions, UVs, normals, image)
            offsets = [0, len(idx_bytes), len(idx_bytes)+len(pos_bytes), len(idx_bytes)+len(pos_bytes)+len(uv_bytes), len(idx_bytes)+len(pos_bytes)+len(uv_bytes)+len(norm_bytes)]
            lengths = [len(idx_bytes), len(pos_bytes), len(uv_bytes), len(norm_bytes), len(pil_tex_bytes)]
            for i, (off, ln) in enumerate(zip(offsets, lengths)):
                gltf.bufferViews.append(GLTFBufferView(buffer=0, byteOffset=off, byteLength=ln))

            # Accessors
            gltf.accessors.append(GLTFAccessor(bufferView=0, componentType=5125, count=len(glb_faces) * 3, type="SCALAR"))
            gltf.accessors.append(GLTFAccessor(bufferView=1, componentType=5126, count=len(glb_vertices), type="VEC3", min=glb_vertices.min(axis=0).tolist(), max=glb_vertices.max(axis=0).tolist()))
            gltf.accessors.append(GLTFAccessor(bufferView=2, componentType=5126, count=len(glb_uvs), type="VEC2", min=[0.0, 0.0], max=[1.0, 1.0]))
            gltf.accessors.append(GLTFAccessor(bufferView=3, componentType=5126, count=len(glb_normals), type="VEC3"))

            # Image, texture, material, mesh, node, scene
            gltf.images.append(GLTFImage(bufferView=4, mimeType="image/png", name="texture"))
            gltf.textures.append(GLTFTexture(source=0))
            gltf.materials.append(GLTFMaterial(
                pbrMetallicRoughness=GLTFPbrMetallicRoughness(baseColorTexture=GLTFTextureInfo(index=0), baseColorFactor=[0.4, 0.4, 0.4, 1.0]),
                doubleSided=False, name="material0"
            ))
            prim = GLTFPrimitive(attributes=GLTFAttributes(POSITION=1, TEXCOORD_0=2, NORMAL=3), indices=0, material=0, mode=4)
            gltf.meshes.append(GLTFMesh(primitives=[prim], name="mesh0"))
            gltf.nodes.append(GLTFNode(mesh=0, name="node0"))
            gltf.scenes.append(GLTFScene(nodes=[0], name="scene0"))
            gltf.scene = 0

            gltf.save_binary(str(glb_path))
            glb_bytes = glb_path.read_bytes()
            paths["glb"] = glb_path
            logger.info(f"Exported GLB: {glb_path} ({len(glb_bytes):,} bytes)")
        except Exception as glb_err:
            logger.warning(f"GLB export fallback: {glb_err}")
            o3d.io.write_triangle_mesh(str(output_dir / "model.gltf"), mesh)

        # 4. Export FBX format
        fbx_path = output_dir / "model.fbx"
        num_v = len(export_vertices)
        num_f = len(export_faces)
        vert_str = ','.join([f"{v[0]:.3f},{v[1]:.3f},{v[2]:.3f}" for v in export_vertices])
        normal_str = ','.join([f"{n[0]:.3f},{n[1]:.3f},{n[2]:.3f}" for n in export_normals])
        poly_indices = []
        for f_idx in export_faces:
            poly_indices.append(str(f_idx[0]))
            poly_indices.append(str(f_idx[1]))
            poly_indices.append(str(-f_idx[2] - 1))
        poly_str = ','.join(poly_indices)
        uv_str = ','.join([f"{uv[0]:.4f},{uv[1]:.4f}" for uv in export_uvs])

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
