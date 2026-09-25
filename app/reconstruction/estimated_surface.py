"""Build a closed, explicitly inferred terrain model from measured dense points.

The single-pass video cannot reveal every facade or the underside. This model
fits the observed upper surface in the aligned scene frame, interpolates small
gaps, and closes the terrain with a flat display base. The original textured
OpenMVS mesh remains available for inspection of its measured triangles.
"""
from __future__ import annotations

import json
import io
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image as PILImage
from pygltflib import GLTF2, Accessor, Attributes, Buffer, BufferView, Image as GLTFImage, Material, Mesh, Node, PbrMetallicRoughness, Primitive, Scene, Texture, TextureInfo
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """glTF vertex colors are linear, while PLY camera colors are sRGB."""
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _project_camera_texture(top_vertices: np.ndarray, source_vertices: np.ndarray,
                            source_faces: np.ndarray, source_uvs: np.ndarray,
                            texture: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample the camera atlas only where the observed mesh supports the surface."""
    observed = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(source_vertices.astype(np.float32)),
        o3d.core.Tensor(source_faces.astype(np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(observed)
    projected = np.zeros((len(top_vertices), 3), dtype=np.float32)
    valid = np.zeros(len(top_vertices), dtype=bool)
    height, width = texture.shape[:2]
    for start in range(0, len(top_vertices), 65536):
        end = min(start + 65536, len(top_vertices))
        result = scene.compute_closest_points(o3d.core.Tensor(top_vertices[start:end]))
        face_ids = result['primitive_ids'].numpy().astype(np.int64)
        nearest = result['points'].numpy()
        bary = result['primitive_uvs'].numpy()
        uv_tri = source_uvs[face_ids]
        uv = uv_tri[:, 0] * (1 - bary[:, 0:1] - bary[:, 1:2])
        uv += uv_tri[:, 1] * bary[:, 0:1] + uv_tri[:, 2] * bary[:, 1:2]
        x = np.clip(uv[:, 0] * (width - 1), 0, width - 1)
        y = np.clip((1 - uv[:, 1]) * (height - 1), 0, height - 1)
        sampled = np.stack([ndi.map_coordinates(texture[:, :, channel].astype(np.float32), [y, x], order=1,
                                                mode='nearest') for channel in range(3)], axis=1)
        distance = np.linalg.norm(top_vertices[start:end] - nearest, axis=1)
        neutral = np.max(np.abs(sampled - 128), axis=1) < 10
        dark = np.max(sampled, axis=1) < 20
        usable = (distance < max(0.30, spacing * 3)) & ~neutral & ~dark
        projected[start:end] = sampled / 255.0
        valid[start:end] = usable
    return projected, valid


def _bake_surface_texture(height: np.ndarray, footprint: np.ndarray, colors: np.ndarray,
                          source: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                          lower: np.ndarray, spacing: float, resolution: int) -> tuple[np.ndarray, int]:
    """Bake observed camera texels onto a regular terrain UV map at source scale."""
    source_vertices, source_faces, source_uvs, atlas = source
    observed = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(source_vertices.astype(np.float32)),
        o3d.core.Tensor(source_faces.astype(np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(observed)
    nz, nx = height.shape
    pixels = np.zeros((resolution * resolution, 3), dtype=np.uint8)
    textured = 0
    ray_height = float(np.nanmax(height) + 10)
    atlas_h, atlas_w = atlas.shape[:2]
    for start in range(0, len(pixels), 65536):
        end = min(start + 65536, len(pixels))
        index = np.arange(start, end)
        col = index % resolution
        row = index // resolution
        gx = col * (nx - 1) / (resolution - 1)
        gz = (resolution - 1 - row) * (nz - 1) / (resolution - 1)
        predicted = ndi.map_coordinates(height, [gz, gx], order=1, mode='nearest')
        fallback = np.stack([
            ndi.map_coordinates(colors[:, :, channel], [gz, gx], order=1, mode='nearest')
            for channel in range(3)
        ], axis=1)
        chunk = np.rint(np.clip(fallback, 0, 1) * 255).astype(np.uint8)
        inside = footprint[np.rint(gz).astype(int), np.rint(gx).astype(int)]
        if inside.any():
            local = np.flatnonzero(inside)
            rays = np.zeros((len(local), 6), dtype=np.float32)
            rays[:, 0] = lower[0] + gx[local] * spacing
            rays[:, 1] = ray_height
            rays[:, 2] = lower[2] + gz[local] * spacing
            rays[:, 4] = -1
            hit = scene.cast_rays(o3d.core.Tensor(rays))
            distance = hit['t_hit'].numpy()
            reliable = np.isfinite(distance) & (np.abs(ray_height - distance - predicted[local]) < 0.35)
            if reliable.any():
                selected = local[reliable]
                face_ids = hit['primitive_ids'].numpy()[reliable].astype(np.int64)
                bary = hit['primitive_uvs'].numpy()[reliable]
                triangle_uv = source_uvs[face_ids]
                uv = triangle_uv[:, 0] * (1 - bary[:, :1] - bary[:, 1:2])
                uv += triangle_uv[:, 1] * bary[:, :1] + triangle_uv[:, 2] * bary[:, 1:2]
                tx = np.clip(uv[:, 0] * (atlas_w - 1), 0, atlas_w - 1)
                ty = np.clip((1 - uv[:, 1]) * (atlas_h - 1), 0, atlas_h - 1)
                sampled = np.stack([
                    ndi.map_coordinates(atlas[:, :, channel], [ty, tx], order=1, mode='nearest')
                    for channel in range(3)
                ], axis=1)
                neutral = np.max(np.abs(sampled.astype(np.int16) - 128), axis=1) < 10
                dark = sampled.max(axis=1) < 20
                good = ~neutral & ~dark
                chunk[selected[good]] = sampled[good]
                textured += int(good.sum())
        pixels[start:end] = chunk
    return pixels.reshape(resolution, resolution, 3), textured


def _refine_roof_planes(height: np.ndarray, texture: np.ndarray | None, points: np.ndarray,
                        point_colors: np.ndarray, lower: np.ndarray, spacing: float) -> int:
    """Regularize well-supported red roof facets without inventing rectangular roofs.

    The mesh atlas is not a map: its tiles are unrelated image patches. Roof
    detection therefore uses colored 3D samples and only edits their connected
    footprint (plus one grid cell for point sparsity). This avoids the previous
    minimum-area rectangle fill, which could flatten nearby trees and walls.
    """
    del texture  # Packed atlas colors have no global spatial meaning.
    nz, nx = height.shape
    if point_colors is None or len(point_colors) != len(points):
        return 0

    rgb = np.clip(point_colors, 0, 1)
    red = (rgb[:, 0] > rgb[:, 1] * 1.12) & (rgb[:, 0] > rgb[:, 2] * 1.15)
    red &= (rgb[:, 0] - rgb[:, 1] > 0.055) & (rgb[:, 0] > 0.16)
    if int(red.sum()) < 30:
        return 0

    gx = np.clip(((points[:, 0] - lower[0]) / spacing).astype(np.int32), 0, nx - 1)
    gz = np.clip(((points[:, 2] - lower[2]) / spacing).astype(np.int32), 0, nz - 1)
    cell = gz * nx + gx
    roof_grid = np.zeros((nz, nx), dtype=np.uint8)
    roof_grid[gz[red], gx[red]] = 1

    # Close sub-decimetre sampling gaps, preserving the measured footprint.
    roof_grid = cv2.morphologyEx(
        roof_grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8),
    )
    count, labels, components, _ = cv2.connectedComponentsWithStats(roof_grid)
    min_cells = max(10, int(np.ceil(0.16 / max(spacing ** 2, 1e-8))))
    support_radius = max(1, int(np.ceil(0.08 / spacing)))
    support_kernel = np.ones((2 * support_radius + 1, 2 * support_radius + 1), np.uint8)
    refined = 0

    for label in range(1, count):
        x, z, width, depth, area = components[label]
        if area < min_cells or max(width, depth) * spacing > 24:
            continue
        if min(width, depth) * spacing < 0.35:
            continue

        component = labels == label
        roof_points_mask = red & component.ravel()[cell]
        roof_points = points[roof_points_mask]
        if len(roof_points) < 18:
            continue

        # Robustly fit a plane to the colored roof samples. Refit twice using
        # a 9 cm residual gate to reject vegetation and mixed boundary points.
        design = np.column_stack((roof_points[:, 0], roof_points[:, 2], np.ones(len(roof_points))))
        inliers = np.ones(len(roof_points), dtype=bool)
        plane = np.zeros(3, dtype=np.float64)
        for _ in range(3):
            if inliers.sum() < 18:
                break
            plane = np.linalg.lstsq(design[inliers], roof_points[inliers, 1], rcond=None)[0]
            residual = np.abs(design @ plane - roof_points[:, 1])
            updated = residual < 0.09
            if np.array_equal(updated, inliers):
                break
            inliers = updated
        if inliers.sum() < 18 or np.median(residual[inliers]) > 0.055:
            continue
        plane = np.linalg.lstsq(design[inliers], roof_points[inliers, 1], rcond=None)[0]
        residual = np.abs(design @ plane - roof_points[:, 1])
        inliers = residual < 0.09
        if inliers.sum() < 18 or np.median(residual[inliers]) > 0.055:
            continue
        if np.max(np.abs(plane[:2])) > 1.8:
            continue

        # Preserve the connected support outline instead of filling its
        # rotated bounding rectangle. Only a narrow measured-edge margin is
        # included to account for the finite point spacing.
        patch = cv2.dilate(component.astype(np.uint8), support_kernel, iterations=1).astype(bool)
        iz, ix = np.nonzero(patch)
        if len(ix) == 0:
            continue
        world_x = lower[0] + ix * spacing
        world_z = lower[2] + iz * spacing
        plane_y = plane[0] * world_x + plane[1] * world_z + plane[2]
        # A roof fit may regularize noisy samples, but may not bridge a large
        # depth jump or pull unrelated vegetation onto the plane.
        plausible = np.abs(plane_y - height[iz, ix]) <= max(0.32, spacing * 3.5)
        if int(plausible.sum()) < min_cells:
            continue
        height[iz[plausible], ix[plausible]] = plane_y[plausible]
        refined += 1
    return refined


def _closed_heightfield(points: np.ndarray, colors: np.ndarray,
                        texture_source: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray | None]:
    """Interpolate the upper surface and close it with a shallow matching underside."""
    lower = points.min(axis=0)
    span = np.ptp(points, axis=0)
    spacing = float(np.clip(max(span[0], span[2]) / 500.0, 0.07, 0.2))
    nx = int(np.ceil(span[0] / spacing)) + 1
    nz = int(np.ceil(span[2] / spacing)) + 1
    ix = np.clip(((points[:, 0] - lower[0]) / spacing).astype(np.int32), 0, nx - 1)
    iz = np.clip(((points[:, 2] - lower[2]) / spacing).astype(np.int32), 0, nz - 1)
    cell = iz * nx + ix

    # Choose a robust upper quantile per cell, avoiding isolated high outliers.
    order = np.lexsort((points[:, 1], cell))
    keys, starts, counts = np.unique(cell[order], return_index=True, return_counts=True)
    rank = np.minimum((counts * 0.85).astype(np.int64), counts - 1)
    height = np.full(nx * nz, np.nan, dtype=np.float32)
    height[keys] = points[order[starts + rank], 1]
    height = height.reshape(nz, nx)
    observed = np.isfinite(height)

    # Keep the closing radius fixed in world units as the grid gets finer.
    footprint = ndi.binary_fill_holes(ndi.binary_closing(
        observed, iterations=max(3, int(np.ceil(0.55 / spacing))),
    ))
    labels, count = ndi.label(footprint)
    if count == 0:
        raise ValueError("Dense cloud has no continuous footprint")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    footprint = labels == int(sizes.argmax())
    if int(footprint.sum()) < 100:
        raise ValueError("Dense cloud footprint is too small for a complete model")

    nearest_cell = ndi.distance_transform_edt(~observed, return_distances=False, return_indices=True)
    interpolated = height[tuple(nearest_cell)].copy()
    missing = footprint & ~observed
    for _ in range(80):
        padded = np.pad(interpolated, 1, mode='edge')
        average = (padded[:-2, 1:-1] + padded[2:, 1:-1]
                   + padded[1:-1, :-2] + padded[1:-1, 2:]) * 0.25
        interpolated[missing] = average[missing]

    # Smooth point noise over a 0.3 m neighbourhood while retaining building
    # edges and terrace breaks. Apply it to measured and interpolated cells.
    interpolated_filtered = cv2.bilateralFilter(interpolated, d=7, sigmaColor=0.22, sigmaSpace=3.0)
    interpolated[footprint] = interpolated_filtered[footprint]

    # Do not extrapolate rectangular roofs from color alone. The camera pass
    # lacks enough side coverage to guarantee their true wall outlines.
    refined_roofs = 0

    xx = lower[0] + np.arange(nx) * spacing
    zz = lower[2] + np.arange(nz) * spacing
    x, z = np.meshgrid(xx, zz)
    grid_vertices = np.stack((x, interpolated, z), axis=-1).reshape(-1, 3)
    cell_ids = np.arange(nx * nz).reshape(nz, nx)
    a, b = cell_ids[:-1, :-1], cell_ids[:-1, 1:]
    c, d = cell_ids[1:, :-1], cell_ids[1:, 1:]
    valid = footprint[:-1, :-1] & footprint[:-1, 1:] & footprint[1:, :-1] & footprint[1:, 1:]
    original_valid = valid.copy()
    # A diagonal pair of raster cells has a bow-tie boundary at their shared
    # corner. Close those tiny gaps at the cell level before meshing; this is
    # what keeps the front, side walls, and inferred underside manifold.
    for _ in range(8):
        va, vb = valid[:-1, :-1], valid[:-1, 1:]
        vc, vd = valid[1:, :-1], valid[1:, 1:]
        diagonal = ((va & vd & ~vb & ~vc) | (vb & vc & ~va & ~vd))
        if not diagonal.any():
            break
        valid[:-1, :-1] |= diagonal
        valid[:-1, 1:] |= diagonal
        valid[1:, :-1] |= diagonal
        valid[1:, 1:] |= diagonal
    top_faces = np.concatenate((
        np.stack((a[valid], d[valid], b[valid]), axis=1),
        np.stack((a[valid], c[valid], d[valid]), axis=1),
    )).astype(np.uint32)
    used = np.unique(top_faces)
    remap = np.full(len(grid_vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    top_faces = remap[top_faces].astype(np.uint32)
    top_vertices = grid_vertices[used].astype(np.float32)
    n = len(top_vertices)

    # Find the perimeter once; the back cap is set after texture-based roof
    # refinement so it remains below every surface vertex.
    directed = np.concatenate((top_faces[:, [0, 1]], top_faces[:, [1, 2]], top_faces[:, [2, 0]]))
    _, inverse, edge_counts = np.unique(np.sort(directed, axis=1), axis=0,
                                        return_inverse=True, return_counts=True)
    boundary = directed[edge_counts[inverse] == 1]
    side_faces = np.concatenate((
        np.stack((boundary[:, 1], boundary[:, 0], boundary[:, 0] + n), axis=1),
        np.stack((boundary[:, 1], boundary[:, 0] + n, boundary[:, 1] + n), axis=1),
    )).astype(np.uint32)

    tree = cKDTree(points)
    distances, neighbors = tree.query(top_vertices, k=8, workers=-1)
    weights = 1 / np.maximum(distances, 0.02) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    top_colors = (colors[neighbors] * weights[..., None]).sum(axis=1)
    textured_vertices = 0
    texture_warning = None
    if texture_source is not None:
        try:
            atlas_colors, usable = _project_camera_texture(top_vertices, *texture_source, spacing)
            top_colors[usable] = atlas_colors[usable]
            textured_vertices = int(usable.sum())
        except (RuntimeError, ValueError, IndexError) as exc:
            texture_warning = f'Camera texture projection unavailable: {exc}'
    baked_texture = None
    textured_pixels = 0
    if texture_source is not None and texture_warning is None:
        try:
            resolution = int(np.clip(np.ceil(max(span[0], span[2]) / 0.0215 / 256) * 256, 1024, 3072))
            color_grid = np.zeros((nz * nx, 3), dtype=np.float32)
            color_grid[used] = top_colors
            color_grid = color_grid.reshape(nz, nx, 3)
            nearest_color = ndi.distance_transform_edt(~np.isin(np.arange(nx * nz), used).reshape(nz, nx),
                                                        return_distances=False, return_indices=True)
            color_grid = color_grid[tuple(nearest_color)]
            baked_texture, textured_pixels = _bake_surface_texture(
                interpolated, footprint, color_grid, texture_source, lower, spacing, resolution,
            )
        except (RuntimeError, ValueError, IndexError) as exc:
            texture_warning = f'High-resolution camera texture unavailable: {exc}'
    # A flat, one-metre-scale base makes the inferred surface a solid object
    # from every orbit angle instead of a pair of nearly coincident skins.
    thickness = float(np.clip(spacing * 1.5, 0.12, 0.3))
    base_y = float(top_vertices[:, 1].min() - thickness)
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 1] = base_y
    faces = np.concatenate((top_faces, top_faces[:, [0, 2, 1]] + n, side_faces))
    vertices = np.concatenate((top_vertices, bottom_vertices))

    # The photo belongs on the measured upper surface; give the cut sides and
    # underside a neutral earth finish instead of stretching the aerial image.
    bottom_colors = np.tile(np.array([0.24, 0.20, 0.15], dtype=np.float32), (n, 1))
    linear_colors = _srgb_to_linear(np.concatenate((top_colors, bottom_colors)))
    rgba = np.concatenate((np.rint(linear_colors * 255).astype(np.uint8),
                           np.full((2 * n, 1), 255, dtype=np.uint8)), axis=1)
    stats = {
        'method': 'Aligned terrain height field with interpolated gaps and flat solid base',
        'grid_spacing_m': round(spacing, 4),
        'observed_grid_cells': int((observed & footprint).sum()),
        'interpolated_grid_cells': int(missing.sum()),
        'manifold_bridge_cells': int((valid & ~original_valid).sum()),
        'nearest_point_distance_p95_m': round(float(np.quantile(distances[:, 0], .95)), 3),
        'camera_textured_vertices': textured_vertices,
        'camera_textured_pixels': textured_pixels,
        'refined_roof_patches': refined_roofs,
        **({'texture_resolution_px': int(baked_texture.shape[0])} if baked_texture is not None else {}),
        **({'camera_texture_warning': texture_warning} if texture_warning else {}),
        'shell_thickness_m': round(thickness, 3),
        'shell_base_elevation_m': round(base_y, 3),
        'shell_material': 'Photo texture on top; neutral earth on sides and base',
        'top_triangles': int(len(top_faces)),
        'bottom_triangles': int(len(top_faces)),
        'side_triangles': int(len(side_faces)),
        'warning': 'The opposite and underside are inferred from the same single-pass points, not measured.',
    }
    return vertices, faces, rgba, stats, baked_texture


def export_estimated_surface(dense_ply: Path, output_glb: Path,
                             rotation: np.ndarray, center: np.ndarray,
                             texture_source: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None) -> dict:

    cloud = o3d.io.read_point_cloud(str(dense_ply))
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    center = np.asarray(center, dtype=np.float64).reshape(1, 3)
    # Fit the terrain in the same upright, centered coordinate frame used by
    # model.glb. Fitting before this transform treats the arbitrary COLMAP
    # camera axes as east/up/north and folds the scene into tall, torn ridges.
    measured = np.asarray(cloud.points, dtype=np.float64) @ rotation.T - center
    if len(measured) < 1000 or not cloud.has_colors():
        raise ValueError("Complete model requires a colored dense point cloud")
    if texture_source is not None:
        source_vertices, source_faces, source_uvs, atlas = texture_source
        source_vertices = np.asarray(source_vertices, dtype=np.float64) @ rotation.T - center
        texture_source = (source_vertices.astype(np.float32), source_faces, source_uvs, atlas)
    vertices, faces, rgba, stats, baked_texture = _closed_heightfield(measured, np.asarray(cloud.colors), texture_source)
    stats['coordinate_frame'] = 'Upright and centered to the textured mesh before surface fitting'
    # Verify geometry before writing a model that claims to be complete.
    validation = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not validation.is_watertight or not validation.is_winding_consistent:
        raise ValueError("Interpolated shell has an open or non-manifold boundary")

    # _closed_heightfield already returned coordinates in the final mesh frame.
    positions = vertices.astype(np.float32)
    top_face_count = int(stats['top_triangles'])
    top_index_bytes = faces[:top_face_count].astype('<u4', copy=False).reshape(-1).tobytes()
    shell_index_bytes = faces[top_face_count:].astype('<u4', copy=False).reshape(-1).tobytes()
    pos_bytes = positions.astype('<f4', copy=False).tobytes()
    normal_bytes = np.asarray(validation.vertex_normals, dtype='<f4').tobytes()
    earth_color = [0.24, 0.20, 0.15, 1.0]
    gltf = GLTF2()
    gltf.bufferViews = [BufferView(buffer=0, byteOffset=0, byteLength=len(pos_bytes), target=34962)]
    gltf.accessors = [Accessor(bufferView=0, componentType=5126, count=len(positions), type='VEC3',
                               min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist())]
    if baked_texture is not None:
        lower = measured.min(axis=0)
        span = np.ptp(measured, axis=0)
        spacing = float(stats['grid_spacing_m'])
        nx, nz = int(np.ceil(span[0] / spacing)) + 1, int(np.ceil(span[2] / spacing)) + 1
        uv = np.column_stack(((vertices[:, 0] - lower[0]) / ((nx - 1) * spacing),
                              1 - (vertices[:, 2] - lower[2]) / ((nz - 1) * spacing))).astype(np.float32)
        image_buffer = io.BytesIO()
        PILImage.fromarray(baked_texture).save(image_buffer, format='PNG')
        uv_bytes, image_bytes = uv.astype('<f4', copy=False).tobytes(), image_buffer.getvalue()
        index_bytes = faces.astype('<u4', copy=False).reshape(-1).tobytes()
        chunks = [pos_bytes, uv_bytes, normal_bytes, index_bytes, image_bytes]
        offsets = np.cumsum([0] + [len(chunk) for chunk in chunks[:-1]])
        gltf.bufferViews += [
            BufferView(buffer=0, byteOffset=int(offsets[1]), byteLength=len(uv_bytes), target=34962),
            BufferView(buffer=0, byteOffset=int(offsets[2]), byteLength=len(normal_bytes), target=34962),
            BufferView(buffer=0, byteOffset=int(offsets[3]), byteLength=len(index_bytes), target=34963),
            BufferView(buffer=0, byteOffset=int(offsets[4]), byteLength=len(image_bytes)),
        ]
        gltf.accessors += [
            Accessor(bufferView=1, componentType=5126, count=len(uv), type='VEC2'),
            Accessor(bufferView=2, componentType=5126, count=len(positions), type='VEC3'),
            Accessor(bufferView=3, componentType=5125, count=len(index_bytes) // 4, type='SCALAR'),
        ]
        gltf.images = [GLTFImage(bufferView=4, mimeType='image/png')]
        gltf.textures = [Texture(source=0)]
        gltf.materials = [
            Material(name='Complete surface with camera texture and terrain shading',
                     doubleSided=True,
                     pbrMetallicRoughness=PbrMetallicRoughness(
                         baseColorTexture=TextureInfo(index=0), metallicFactor=0, roughnessFactor=1.0),
                     extensions={'KHR_materials_unlit': {}}),
        ]
        gltf.extensionsUsed = ['KHR_materials_unlit']
        primitives = [
            Primitive(attributes=Attributes(POSITION=0, TEXCOORD_0=1, NORMAL=2), indices=3, material=0),
        ]
    else:
        color_bytes = rgba.tobytes()
        # Keep the closed vertex-colored shell in one glTF primitive. Splitting
        # top and sides into separate primitives makes GLB readers duplicate
        # shared boundary vertices and report a visually closed model as open.
        index_bytes = faces.astype('<u4', copy=False).reshape(-1).tobytes()
        chunks = [pos_bytes, normal_bytes, color_bytes, index_bytes]
        offsets = np.cumsum([0] + [len(chunk) for chunk in chunks[:-1]])
        gltf.bufferViews += [
            BufferView(buffer=0, byteOffset=int(offsets[1]), byteLength=len(normal_bytes), target=34962),
            BufferView(buffer=0, byteOffset=int(offsets[2]), byteLength=len(color_bytes), target=34962),
            BufferView(buffer=0, byteOffset=int(offsets[3]), byteLength=len(index_bytes), target=34963),
        ]
        gltf.accessors += [
            Accessor(bufferView=1, componentType=5126, count=len(positions), type='VEC3'),
            Accessor(bufferView=2, componentType=5121, normalized=True, count=len(rgba), type='VEC4'),
            Accessor(bufferView=3, componentType=5125, count=len(index_bytes) // 4, type='SCALAR'),
        ]
        gltf.materials = [
            Material(name='Measured terrain colors',
                     pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0, roughnessFactor=0.96)),
        ]
        primitives = [
            Primitive(attributes=Attributes(POSITION=0, NORMAL=1, COLOR_0=2), indices=3, material=0),
        ]
    gltf.meshes = [Mesh(primitives=primitives)]
    gltf.nodes = [Node(mesh=0)]
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0
    blob = b''.join(chunks)
    gltf.buffers = [Buffer(byteLength=len(blob))]
    gltf.set_binary_blob(blob)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(output_glb))
    stats.update(estimated_vertices=int(len(vertices)), estimated_triangles=int(len(faces)), watertight=True)
    output_glb.with_suffix('.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    return stats
