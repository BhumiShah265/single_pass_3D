import numpy as np
import open3d as o3d
from pygltflib import GLTF2
import trimesh

from app.reconstruction.estimated_surface import export_estimated_surface, _bake_surface_texture, _project_camera_texture


def test_camera_atlas_projection_keeps_detail_only_near_measured_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)[faces]
    atlas = np.zeros((16, 16, 3), dtype=np.uint8)
    atlas[:, :8] = [220, 30, 30]
    atlas[:, 8:] = [30, 180, 40]
    target = np.array([[0.2, 0, 0.5], [0.8, 0, 0.5], [0.2, 1, 0.5]], dtype=np.float32)

    color, valid = _project_camera_texture(target, vertices, faces, uvs, atlas, 0.1)

    assert valid.tolist() == [True, True, False]
    assert color[0, 0] > color[0, 1]
    assert color[1, 1] > color[1, 0]


def test_baked_texture_keeps_camera_pixels_between_mesh_vertices():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)[faces]
    atlas = np.zeros((16, 16, 3), dtype=np.uint8)
    atlas[:, :8] = [220, 30, 30]
    atlas[:, 8:] = [30, 180, 40]
    fallback = np.full((5, 5, 3), [0.2, 0.2, 0.8], dtype=np.float32)

    baked, textured = _bake_surface_texture(
        np.zeros((5, 5), dtype=np.float32), np.ones((5, 5), dtype=bool), fallback,
        (vertices, faces, uvs, atlas), np.zeros(3), 0.25, 64,
    )

    assert textured > 3000
    assert baked[32, 12, 0] > baked[32, 12, 1]
    assert baked[32, 50, 1] > baked[32, 50, 0]


def test_estimate_closes_a_missing_region_without_a_tall_backing_wall(tmp_path):
    x, z = np.meshgrid(np.linspace(-2, 2, 80), np.linspace(-2, 2, 80))
    keep = x*x + z*z > 0.5**2
    points = np.stack((x[keep], 0.15 * np.sin(x[keep]), z[keep]), axis=1)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.tile([0.3, 0.6, 0.4], (len(points), 1)))
    dense = tmp_path / 'dense.ply'
    o3d.io.write_point_cloud(str(dense), cloud)

    output = tmp_path / 'estimated.glb'
    stats = export_estimated_surface(dense, output, np.eye(3), np.zeros(3))
    glb = GLTF2().load(str(output))
    view = glb.bufferViews[0]
    positions = np.frombuffer(glb.binary_blob(), dtype='<f4',
                              count=glb.accessors[0].count * 3,
                              offset=view.byteOffset).reshape(-1, 3)
    color_accessor = glb.accessors[glb.meshes[0].primitives[0].attributes.COLOR_0]
    color_view = glb.bufferViews[color_accessor.bufferView]
    vertex_colors = np.frombuffer(glb.binary_blob(), dtype=np.uint8,
                                  count=color_accessor.count * 4,
                                  offset=color_view.byteOffset + (color_accessor.byteOffset or 0)).reshape(-1, 4)
    loaded = trimesh.load(output)
    model = next(iter(loaded.geometry.values()))

    assert len(glb.meshes[0].primitives) == 1
    assert color_accessor.type == 'VEC4'
    assert stats['estimated_triangles'] > 100
    assert stats['interpolated_grid_cells'] > 0
    assert model.is_watertight
    assert model.is_winding_consistent
    assert np.ptp(positions[:, 1]) < 1.0
    # PLY RGB is sRGB; glTF COLOR_0 must contain linear-light values.
    expected_linear = np.where(np.array([0.3, 0.6, 0.4]) <= 0.04045,
                               np.array([0.3, 0.6, 0.4]) / 12.92,
                               ((np.array([0.3, 0.6, 0.4]) + 0.055) / 1.055) ** 2.4)
    assert np.allclose(np.median(vertex_colors[:len(vertex_colors)//2, :3], axis=0) / 255,
                       expected_linear, atol=0.01)
