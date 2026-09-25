import numpy as np
import open3d as o3d
from pygltflib import GLTF2

from app.reconstruction.meshing import MeshProcessor


def test_export_has_only_observed_surface_without_backing_wall(tmp_path):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array([
        [0, 0, 0], [1, 0, 0], [0, 0, 1],
        [1, 0, 0], [1, 0, 1], [0, 0, 1],
    ]))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [3, 4, 5]]))
    mesh.compute_vertex_normals()
    output = tmp_path / 'output'
    MeshProcessor(tmp_path).export_mesh_deliverables(
        mesh,
        np.zeros((6, 2), dtype=np.float32),
        np.full((8, 8, 3), 128, dtype=np.uint8),
        output,
    )

    glb = GLTF2().load(str(output / 'model.glb'))
    assert len(glb.meshes[0].primitives) == 1
    assert len(glb.materials) == 1
    assert glb.materials[0].doubleSided is True
    assert glb.accessors[0].count == 6
    assert 'material_backing' not in (output / 'model.mtl').read_text()
