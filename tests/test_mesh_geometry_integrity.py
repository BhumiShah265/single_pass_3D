"""Test suite for mesh geometric integrity and boundary properties."""
import numpy as np
import pytest
import trimesh

from app.reconstruction.estimated_surface import _closed_heightfield, _srgb_to_linear


def test_closed_heightfield_produces_solid_plinth():
    """Verify that _closed_heightfield creates a watertight solid block with a flat datum base."""
    x = np.linspace(-5, 5, 50)
    z = np.linspace(-5, 5, 50)
    xx, zz = np.meshgrid(x, z)
    yy = 2.0 + 0.8 * np.sin(xx * 0.5) * np.cos(zz * 0.5)

    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    colors = np.full((len(points), 3), [0.3, 0.7, 0.4], dtype=np.float32)

    vertices, faces, rgba, stats, _ = _closed_heightfield(points, colors)

    # 1. Output must be watertight
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight, "Estimated surface plinth must be watertight"
    assert mesh.is_winding_consistent, "Faces must have consistent winding"

    # 2. The inferred underside is a flat base below every surface vertex.
    n_top = len(vertices) // 2
    top_v = vertices[:n_top]
    bottom_v = vertices[n_top:]
    offsets = top_v[:, 1] - bottom_v[:, 1]
    assert np.all(offsets > 0), "Underside vertices must sit strictly below top surface"
    assert np.ptp(bottom_v[:, 1]) < 1e-5, "The display base must be level"
    assert offsets.min() >= stats['shell_thickness_m'] - 0.01


def test_srgb_to_linear_color_conversion():
    """Verify sRGB to linear conversion conforms to IEC 61966-2-1."""
    assert np.allclose(_srgb_to_linear(np.array([0.0, 1.0])), [0.0, 1.0], atol=1e-5)
    mid = _srgb_to_linear(np.array([0.5]))
    assert 0.21 < mid[0] < 0.22


def test_mesh_boundary_normals_no_spikes():
    """Verify boundary faces don't produce upward-facing spikes."""
    x = np.linspace(-3, 3, 30)
    z = np.linspace(-3, 3, 30)
    xx, zz = np.meshgrid(x, z)
    yy = 1.0 + 0.3 * np.cos(xx)

    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    colors = np.full((len(points), 3), [0.4, 0.6, 0.5], dtype=np.float32)

    vertices, faces, rgba, stats, _ = _closed_heightfield(points, colors)
    n_top = len(vertices) // 2
    top_v = vertices[:n_top]
    assert np.ptp(top_v[:, 1]) < 3.5, "Top mesh elevation must not contain wild spike artifacts"
