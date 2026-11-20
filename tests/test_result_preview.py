from types import SimpleNamespace

import numpy as np

from geogarmodel.inversion import Mesh
from geogarmodel.result_preview import preview_volume, surface_faces


def test_volume_weighted_preview_respects_clipping_mask_and_vectors():
    mesh = Mesh([0, 1, 4], [0, 2], [0, 1, 2])
    bounds = mesh.cells()
    bounds[0, 2] = .5
    mesh = Mesh(mesh.east, mesh.north, mesh.depth, [1, 0, 1, 0], bounds)
    vectors = np.array([[3, 4, 0], [800, 0, 0], [0, 0, 2], [900, 0, 0]])
    result = SimpleNamespace(mesh=mesh, model=vectors)
    axes, values = preview_volume(result, limit=1)
    np.testing.assert_allclose(values, [[[(5 + 6 * 2) / 7]]])
    np.testing.assert_equal(axes[0], [0, 4])
    _, exact = preview_volume(result)
    np.testing.assert_equal(exact.ravel(), [5, np.nan, 2, np.nan])


def test_preview_only_draws_exterior_and_exposes_deeper_values_on_cut():
    axes = (np.arange(2), np.arange(2), np.arange(3))
    cube = np.array([[[1., 2.]]])
    faces, colors = surface_faces(axes, cube)
    assert faces.shape == (10, 4, 3)
    np.testing.assert_equal(np.sort(colors), [1.] * 5 + [2.] * 5)
    cut, colors = surface_faces(axes, cube, top=1)
    assert cut.shape == (6, 4, 3) and cut[..., 2].min() == 1
    np.testing.assert_equal(colors, np.full(6, 2.))
