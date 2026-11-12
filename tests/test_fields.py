import numpy as np
import pytest
from geogarmodel.grids import Grid, save_grid, load_grid, read_zmap, points_to_grid
from geogarmodel.fields import transform, TRANSFORMS


@pytest.fixture
def wave():
    x = np.arange(64.) * 100
    y = np.arange(48.) * 150
    xx, yy = np.meshgrid(x, y)
    z = 12 + 3 * np.cos(2 * np.pi * xx / 6400) + 2 * np.sin(2 * np.pi * yy / 7200)
    return Grid(x, y, z, "synthetic", "mGal", "EPSG:20910", "m")


def test_spectral_derivatives_and_sign(wave):
    xx, yy = np.meshgrid(wave.x, wave.y)
    kx, ky = 2 * np.pi / 6400, 2 * np.pi / 7200
    expected_x = -3 * kx * np.sin(kx * xx)
    expected_y = 2 * ky * np.cos(ky * yy)
    expected_z = 3 * kx * np.cos(kx * xx) + 2 * ky * np.sin(ky * yy)
    for op, expected in [("dx", expected_x), ("dy", expected_y), ("dz", expected_z)]:
        np.testing.assert_allclose(transform(wave, op, pad=0).values, expected, atol=1e-14)
    analytic = np.sqrt(expected_x ** 2 + expected_y ** 2 + expected_z ** 2)
    np.testing.assert_allclose(transform(wave, "analytic", pad=0).values, analytic, atol=1e-14)
    np.testing.assert_allclose(transform(wave, "tilt", pad=0).values,
                               np.arctan2(expected_z, np.hypot(expected_x, expected_y)), atol=1e-12)


def test_upward_attenuation_and_residual(wave):
    xx, yy = np.meshgrid(wave.x, wave.y)
    h = 750
    expected = 12 + 3 * np.cos(2 * np.pi * xx / 6400) * np.exp(-2 * np.pi * h / 6400)
    expected += 2 * np.sin(2 * np.pi * yy / 7200) * np.exp(-2 * np.pi * h / 7200)
    regional = transform(wave, "upward", height=h, pad=0)
    np.testing.assert_allclose(regional.values, expected, atol=1e-12)
    residual = transform(wave, "residual", height=h, pad=0)
    np.testing.assert_allclose(regional.values + residual.values, wave.values)
    np.testing.assert_allclose(transform(wave, "upward", height=0).values, wave.values, atol=1e-12)


@pytest.mark.parametrize("op", list(TRANSFORMS))
def test_constant_field(op):
    grid = Grid(np.arange(10.), np.arange(12.), np.full((12, 10), 7.), coordinate_unit="m")
    result = transform(grid, op)
    expected = 7 if op in {"trend", "upward", "lowpass"} else 0
    np.testing.assert_allclose(result.values, expected, atol=1e-12)


def test_polynomial_and_mask(wave):
    xx, yy = np.meshgrid(wave.x, wave.y)
    wave.values = 2 + .5 * xx - 3 * yy + .002 * xx * yy
    wave.values[4:7, 5:8] = np.nan
    result = transform(wave, "detrend", degree=2)
    assert np.array_equal(np.isnan(result.values), np.isnan(wave.values))
    np.testing.assert_allclose(result.values[np.isfinite(result.values)], 0, atol=1e-7)
    fft = transform(wave, "dx")
    assert np.array_equal(np.isnan(fft.values), np.isnan(wave.values))


def test_reject_geographic_and_bad_parameters(wave):
    wave.coordinate_unit = "degrees"
    with pytest.raises(ValueError, match="метрах"):
        transform(wave, "dx")
    wave.coordinate_unit = "m"
    for args in [dict(height=-1), dict(height=float("nan")), dict(pad=-.1), dict(wavelength=0)]:
        with pytest.raises(ValueError):
            transform(wave, "upward", **args)


def test_npz_roundtrip(wave, tmp_path):
    wave.values[0, 0] = np.nan
    result = transform(wave, "analytic")
    path = tmp_path / "map.npz"
    save_grid(result, path)
    loaded = load_grid(path)
    np.testing.assert_allclose(loaded.values, result.values)
    assert loaded.history == result.history
    assert loaded.crs == wave.crs


def test_zmap_column_major_north_to_south(tmp_path):
    path = tmp_path / "test.zmap"
    path.write_text("! demo\n@grid, GRID, 3\n10,-9999,,4,1\n3,2,100,200,0,20\n0,0,0\n@\n"
                    + "".join(f"{v:10.4f}" for v in [3, 2, 1]) + "\n"
                    + "".join(f"{v:10.4f}" for v in [6, -9999, 4]) + "\n")
    grid = read_zmap(path)
    np.testing.assert_allclose(grid.values, [[1, 4], [2, np.nan], [3, 6]])


def test_xyz_convex_hull_and_duplicates():
    grid = points_to_grid([0, 1, 0, 0], [0, 0, 1, 0], [2, 2, 2, 4], nx=3, ny=3)
    assert grid.values[0, 0] == 3
    assert np.isnan(grid.values[-1, -1])
