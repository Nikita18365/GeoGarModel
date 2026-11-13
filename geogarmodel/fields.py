"""Potential-field transforms. z derivative is positive DOWNWARD (geophysical convention)."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .grids import Grid
from .coordinates import validate_crs

TRANSFORMS = {
    "dx": "Производная X", "dy": "Производная Y",
    "dz": "1-я вертикальная производная", "dzz": "2-я вертикальная производная",
    "horizontal": "Полный горизонтальный градиент", "analytic": "Аналитический сигнал",
    "tilt": "Tilt (радианы)", "upward": "Продолжение вверх / региональное поле",
    "residual": "Локальное поле (исходное − продолженное вверх)",
    "trend": "Полиномиальный региональный тренд", "detrend": "Локальное поле (− тренд)",
    "lowpass": "Гауссов низкочастотный фильтр", "highpass": "Гауссов высокочастотный фильтр",
}


def transform(grid: Grid, operation, *, height=1000., wavelength=5000., degree=1, pad=0.25):
    if operation not in TRANSFORMS:
        raise ValueError("Неизвестная трансформанта.")
    if grid.coordinate_unit != "m":
        raise ValueError("Для трансформант задайте плоские координаты в метрах; градусы недопустимы.")
    validate_crs(grid.crs, grid.coordinate_unit)
    if not np.isfinite(pad) or not 0 <= pad <= 1:
        raise ValueError("pad должен лежать в [0, 1].")
    mask = ~np.isfinite(grid.values)
    if mask.mean() > .5:
        raise ValueError("Более половины узлов пусты: сначала подготовьте сетку.")
    if operation in {"trend", "detrend"}:
        if degree not in (1, 2, 3):
            raise ValueError("Степень тренда: 1–3.")
        xx, yy = np.meshgrid(np.linspace(-1, 1, len(grid.x)), np.linspace(-1, 1, len(grid.y)))
        design = np.stack([xx ** i * yy ** j for i in range(degree + 1)
                           for j in range(degree + 1 - i)], axis=-1)
        if np.linalg.matrix_rank(design[~mask]) < design.shape[-1]:
            raise ValueError("Недостаточно независимых узлов для тренда.")
        coef = np.linalg.lstsq(design[~mask], grid.values[~mask], rcond=None)[0]
        regional = design @ coef
        result = regional if operation == "trend" else grid.values - regional
        params = {"degree": degree}
        unit = grid.unit
    else:
        if not np.isfinite(height) or height < 0:
            raise ValueError("Высота продолжения должна быть конечной и ≥ 0 м.")
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("Длина волны должна быть конечной и > 0 м.")
        # Fill holes only internally; restore the original data mask on output.
        filled = grid.values.copy()
        if mask.any():
            indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
            filled[mask] = filled[tuple(indices[:, mask])]
        py, px = [int(np.ceil(n * pad)) for n in filled.shape]
        padded = np.pad(filled, ((py, py), (px, px)), mode="reflect")
        mean = padded.mean()
        spectrum = np.fft.fft2(padded - mean)
        kx = 2 * np.pi * np.fft.fftfreq(padded.shape[1], grid.dx)[None, :]
        ky = 2 * np.pi * np.fft.fftfreq(padded.shape[0], grid.dy)[:, None]
        k = np.hypot(kx, ky)
        def inverse(multiplier):
            full = np.fft.ifft2(spectrum * multiplier).real
            return full[py:py + len(grid.y), px:px + len(grid.x)]
        if operation in {"upward", "residual"}:
            regional = inverse(np.exp(-k * height)) + mean
            result = regional if operation == "upward" else grid.values - regional
            unit = grid.unit
        elif operation in {"lowpass", "highpass"}:
            regional = inverse(np.exp(-.5 * (k * wavelength / (2 * np.pi)) ** 2)) + mean
            result = regional if operation == "lowpass" else grid.values - regional
            unit = grid.unit
        elif operation == "dzz":
            result, unit = inverse(k ** 2), f"{grid.unit}/m²"
        else:
            dx, dy, dz = inverse(1j * kx), inverse(1j * ky), inverse(k)
            horizontal = np.hypot(dx, dy)
            result = {"dx": dx, "dy": dy, "dz": dz, "horizontal": horizontal,
                      "analytic": np.sqrt(dx ** 2 + dy ** 2 + dz ** 2),
                      "tilt": np.arctan2(dz, horizontal)}[operation]
            unit = "rad" if operation == "tilt" else f"{grid.unit}/m"
        params = {"height_m": height, "wavelength_m": wavelength, "pad_fraction": pad,
                  "hole_fill": "nearest (output mask restored)", "z_positive": "down"}
    result[mask] = np.nan
    return grid.derived(result, TRANSFORMS[operation], unit, params)
