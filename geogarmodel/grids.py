"""Regular grids: ascending X/Y, values indexed [y, x], node coordinates."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import griddata


@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    name: str = "Field"
    unit: str = ""
    crs: str = ""
    coordinate_unit: str = "unknown"
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.x.ndim != 1 or self.y.ndim != 1 or min(len(self.x), len(self.y)) < 2:
            raise ValueError("Сетка должна содержать минимум 2×2 узла.")
        if self.values.shape != (len(self.y), len(self.x)):
            raise ValueError("Размер values должен быть (ny, nx).")
        for axis in (self.x, self.y):
            d = np.diff(axis)
            if not np.isfinite(axis).all() or not (d > 0).all() or not np.allclose(d, d[0], rtol=1e-5, atol=1e-9):
                raise ValueError("Оси должны быть возрастающими и равномерными.")
        if np.isinf(self.values).any() or not np.isfinite(self.values).any():
            raise ValueError("Сетка не содержит конечных данных или содержит Inf.")

    @property
    def dx(self):
        return float(self.x[1] - self.x[0])

    @property
    def dy(self):
        return float(self.y[1] - self.y[0])

    def derived(self, values, name, unit=None, parameters=None):
        return Grid(self.x.copy(), self.y.copy(), values, name,
                    self.unit if unit is None else unit, self.crs, self.coordinate_unit,
                    self.history + [{"operation": name, "parameters": parameters or {}}])

    def crop(self, xmin, xmax, ymin, ymax):
        ix = (self.x >= min(xmin, xmax)) & (self.x <= max(xmin, xmax))
        iy = (self.y >= min(ymin, ymax)) & (self.y <= max(ymin, ymax))
        return Grid(self.x[ix], self.y[iy], self.values[np.ix_(iy, ix)].copy(),
                    self.name + " · область", self.unit, self.crs, self.coordinate_unit,
                    self.history + [{"operation": "crop", "bounds": [xmin, xmax, ymin, ymax]}])


def save_grid(grid, path):
    """Lossless NPZ includes mask, CRS, units and processing provenance."""
    metadata = {k: getattr(grid, k) for k in ("name", "unit", "crs", "coordinate_unit", "history")}
    np.savez_compressed(path, x=grid.x, y=grid.y, values=grid.values,
                        metadata=json.dumps(metadata, ensure_ascii=False))


def load_grid(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return Grid(data["x"], data["y"], data["values"], **json.loads(str(data["metadata"])))
    return read_zmap(path)


def read_zmap(path):
    """ZMAP+ ASCII, fixed-width values, columns north→south; D exponents supported."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = Path(path).read_text(encoding="cp1251")
    lines = [s.rstrip("\n") for s in text.splitlines()
             if s.strip() and not s.lstrip().startswith("!")]
    start = next((i for i, s in enumerate(lines) if s.lstrip().startswith("@")), None)
    if start is None or start + 4 >= len(lines):
        raise ValueError("Не найден заголовок ZMAP+.")
    parts = lines[start + 1].split(",")
    width, null = int(parts[0]), float(parts[1])
    header = lines[start + 2].split(",")
    ny, nx = int(header[0]), int(header[1])
    xmin, xmax, ymin, ymax = map(float, header[2:6])
    end = next(i for i in range(start + 3, len(lines)) if lines[i].strip() == "@")
    values = []
    for line in lines[end + 1:]:
        # Fixed-width is essential when negative numbers touch their neighbours.
        chunks = [line[i:i + width].strip() for i in range(0, len(line), width)]
        try:
            values.extend(float(c.replace("D", "E").replace("d", "e")) for c in chunks if c)
        except ValueError as exc:
            raise ValueError("Некорректные данные ZMAP (ширина поля / число).") from exc
    if len(values) != nx * ny:
        raise ValueError(f"ZMAP: ожидалось {nx * ny} значений, получено {len(values)}.")
    array = np.asarray(values).reshape(nx, ny).T[::-1]
    array[array == null] = np.nan
    return Grid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), array, Path(path).stem)


def points_to_grid(x, y, values, nx=200, ny=200, method="linear", **metadata):
    if method not in {"linear", "nearest", "cubic"}:
        raise ValueError("Метод интерполяции: linear, nearest или cubic.")
    points = np.column_stack([x, y, values]).astype(float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3 or np.linalg.matrix_rank(points[:, :2] - points[0, :2]) < 2:
        raise ValueError("Для карты нужны минимум три неколлинеарные точки.")
    # Average duplicate positions rather than hand arbitrary duplicates to QHull.
    xy, inverse = np.unique(points[:, :2], axis=0, return_inverse=True)
    z = np.bincount(inverse, weights=points[:, 2]) / np.bincount(inverse)
    gx = np.linspace(xy[:, 0].min(), xy[:, 0].max(), nx)
    gy = np.linspace(xy[:, 1].min(), xy[:, 1].max(), ny)
    xx, yy = np.meshgrid(gx, gy)
    return Grid(gx, gy, griddata(xy, z, (xx, yy), method=method), **metadata)
