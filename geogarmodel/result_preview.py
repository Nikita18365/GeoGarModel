"""Bounded display geometry. Full resolution arrays belong to the inversion result."""
from .work_progress import report
import numpy as np


def preview_volume(result, limit=32):
    """Volume-weighted property on a small display grid, including clipped prisms."""
    mesh = result.mesh
    axes = (mesh.east, mesh.north, mesh.depth)
    cuts = [np.linspace(0, n, min(n, limit) + 1, dtype=int) for n in mesh.shape]
    shape = tuple(len(edges) - 1 for edges in cuts)
    total = np.zeros(shape)
    weight = np.zeros(shape)
    model = result.model.reshape(mesh.shape + result.model.shape[1:])
    active = mesh.active.reshape(mesh.shape)
    bounds = None if mesh.bounds is None else mesh.bounds.reshape(mesh.shape + (6,))
    dx, dy, dz = map(np.diff, axes)
    # Limit temporary arrays even for wide, thin grids. Each tile starts/ends at bin edges.
    for bx, (x0, x1) in enumerate(zip(cuts[0][:-1], cuts[0][1:])):
        report(bx,shape[0],'Подготовка обзора')
        for by, (y0, y1) in enumerate(zip(cuts[1][:-1], cuts[1][1:])):
            for x in range(x0, x1):
                rows = max(1, 131072 // len(dz))
                for y in range(y0, y1, rows):
                    ys = slice(y, min(y + rows, y1))
                    if bounds is None:
                        volumes = dx[x] * dy[ys, None] * dz
                    else:
                        b = bounds[x, ys]
                        volumes = np.prod(b[..., 3:] - b[..., :3], axis=-1)
                    volumes = np.where(active[x, ys], volumes, 0.)
                    values = model[x, ys]
                    if result.model.ndim == 2:
                        values = np.linalg.norm(values, axis=-1)
                    weight[bx, by] += np.add.reduceat(volumes.sum(axis=0), cuts[2][:-1])
                    total[bx, by] += np.add.reduceat((volumes * values).sum(axis=0), cuts[2][:-1])
    values = np.divide(total, weight, out=np.full(shape, np.nan), where=weight > 0)
    return tuple(axis[edges] for axis, edges in zip(axes, cuts)), values


def surface_faces(axes, values, top=0):
    """Only the exposed faces of occupied display cells at or below the selected layer."""
    occupied = np.isfinite(values)
    occupied[:, :, :top] = False
    padded = np.pad(occupied, 1)
    faces, colors = [], []
    for axis in range(3):
        for side in (0, 1):
            neighbour = [slice(1, -1)] * 3
            neighbour[axis] = slice(2, None) if side else slice(None, -2)
            cells = np.nonzero(occupied & ~padded[tuple(neighbour)])
            offsets = np.zeros((4, 3), dtype=int)
            offsets[:, axis] = side
            offsets[:, (axis + 1) % 3] = [0, 1, 1, 0]
            offsets[:, (axis + 2) % 3] = [0, 0, 1, 1]
            faces.append(np.stack([a[i[:, None] + offsets[:, j]] for j, (a, i)
                                   in enumerate(zip(axes, cells))], axis=-1))
            colors.append(values[cells])
    return np.concatenate(faces), np.concatenate(colors)
