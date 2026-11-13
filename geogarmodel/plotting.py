"""Qt-independent map renderer, shared with headless tests."""
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm, LightSource
from matplotlib.ticker import MaxNLocator
from .theme import style_figure


def draw_map(figure, grid, *, cmap="viridis", mode="linear", vmin=None, vmax=None,
             contours=False, hillshade=False, title=None, overlays=(), points=None):
    finite = grid.values[np.isfinite(grid.values)]
    if mode == "percentile":
        low, high = np.percentile(finite, [2, 98])
    else:
        low, high = finite.min(), finite.max()
    low = float(low if vmin is None else vmin)
    high = float(high if vmax is None else vmax)
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("Пределы палитры должны быть конечными, min ≤ max.")
    if low == high:
        high = low + max(abs(low) * 1e-6, 1e-6)
    if mode == "symmetric":
        limit = max(abs(low), abs(high))
        norm = TwoSlopeNorm(vcenter=0., vmin=-limit, vmax=limit)
    else:
        norm = Normalize(low, high)
    fast = not contours and not hillshade and not overlays and points is None
    cached = getattr(figure, "_map_cache", None)
    if fast and cached is not None and cached[0] is grid:
        _, ax, image, bar = cached
        image.set_cmap(cmap)
        image.set_norm(norm)
        bar.update_normal(image)
        bar.set_label(grid.unit)
        ax.set_title(grid.name if title is None else title)
        ax.set_xlabel(f"X ({grid.coordinate_unit})")
        ax.set_ylabel(f"Y ({grid.coordinate_unit})")
        style_figure(figure)
        return ax
    figure._map_cache = None
    figure.clear()
    ax = figure.add_subplot(111)
    extent = (grid.x[0] - grid.dx / 2, grid.x[-1] + grid.dx / 2,
              grid.y[0] - grid.dy / 2, grid.y[-1] + grid.dy / 2)
    image = ax.imshow(np.ma.masked_invalid(grid.values), origin="lower", extent=extent,
                      interpolation="nearest", cmap=cmap, norm=norm, rasterized=True)
    if hillshade:
        # Shade an independent intensity layer so the colourbar still represents values.
        filled = np.where(np.isfinite(grid.values), grid.values, np.nanmedian(grid.values))
        shade = LightSource(315, 45).hillshade(filled, dx=grid.dx, dy=grid.dy, vert_exag=1)
        shade = np.ma.array(shade, mask=~np.isfinite(grid.values))
        ax.imshow(shade, origin="lower", extent=extent, interpolation="nearest", cmap="gray", alpha=.25, rasterized=True)
    if contours and np.ptp(finite) > 0:
        sy, sx = max(1, int(np.ceil(len(grid.y) / 512))), max(1, int(np.ceil(len(grid.x) / 512)))
        cs = ax.contour(grid.x[::sx], grid.y[::sy], grid.values[::sy, ::sx], levels=10, colors="#dbe7f6", linewidths=.5, alpha=.6)
        ax.clabel(cs, fontsize=7, inline=True)
    for layer in overlays:
        if not grid.crs or layer.crs is None:
            raise ValueError("Для наложения векторных данных задайте CRS карты и вектора.")
        layer.to_crs(grid.crs).boundary.plot(ax=ax, color="#333333", linewidth=.8)
    if points is not None:
        ax.scatter(points.iloc[:, 0], points.iloc[:, 1], s=9, c="black", label="Контрольные точки")
        ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(grid.name if title is None else title)
    ax.set_xlabel(f"X ({grid.coordinate_unit})")
    ax.set_ylabel(f"Y ({grid.coordinate_unit})")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    bar = figure.colorbar(image, ax=ax, label=grid.unit, fraction=.045, pad=.04)
    style_figure(figure)
    figure.tight_layout()
    if fast:
        figure._map_cache = (grid, ax, image, bar)
    return ax
