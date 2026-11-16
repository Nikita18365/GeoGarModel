"""Qt-independent map renderer, shared with headless tests."""
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm, LightSource
from matplotlib.ticker import MaxNLocator
from .theme import style_figure


def draw_map(figure, grid, *, cmap="viridis", mode="linear", vmin=None, vmax=None,
             contours=False, hillshade=False, title=None, overlays=(), points=None,
             contour_step=0, contour_color='#dbe7f6', contour_width=.5,
             contour_labels=True, alpha=1., target_ax=None):
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
    fast = target_ax is None and not contours and not hillshade and not overlays and points is None
    cached = getattr(figure, "_map_cache", None)
    if fast and cached is not None and cached[0] is grid:
        _, ax, image, bar = cached
        image.set_cmap(cmap)
        image.set_norm(norm)
        bar.update_normal(image)
        bar.set_label(grid.unit)
        ax.set_title(grid.name if title is None else title)
        ax.set_xlabel('X, м' if grid.coordinate_unit in ('m','unknown') else 'X, °')
        ax.set_ylabel('Y, м' if grid.coordinate_unit in ('m','unknown') else 'Y, °')
        style_figure(figure)
        return ax
    figure._map_cache = None
    if target_ax is None:
        figure.clear()
        ax = figure.add_subplot(111)
    else: ax=target_ax
    extent = (grid.x[0] - grid.dx / 2, grid.x[-1] + grid.dx / 2,
              grid.y[0] - grid.dy / 2, grid.y[-1] + grid.dy / 2)
    image = ax.imshow(np.ma.masked_invalid(grid.values), origin="lower", extent=extent,
                      interpolation="nearest", cmap=cmap, norm=norm, rasterized=True,alpha=alpha)
    image.set_gid('map-field')
    if hillshade:
        # Shade an independent intensity layer so the colourbar still represents values.
        filled = np.where(np.isfinite(grid.values), grid.values, np.nanmedian(grid.values))
        shade = LightSource(315, 45).hillshade(filled, dx=grid.dx, dy=grid.dy, vert_exag=1)
        shade = np.ma.array(shade, mask=~np.isfinite(grid.values))
        ax.imshow(shade, origin="lower", extent=extent, interpolation="nearest", cmap="gray", alpha=.25, rasterized=True)
    if contours and np.ptp(finite) > 0:
        sy, sx = max(1, int(np.ceil(len(grid.y) / 512))), max(1, int(np.ceil(len(grid.x) / 512)))
        levels=10
        if contour_step>0:
            start=np.ceil(finite.min()/contour_step); end=np.floor(finite.max()/contour_step)
            if end-start+1>1000: raise ValueError('Слишком много изолиний: увеличьте сечение (не более 1000 уровней).')
            levels=np.arange(start,end+1)*contour_step
        if np.isscalar(levels) or len(levels):
            cs = ax.contour(grid.x[::sx], grid.y[::sy], grid.values[::sy, ::sx], levels=levels, colors=contour_color, linewidths=contour_width)
            if contour_labels: ax.clabel(cs, fontsize=8, inline=True)
    for layer in overlays:
        if not grid.crs or layer.crs is None:
            raise ValueError("Для наложения векторных данных задайте CRS карты и вектора.")
        if hasattr(layer,'draw'): layer.draw(ax,grid.crs)
        else: layer.to_crs(grid.crs).boundary.plot(ax=ax, color="#333333", linewidth=.8)
    if points is not None:
        ax.scatter(points.iloc[:, 0], points.iloc[:, 1], s=9, c="black", label="Контрольные точки")
        ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(grid.name if title is None else title)
    ax.set_xlabel('X, м' if grid.coordinate_unit in ('m','unknown') else 'X, °')
    ax.set_ylabel('Y, м' if grid.coordinate_unit in ('m','unknown') else 'Y, °')
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    bar = figure.colorbar(image, ax=ax, label=grid.unit, fraction=.045, pad=.04) if target_ax is None else None
    style_figure(figure)
    figure.tight_layout()
    if fast:
        figure._map_cache = (grid, ax, image, bar)
    return ax
