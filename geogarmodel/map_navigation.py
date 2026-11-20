"""Map navigation and geographic north in display coordinates."""
import numpy as np
from pyproj import CRS,Transformer
from .figure_toolbar import FigureToolbar
from matplotlib.backends.backend_qt import SubplotToolQt


class MapSubplotTool(SubplotToolQt):
    def __init__(self,workbench):
        self.workbench=workbench
        super().__init__(workbench.figure,workbench)

    def _on_value_changed(self):
        views=[(ax,ax.get_xlim(),ax.get_ylim()) for ax in self._figure.axes]
        super()._on_value_changed()
        for ax,x,y in views: ax.set_xlim(x); ax.set_ylim(y)
        self.workbench.remember_map_layout()

    def _tight_layout(self):
        views=[(ax,ax.get_xlim(),ax.get_ylim()) for ax in self._figure.axes]
        super()._tight_layout()
        for ax,x,y in views: ax.set_xlim(x); ax.set_ylim(y)
        self.workbench.remember_map_layout()


class MapToolbar(FigureToolbar):
    def __init__(self,canvas,workbench):
        self.workbench=workbench
        super().__init__(canvas,workbench)

    def home(self,*args):
        self.workbench.guard(self.workbench.fit_maps)

    def configure_subplots(self):
        self.workbench.auto_map_layout.setChecked(False)
        if self._subplot_dialog is None: self._subplot_dialog=MapSubplotTool(self.workbench)
        self._subplot_dialog.update_from_current_subplotpars()
        self._subplot_dialog.setModal(True); self._subplot_dialog.show()
        return self._subplot_dialog


def north_direction(crs,x,y):
    """Unit direction of geographic north in map XY (always east/north order)."""
    source=CRS.from_user_input(crs)
    geodetic=source.geodetic_crs
    inverse=Transformer.from_crs(source,geodetic,always_xy=True)
    forward=Transformer.from_crs(geodetic,source,always_xy=True)
    lon,lat=inverse.transform(x,y)
    lon2,lat2,_=source.get_geod().fwd(lon,lat,0,100.)
    xx,yy=forward.transform(lon2,lat2)
    delta=np.array([xx-x,yy-y]); size=np.linalg.norm(delta)
    if not np.isfinite(delta).all() or size<=0: raise ValueError('Направление севера не определено в этой точке.')
    return delta/size
