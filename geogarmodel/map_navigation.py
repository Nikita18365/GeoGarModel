"""Map navigation and geographic north in display coordinates."""
import numpy as np
from pyproj import CRS,Transformer
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


class MapToolbar(NavigationToolbar2QT):
    def __init__(self,canvas,workbench):
        self.workbench=workbench
        super().__init__(canvas,workbench)

    def home(self,*args):
        self.workbench.guard(self.workbench.fit_maps)


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
