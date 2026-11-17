"""Bounded display sampling in field-value space, before colour mapping."""
import numpy as np


def sample_display(grid,bounds,width,height,smooth=True):
    """Return screen samples and pixel-edge extent; never modify source values.

    Bilinear interpolation is nodata-aware. A 2x2 footprint when minifying
    suppresses stride aliasing. The nearest source-cell mask preserves holes.
    """
    xmin,xmax,ymin,ymax=bounds
    xmin=max(xmin,grid.x[0]-grid.dx/2); xmax=min(xmax,grid.x[-1]+grid.dx/2)
    ymin=max(ymin,grid.y[0]-grid.dy/2); ymax=min(ymax,grid.y[-1]+grid.dy/2)
    if xmax<=xmin or ymax<=ymin: return None
    nx=max(2,min(1600,int(width))); ny=max(2,min(1600,int(height)))
    if not smooth:
        nx=min(nx,max(2,int(np.ceil((xmax-xmin)/grid.dx))))
        ny=min(ny,max(2,int(np.ceil((ymax-ymin)/grid.dy))))
    px=(xmax-xmin)/nx/grid.dx; py=(ymax-ymin)/ny/grid.dy
    x=(xmin-grid.x[0])/grid.dx+(np.arange(nx)+.5)*px
    y=(ymin-grid.y[0])/grid.dy+(np.arange(ny)+.5)*py
    def nearest(xx,yy):
        ix=np.clip(np.floor(xx+.5).astype(int),0,len(grid.x)-1)
        iy=np.clip(np.floor(yy+.5).astype(int),0,len(grid.y)-1)
        return grid.values[np.ix_(iy,ix)]
    mask=np.isfinite(nearest(x,y))
    if not smooth: values=nearest(x,y).astype(np.float32)
    else:
        from scipy.ndimage import map_coordinates
        left=max(0,int(np.floor(x.min()-px))); right=min(len(grid.x),int(np.ceil(x.max()+px))+2)
        bottom=max(0,int(np.floor(y.min()-py))); top=min(len(grid.y),int(np.ceil(y.max()+py))+2)
        tile=grid.values[bottom:top,left:right]; finite=np.isfinite(tile)
        weights=None if finite.all() else finite.astype(np.float32)
        tile=tile if weights is None else np.where(finite,tile,0)
        def bilinear(xx,yy):
            coordinates=np.broadcast_arrays(yy[:,None]-bottom,xx[None,:]-left)
            result=map_coordinates(tile,coordinates,order=1,prefilter=False,mode='nearest',output=np.float32)
            if weights is not None:
                support=map_coordinates(weights,coordinates,order=1,prefilter=False,mode='nearest')
                result=np.divide(result,support,out=np.full_like(result,np.nan),where=support>1e-12)
            return result
        offsets=[(-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)] if max(px,py)>1.5 else [(0,0)]
        total=np.zeros((ny,nx)); count=np.zeros((ny,nx))
        for ox,oy in offsets:
            v=bilinear(x+ox*px,y+oy*py); valid=np.isfinite(v)
            total+=np.where(valid,v,0); count+=valid
        values=np.divide(total,count,out=np.full_like(total,np.nan),where=count>0).astype(np.float32)
    values[~mask]=np.nan
    return values,(xmin,xmax,ymin,ymax)
