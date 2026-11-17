"""Bounded, mask-aware section sampling in property space for display only."""
import numpy as np
from scipy.ndimage import map_coordinates
from .model_slices import property_values


def sample_section(result,axis,coordinate,width=768,height=768,smooth=True,component='amplitude'):
    """Sample an East/North/Down plane, preserving exact active prism footprints.

    The horizontal section has rows North, columns East. Vertical sections have
    rows Down and columns North (X plane) or East (Y plane). No depth averaging.
    """
    mesh=result.mesh; axes=(mesh.east,mesh.north,mesh.depth)
    if axis not in (0,1,2) or not np.isfinite(coordinate) or not axes[axis][0]<=coordinate<=axes[axis][-1]:
        raise ValueError('Координата сечения вне модели.')
    free=[i for i in range(3) if i!=axis]; u,v=[axes[i] for i in free]
    index=min(np.searchsorted(axes[axis],coordinate,side='right')-1,mesh.shape[axis]-1)
    cube=property_values(result,component).reshape(mesh.shape)
    layer=np.take(cube,index,axis=axis).T
    valid=np.take(mesh.active.reshape(mesh.shape),index,axis=axis).T.copy()
    cells=None
    if mesh.bounds is not None:
        cells=np.take(mesh.bounds.reshape(mesh.shape+(6,)),index,axis=axis).transpose(1,0,2)
        valid &= (coordinate>=cells[:,:,axis])&((coordinate<cells[:,:,axis+3])|((coordinate==axes[axis][-1])&(coordinate==cells[:,:,axis+3])))
    nx=max(2,min(1600,int(width))); ny=max(2,min(1600,int(height)))
    x=u[0]+(np.arange(nx)+.5)*(u[-1]-u[0])/nx
    y=v[0]+(np.arange(ny)+.5)*(v[-1]-v[0])/ny
    ix=np.clip(np.searchsorted(u,x,side='right')-1,0,len(u)-2)
    iy=np.clip(np.searchsorted(v,y,side='right')-1,0,len(v)-2)
    mask=valid[np.ix_(iy,ix)].copy()
    if cells is not None:
        chosen=cells[np.ix_(iy,ix)]
        mask &= (x[None,:]>=chosen[:,:,free[0]])&(x[None,:]<=chosen[:,:,free[0]+3])
        mask &= (y[:,None]>=chosen[:,:,free[1]])&(y[:,None]<=chosen[:,:,free[1]+3])
    if smooth:
        # Interpolate scalar values before colour mapping, including nonuniform grids.
        centers=[(a[:-1]+a[1:])/2 for a in (u,v)]
        xx=np.interp(x,centers[0],np.arange(len(centers[0])))
        yy=np.interp(y,centers[1],np.arange(len(centers[1])))
        coordinates=np.broadcast_arrays(yy[:,None],xx[None,:])
        numerator=map_coordinates(np.where(valid,layer,0),coordinates,order=1,prefilter=False,mode='nearest',output=np.float32)
        weight=map_coordinates(valid.astype(np.float32),coordinates,order=1,prefilter=False,mode='nearest')
        values=np.divide(numerator,weight,out=np.full_like(numerator,np.nan),where=weight>1e-8)
    else: values=layer[np.ix_(iy,ix)].astype(np.float32)
    values[~mask]=np.nan
    return values,(u[0],u[-1],v[0],v[-1])
