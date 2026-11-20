"""Horizontal sections through active, possibly vertically clipped prisms."""
from dataclasses import dataclass
import re
import numpy as np


def property_values(result,component='amplitude'):
    values=result.model
    if values.ndim==2:
        values=np.linalg.norm(values,axis=1) if component=='amplitude' else values[:,{'East':0,'North':1,'Down':2}[component]]
    return values


def property_unit(result):
    return {'gravity':'г/см³','induced':'SI','vector':'А/м','J_parallel':'А/м','J_magnitude':'А/м'}[result.settings['mode']]


def select_depths(mesh,mode='manual',text='',top=None,bottom=None,step=100,every=1):
    lo,hi=float(mesh.depth[0]),float(mesh.depth[-1])
    top=lo if top is None else float(top); bottom=hi if bottom is None else float(bottom)
    if mode=='manual': top,bottom=lo,hi
    if not np.isfinite([top,bottom]).all() or top<lo or bottom>hi or top>bottom:
        raise ValueError(f'Границы глубины должны лежать в пределах {lo:g}…{hi:g} м; верх не глубже низа.')
    if mode=='manual':
        try: depths=np.array([float(v.replace(',','.')) for v in re.split(r'[;\s]+',text.strip()) if v])
        except ValueError: raise ValueError('Задайте глубины числами через точку с запятой.') from None
    elif mode=='layers':
        if every<1 or int(every)!=every: raise ValueError('Частота слоёв должна быть целым числом ≥ 1.')
        centers=(mesh.depth[:-1]+mesh.depth[1:])/2
        depths=centers[(centers>=top)&(centers<=bottom)][::int(every)]
    elif mode=='step':
        if not np.isfinite(step) or step<=0: raise ValueError('Шаг глубины должен быть больше нуля.')
        count=int(np.floor((bottom-top)/step+1e-10))+1
        if count>10000: raise ValueError('Слишком много срезов (предел 10 000); увеличьте шаг.')
        depths=np.minimum(top+np.arange(count)*step,bottom)
    else: raise ValueError('Неизвестный способ выбора глубин.')
    if not len(depths): raise ValueError('Не выбрана ни одна глубина.')
    if len(depths)>10000 or not np.isfinite(depths).all() or np.any(depths<lo) or np.any(depths>hi):
        raise ValueError(f'Нужно не более 10 000 конечных глубин в пределах {lo:g}…{hi:g} м.')
    return np.unique(depths)


@dataclass
class HorizontalSlice:
    depth: float
    values: np.ndarray
    rectangles: np.ndarray | None = None
    rectangle_values: np.ndarray | None = None


def horizontal_slice(result,depth,component='amplitude'):
    mesh=result.mesh
    if not np.isfinite(depth) or not mesh.depth[0]<=depth<=mesh.depth[-1]: raise ValueError('Глубина вне модели.')
    iz=min(int(np.searchsorted(mesh.depth,depth,side='right')-1),mesh.shape[2]-1)
    values=property_values(result,component)
    cube=values.reshape(mesh.shape); active=mesh.active.reshape(mesh.shape)
    layer=np.where(active[:,:,iz],cube[:,:,iz],np.nan).T.copy()
    if mesh.bounds is None: return HorizontalSlice(depth,layer)
    # Actual prism intersections, including partial XY cells, not nominal cell centres.
    cells=mesh.bounds; mask=mesh.active&(cells[:,2]<=depth)&((cells[:,5]>depth)|((depth==mesh.depth[-1])&(cells[:,5]==depth)))
    grid_mask=mask.reshape(mesh.shape)[:,:,iz]
    layer=np.where(grid_mask,cube[:,:,iz],np.nan).T.copy()
    return HorizontalSlice(depth,layer,cells[mask][:,[0,1,3,4]],values[mask])
