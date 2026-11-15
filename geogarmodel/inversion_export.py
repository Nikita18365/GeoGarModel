"""Exports preserve observation samples and tensor-cell geometry explicitly."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .grids import Grid


def field_table(result):
    return pd.DataFrame(dict(East=result.observations[:,0], North=result.observations[:,1],
        height=-result.observations[:,2], observed=result.observed, predicted=result.predicted,
        residual=result.predicted-result.observed))


def export_zmap(result, path):
    """Write samples on a regular XY lattice; missing samples remain NULL."""
    xy=result.observations[:,:2]
    x,y=np.unique(xy[:,0]),np.unique(xy[:,1])
    if len(x)*len(y)>10_000_000:
        raise ValueError('Слишком большая XY-сетка. Сохраните поле в CSV.')
    if len(np.unique(xy,axis=0))!=len(xy):
        raise ValueError('ZMAP: повторяющиеся XY (возможно, разные высоты). Используйте CSV.')
    values=np.full((len(y),len(x)),np.nan)
    values[np.searchsorted(y,xy[:,1]),np.searchsorted(x,xy[:,0])]=result.predicted
    try:
        grid=Grid(x,y,values)
    except ValueError as exc:
        raise ValueError('ZMAP требует равномерную XY-сетку минимум 2×2. Для произвольных точек используйте CSV; интерполяция автоматически не выполняется.') from exc
    null=-1e32
    if np.any(grid.values==null):
        raise ValueError('Значение поля совпадает с NULL ZMAP; используйте CSV.')
    lines=['! Calculated field; units: '+('mGal' if result.settings['mode']=='gravity' else 'nT'),
           '! CRS: '+result.settings.get('crs',''), '@GeoGarModel, GRID, 5',
           f'24, {null}, , 16, 1',f'{len(y)}, {len(x)}, {x[0]:.17g}, {x[-1]:.17g}, {y[0]:.17g}, {y[-1]:.17g}', '0, 0, 0', '@']
    for column in grid.values[::-1].T:
        for start in range(0,len(column),5):
            lines.append(''.join(f'{v if np.isfinite(v) else null:24.16E}' for v in column[start:start+5]))
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def export_ubc(result, path):
    """Tensor UBC mesh + scalar properties; vector exported as JE/JN/JD files."""
    mesh=result.mesh
    if mesh.bounds is not None:
        raise ValueError('Обрезанные призмы нельзя точно записать в UBC tensor mesh. Выберите NPZ или VTK/VTU.')
    path=Path(path).with_suffix('.msh')
    dummy=-1e32
    if np.any(result.model[mesh.active]==dummy):
        raise ValueError('Свойство совпадает с UBC dummy; выберите NPZ.')
    lines=[' '.join(map(str,mesh.shape)),f'{mesh.east[0]:.17g} {mesh.north[0]:.17g} {-mesh.depth[0]:.17g}']
    lines += [' '.join(f'{v:.17g}' for v in np.diff(a)) for a in (mesh.east,mesh.north,mesh.depth)]
    values=result.model.copy(); values[~mesh.active]=dummy
    components=[('',values)] if values.ndim==1 else [(name,values[:,i]) for i,name in enumerate(('JE','JN','JD'))]
    files=[]
    for name,v in components:
        dest=path.with_name(path.stem+('_'+name if name else '')+'.mod')
        ordered=v.reshape(mesh.shape).swapaxes(0,1).ravel()
        files.append((dest,ordered))
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for dest,v in files:
        np.savetxt(dest,v,fmt='%.17g')
    meta=dict(mode=result.settings['mode'],crs=result.settings.get('crs',''),dummy=dummy,
              density_unit='g/cm3',susceptibility_unit='SI',magnetization_unit='A/m',
              components='JE East, JN North, JD Down',selected_iteration=result.settings.get('selected_iteration'),
              model_files=[p.name for p,_ in files])
    path.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
