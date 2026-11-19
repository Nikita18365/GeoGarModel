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


def clipped_count(mesh):
    from .model_archive import cell_blocks
    if mesh.bounds is None:
        return 0
    return sum(int(np.count_nonzero(mesh.active[ids] & (mesh.bounds[ids]!=regular).any(axis=1)))
               for ids,regular in cell_blocks(mesh))


def export_ubc(result, path, policy='exact'):
    """Stage the whole bundle before replacing any previous export."""
    import tempfile
    path=Path(path).with_suffix('.msh')
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        root=Path(folder)
        _write_ubc(result,root/path.name,policy)
        outputs=list(root.iterdir())
        backups={}
        installed=[]
        try:
            for source in outputs:
                dest=path.parent/source.name
                if dest.is_file():
                    backup=root/(source.name+'.bak')
                    dest.replace(backup)
                    backups[dest]=backup
                source.replace(dest)
                installed.append(dest)
        except OSError:
            for dest in installed: dest.unlink(missing_ok=True)
            for dest,backup in backups.items(): backup.replace(dest)
            raise


def _write_ubc(result, path, policy):
    """Write bounded strips in UBC order: depth, East, North."""
    from .model_archive import regular_cells
    mesh=result.mesh
    count=clipped_count(mesh)
    if policy not in ('exact','keep','mass') or (policy=='mass' and result.settings['mode']!='gravity'):
        raise ValueError('Недопустимый способ экспорта UBC.')
    if count and policy=='exact':
        raise ValueError('Обрезанные призмы требуют явного выбора ступенчатого приближения.')
    path=Path(path).with_suffix('.msh')
    dummy=-1e32
    nx,ny,nz=mesh.shape
    names=[''] if result.model.ndim==1 else ['_JE','_JN','_JD']
    files=[]
    for component,name in enumerate(names):
        dest=path.with_name(path.stem+name+'.mod')
        files.append(dest.name)
        with dest.open('w',encoding='ascii') as stream:
            step=max(1,65536//nz)
            for y in range(ny):
                for x in range(0,nx,step):
                    ids=((np.arange(x,min(nx,x+step))[:,None]*ny+y)*nz+np.arange(nz)).ravel()
                    values=result.model[ids] if not name else result.model[ids,component]
                    if count and policy=='mass':
                        b=mesh.bounds[ids]
                        base=regular_cells(mesh,ids)
                        values*=np.prod(b[:,3:]-b[:,:3],axis=1)/np.prod(base[:,3:]-base[:,:3],axis=1)
                    if np.any((values==dummy)&mesh.active[ids]):
                        raise ValueError('Свойство совпадает с UBC dummy.')
                    values[~mesh.active[ids]]=dummy
                    np.savetxt(stream,values,fmt='%.17g')
    lines=[' '.join(map(str,mesh.shape)),f'{mesh.east[0]:.17g} {mesh.north[0]:.17g} {-mesh.depth[0]:.17g}']
    lines+=[' '.join(f'{v:.17g}' for v in np.diff(a)) for a in (mesh.east,mesh.north,mesh.depth)]
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    meta=dict(mode=result.settings['mode'],crs=result.settings.get('crs',''),dummy=dummy,
              density_unit='g/cm3',susceptibility_unit='SI',magnetization_unit='A/m',
              components='JE East, JN North, JD Down',selected_iteration=result.settings.get('selected_iteration'),
              clipped_cells=count,geometry_policy=policy,model_files=files)
    path.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


def oasis_package(result,directory,policy):
    """Portable script uses ASCII relative names and runs inside the user's Oasis project."""
    import tempfile
    import os
    directory=Path(directory)
    if directory.exists():
        raise ValueError('Выберите новое имя папки комплекта Oasis montaj.')
    with tempfile.TemporaryDirectory(dir=directory.parent) as staging:
        folder=Path(staging)/'package'
        folder.mkdir()
        _write_ubc(result,folder/'model.msh',policy)
        meta=json.loads((folder/'model.json').read_text(encoding='utf-8'))
        commands=[]
        for mod in meta['model_files']:
            for key,value in [('MODEL_FILE',mod),('MSH_HEADER_FILE','model.msh'),
                              ('OUTPUT_FILE',Path(mod).stem+'.geosoft_voxel'),('DUMMY_VALUE','-1e32')]:
                commands.append(f'SETINI IMPORTUBCTOVOX.{key}="{value}"')
            commands.append('GX geogxnet.dll(Geosoft.GX.VoxUtils.ImportUBCToVox;Run)')
        (folder/'convert.gs').write_text('\n'.join(commands)+'\n',encoding='ascii')
        (folder/'README.txt').write_text(
            'Откройте проект Oasis montaj в этой папке и запустите convert.gs (Run Script).\n'
            'Проверьте систему координат: '+str(meta['crs'])+'\n'
            'Вручную: Voxel > Conversions > UBC to Voxel. Выберите model.msh и соответствующий MOD. '
            'Dummy = -1e32. Для векторной модели создаются отдельные JE, JN, JD.\n',encoding='utf-8-sig')
        os.rename(folder,directory)
