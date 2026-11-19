"""GOCAD Voxet with external RAW IEEE properties, U varying fastest."""
import json
from pathlib import Path
import shlex
import tempfile
import numpy as np
from .inversion import Mesh
from .inversion_export import clipped_count
from .voxel import VoxelModel


def header(path):
    axes,props,meta={}, {}, {}
    with Path(path).open(encoding='utf-8-sig',errors='replace') as f:
        if not f.readline().strip().upper().startswith('GOCAD VOXET'):
            raise ValueError('Файл не содержит заголовок GOCAD Voxet.')
        for line in f:
            line=line.strip()
            if line.startswith('# GeoGarModel '):
                meta=json.loads(line[len('# GeoGarModel '):])
                continue
            if not line: continue
            key=line.split(maxsplit=1)[0].upper()
            if key=='END': break
            if not key.startswith(('AXIS_','PROP_')) and key not in ('PROPERTY','ZPOSITIVE'): continue
            args=[v.strip('"') for v in shlex.split(line,posix=False,comments=True)[1:]]
            if key.startswith('AXIS_') or key=='ZPOSITIVE': axes[key]=args
            else:
                idx, *values=args
                props.setdefault(idx,{})[key]=' '.join(values)
    if not props: raise ValueError('В Voxet отсутствуют внешние свойства.')
    return axes,props,meta


def read_voxet(path,mode='gravity',crs='',property_id=None):
    path=Path(path)
    a,props,meta=header(path)
    if 'AXIS_N' in a:
        n=tuple(map(int,a['AXIS_N']))
    elif 'AXIS_D' in a:
        span=np.array(a.get('AXIS_MAX',[1]*3),float)-np.array(a.get('AXIS_MIN',[0]*3),float)
        steps=span/np.array(a['AXIS_D'],float)
        if not np.isfinite(steps).all() or not np.allclose(steps,steps.round(),rtol=0,atol=1e-6):
            raise ValueError('AXIS_D не согласован с диапазоном осей Voxet.')
        n=tuple(int(v)+1 for v in steps.round())
    else:
        raise ValueError('В заголовке Voxet отсутствуют AXIS_N и AXIS_D.')
    if len(n)!=3 or min(n)<1: raise ValueError('Некорректное число узлов Voxet.')
    vectors=np.array([a['AXIS_'+key] for key in 'UVW'],dtype=float)
    if vectors.shape!=(3,3) or not np.isfinite(vectors).all(): raise ValueError('Некорректные оси Voxet.')
    xyz=np.argmax(abs(vectors),axis=1)
    if len(set(xyz))!=3 or not np.allclose(vectors,np.eye(3)[xyz]*vectors[np.arange(3),xyz,None],rtol=0,atol=1e-12):
        raise ValueError('Поворот или наклон Voxet требует переноса на осевую сетку.')
    if any(v.lower()!='even' for v in a.get('AXIS_TYPE',[])):
        raise ValueError('Поддерживаются равномерные оси Voxet.')
    if any(v.lower() not in ('m','metre','meter','metres','meters','') for v in a.get('AXIS_UNIT',[])):
        raise ValueError('Координаты Voxet должны быть в метрах.')
    lo=np.array(a.get('AXIS_MIN',[0]*3),float)
    hi=np.array(a.get('AXIS_MAX',[1]*3),float)
    origin=np.array(a['AXIS_O'],float)+lo@vectors
    down=a.get('ZPOSITIVE',['Elevation'])[0].lower()=='depth'
    axes=[None]*3
    rev=[slice(None)]*3
    for i,j in enumerate(xyz):
        delta=(hi[i]-lo[i])/(n[i]-1) if n[i]>1 else float(a.get('AXIS_D',[1]*3)[i])
        sign=-1 if j==2 and not down else 1
        step=delta*vectors[i,j]*sign
        edge=origin[j]*sign+(np.arange(n[i]+1)-.5)*step
        if step<0:
            edge=edge[::-1]
            rev[j]=slice(None,None,-1)
        axes[j]=edge
    mesh=Mesh(*axes)
    mode=meta.get('mode',mode) if property_id is None else mode
    if mode=='vector':
        by_name={p.get('PROPERTY','').upper():k for k,p in props.items()}
        if not all(k in by_name for k in ('JE','JN','JD')):
            raise ValueError('Для векторной модели нужны три свойства JE, JN и JD.')
        selected=[by_name[k] for k in ('JE','JN','JD')]
    else:
        selected=[property_id or next(iter(props))]
    values=[]
    for key in selected:
        p=props[key]
        size=int(p.get('PROP_ESIZE','4'))
        if size not in (4,8) or p.get('PROP_ETYPE','IEEE').upper()!='IEEE' or p.get('PROP_FORMAT','RAW').upper()!='RAW':
            raise ValueError('Поддерживаются внешние RAW IEEE float32 и float64.')
        order=p.get('PROP_BYTE_ORDER','BIG_ENDIAN').upper()
        if order not in ('BIG_ENDIAN','LITTLE_ENDIAN','BIG','LITTLE'):
            raise ValueError('Неизвестный порядок байтов Voxet.')
        dtype=np.dtype(('>' if order.startswith('BIG') else '<')+'f'+str(size))
        offset=int(p.get('PROP_OFFSET',0))
        binary=path.parent/p['PROP_FILE']
        if offset<0 or binary.stat().st_size-offset<mesh.size*size:
            raise ValueError('Неверное смещение или неполный бинарный файл Voxet.')
        raw=np.memmap(binary,dtype=dtype,mode='r',offset=offset,shape=n,order='F')
        v=np.asarray(raw.transpose(np.argsort(xyz))[tuple(rev)],dtype=float).ravel()
        mask=np.isfinite(v)
        if 'PROP_NO_DATA_VALUE' in p:
            mask &= v!=np.array(float(p['PROP_NO_DATA_VALUE']),dtype=dtype).item()
        mesh.active &= mask
        values.append(np.where(mask,v,0.))
    return VoxelModel(mesh,values[0] if len(values)==1 else np.column_stack(values),
                      mode,meta.get('crs') or crs,[dict(operation='GOCAD Voxet',properties=selected)])


def write_voxet(result,path,policy='exact'):
    path=Path(path).with_suffix('.vo')
    mesh=result.mesh
    axes=(mesh.east,mesh.north,mesh.depth)
    widths=[np.diff(axis) for axis in axes]
    if any(not np.allclose(w,w[0],rtol=1e-9,atol=1e-10) for w in widths):
        raise ValueError('Для GOCAD Voxet нужна равномерная сетка по каждой оси. Выполните перенос модели или сохраните GGModel.')
    count=clipped_count(mesh)
    if count and policy=='exact': raise ValueError('Обрезанные ячейки требуют явного выбора приближения.')
    mode=result.settings['mode']
    if policy not in ('exact','keep','mass') or (policy=='mass' and mode!='gravity'): raise ValueError('Некорректный способ переноса свойств.')
    props={'gravity':['density_contrast'],'induced':['susceptibility'],'vector':['JE','JN','JD']}[mode]
    unit={'gravity':'g/cm3','induced':'SI','vector':'A/m'}[mode]
    meta=dict(mode=mode,crs=result.settings.get('crs',''),geometry_policy=policy,clipped_cells=count)
    origin=[a[0]+w[0]/2 for a,w in zip(axes,widths)]
    origin[2]*=-1
    nx,ny,nz=mesh.shape
    lines=['GOCAD Voxet 1','HEADER {','name:GeoGarModel','}','# GeoGarModel '+json.dumps(meta),
           'AXIS_O '+' '.join(format(v,'.17g') for v in origin),
           f'AXIS_U {widths[0][0]:.17g} 0 0',f'AXIS_V 0 {widths[1][0]:.17g} 0',f'AXIS_W 0 0 {-widths[2][0]:.17g}',
           'AXIS_MIN 0 0 0',f'AXIS_MAX {nx-1} {ny-1} {nz-1}',f'AXIS_N {nx} {ny} {nz}',
           'AXIS_D 1 1 1','AXIS_TYPE even even even','AXIS_UNIT "m" "m" "m"']
    nodata=float(np.finfo('f4').min)
    binary=path.with_suffix('.bin')
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        root=Path(folder)
        with (root/binary.name).open('wb') as stream:
            for i,name in enumerate(props,1):
                lines.extend([f'PROPERTY {i} "{name}"',f'PROP_UNIT {i} "{unit}"',f'PROP_ESIZE {i} 4',
                              f'PROP_ETYPE {i} IEEE',f'PROP_FORMAT {i} RAW',f'PROP_OFFSET {i} {stream.tell()}',
                              f'PROP_NO_DATA_VALUE {i} {nodata:.17g}',f'PROP_FILE {i} "{binary.name}"'])
                for k in range(nz):
                    for y in range(0,ny,max(1,65536//nx)):
                        stop=min(ny,y+max(1,65536//nx))
                        ids=(np.arange(nx)[None,:]*ny+np.arange(y,stop)[:,None])*nz+k
                        values=result.model[ids] if len(props)==1 else result.model[ids,i-1]
                        active=mesh.active[ids]
                        if count and policy=='mass':
                            b=mesh.bounds[ids]
                            values*=np.prod(b[...,3:]-b[...,:3],axis=-1)/(widths[0][:,None]*widths[1][y:stop]).T/widths[2][k]
                        v=values.astype('>f4')
                        if np.any(active & (~np.isfinite(v)|(v==nodata))):
                            raise ValueError('Свойство выходит за диапазон float32 или совпадает с NoData.')
                        v[~active]=nodata
                        stream.write(v.tobytes())
        (root/path.name).write_text('\n'.join(lines+['END','']),encoding='utf-8')
        from .inversion_export import install_bundle
        install_bundle(root,path.parent)
