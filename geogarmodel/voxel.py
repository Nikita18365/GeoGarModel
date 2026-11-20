"""Initial voxel models, explicit coordinate conventions and reversible operations."""
from dataclasses import dataclass, field, InitVar
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
from .work_progress import report
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from .inversion import Mesh, InversionResult


@dataclass
class VoxelModel:
    mesh: Mesh
    values: np.ndarray
    mode: str = 'gravity'
    crs: str = ''
    history: list = field(default_factory=list)
    source_path: str = ''
    source_array: str = ''

    own_values: InitVar[bool] = False

    def __post_init__(self,own_values):
        self.values = np.asarray(self.values, dtype=float)
        if not own_values: self.values=self.values.copy()
        shape = (self.mesh.size,3) if self.mode == 'vector' else (self.mesh.size,)
        if self.values.shape != shape:
            raise ValueError(f'Ожидались свойства {shape}, получено {self.values.shape}.')
        if not self.mesh.active.any(): raise ValueError('Нет активных ячеек в модели.')
        for start in range(0,self.mesh.size,131072):
            block=slice(start,start+131072)
            if not np.isfinite(self.values[block][self.mesh.active[block]]).all():
                raise ValueError('Активные ячейки содержат NaN/Inf.')
        self.values[~self.mesh.active] = 0

    def copy(self):
        m = self.mesh
        return VoxelModel(Mesh(m.east.copy(),m.north.copy(),m.depth.copy(),m.active.copy(),None if m.bounds is None else m.bounds.copy()),self.values,self.mode,self.crs,list(self.history),self.source_path,self.source_array)

    def save(self,path):
        np.savez_compressed(path,east=self.mesh.east,north=self.mesh.north,depth=self.mesh.depth,
                            active=self.mesh.active,bounds=np.empty(0) if self.mesh.bounds is None else self.mesh.bounds,model=self.values,
                            voxel_metadata=json.dumps(dict(mode=self.mode,crs=self.crs,history=self.history),ensure_ascii=False))

    def result(self):
        return InversionResult(self.mesh,self.values,np.empty((0,3)),np.empty(0),np.empty(0),[],dict(mode=self.mode,crs=self.crs,source_path=self.source_path,source_array=self.source_array,voxel_history=list(self.history)),'initial_model')


def crop_box(model, xmin, xmax, ymin, ymax, zmin, zmax):
    """Cut out a solid rectangular volume; newly activated material starts at zero."""
    mesh = model.mesh
    axes = []
    selections = []
    for axis, low, high in zip((mesh.east, mesh.north, mesh.depth),
                               (xmin,ymin,zmin), (xmax,ymax,zmax)):
        if not np.isfinite([low,high]).all() or not axis[0] <= low < high <= axis[-1]:
            raise ValueError('Задайте минимум и максимум внутри границ исходной сетки.')
        start = axis.searchsorted(low,side='right')-1
        stop = axis.searchsorted(high,side='left')
        selections.append(slice(start,stop))
        edges = axis[start:stop+1].copy()
        edges[0],edges[-1] = low,high
        axes.append(edges)
    keep = tuple(selections)
    result_mesh = Mesh(*axes)
    components = model.values.shape[1:]
    values = model.values.reshape(mesh.shape+components)[keep].copy()
    valid = mesh.active.reshape(mesh.shape)[keep].copy()
    if mesh.bounds is not None:
        cells = mesh.bounds.reshape(mesh.shape+(6,))[keep]
        for dim,(low,high) in enumerate(((xmin,xmax),(ymin,ymax),(zmin,zmax))):
            valid &= (cells[...,dim]<high) & (cells[...,dim+3]>low)
    values[~valid] = 0
    history = model.history+[dict(operation='solid box crop',limits=[[xmin,xmax],[ymin,ymax],[zmin,zmax]],fill=0)]
    return VoxelModel(result_mesh,values.reshape((-1,)+components),model.mode,model.crs,history,own_values=True)


def crop_xy(model, east_min, east_max, north_min, north_max):
    """Keep the original vertical extent of each prism intersecting the XY rectangle."""
    mesh = model.mesh
    selection = []
    axes = []
    partial = False
    for axis, low, high in ((mesh.east, east_min, east_max), (mesh.north, north_min, north_max)):
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError('Для каждой оси укажите конечные границы по возрастанию.')
        first = max(0, axis.searchsorted(low, side='right') - 1)
        stop = min(len(axis)-1, axis.searchsorted(high, side='left'))
        if first >= stop:
            raise ValueError('Прямоугольник не пересекает модель.')
        selection.append(slice(first, stop))
        edges = axis[first:stop+1]
        axes.append(edges.copy())
        partial |= low > edges[0] or high < edges[-1]

    keep = tuple(selection)
    cropped = Mesh(*axes, mesh.depth.copy(), mesh.active.reshape(mesh.shape)[keep])
    if mesh.bounds is not None or partial:
        cells = (mesh.bounds.reshape(mesh.shape+(6,))[keep].reshape(-1,6).copy()
                 if mesh.bounds is not None else cropped.cells())
        for dim, low, high in ((0,east_min,east_max),(1,north_min,north_max)):
            np.maximum(cells[:,dim],low,out=cells[:,dim])
            np.minimum(cells[:,dim+3],high,out=cells[:,dim+3])
            cropped.active &= cells[:,dim+3] > cells[:,dim]
        # Inactive prisms must still have usable geometry for subsequent editing.
        ids = np.flatnonzero(~cropped.active)
        ijk = np.unravel_index(ids,cropped.shape)
        for dim, (axis,index) in enumerate(zip((*axes,cropped.depth),ijk)):
            cells[ids,dim] = axis[index]
            cells[ids,dim+3] = axis[index+1]
        cropped.bounds = cells
    if not cropped.active.any():
        raise ValueError('Прямоугольник не пересекает активные ячейки.')
    components = model.values.shape[1:]
    values = model.values.reshape(mesh.shape+components)[keep].reshape((-1,)+components)
    history = model.history+[dict(operation='XY crop',east=[east_min,east_max],north=[north_min,north_max])]
    return VoxelModel(cropped,values,model.mode,model.crs,history)


def extend_density_model(model, east, north, padding, initial_density):
    """Add full prisms around a regular XY model, preserving interior masks and bounds."""
    from .layer_fft import uniform_step
    import psutil

    mesh = model.mesh
    axes = (mesh.east, mesh.north)
    steps = [uniform_step(edges) for edges in axes]
    added = []
    for edges, step, coordinates in zip(axes, steps, (east, north)):
        lo, hi = coordinates.min(), coordinates.max()
        margin = (hi - lo) * padding
        before = max(0, int(np.ceil((edges[0] - (lo - margin)) / step - 1e-8)))
        after = max(0, int(np.ceil(((hi + margin) - edges[-1]) / step - 1e-8)))
        added.append((before, after))
    if not any(left or right for left, right in added):
        return model
    nx, ny, nz = mesh.shape
    shape = nx + sum(added[0]), ny + sum(added[1]), nz
    per_cell = 240 if mesh.bounds is not None else 40
    if shape[0]*shape[1]*shape[2]*per_cell > psutil.virtual_memory().available*.8:
        raise ValueError('Недостаточно RAM для расширения начальной модели.')
    axes = [np.r_[edges[0] - step*np.arange(before,0,-1), edges,
                  edges[-1] + step*np.arange(1,after+1)]
            for edges,step,(before,after) in zip(axes,steps,added)]
    interior = tuple(slice(left, left + count) for (left, _), count in zip(added, mesh.shape))
    active = np.ones(shape, dtype=bool)
    active[interior] = mesh.active.reshape(mesh.shape)
    components = model.values.shape[1:]
    values = np.full(shape + components, initial_density, dtype=float)
    values[interior] = model.values.reshape(mesh.shape + components)
    extended = Mesh(*axes, mesh.depth, active)
    if mesh.bounds is not None:
        bounds = extended.cells().reshape(shape + (6,))
        bounds[interior] = mesh.bounds.reshape(mesh.shape + (6,))
        extended = Mesh(*axes, mesh.depth, active, bounds)
    history = model.history + [dict(operation='fft_padding', east=added[0], north=added[1])]
    return VoxelModel(extended, values.reshape((-1,) + components), model.mode, model.crs, history)


def _numbers(path):
    from itertools import repeat
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            for token in line.partition('!')[0].partition('#')[0].split():
                count,value = token.split('*') if '*' in token else (1,token)
                yield from repeat(float(value.replace('D','E').replace('d','e')),int(count))


def _ubc_blocks(path):
    from itertools import islice, repeat
    import warnings

    def expanded(lines):
        for line in lines:
            for token in line.partition(b'!')[0].partition(b'#')[0].split():
                count,value=token.split(b'*') if b'*' in token else (1,token)
                count=int(count)
                if count<0: raise ValueError('Неверный повтор UBC.')
                yield from repeat(float(value.upper().replace(b'D',b'E')),count)

    with Path(path).open('rb') as stream:
        while lines:=list(islice(stream,32768)):
            lines[0]=lines[0].removeprefix(b'\xef\xbb\xbf')
            data=b' '.join(lines)
            if any(marker in data for marker in (b'*',b'#',b'!')):
                tokens=expanded(lines)
                while (block:=np.fromiter(islice(tokens,65536),dtype=float)).size:
                    yield block
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('error',DeprecationWarning)
                    yield np.fromstring(data.replace(b'D',b'E').replace(b'd',b'e'),sep=' ')


def read_ubc(mesh_path,model_path,mode='gravity',dummy=None,crs=''):
    """UBC tensor mesh: SW top elevation; z fastest, East next, North last."""
    a = np.asarray(list(_numbers(mesh_path)))
    if len(a)<9: raise ValueError('Неполный UBC mesh.')
    counts = a[:3].astype(int)
    if np.any(counts<1) or np.any(counts != a[:3]) or len(a)!=6+sum(counts):
        raise ValueError('UBC: проверьте Nx Ny Nz и списки размеров (octree не поддерживается).')
    nx,ny,nz = counts; widths = np.split(a[6:],[nx,nx+ny])
    if any(np.any(w<=0) for w in widths): raise ValueError('Размеры UBC ячеек должны быть >0.')
    mesh = Mesh(a[3]+np.r_[0,np.cumsum(widths[0])],a[4]+np.r_[0,np.cumsum(widths[1])],-a[5]+np.r_[0,np.cumsum(widths[2])])
    components=3 if mode=='vector' else 1
    v=np.empty((mesh.size,components))
    cursor=0
    for block in _ubc_blocks(model_path):
        end=cursor+len(block)
        if end>v.size: raise ValueError('UBC model: число свойств не соответствует mesh.')
        src=np.arange(cursor,end)
        cell=src//components
        y,remainder=np.divmod(cell,nx*nz)
        x,z=np.divmod(remainder,nz)
        v[(x*ny+y)*nz+z,src%components]=block
        cursor=end
        report(cursor,v.size,'Свойства UBC')
    if cursor!=v.size: raise ValueError('UBC model: число свойств не соответствует mesh.')
    for start in range(0,mesh.size,131072):
        s=slice(start,start+131072)
        valid=np.isfinite(v[s])
        if dummy is not None: valid &= v[s]!=dummy
        mesh.active[s]=valid.all(axis=1)
    return VoxelModel(mesh,v[:,0] if components==1 else v,mode,crs,
                      [dict(operation='UBC import',source=Path(model_path).name,dummy=dummy)],own_values=True)


def read_centers(path,mode='gravity',vertical='Down',crs=''):
    from .models import read_table
    t=read_table(path)
    required=['East','North',vertical]+(['JE','JN','JD'] if mode=='vector' else ['Property'])
    if not set(required)<=set(t.columns): raise ValueError('Нужны столбцы: '+', '.join(required))
    xyz=t[required[:3]].to_numpy(float); xyz[:,2]*=1 if vertical=='Down' else -1
    axes=[np.unique(xyz[:,i]) for i in range(3)]
    if any(len(a)<2 for a in axes): raise ValueError('Таблица центров: минимум 2 центра по каждой оси; иначе используйте UBC/VTR с границами.')
    edges=[]
    for a in axes:
        if not np.allclose(np.diff(a),np.diff(a)[0],rtol=1e-6): raise ValueError('Из неравномерных центров нельзя однозначно вывести грани; используйте UBC/VTR.')
        step=a[1]-a[0]; edges.append(np.r_[a-step/2,a[-1]+step/2])
    mesh=Mesh(*edges); indices=tuple(np.searchsorted(a,xyz[:,i]) for i,a in enumerate(axes))
    flat=np.ravel_multi_index(indices,mesh.shape)
    if len(np.unique(flat))!=len(flat): raise ValueError('Повторяющиеся центры ячеек.')
    values=np.zeros((mesh.size,3) if mode=='vector' else mesh.size)
    raw=t[required[3:]].to_numpy(float); raw=raw if mode=='vector' else raw[:,0]
    mesh.active[:]=False; mesh.active[flat]=np.isfinite(raw).all(axis=1) if mode=='vector' else np.isfinite(raw)
    values[flat]=raw
    return VoxelModel(mesh,values,mode,crs,[dict(operation='centers import',vertical=vertical)])


def _read_prism_grid(grid,mode,array,crs):
    """Reconstruct a tensor mesh from exported axis-aligned VTK hexahedra."""
    from .inversion_resources import available_memory
    if not grid.n_cells or not np.all(grid.celltypes==12):
        raise ValueError('Импорт VTU поддерживает осевые прямоугольные призмы.')
    connectivity = grid.cell_connectivity.reshape(-1,8)
    raw = np.asarray(grid.cell_data[array])
    shape = (grid.n_cells,3) if mode=='vector' else (grid.n_cells,)
    if raw.shape!=shape:
        raise ValueError('Неверное число компонент VTK массива.')
    bounds = np.empty((grid.n_cells,6))
    for start in range(0,grid.n_cells,65536):
        stop = min(start+65536,grid.n_cells)
        points = grid.points[connectivity[start:stop]].astype(float)
        points[...,2] *= -1
        low,high = points.min(axis=1),points.max(axis=1)
        upper = points==high[:,None,:]
        corner = upper @ np.array([1,2,4])
        if (not np.isfinite(points).all() or np.any(high<=low)
                or not (upper | (points==low[:,None,:])).all()
                or not (np.sort(corner,axis=1)==np.arange(8)).all()):
            raise ValueError('Импорт VTU поддерживает осевые прямоугольные призмы.')
        bounds[start:stop,:3],bounds[start:stop,3:] = low,high
    axes = [np.unique(bounds[:,[i,i+3]]) for i in range(3)]
    import math
    size = math.prod(len(axis)-1 for axis in axes)
    if size*(8*(3 if mode=='vector' else 1)+4) > .65*available_memory(np):
        raise ValueError('Для восстановления сетки VTU недостаточно памяти. Откройте исходный результат GGModel или NPZ.')
    mesh = Mesh(*axes,np.zeros(size,dtype=bool))
    values=np.zeros((size,)+raw.shape[1:],dtype=float)
    lo = np.column_stack([axis.searchsorted(bounds[:,i]) for i,axis in enumerate(axes)])
    hi = np.column_stack([axis.searchsorted(bounds[:,i+3]) for i,axis in enumerate(axes)])
    whole = (hi==lo+1).all(axis=1)
    valid = np.isfinite(raw).reshape(len(raw),-1).all(axis=1)
    ids = np.ravel_multi_index(lo[whole & valid].T,mesh.shape)
    if len(ids)!=len(np.unique(ids)):
        raise ValueError('Призмы VTU перекрываются.')
    values[ids],mesh.active[ids] = raw[whole & valid],True
    cube = values.reshape(mesh.shape+raw.shape[1:])
    active = mesh.active.reshape(mesh.shape)
    for i in np.flatnonzero(~whole & valid):
        region = tuple(slice(l,h) for l,h in zip(lo[i],hi[i]))
        if active[region].any():
            raise ValueError('Призмы VTU перекрываются.')
        cube[region],active[region] = raw[i],True
    return VoxelModel(mesh,values,mode,crs,[dict(operation='VTU import',array=array)],own_values=True)


def read_vtk(path,mode='gravity',array=None,crs=''):
    import pyvista as pv
    g=pv.read(path)
    array=array or ('M_END' if mode=='vector' else 'property')
    if array not in g.cell_data: raise ValueError(f'Нет cell-массива {array}. Доступны: {list(g.cell_data.keys())}')
    if isinstance(g,pv.UnstructuredGrid):
        return _read_prism_grid(g,mode,array,crs)
    if not isinstance(g,(pv.RectilinearGrid,pv.ImageData)):
        raise ValueError('Импорт VTK: требуется осевая RectilinearGrid/ImageData. Для обрезанной модели используйте NPZ.')
    if isinstance(g,pv.ImageData):
        if not np.allclose(g.direction_matrix,np.eye(3)): raise ValueError('Поворот VTK сетки не поддерживается.')
        axes=[g.origin[i]+np.arange(g.dimensions[i])*g.spacing[i] for i in range(3)]
    else: axes=[np.asarray(g.x),np.asarray(g.y),np.asarray(g.z)]
    if any(np.any(np.diff(a)<=0) for a in axes): raise ValueError('VTK: оси должны возрастать; Z трактуется как высота Up.')
    mesh=Mesh(axes[0],axes[1],-axes[2][::-1])
    raw=np.asarray(g.cell_data[array]); extra=(3,) if mode=='vector' else ()
    if raw.shape != (mesh.size,)+extra: raise ValueError('Неверное число компонент VTK массива.')
    v=raw.reshape(mesh.shape+extra,order='F')[:,:,::-1].reshape((mesh.size,)+extra)
    valid=np.isfinite(v) if mode!='vector' else np.isfinite(v).all(axis=1)
    mesh.active=valid
    return VoxelModel(mesh,np.nan_to_num(v),mode,crs,[dict(operation='VTK import',array=array,vertical='Up')])


def _read_geosoft_native(path,mode='gravity',crs=''):
    """Official GX runtime only; never guess a proprietary binary layout."""
    try:
        from geosoft.gxpy import gx,vox
    except ImportError as exc:
        raise ValueError('Для .geosoft_voxel нужен официальный Geosoft GX Python/runtime в этом Python. Альтернатива: Oasis montaj → Voxel → Conversions → Voxel to UBC; загрузите .msh + .mod. Автономный бинарный декодер пока недоступен.') from exc
    runtime=os.environ.get('GEOGARMODEL_GX_REDIST','')
    context_options=dict(redist_override=True,redist_dir=runtime) if runtime else {}
    try:
        context=gx.GXpy(suppress_progress=True,**context_options)
    except Exception as exc:
        raise ValueError('Geosoft GX runtime недоступен. Укажите GEOGARMODEL_GX_REDIST (каталог с bin/geodist.dll) или используйте UBC .msh + .mod. '+str(exc)) from exc
    with context:
        with vox.Vox.open(str(path),depth=True) as v:
            axes=[]
            for centers,widths in ((v.locations_x,v.cells_x),(v.locations_y,v.cells_y),(v.locations_z,v.cells_z)):
                centers=np.asarray(centers); widths=np.asarray(widths)
                edges=np.r_[centers-widths/2,centers[-1]+widths[-1]/2]
                if not np.allclose(edges[1:-1],centers[:-1]+widths[:-1]/2): raise ValueError('Geosoft: ячейки не смежные.')
                axes.append(edges)
            raw=v.np(dtype=np.float64)
            if raw.ndim!=3: raise ValueError('Импортируйте JE, JN, JD отдельно: знаки вертикальной компоненты должны быть заданы в Down.')
            values=raw.transpose(2,1,0).ravel()
    if mode=='vector': raise ValueError('Для векторного режима загрузите отдельно JE, JN, JD.')
    mesh=Mesh(*axes); mesh.active=np.isfinite(values)&(np.abs(values)<1e30)
    return VoxelModel(mesh,np.where(mesh.active,values,0),mode,crs,[dict(operation='Geosoft GX import')])


def read_geosoft(path,mode='gravity',crs=''):
    # A native GX failure must never terminate the application process.
    with tempfile.TemporaryDirectory(prefix='geogarmodel-gx-') as folder:
        output=Path(folder)/'model.npz'
        try:
            proc=subprocess.run([sys.executable,'-m','geogarmodel.geosoft_bridge',str(Path(path).resolve()),str(output),mode,crs],capture_output=True,text=True,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except subprocess.TimeoutExpired as exc:
            raise ValueError('Geosoft GX не завершил чтение за 180 с. Проверьте runtime или экспортируйте UBC .msh + .mod.') from exc
        if proc.returncode or not output.exists():
            detail=(proc.stderr or proc.stdout).strip()[-1800:]
            raise ValueError(f'Geosoft GX не прочитал воксель (код {proc.returncode}). Установите официальный GX runtime и пакет geosoft либо экспортируйте UBC .msh + .mod.\n{detail}')
        return load_voxel(output)


def load_voxel(path,mode='gravity',crs='',**kwargs):
    ext=Path(path).suffix.lower()
    if ext=='.vo':
        from .gocad_voxet import read_voxet
        return read_voxet(path,mode,crs,kwargs.get('property_id'),kwargs.get('binary_paths'))
    if ext=='.ggmodel':
        from .model_archive import ModelArchive
        r=ModelArchive(path).load()
        return VoxelModel(r.mesh,r.model,r.settings['mode'],r.settings.get('crs',''),[dict(operation='previous result')])
    if ext=='.npz':
        with np.load(path,allow_pickle=False) as s:
            if 'voxel_metadata' in s:
                meta=json.loads(str(s['voxel_metadata']))
                return VoxelModel(Mesh(s['east'],s['north'],s['depth'],s['active'],s['bounds'] if s['bounds'].size else None),s['model'],**meta)
        r=InversionResult.load(path)
        return VoxelModel(r.mesh,r.model,r.settings['mode'],r.settings.get('crs',''),[dict(operation='previous result')])
    if ext in ('.msh','.mesh'):
        components = kwargs.get('component_paths')
        if components is not None:
            return combine_components([read_ubc(path,p,'J_component',kwargs.get('dummy'),crs) for p in components])
        return read_ubc(path,kwargs['model_path'],mode,kwargs.get('dummy'),crs)
    if ext in ('.vtk','.vtr','.vti','.vtu'): return read_vtk(path,mode,kwargs.get('array'),crs)
    if ext=='.geosoft_voxel': return read_geosoft(path,mode,crs)
    return read_centers(path,mode,kwargs.get('vertical','Down'),crs)


def combine_components(models):
    first=models[0]
    for m in models:
        if m.mode=='vector' or any(not np.array_equal(a,b) for a,b in zip((first.mesh.east,first.mesh.north,first.mesh.depth),(m.mesh.east,m.mesh.north,m.mesh.depth))) or not np.array_equal(first.mesh.cells(),m.mesh.cells()) or m.crs!=first.crs:
            raise ValueError('JE/JN/JD должны иметь одинаковые грани сетки и CRS.')
    result=first.copy(); result.mode='vector'; result.mesh.active=np.logical_and.reduce([m.mesh.active for m in models])
    result.values=np.column_stack([m.values for m in models]); result.values[~result.mesh.active]=0
    result.source_path=''; result.source_array=''
    result.history.append(dict(operation='combine JE JN JD, A/m',sources=[m.source_path for m in models]))
    return result


def resample(model,target,method='nearest',outside=0.):
    """Explicit sampling of cell properties, not conservative mass transfer."""
    if method not in ('nearest','linear'): raise ValueError('Выберите nearest или linear.')
    axes=[(a[1:]+a[:-1])/2 for a in (model.mesh.east,model.mesh.north,model.mesh.depth)]
    cube=model.values.reshape(model.mesh.shape+(() if model.mode!='vector' else (3,)))
    points=target.centers()
    # Nearest cell and active mask are defined by actual cell edges, not center hull.
    indices=[np.searchsorted(a,points[:,i],side='right')-1 for i,a in enumerate((model.mesh.east,model.mesh.north,model.mesh.depth))]
    inside=np.ones(len(points),bool)
    for i,n in zip(indices,model.mesh.shape): inside &= (i>=0)&(i<n)
    clipped=[np.clip(i,0,n-1) for i,n in zip(indices,model.mesh.shape)]
    flat=np.ravel_multi_index(tuple(clipped),model.mesh.shape)
    cells=model.mesh.cells()[flat]
    inside &= model.mesh.active[flat]&np.all(points>=cells[:,:3],axis=1)&np.all(points<=cells[:,3:],axis=1)
    values=model.values[flat].copy() if method=='nearest' else RegularGridInterpolator(axes,cube,method='linear',bounds_error=False,fill_value=None)(points)
    values[~inside]=outside
    target=Mesh(target.east,target.north,target.depth,target.active.copy(),target.cells())
    target.active &= inside
    return VoxelModel(target,values,model.mode,model.crs,model.history+[dict(operation='resample',method=method,outside=outside,note='Not mass conserving; outside cells inactive')])


def edit(model,operation,a=0.,b=1.,seed=42,component=None):
    result=model.copy(); v=result.values
    target=v if component is None or v.ndim==1 else v[:,component]
    active=result.mesh.active
    if operation=='constant': target[active]=a
    elif operation=='multiply': target[active]*=a
    elif operation=='add': target[active]+=a
    elif operation=='clamp':
        if b<a: raise ValueError('Максимум меньше минимума.')
        target[active]=np.clip(target[active],a,b)
    elif operation in ('normal','uniform'):
        if b<0 and operation=='normal': raise ValueError('Стандартное отклонение должно быть ≥0.')
        if b<a and operation=='uniform': raise ValueError('Максимум меньше минимума.')
        rng=np.random.default_rng(seed)
        target[active]=rng.normal(a,b,target[active].shape) if operation=='normal' else rng.uniform(a,b,target[active].shape)
    elif operation=='depth_gradient': target[active]=a+b*result.mesh.centers()[active,2].reshape((-1,)+( (1,) if target.ndim==2 else ()))
    else: raise ValueError('Неизвестная операция.')
    if not np.isfinite(v).all(): raise ValueError('Операция создала NaN/Inf.')
    result.history.append(dict(operation=operation,a=a,b=b,seed=seed,component=component))
    return result


def clip_surface(model,east,north,z,keep='below',vertical='Down'):
    """Column-wise vertical prism cut at surface interpolated at XY cell centers.

    Unknown surface outside convex hull is rejected. Partial prisms stay partial
    in the forward operator and exports; no smoothing across inactive neighbors.
    """
    if keep not in ('below','above'): raise ValueError('Выберите выше или ниже поверхности.')
    xyz=np.column_stack((east,north,z)).astype(float)
    if not np.isfinite(xyz).all(): raise ValueError('Поверхность содержит NaN/Inf.')
    result=model.copy(); centers=result.mesh.centers()
    surface=LinearNDInterpolator(xyz[:,:2],xyz[:,2])(centers[:,:2])
    if vertical=='Up': surface=-surface
    if np.any(~np.isfinite(surface[result.mesh.active])): raise ValueError('Поверхность не покрывает все активные XY модели. Обрежьте модель по площади или дополните поверхность.')
    cells=result.mesh.cells()
    if keep=='below': cells[:,2]=np.maximum(cells[:,2],surface)
    else: cells[:,5]=np.minimum(cells[:,5],surface)
    result.mesh.active &= np.isfinite(surface)&(cells[:,5]>cells[:,2])
    if not result.mesh.active.any(): raise ValueError('Обрезка удаляет все ячейки.')
    cells[~result.mesh.active]=model.mesh.cells()[~result.mesh.active]
    result.mesh.bounds=cells; result.values[~result.mesh.active]=0
    result.history.append(dict(operation='surface clip',keep=keep,vertical=vertical,approximation='surface at XY cell center, vertical prism intersection'))
    return result
