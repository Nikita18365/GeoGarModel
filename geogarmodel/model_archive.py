"""Seekable, uncompressed NumPy members. Each horizontal layer is independent."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile
import numpy as np
from .inversion import InversionResult, Mesh
from .result_preview import preview_volume


@contextmanager
def atomic_output(path):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.tmp')
    os.close(fd)
    try:
        yield Path(name)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def regular_cells(mesh, indices):
    ijk = np.unravel_index(indices, mesh.shape)
    axes = (mesh.east, mesh.north, mesh.depth)
    return np.column_stack([a[i+end] for end in (0,1) for a,i in zip(axes,ijk)])


def cell_blocks(mesh, size=65536):
    for start in range(0,mesh.size,size):
        ids = np.arange(start,min(start+size,mesh.size))
        yield ids, regular_cells(mesh,ids)


def write_archive(path, result):
    mesh = result.mesh
    axes, preview = preview_volume(result)
    with atomic_output(path) as tmp, ZipFile(tmp,'w',allowZip64=True) as archive:
        def put(key, values):
            with archive.open(key+'.npy','w',force_zip64=True) as stream:
                np.lib.format.write_array(stream,np.ascontiguousarray(values),allow_pickle=False)
        for key, values in zip(('east','north','depth'),(mesh.east,mesh.north,mesh.depth)):
            put(key, values)
        for key, values in zip(('px','py','pz','preview'),(*axes,preview)):
            put(key,values)
        for key in ('observations','observed','predicted'):
            put(key,getattr(result,key))
        cube = result.model.reshape(mesh.shape+result.model.shape[1:])
        mask = mesh.active.reshape(mesh.shape)
        for k in range(mesh.shape[2]):
            put(f'value{k}',cube[:,:,k])
            put(f'mask{k}',np.packbits(mask[:,:,k]))
        blocks = []
        if mesh.bounds is not None:
            for ids, regular in cell_blocks(mesh):
                bounds = mesh.bounds[ids]
                different = (regular != bounds).any(axis=1)
                if different.any():
                    key = str(len(blocks))
                    put('ids'+key,ids[different])
                    put('bounds'+key,bounds[different])
                    blocks.append(key)
        archive.writestr('manifest.json',json.dumps(dict(version=1,blocks=blocks,settings=result.settings,
                            status=result.status,history=result.history),ensure_ascii=False).encode('utf-8'))


class ModelArchive:
    def __init__(self,path):
        self.path = Path(path)
        stat=self.path.stat()
        self.signature=(stat.st_ino,stat.st_size,stat.st_mtime_ns)
        with ZipFile(path) as z:
            meta=json.loads(z.read('manifest.json'))
        if meta['version'] != 1:
            raise ValueError('Неизвестная версия файла модели.')
        self.meta=meta
        self.settings=meta['settings']
        with self.arrays() as f:
            self.axes=tuple(f[key] for key in ('east','north','depth'))
            self.shape=tuple(len(a)-1 for a in self.axes)
            self.preview=tuple(f[key] for key in ('px','py','pz')),f['preview']
            self.observations,self.predicted=f['observations'],f['predicted']

    @contextmanager
    def arrays(self):
        with self.path.open('rb') as stream:
            stat=os.fstat(stream.fileno())
            if self.signature!=(stat.st_ino,stat.st_size,stat.st_mtime_ns):
                raise ValueError('Файл модели изменился. Откройте результат заново.')
            with np.load(stream,allow_pickle=False) as f:
                yield f

    @property
    def observed(self):
        with self.arrays() as f:
            return f['observed']

    def layer(self,k):
        with self.arrays() as f:
            return self.read_layer(f,k)

    def read_layer(self,f,k):
        shape=self.shape[:2]
        values=f[f'value{k}']
        active=np.unpackbits(f[f'mask{k}'],count=shape[0]*shape[1]).reshape(shape).astype(bool)
        expected=shape+((3,) if self.settings['mode']=='vector' else ())
        if values.shape!=expected or not np.isfinite(values).all():
            raise ValueError('Некорректный слой модели.')
        return values,active

    def load(self):
        mesh=Mesh(*self.axes)
        with self.arrays() as f:
            shape=mesh.shape+((3,) if self.settings['mode']=='vector' else ())
            cube=np.empty(shape,dtype=f['value0'].dtype)
            mask=mesh.active.reshape(mesh.shape)
            for k in range(mesh.shape[2]):
                cube[:,:,k],mask[:,:,k]=self.read_layer(f,k)
            if self.meta['blocks']:
                bounds=np.empty((mesh.size,6))
                for ids,regular in cell_blocks(mesh):
                    bounds[ids]=regular
                for key in self.meta['blocks']:
                    ids=f['ids'+key]
                    if (ids<0).any() or (ids>=mesh.size).any():
                        raise ValueError('Некорректные индексы обрезки.')
                    clipped=f['bounds'+key]
                    base=regular_cells(mesh,ids)
                    active=mesh.active[ids,None]
                    if (clipped.shape!=base.shape or not np.isfinite(clipped).all()
                            or np.any(active & ((clipped[:,:3]<base[:,:3]) | (clipped[:,3:]>base[:,3:])
                                                | (clipped[:,3:]<=clipped[:,:3])))):
                        raise ValueError('Некорректные границы обрезанных ячеек.')
                    bounds[ids]=clipped
                mesh.bounds=bounds
            model=cube.reshape((mesh.size,)+shape[3:])
            return InversionResult(mesh,model,self.observations,f['observed'],self.predicted,
                                   self.meta['history'],dict(self.settings),self.meta['status'])
