"""Independent Geosoft v2 scalar GRD reader (no Geosoft runtime).

Format reference: Seequent Geosoft Grid File Format; compressed block layout
also documented by Fatiando Harmonica's oasis_montaj_grd reader.
"""
from pathlib import Path
import struct
import zlib
import numpy as np
from .grids import Grid


def read_binary_grid(path):
    path=Path(path)
    with path.open('rb') as stream:
        header=stream.read(512)
        if len(header)!=512: raise ValueError('Повреждённый заголовок Geosoft GRD.')
        es,sf,nx,ny,order=struct.unpack_from('<5i',header)
        size=es-1024 if es>1024 else es
        if size not in (1,2,4,8) or sf not in (0,1,2) or order not in (-1,1):
            raise ValueError('Это не поддерживаемый числовой Geosoft GRD v2 (Surfer и RGB GRD не поддерживаются).')
        if nx<2 or ny<2 or nx*ny>500_000_000: raise ValueError('Некорректный или слишком большой размер GRD.')
        dx,dy,x0,y0,rotation,base,factor=struct.unpack_from('<7d',header,20)
        if not np.isfinite([dx,dy,x0,y0,rotation,base,factor]).all() or dx<=0 or dy<=0 or factor==0:
            raise ValueError('Некорректная геометрия или масштаб данных GRD.')
        expected=nx*ny*size
        if es>1024:
            data=stream.read(); count=struct.unpack_from('<i',data,8)[0]
            if count<1 or 16+count*12>len(data): raise ValueError('Повреждённая таблица блоков GRD.')
            offsets=struct.unpack_from('<'+'q'*count,data,16)
            lengths=struct.unpack_from('<'+'i'*count,data,16+count*8)
            blocks=[]; total=0
            for offset,length in zip(offsets,lengths):
                start=offset-512+16; end=offset-512+length
                if start<0 or end>len(data) or end<=start: raise ValueError('Повреждённый блок GRD.')
                decoder=zlib.decompressobj(); block=decoder.decompress(data[start:end],expected-total+1)
                total+=len(block)
                if total>expected or not decoder.eof: raise ValueError('Неверный размер распакованного GRD.')
                blocks.append(block)
            raw=b''.join(blocks)
        else: raw=stream.read(expected+1)
    if len(raw)!=expected: raise ValueError('Размер массива GRD не соответствует заголовку.')
    kind='f' if sf==2 else 'u' if sf==0 else 'i'
    dtype=np.dtype('<'+kind+str(size))
    values=np.frombuffer(raw,dtype=dtype).astype(float)
    dummy=-1e32 if kind=='f' else np.iinfo(dtype).max if kind=='u' else -np.iinfo(dtype).max
    values[values<=dummy if kind=='f' else values==dummy]=np.nan
    values=values/factor+base
    if order==1: values=values.reshape(ny,nx)
    else: values=values.reshape(nx,ny,order='F'); nx,ny=ny,nx; dx,dy=dy,dx
    x=x0+np.arange(nx)*dx; y=y0+np.arange(ny)*dy
    history=[dict(operation='Geosoft GRD v2',rotation=rotation,compressed=es>1024)]
    if rotation:
        from scipy.interpolate import RegularGridInterpolator
        angle=np.deg2rad(rotation); c,s=np.cos(angle),np.sin(angle)
        corners=np.array([[0,0],[(nx-1)*dx,0],[0,(ny-1)*dy],[(nx-1)*dx,(ny-1)*dy]])
        xx=x0+corners[:,0]*c-corners[:,1]*s; yy=y0+corners[:,0]*s+corners[:,1]*c
        x=np.linspace(xx.min(),xx.max(),nx); y=np.linspace(yy.min(),yy.max(),ny)
        sample=RegularGridInterpolator((np.arange(ny)*dy,np.arange(nx)*dx),values,bounds_error=False,fill_value=np.nan)
        rotated=np.empty_like(values)
        for start in range(0,ny,128):
            xx,yy=np.meshgrid(x-x0,y[start:start+128]-y0)
            rotated[start:start+128]=sample(np.stack((-s*xx+c*yy,c*xx+s*yy),axis=-1))
        values=rotated; history.append(dict(operation='rotate_to_xy',interpolation='linear'))
    grid=Grid(x,y,values,path.stem,coordinate_unit='m',history=history)
    # Only trust actual WKT sidecars. Binary GI is deliberately not guessed.
    sidecars={p.name.lower():p for p in path.parent.iterdir() if p.is_file()}
    for name in (path.stem+'.prj',path.name+'.prj'):
        sidecar=sidecars.get(name.lower())
        if sidecar:
            from .crs_editor import read_prj
            grid.crs=read_prj(sidecar); break
    gi=sidecars.get((path.name+'.gi').lower())
    if gi: grid.history.append(dict(operation='GI companion',file=gi.name,crs_status='unresolved; use project or select source CRS'))
    return grid
