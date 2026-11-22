"""Portable map projects and lightweight SHP/GeoJSON overlays."""
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
from pyproj import CRS,Transformer
from .grids import Grid
from scipy.interpolate import RegularGridInterpolator


def _load_pyshp():
    """Import pyshp and report the interpreter that actually needs it."""
    try:
        return importlib.import_module('shapefile')
    except ImportError as error:
        executable="& '"+sys.executable.replace("'","''")+"'"
        detail=(f'\nПричина импорта: {error}' if not isinstance(error,ModuleNotFoundError)
                or error.name!='shapefile' else '')
        raise ValueError(
            'Библиотека pyshp не доступна в Python, которым запущен GeoGarModel:\n'
            f'{sys.executable}\n\n'
            'Закройте программу, выполните в PowerShell:\n'
            f'{executable} -m pip install "pyshp>=2.3,<3"\n\n'
            'Затем снова запустите GeoGarModel этим же Python.'+detail
        ) from None


def reproject_grid(grid,target_crs,source_crs=None):
    source=CRS.from_user_input(source_crs or grid.crs); target=CRS.from_user_input(target_crs)
    forward=None if source.equals(target) else Transformer.from_crs(source,target,always_xy=True)
    # An ArcGIS WKT and the project EPSG can describe the same XY transform.
    if forward is None or forward.definition.split()[:1]==['proj=noop']:
        return Grid(grid.x.copy(),grid.y.copy(),grid.values.copy(),grid.name,grid.unit,target.to_string(),
                    'degrees' if target.is_geographic else 'm',list(grid.history))
    if not target.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in target.axis_info[:2]):
        raise ValueError('Для проекта выберите плоскую СК в метрах.')
    inverse=Transformer.from_crs(target,source,always_xy=True)
    bounds=forward.transform_bounds(grid.x[0],grid.y[0],grid.x[-1],grid.y[-1],densify_pts=64,errcheck=True)
    if not np.isfinite(bounds).all() or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
        raise ValueError('Карта не попадает в область выбранной СК. Проверьте исходную СК и координаты.')
    x=np.linspace(bounds[0],bounds[2],max(3,len(grid.x))); y=np.linspace(bounds[1],bounds[3],max(3,len(grid.y)))
    sampler=RegularGridInterpolator((grid.y,grid.x),grid.values,bounds_error=False,fill_value=np.nan)
    values=np.empty((len(y),len(x)))
    # Bounded temporary arrays even for large input maps.
    for start in range(0,len(y),128):
        xx,yy=np.meshgrid(x,y[start:start+128]); sx,sy=inverse.transform(xx,yy)
        sx=np.where((sx>=grid.x[0]-grid.dx*1e-7)&(sx<=grid.x[-1]+grid.dx*1e-7),np.clip(sx,grid.x[0],grid.x[-1]),sx)
        sy=np.where((sy>=grid.y[0]-grid.dy*1e-7)&(sy<=grid.y[-1]+grid.dy*1e-7),np.clip(sy,grid.y[0],grid.y[-1]),sy)
        values[start:start+128]=sampler(np.stack((sy,sx),axis=-1))
    return Grid(x,y,values,grid.name,grid.unit,target.to_string(),'m',grid.history+[
        dict(operation='reproject',source_crs=source.to_string(),target_crs=target.to_string(),
             interpolation='linear',shape=list(values.shape))])


@dataclass
class Vector:
    features: list
    crs: str

    def reproject(self,target_crs):
        transform=Transformer.from_crs(self.crs,target_crs,always_xy=True)
        def coordinates(c):
            if len(c)>=2 and isinstance(c[0],(float,int)):
                x,y=transform.transform(c[0],c[1],errcheck=True); return [x,y,*c[2:]]
            return [coordinates(v) for v in c]
        def geometry(g):
            if g is None: return None
            if g['type']=='GeometryCollection': return dict(type=g['type'],geometries=[geometry(v) for v in g['geometries']])
            return dict(type=g['type'],coordinates=coordinates(g['coordinates']))
        return Vector([geometry(g) for g in self.features],CRS.from_user_input(target_crs).to_string())

    def draw(self,ax,crs,style=None):
        from matplotlib.path import Path as PlotPath
        from matplotlib.patches import PathPatch
        from matplotlib.colors import to_rgba
        style=style or {}; color=style.get('line_color','#fbbf24'); width=style.get('line_width',1.)
        transform=Transformer.from_crs(self.crs,crs,always_xy=True)
        def geometry(g):
            kind=g['type']; coords=g.get('coordinates',[])
            if kind=='GeometryCollection':
                for child in g['geometries']: geometry(child)
            elif kind in ('MultiPolygon','MultiLineString','MultiPoint'):
                child={'MultiPolygon':'Polygon','MultiLineString':'LineString','MultiPoint':'Point'}[kind]
                for c in coords: geometry(dict(type=child,coordinates=c))
            elif kind=='Polygon':
                if style.get('fill',False):
                    vertices=[]; codes=[]
                    for i,ring in enumerate(coords):
                        if len(ring)<3: continue
                        a=np.asarray(ring)[:,:2]; x,y=transform.transform(a[:,0],a[:,1]); a=np.column_stack((x,y))
                        if not np.array_equal(a[0],a[-1]): a=np.vstack((a,a[0]))
                        area=np.sum(a[:-1,0]*a[1:,1]-a[1:,0]*a[:-1,1])
                        if (area>0)!=(i==0): a=a[::-1]
                        vertices.extend(a); codes.extend([PlotPath.MOVETO]+[PlotPath.LINETO]*(len(a)-2)+[PlotPath.CLOSEPOLY])
                    if vertices:
                        ax.add_patch(PathPatch(PlotPath(vertices,codes),facecolor=to_rgba(style.get('fill_color',color),style.get('fill_alpha',.3)),edgecolor='none',zorder=100))
                for ring in coords: geometry(dict(type='LineString',coordinates=ring))
            elif kind=='Point':
                x,y=transform.transform(*coords[:2]); ax.scatter([x],[y],s=18,c=color,zorder=100)
            elif kind=='LineString' and coords:
                a=np.asarray(coords); x,y=transform.transform(a[:,0],a[:,1]); ax.plot(x,y,color=color,linewidth=width,zorder=100)
        for feature in self.features:
            if feature: geometry(feature)


def read_vector(path,source_crs=None):
    path=Path(path)
    if path.suffix.lower()=='.shp':
        shapefile=_load_pyshp()
        prj=path.with_suffix('.prj')
        if not prj.exists() and not source_crs: raise ValueError('Рядом с SHP нужен файл .prj с системой координат.')
        crs=source_crs or CRS.from_wkt(prj.read_text()).to_string()
        with shapefile.Reader(str(path)) as reader:
            features=[s.__geo_interface__ for s in reader.shapes() if s.shapeType!=0]
    else:
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        crs=data.get('crs',{}).get('properties',{}).get('name','EPSG:4326')
        features=([f.get('geometry') for f in data['features']] if data['type']=='FeatureCollection'
                  else [data.get('geometry')] if data['type']=='Feature' else [data])
    CRS.from_user_input(crs)
    return Vector(features,crs)


def read_xyz_table(path):
    """Read named columns or number the columns of a raw XYZ point file."""
    import csv
    import re
    import pandas as pd
    if Path(path).suffix.lower()=='.xlsx':
        table=pd.read_excel(path)
    else:
        with Path(path).open(encoding='utf-8-sig') as stream:
            line=next((row.strip() for row in stream if row.strip() and not row.lstrip().startswith('#')),'')
        if not line:
            raise ValueError('Таблица XYZ пуста.')
        dialect=next((separator for separator in ('\t',';',',') if separator in line),None)
        first=next(csv.reader([line],delimiter=dialect)) if dialect else re.split(r'\s+',line)
        try:
            for value in first:
                if value.strip(): float(value)
            header=None
        except ValueError:
            header=0
        table=pd.read_csv(path,sep=dialect or r'\s+',header=header,comment='#',
                          encoding='utf-8-sig',float_precision='round_trip')
        if header is None:
            table.columns=[f'Столбец {i+1}' for i in range(len(table.columns))]
    table.columns=table.columns.map(str)
    return table


def save_project(path,name,layers,view,scale,crs='',presentation=None):
    arrays={}; rows=[]; tables={}
    for i,layer in enumerate(layers):
        row={k:layer[k] for k in ('name','visible','style','kind')}
        value=layer['data']
        if row['kind']=='grid':
            for key in ('x','y','values'): arrays[f'{i}_{key}']=getattr(value,key)
            row['metadata']={k:getattr(value,k) for k in ('name','unit','crs','coordinate_unit','history')}
        elif row['kind']=='raster':
            arrays[f'{i}_image']=value.image
            row.update(bounds=value.bounds,crs=value.crs,control_points=value.control_points)
        else: row.update(features=value.features,crs=value.crs)
        if 'xyz' in layer:
            source=layer['xyz']
            table=source['table']
            key=tables.get(id(table))
            if key is None:
                key=f'xyz_{len(tables)}'
                tables[id(table)]=key
                arrays[key]=np.frombuffer(table.to_csv(index=False).encode('utf-8'),dtype=np.uint8)
            row['xyz']={**{k:source[k] for k in ('stem','columns','crs')},'table_key':key}
        rows.append(row)
    arrays['manifest']=json.dumps(dict(version=1,name=name,layers=rows,view=view,scale=scale,crs=crs,presentation=presentation or {}),ensure_ascii=False)
    if hasattr(path,'write'):
        np.savez_compressed(path,**arrays); return
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as stream:
        temporary=stream.name
        try:
            np.savez_compressed(stream,**arrays); stream.close(); os.replace(temporary,path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)


def load_project(path):
    # The map reader also accepts a complete application project.
    from zipfile import ZipFile
    from io import BytesIO
    with ZipFile(path) as archive:
        if 'manifest.json' in archive.namelist(): return load_project(BytesIO(archive.read('maps.npz')))
    if hasattr(path,'seek'): path.seek(0)
    from io import StringIO
    import pandas as pd
    layers=[]; tables={}
    with np.load(path,allow_pickle=False) as data:
        manifest=json.loads(str(data['manifest']))
        if manifest['version']!=1: raise ValueError('Неподдерживаемая версия проекта карт.')
        for i,row in enumerate(manifest['layers']):
            if row['kind']=='grid':
                value=Grid(*(data[f'{i}_{k}'] for k in ('x','y','values')),**row['metadata'])
            elif row['kind']=='raster':
                from .raster_map import RasterLayer
                value=RasterLayer(data[f'{i}_image'],tuple(row['bounds']),row['crs'],row.get('control_points'))
            else:
                value=Vector(row['features'],row['crs'])
            layer={**{k:row[k] for k in ('name','visible','style','kind')},'data':value}
            if 'xyz' in row:
                source=dict(row['xyz'])
                key=source.pop('table_key')
                if key not in tables:
                    tables[key]=pd.read_csv(StringIO(data[key].tobytes().decode('utf-8')),float_precision='round_trip')
                layer['xyz']=dict(source,table=tables[key])
            layers.append(layer)
    return manifest,layers
