"""Portable map projects and lightweight SHP/GeoJSON overlays."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from pyproj import CRS,Transformer
from .grids import Grid
from scipy.interpolate import RegularGridInterpolator


def reproject_grid(grid,target_crs,source_crs=None):
    source=CRS.from_user_input(source_crs or grid.crs); target=CRS.from_user_input(target_crs)
    if source.equals(target):
        return Grid(grid.x.copy(),grid.y.copy(),grid.values.copy(),grid.name,grid.unit,target.to_string(),
                    'degrees' if target.is_geographic else 'm',list(grid.history))
    if not target.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in target.axis_info[:2]):
        raise ValueError('Для проекта выберите плоскую СК в метрах.')
    forward=Transformer.from_crs(source,target,always_xy=True)
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

    def draw(self,ax,crs):
        transform=Transformer.from_crs(self.crs,crs,always_xy=True)
        def geometry(g):
            kind=g['type']; coords=g.get('coordinates',[])
            if kind=='GeometryCollection':
                for child in g['geometries']: geometry(child)
            elif kind in ('MultiPolygon','MultiLineString','MultiPoint'):
                child={'MultiPolygon':'Polygon','MultiLineString':'LineString','MultiPoint':'Point'}[kind]
                for c in coords: geometry(dict(type=child,coordinates=c))
            elif kind=='Polygon':
                for ring in coords: geometry(dict(type='LineString',coordinates=ring))
            elif kind=='Point':
                x,y=transform.transform(*coords[:2]); ax.scatter([x],[y],s=18,c='#fbbf24',zorder=100)
            elif kind=='LineString' and coords:
                a=np.asarray(coords); x,y=transform.transform(a[:,0],a[:,1]); ax.plot(x,y,color='#fbbf24',linewidth=1,zorder=100)
        for feature in self.features:
            if feature: geometry(feature)


def read_vector(path,source_crs=None):
    path=Path(path)
    if path.suffix.lower()=='.shp':
        try: import shapefile
        except ImportError: raise ValueError('Для SHP обновите зависимости: python -m pip install -e .') from None
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


def save_project(path,name,layers,view,scale,crs='',presentation=None):
    arrays={}; rows=[]
    for i,layer in enumerate(layers):
        row={k:layer[k] for k in ('name','visible','style','kind')}
        value=layer['data']
        if row['kind']=='grid':
            for key in ('x','y','values'): arrays[f'{i}_{key}']=getattr(value,key)
            row['metadata']={k:getattr(value,k) for k in ('name','unit','crs','coordinate_unit','history')}
        else: row.update(features=value.features,crs=value.crs)
        rows.append(row)
    arrays['manifest']=json.dumps(dict(version=1,name=name,layers=rows,view=view,scale=scale,crs=crs,presentation=presentation or {}),ensure_ascii=False)
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as stream:
        temporary=stream.name
        try:
            np.savez_compressed(stream,**arrays); stream.close(); os.replace(temporary,path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)


def load_project(path):
    layers=[]
    with np.load(path,allow_pickle=False) as data:
        manifest=json.loads(str(data['manifest']))
        if manifest['version']!=1: raise ValueError('Неподдерживаемая версия проекта карт.')
        for i,row in enumerate(manifest['layers']):
            value=(Grid(*(data[f'{i}_{k}'].copy() for k in ('x','y','values')),**row['metadata'])
                   if row['kind']=='grid' else Vector(row['features'],row['crs']))
            layers.append({**{k:row[k] for k in ('name','visible','style','kind')},'data':value})
    return manifest,layers
