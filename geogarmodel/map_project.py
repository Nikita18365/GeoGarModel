"""Portable map projects and lightweight SHP/GeoJSON overlays."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from pyproj import CRS,Transformer
from .grids import Grid


@dataclass
class Vector:
    features: list
    crs: str

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


def read_vector(path):
    path=Path(path)
    if path.suffix.lower()=='.shp':
        try: import shapefile
        except ImportError: raise ValueError('Для SHP обновите зависимости: python -m pip install -e .') from None
        prj=path.with_suffix('.prj')
        if not prj.exists(): raise ValueError('Рядом с SHP нужен файл .prj с системой координат.')
        crs=CRS.from_wkt(prj.read_text()).to_string()
        with shapefile.Reader(str(path)) as reader:
            features=[s.__geo_interface__ for s in reader.shapes() if s.shapeType!=0]
    else:
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        crs=data.get('crs',{}).get('properties',{}).get('name','EPSG:4326')
        features=([f.get('geometry') for f in data['features']] if data['type']=='FeatureCollection'
                  else [data.get('geometry')] if data['type']=='Feature' else [data])
    CRS.from_user_input(crs)
    return Vector(features,crs)


def save_project(path,name,layers,view,scale):
    arrays={}; rows=[]
    for i,layer in enumerate(layers):
        row={k:layer[k] for k in ('name','visible','style','kind')}
        value=layer['data']
        if row['kind']=='grid':
            for key in ('x','y','values'): arrays[f'{i}_{key}']=getattr(value,key)
            row['metadata']={k:getattr(value,k) for k in ('name','unit','crs','coordinate_unit','history')}
        else: row.update(features=value.features,crs=value.crs)
        rows.append(row)
    arrays['manifest']=json.dumps(dict(version=1,name=name,layers=rows,view=view,scale=scale),ensure_ascii=False)
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
