"""Image layers and affine georeferencing for the map workspace."""
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from pyproj import CRS


def read_raster(path):
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.errors import NotGeoreferencedWarning

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            transform = source.transform if not source.transform.is_identity else None
            crs = source.crs.to_wkt() if source.crs else ''
            prj = Path(path).with_suffix('.prj')
            if prj.is_file() and not crs:
                crs = CRS.from_wkt(prj.read_text(encoding='utf-8-sig')).to_wkt()
            rgb = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            bands = source.colorinterp
            alpha = source.dataset_mask()
            if ColorInterp.palette in bands:
                index = bands.index(ColorInterp.palette)+1
                raw = source.read(index)
                palette = source.colormap(index)
                colors = np.zeros((max(palette)+1,4),dtype=np.uint8)
                for key,color in palette.items():
                    colors[key] = color
                image = colors[np.minimum(raw,len(colors)-1)]
                image[...,3] = np.minimum(image[...,3],alpha)
            else:
                indexes = [bands.index(color)+1 for color in rgb] if all(color in bands for color in rgb) else [1]
                raw = source.read(indexes)
                if raw.dtype!=np.uint8:
                    data = raw[:,alpha>0]
                    finite = data[np.isfinite(data)]
                    low,high = (finite.min(),finite.max()) if finite.size else (0,1)
                    raw = np.nan_to_num(np.clip((raw.astype(float)-low)/(high-low or 1),0,1)*255).astype(np.uint8)
                image = np.empty((source.height,source.width,4),dtype=np.uint8)
                image[...,:3] = np.moveaxis(raw,0,-1)
                image[...,3] = alpha
            return image,transform,crs


def fit_affine(pixel, world):
    from rasterio.transform import Affine
    pixel,world = np.asarray(pixel,dtype=float),np.asarray(world,dtype=float)
    if len(pixel)<3 or not np.isfinite(pixel).all() or not np.isfinite(world).all():
        raise ValueError('Введите координаты X и Y как минимум для трёх точек.')
    origin = pixel.mean(axis=0)
    model_origin = world.mean(axis=0)
    design = np.column_stack((pixel-origin,np.ones(len(pixel))))
    coef,_,rank,_ = np.linalg.lstsq(design,world-model_origin,rcond=None)
    if rank!=3 or np.linalg.matrix_rank(coef[:2])!=2:
        raise ValueError('Опорные точки на изображении и на карте должны задавать площадь.')
    offset = model_origin+coef[2]-origin@coef[:2]
    return Affine(coef[0,0],coef[1,0],offset[0],coef[0,1],coef[1,1],offset[1])


@dataclass
class RasterLayer:
    image: np.ndarray
    bounds: tuple
    crs: str
    control_points: list | None = None

    def reproject(self, target):
        from rasterio.transform import from_bounds

        transform = from_bounds(*self.bounds,self.image.shape[1],self.image.shape[0])
        return place_raster(self.image,transform,self.crs,target,self.control_points)


def place_raster(image, transform, source, target, control_points=None):
    from rasterio.warp import reproject, transform_bounds
    from rasterio.transform import from_bounds
    from .work_progress import report

    if transform is None or not np.isfinite(tuple(transform)).all() or transform.determinant==0:
        raise ValueError('Для растра нужна действительная пространственная привязка.')
    source,target = CRS.from_user_input(source),CRS.from_user_input(target)
    height,width = image.shape[:2]
    x,y = zip(*(transform*pixel for pixel in ((0,0),(width,0),(0,height),(width,height))))
    bounds = (min(x),min(y),max(x),max(y))
    if source.equals(target,ignore_axis_order=True) and transform.b==transform.d==0 and transform.a>0>transform.e:
        return RasterLayer(image,bounds,target.to_string(),control_points)
    if not source.equals(target,ignore_axis_order=True):
        bounds = transform_bounds(source.to_wkt(),target.to_wkt(),*bounds,densify_pts=21)
    if not np.isfinite(bounds).all() or bounds[0]>=bounds[2] or bounds[1]>=bounds[3]:
        raise ValueError('Проверьте исходную СК и координаты привязки растра.')
    result = np.zeros_like(image)
    target_transform = from_bounds(*bounds,width,height)
    for band in range(4):
        reproject(image[...,band], result[...,band],src_transform=transform,src_crs=source.to_wkt(),
                  dst_transform=target_transform,dst_crs=target.to_wkt())
        report(band+1,4,'Преобразование растра')
    if not result[...,3].any():
        raise ValueError('После преобразования растр не содержит видимых пикселей.')
    return RasterLayer(result,bounds,target.to_string(),control_points)
