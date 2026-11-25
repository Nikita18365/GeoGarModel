"""Read-only properties for full-resolution map, image and vector layers."""
from collections import Counter
from pathlib import PurePath, PureWindowsPath
import math
import numpy as np
from pyproj import CRS
from .work_progress import report


UNKNOWN = 'Не указано'
NO_PATH = 'Не сохранён (расчётный слой или проект без сведений об источнике)'
KIND_LABELS = {'grid':'Числовая карта (грид)', 'raster':'Растровое изображение',
               'vector':'Векторный слой'}


def _number(value):
    return format(float(value), '.15g') if math.isfinite(float(value)) else UNKNOWN


def _coordinates(value):
    return format(float(value), '.6f').rstrip('0').rstrip('.') if math.isfinite(float(value)) else UNKNOWN


def _count(value):
    return str(int(value))


def _percent(n,total):
    return f'{100*n/total:.2f} %' if total else UNKNOWN


def _crs_text(value):
    if not value:
        return UNKNOWN
    try:
        crs = CRS.from_user_input(value)
        authority = crs.to_authority()
        identifier = ':'.join(authority) if authority else crs.to_string()
        return f'{identifier} | {crs.name}'
    except (ValueError, TypeError, RuntimeError):
        return str(value)


def _axis_units(data):
    if data.crs:
        try:
            axes = CRS.from_user_input(data.crs).axis_info[:2]
            if axes:
                return ' / '.join(dict.fromkeys(axis.unit_name for axis in axes))
        except (ValueError, TypeError, RuntimeError):
            pass
    return getattr(data,'coordinate_unit',UNKNOWN)


def _bounds(xmin,ymin,xmax,ymax):
    return [(key,_coordinates(value)) for key,value in
            [('xmin',xmin),('xmax',xmax),('ymin',ymin),('ymax',ymax)]]


def _grid_sections(grid,row):
    ny,nx = grid.values.shape
    bounds = _bounds(grid.x[0],grid.y[0],grid.x[-1],grid.y[-1])
    bounds += [('x_space',_coordinates(grid.dx)),('y_space',_coordinates(grid.dy)),
               ('Nx',_count(nx)),('Ny',_count(ny)),('Единицы поля',grid.unit or UNKNOWN)]
    # One finite-value copy. Median operates in place on that copy only.
    valid = grid.values[np.isfinite(grid.values)]
    n,total = valid.size,grid.values.size
    report(1,3,'Статистика карты')
    stats = [('Количество узлов (всего)',_count(total)),
             ('Количество значений (конечных)',_count(n)),
             ('Пропуски (NaN)',_count(total-n)),('Покрытие',_percent(n,total))]
    if n:
        stats.extend([('Минимум',_number(valid.min())),('Максимум',_number(valid.max())),
                      ('Среднее значение',_number(valid.mean())),
                      ('Стандартное отклонение (σ)',_number(valid.std(ddof=0)))])
        stats.append(('Медианное значение',_number(np.median(valid,overwrite_input=True))))
    report(3,3,'Статистика карты')
    result = [('Геометрия сетки (узлы)',bounds),('Статистика значений Z',stats)]
    source = row.get('xyz')
    if source:
        table = source['table']
        result.append(('Исходная таблица XYZ',[
            ('Строк',_count(len(table))),('Столбцов',_count(len(table.columns))),
            *[(axis,str(column)) for axis,column in zip(('X','Y','Z'),source['columns'])],
            ('СК таблицы',_crs_text(source.get('crs')))]))
    return result


def _raster_sections(raster):
    image = raster.image
    height,width = image.shape[:2]
    xmin,ymin,xmax,ymax = raster.bounds
    total = height*width
    alpha = image[...,3]
    visible = int(np.count_nonzero(alpha))
    opaque = int(np.count_nonzero(alpha==255))
    stats = [('Ширина, пикселей',_count(width)),('Высота, пикселей',_count(height)),
             ('Количество пикселей',_count(total)),
             ('Размер пикселя X',_coordinates((xmax-xmin)/width)),
             ('Размер пикселя Y',_coordinates((ymax-ymin)/height)),
             ('Каналы в проекте','RGBA (4)'),('Тип данных в проекте',str(image.dtype)),
             ('Видимые пиксели (alpha > 0)',_count(visible)),
             ('Полностью прозрачные (alpha = 0)',_count(total-visible)),
             ('Полностью непрозрачные (alpha = 255)',_count(opaque)),
             ('Частично прозрачные',_count(visible-opaque)),
             ('Покрытие',_percent(visible,total)),
             ('Ручных опорных точек',_count(len(raster.control_points or [])))]
    return [('Геометрия растра (границы пикселей)',_bounds(xmin,ymin,xmax,ymax)),
            ('Характеристики изображения в проекте',stats)]


def _vector_sections(vector):
    kinds = Counter()
    parts = Counter()
    xmin,ymin,xmax,ymax = math.inf,math.inf,-math.inf,-math.inf
    vertices = 0

    def visit_coordinates(coords):
        nonlocal xmin,ymin,xmax,ymax,vertices
        if not isinstance(coords,(list,tuple,np.ndarray)) or len(coords)==0:
            return
        if isinstance(coords[0],(int,float,np.number)):
            if len(coords)>=2 and math.isfinite(coords[0]) and math.isfinite(coords[1]):
                x,y = float(coords[0]),float(coords[1])
                xmin,ymin,xmax,ymax = min(xmin,x),min(ymin,y),max(xmax,x),max(ymax,y)
                vertices += 1
        else:
            for child in coords:
                visit_coordinates(child)

    def visit_geometry(geometry):
        if not geometry:
            return
        kind = geometry.get('type','Unknown')
        coords = geometry.get('coordinates') or []
        if kind=='GeometryCollection':
            for child in geometry.get('geometries') or []:
                visit_geometry(child)
        elif kind in ('MultiPoint','MultiLineString','MultiPolygon'):
            primitive = {'MultiPoint':'Point','MultiLineString':'LineString','MultiPolygon':'Polygon'}[kind]
            for child in coords:
                visit_geometry(dict(type=primitive,coordinates=child))
        else:
            before = vertices
            visit_coordinates(coords)
            if vertices>before:
                parts[kind] += 1

    empty = 0
    total = len(vector.features)
    for i,geometry in enumerate(vector.features):
        before = vertices
        if geometry:
            kinds[geometry.get('type','Unknown')] += 1
            visit_geometry(geometry)
        if vertices==before:
            empty += 1
        if i%256==0:
            report(i,total,'Статистика векторного слоя')
    report(total,total,'Статистика векторного слоя')
    bounds = _bounds(xmin,ymin,xmax,ymax)
    bounds += [('Охват по X',_coordinates(xmax-xmin)),('Охват по Y',_coordinates(ymax-ymin))]
    stats = [('Количество объектов в проекте',_count(total)),
             ('Объектов без координат',_count(empty)),
             ('Типы геометрий',', '.join(f'{k}: {n}' for k,n in sorted(kinds.items())) or UNKNOWN),
             ('Точечных частей',_count(parts['Point'])),
             ('Линейных частей',_count(parts['LineString'])),
             ('Полигональных частей',_count(parts['Polygon'])),
             ('Количество записей вершин',_count(vertices))]
    return [('Пространственный охват (вершины)',bounds),('Геометрия объектов',stats)]


def describe_layer(row):
    """Return ordered labelled sections without reading any original file."""
    kind,data = row['kind'],row['data']
    source_path = str(row.get('source_path') or '')
    # PureWindowsPath also recognizes suffixes in a project moved to Linux.
    suffix = (PureWindowsPath(source_path).suffix if '\\' in source_path
              else PurePath(source_path).suffix)
    common = [('Название',str(row['name'])),
              ('Тип слоя',KIND_LABELS[kind]),
              ('Формат исходного файла',suffix.upper().lstrip('.') or UNKNOWN),
              ('Полный путь к исходному файлу',source_path or NO_PATH),
              ('Система координат (текущая)',_crs_text(data.crs)),
              ('Исходная система координат',_crs_text(row.get('source_crs'))),
              ('Единицы координат',_axis_units(data)),
              ('Видимость','Включена' if row.get('visible',True) else 'Выключена')]
    if kind=='grid':
        specific = _grid_sections(data,row)
    elif kind=='raster':
        specific = _raster_sections(data)
    else:
        specific = _vector_sections(data)
    return [('Общие сведения',common),*specific]
