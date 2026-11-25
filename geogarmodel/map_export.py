"""Export scalar grids and original XYZ tables without modifying project layers."""
from contextlib import ExitStack
from pathlib import Path
import struct

import numpy as np
import pandas as pd

from .grids import Grid, points_to_grid, save_grid
from .model_archive import atomic_output
from .work_progress import report


FORMATS = {
    'npz': 'Graticule NPZ (*.npz)',
    'xlsx': 'Xlsx file (East, North, Z) (*.xlsx)',
    'csv': 'CSV file (East, North, Z) (*.csv)',
    'txt': 'TXT file (East, North, Z) (*.txt)',
    'grd': 'GRD Geosoft Binary Grid file (*.grd)',
    'zmap': 'Zmap Petrel file format (*.zmap)',
    'flt': 'Flt ESRI ArcGIS (*.flt)',
}
TABLE_FORMATS = frozenset(('xlsx', 'csv', 'txt'))
XYZ_HEADERS = ['East', 'North', 'Z']


def xyz_columns(table, columns):
    if len(columns) != 3 or len(set(columns)) != 3 or any(key not in table.columns for key in columns):
        raise ValueError('Выберите три разных столбца для East (X), North (Y) и Z.')
    result = table.loc[:, list(columns)].copy()
    try:
        result = result.apply(pd.to_numeric, errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError('Столбцы X, Y и Z должны содержать числа.') from exc
    if np.isinf(result.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError('В выбранных столбцах есть бесконечные значения.')
    result.columns = XYZ_HEADERS
    return result


def grid_table(grid):
    x, y = np.meshgrid(grid.x, grid.y)
    return pd.DataFrame(dict(East=x.ravel(), North=y.ravel(), Z=grid.values.ravel()))


def table_for_export(row, columns=None):
    source = row.get('xyz')
    if source is None:
        return grid_table(row['data'])
    if columns is None:
        return source['table']
    return xyz_columns(source['table'], columns)


def grid_for_export(row, columns=None):
    grid = row['data']
    source = row.get('xyz')
    if source is None:
        return grid
    if columns is None:
        columns = source['columns']
    # The current triple is the map already visible in the workspace. Validate
    # the selection, then retain its actual geometry and transformations.
    table = xyz_columns(source['table'], columns)
    if list(columns) == list(source['columns']):
        return grid
    from pyproj import CRS
    from .map_project import reproject_grid

    crs = CRS.from_user_input(source['crs'])
    unit = 'degrees' if crs.is_geographic else 'm'
    metadata = dict(name=str(columns[2]), crs=crs.to_string(), coordinate_unit=unit)
    points = table.dropna(subset=['East', 'North'])
    x, y = np.unique(points.East), np.unique(points.North)
    def regular(axis):
        return len(axis) >= 2 and np.allclose(np.diff(axis), axis[1]-axis[0], rtol=1e-5, atol=1e-9)

    # Exact rectangular data: no artificial resampling, including missing Z.
    if (regular(x) and regular(y) and len(x)*len(y) == len(points)
            and not points.duplicated(['East', 'North']).any()):
        values = points.pivot(index='North', columns='East', values='Z').reindex(index=y, columns=x)
        selected = Grid(x, y, values.to_numpy(dtype=float), **metadata)
    else:
        selected = points_to_grid(table.East, table.North, table.Z,
                                  nx=len(grid.x), ny=len(grid.y), **metadata)
    return reproject_grid(selected, grid.crs) if grid.crs else selected


def write_table(table, path, fmt):
    if fmt not in TABLE_FORMATS:
        raise ValueError('Неизвестный табличный формат.')
    if fmt == 'xlsx':
        if len(table) > 1048575 or len(table.columns) > 16384:
            raise ValueError('Таблица превышает лимит XLSX. Используйте CSV или TXT.')
        # File handle lets openpyxl write to our atomic .tmp path.
        with atomic_output(path) as tmp, tmp.open('wb') as out:
            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                table.to_excel(writer, index=False)
            report(len(table), len(table), 'Строки XLSX')
    else:
        with atomic_output(path) as tmp, tmp.open('w', encoding='utf-8-sig', newline='') as out:
            size = max(1, 100000 // max(1, len(table.columns)))
            for start in range(0, max(1, len(table)), size):
                end = min(start+size, len(table))
                table.iloc[start:end].to_csv(out, index=False, header=start==0,
                                             sep=',' if fmt == 'csv' else '\t')
                report(end, len(table), 'Строки '+fmt.upper())


def square_cells(grid):
    return np.isclose(grid.dx, grid.dy, rtol=1e-8, atol=0)


def square_grid(grid):
    """Explicitly resample a rectangular grid; never extrapolate outside it."""
    if square_cells(grid):
        return grid
    from scipy.interpolate import RegularGridInterpolator

    step = min(grid.dx, grid.dy)
    nx = int(np.ceil((grid.x[-1]-grid.x[0])/step - 1e-9))+1
    ny = int(np.ceil((grid.y[-1]-grid.y[0])/step - 1e-9))+1
    if nx*ny > 10000000:
        raise ValueError('Квадратная сетка FLT превышает 10 млн узлов. Выберите GRD, ZMAP или NPZ.')
    x, y = grid.x[0]+step*np.arange(nx), grid.y[0]+step*np.arange(ny)
    sample = RegularGridInterpolator((grid.y, grid.x), grid.values,
                                      bounds_error=False, fill_value=np.nan)
    values = np.empty((ny, nx))
    for start in range(0, ny, 128):
        xx, yy = np.meshgrid(x, y[start:start+128])
        # Snap round-off at a shared boundary, but do not extrapolate.
        xx = np.where(np.abs(xx-grid.x[-1]) <= step*1e-8, grid.x[-1], xx)
        yy = np.where(np.abs(yy-grid.y[-1]) <= step*1e-8, grid.y[-1], yy)
        values[start:start+128] = sample(np.stack((yy, xx), axis=-1))
        report(min(start+128, ny), ny, 'Квадратная сетка FLT')
    return Grid(x, y, values, grid.name, grid.unit, grid.crs, grid.coordinate_unit,
                grid.history+[dict(operation='export_flt_resample', step=float(step))])


def _write_grd(grid, path):
    """Geosoft v2, 512-byte header, uncompressed IEEE little-endian double.

    Storage KX=1: west to east, south to north. The optional numeric
    header fields use the format's dummy markers (except valid point count).
    """
    finite = np.isfinite(grid.values)
    if np.any(grid.values[finite] <= -1e32) or np.any(grid.values[finite] > 1e30):
        raise ValueError('Значения выходят за допустимый диапазон Geosoft GRD.')
    header = bytearray(512)
    struct.pack_into('<5i7d', header, 0, 8, 2, len(grid.x), len(grid.y), 1,
                     grid.dx, grid.dy, grid.x[0], grid.y[0], 0., 0., 1.)
    header[76:140] = b' '*64
    struct.pack_into('<5i4fdi', header, 140, *([-2147483647]*4),
                     int(finite.sum()), *([-1e32]*4), -1e32, -2147483647)
    with path.open('wb') as out:
        out.write(header)
        for start in range(0, len(grid.y), 128):
            block = grid.values[start:start+128]
            np.where(np.isfinite(block), block, -1e32).astype('<f8').tofile(out)
            report(min(start+128, len(grid.y)), len(grid.y), 'Строки GRD')


def _write_zmap(grid, path):
    """ZMAP+ fixed width, columns west→east and nodes north→south."""
    null = 1e30
    if np.any(np.abs(grid.values[np.isfinite(grid.values)]) >= null):
        raise ValueError('Значения выходят за допустимый диапазон ZMAP+.')
    nx, ny = len(grid.x), len(grid.y)
    width, count = 20, 5
    with path.open('w', encoding='ascii', newline='\n') as out:
        out.write('! GeoGarModel ZMAP+ node coordinates\n@GRID, GRID, 5\n')
        out.write('20, 1.000000000000E+30, , 12, 1\n')
        out.write(f'{ny}, {nx}, {grid.x[0]:.17g}, {grid.x[-1]:.17g}, {grid.y[0]:.17g}, {grid.y[-1]:.17g}\n')
        out.write('0.0, 0.0, 0.0\n@\n')
        for j in range(nx):
            column = grid.values[::-1, j]
            for start in range(0, ny, count):
                values = column[start:start+count]
                fields = [f'{value if np.isfinite(value) else null:{width}.12E}' for value in values]
                # Do not turn a valid value into the declared null through rounding.
                if any(np.isfinite(v) and float(s) == null for v, s in zip(values, fields)):
                    raise ValueError('Значение Z совпадает с кодом пропуска ZMAP после округления.')
                out.write(''.join(fields)+'\n')
            report(j+1, nx, 'Столбцы ZMAP')


def _write_flt(grid, path, header):
    if not square_cells(grid):
        raise ValueError('FLT требует одинаковый шаг X и Y.')
    nodata = np.float32(-np.finfo(np.float32).max)
    if np.any(np.abs(grid.values[np.isfinite(grid.values)]) >= float(-nodata)):
        raise ValueError('Значения выходят за диапазон float32 формата FLT.')
    nx, ny = len(grid.x), len(grid.y)
    with path.open('wb') as out:
        for start in range(0, ny, 128):
            block = grid.values[::-1][start:start+128]
            data = np.where(np.isfinite(block), block, nodata).astype('<f4')
            if np.any(np.isfinite(block) & (data == nodata)):
                raise ValueError('Значение Z совпадает с кодом пропуска FLT.')
            data.tofile(out)
            report(min(start+128, ny), ny, 'Строки FLT')
    header.write_text(f'NCOLS {nx}\nNROWS {ny}\n'
                      f'XLLCORNER {grid.x[0]-grid.dx/2:.17g}\n'
                      f'YLLCORNER {grid.y[0]-grid.dx/2:.17g}\n'
                      f'CELLSIZE {grid.dx:.17g}\nNODATA_VALUE {float(nodata):.17g}\n'
                      'BYTEORDER LSBFIRST\n', encoding='ascii')


def companion_paths(path, fmt):
    path = Path(path)
    return ([path.with_suffix('.hdr')] if fmt == 'flt' else []) + (
        [path.with_suffix('.prj')] if fmt in ('grd', 'zmap', 'flt') else [])


def write_grid(grid, path, fmt):
    """Stage the data and its companions before replacing individual files."""
    path = Path(path)
    if fmt not in FORMATS or fmt in TABLE_FORMATS:
        raise ValueError('Неизвестный формат сетки.')
    if fmt == 'npz':
        with atomic_output(path) as tmp, tmp.open('wb') as out:
            save_grid(grid, out)
        return
    if fmt == 'flt' and not square_cells(grid):
        raise ValueError('FLT требует квадратные ячейки. Сначала выполните пересчёт.')
    prj = path.with_suffix('.prj')
    if not grid.crs and prj.exists():
        raise ValueError('У карты нет СК, но рядом существует PRJ. Выберите другое имя файла.')
    with ExitStack() as stack:
        out = stack.enter_context(atomic_output(path))
        if grid.crs:
            from pyproj import CRS
            wkt = CRS.from_user_input(grid.crs).to_wkt('WKT1_ESRI' if fmt == 'flt' else 'WKT2_2019')
            projection = stack.enter_context(atomic_output(prj))
            projection.write_text(wkt, encoding='utf-8')
        if fmt == 'grd':
            _write_grd(grid, out)
        elif fmt == 'zmap':
            _write_zmap(grid, out)
        elif fmt == 'flt':
            header = stack.enter_context(atomic_output(path.with_suffix('.hdr')))
            _write_flt(grid, out, header)
