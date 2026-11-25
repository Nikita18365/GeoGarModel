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
COORDINATE_DECIMALS = 6


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


def _round_coordinates(table, columns):
    """Copy only identified coordinate columns; never round Z or features."""
    result = table.copy(deep=False)
    for name in columns:
        values = pd.to_numeric(table[name], errors='raise')
        if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError('В координатах есть бесконечные значения.')
        result[name] = values.round(COORDINATE_DECIMALS)
    return result


def _coordinate_text(value):
    if pd.isna(value):
        return ''
    text = f'{value:.6f}'.rstrip('0').rstrip('.')
    return '0' if text == '-0' else text


def write_table(table, path, fmt, coordinate_columns=None):
    if fmt not in TABLE_FORMATS:
        raise ValueError('Неизвестный табличный формат.')
    if coordinate_columns is None:
        coordinate_columns = XYZ_HEADERS[:2] if all(c in table.columns for c in XYZ_HEADERS[:2]) else ()
    coordinate_columns = tuple(coordinate_columns)
    table = _round_coordinates(table, coordinate_columns)
    if fmt == 'xlsx':
        if len(table) > 1048575 or len(table.columns) > 16384:
            raise ValueError('Таблица превышает лимит XLSX. Используйте CSV или TXT.')
        # File handle lets openpyxl write to our atomic .tmp path.
        with atomic_output(path) as tmp, tmp.open('wb') as out:
            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                table.to_excel(writer, index=False)
                sheet = writer.sheets['Sheet1']
                if len(table):
                    for name in coordinate_columns:
                        index = table.columns.get_loc(name)+1
                        for cells in sheet.iter_cols(min_col=index, max_col=index, min_row=2, max_row=len(table)+1):
                            for cell in cells:
                                cell.number_format = '0.######'
            report(len(table), len(table), 'Строки XLSX')
    else:
        with atomic_output(path) as tmp, tmp.open('w', encoding='utf-8-sig', newline='') as out:
            size = max(1, 100000 // max(1, len(table.columns)))
            for start in range(0, max(1, len(table)), size):
                end = min(start+size, len(table))
                block = table.iloc[start:end].copy(deep=False)
                for name in coordinate_columns:
                    block[name] = block[name].map(_coordinate_text)
                block.to_csv(out, index=False, header=start==0,
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


def rounded_export_grid(grid, fmt):
    """Round regular-lattice origin and step, not Z or individual intervals."""
    dx, dy = round(grid.dx, COORDINATE_DECIMALS), round(grid.dy, COORDINATE_DECIMALS)
    if fmt == 'flt':
        dy = dx
    if min(dx, dy) <= 0:
        raise ValueError('Шаг сетки слишком мал для точности координат 6 знаков после запятой.')
    x = np.round(round(float(grid.x[0]), COORDINATE_DECIMALS)+dx*np.arange(len(grid.x)),
                 COORDINATE_DECIMALS)
    y = np.round(round(float(grid.y[0]), COORDINATE_DECIMALS)+dy*np.arange(len(grid.y)),
                 COORDINATE_DECIMALS)
    return Grid(x, y, grid.values, grid.name, grid.unit, grid.crs, grid.coordinate_unit,
                grid.history+[dict(operation='export_coordinate_precision', decimals=COORDINATE_DECIMALS)])


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
                     round(grid.dx, 6), round(grid.dy, 6), float(grid.x[0]), float(grid.y[0]), 0., 0., 1.)
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
    # 25 characters fit round-trip float64; three fields stay below 80 characters.
    width, count = 25, 3
    with path.open('w', encoding='ascii', newline='\n') as out:
        out.write('! GeoGarModel ZMAP+ node coordinates\n@GRID, GRID, 3\n')
        out.write('25, 1.0E+30, , 17, 1\n')
        out.write(f'{ny}, {nx}, {grid.x[0]:.6f}, {grid.x[-1]:.6f}, {grid.y[0]:.6f}, {grid.y[-1]:.6f}\n')
        out.write('0.0, 0.0, 0.0\n@\n')
        for j in range(nx):
            column = grid.values[::-1, j]
            for start in range(0, ny, count):
                values = column[start:start+count]
                fields = [repr(float(value if np.isfinite(value) else null)).rjust(width)
                          for value in values]
                if any(len(field) != width for field in fields):
                    raise ValueError('Число Z не помещается в поле ZMAP+.')
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
    step = round(grid.dx, 6)
    # An odd number of micro-units requires a half-micro-unit corner.
    # Write the equivalent cell centre instead to keep six decimal places.
    if round(step*10**COORDINATE_DECIMALS) % 2:
        origin = f'XLLCENTER {grid.x[0]:.6f}\nYLLCENTER {grid.y[0]:.6f}\n'
    else:
        origin = f'XLLCORNER {grid.x[0]-step/2:.6f}\nYLLCORNER {grid.y[0]-step/2:.6f}\n'
    header.write_text(f'NCOLS {nx}\nNROWS {ny}\n'+origin+
                      f'CELLSIZE {step:.6f}\nNODATA_VALUE {float(nodata):.17g}\n'
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
    if fmt == 'flt' and not square_cells(grid):
        raise ValueError('FLT требует квадратные ячейки. Сначала выполните пересчёт.')
    grid = rounded_export_grid(grid, fmt)
    if fmt == 'npz':
        with atomic_output(path) as tmp, tmp.open('wb') as out:
            save_grid(grid, out)
        return
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
