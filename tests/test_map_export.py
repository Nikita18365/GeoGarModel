import struct
import numpy as np
import pandas as pd
import pytest
from pyproj import CRS
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QDialog, QComboBox, QMessageBox

from geogarmodel.app import MainWindow
from geogarmodel.grids import Grid, load_grid
from geogarmodel.map_export import (FORMATS, grid_for_export, grid_table, square_grid,
                                   table_for_export, write_grid, write_table)
from geogarmodel.map_project import Vector, read_xyz_table
from geogarmodel.map_export_ui import FORMAT_COLORS, format_icon


def sample(dx=10, dy=10):
    return Grid(1000+dx*np.arange(3), 2000+dy*np.arange(2),
                [[1.25,-9999,np.nan],[4.5,-2.75,6.]], 'field', 'nT','EPSG:3857','m')


@pytest.mark.parametrize('fmt',['npz','grd','zmap','flt'])
def test_grid_formats_roundtrip(tmp_path,fmt):
    grid=sample()
    path=tmp_path/('field.'+fmt)
    write_grid(grid,path,fmt)
    loaded=load_grid(path)
    np.testing.assert_allclose(loaded.x,grid.x)
    np.testing.assert_allclose(loaded.y,grid.y)
    np.testing.assert_allclose(loaded.values,grid.values,equal_nan=True)
    if fmt=='npz':
        assert loaded.crs==grid.crs and loaded.unit==grid.unit
    else:
        assert CRS.from_wkt(path.with_suffix('.prj').read_text(encoding='utf-8'))==CRS(grid.crs)
    if fmt=='grd':
        assert struct.unpack_from('<5i',path.read_bytes())==(8,2,3,2,1)
        assert path.stat().st_size==512+6*8
        np.testing.assert_array_equal(np.fromfile(path,dtype='<f8',offset=512)[:2],[1.25,-9999])
    if fmt=='flt':
        hdr=path.with_suffix('.hdr').read_text()
        assert 'XLLCORNER 995' in hdr and 'YLLCORNER 1995' in hdr and 'BYTEORDER LSBFIRST' in hdr
        np.testing.assert_array_equal(np.fromfile(path,dtype='<f4')[:3],[4.5,-2.75,6.])


def test_flt_rectangular_requires_explicit_resample(tmp_path):
    grid=sample(10,20)
    path=tmp_path/'field.flt'
    with pytest.raises(ValueError,match='квадратные'):
        write_grid(grid,path,'flt')
    assert not path.exists()
    grid=Grid(grid.x,grid.y,[[1,2,3],[5,6,7]],crs=grid.crs)
    square=square_grid(grid)
    assert square.dx==square.dy==10
    np.testing.assert_allclose(square.values[1],[3,4,5])
    write_grid(square,path,'flt')
    np.testing.assert_allclose(load_grid(path).values,square.values)


def source_row():
    grid=sample()
    table=grid_table(grid).rename(columns={'East':'X1','North':'Y1','Z':'Signal'})
    table['X2']=table.X1+100
    table['Other']=np.arange(len(table))*2.
    table['Label']=['a','b','c','d','e','f']
    return dict(kind='grid',name=grid.name,visible=True,style={},data=grid,
                xyz=dict(table=table,stem='source',columns=['X1','Y1','Signal'],crs=grid.crs))


@pytest.mark.parametrize('fmt',['xlsx','csv','txt'])
@pytest.mark.parametrize('all_columns',[True,False])
def test_original_table_exports(tmp_path,fmt,all_columns):
    row=source_row()
    columns=None if all_columns else ['X2','Y1','Other']
    table=table_for_export(row,columns)
    path=tmp_path/('table.'+fmt)
    write_table(table,path,fmt)
    result=read_xyz_table(path)
    expected=row['xyz']['table'] if all_columns else table
    pd.testing.assert_frame_equal(result,expected,check_dtype=False)
    if not all_columns:
        assert list(result.columns)==['East','North','Z']
        np.testing.assert_array_equal(result.East,row['xyz']['table'].X2)


def test_xyz_grid_selection():
    row=source_row()
    assert grid_for_export(row,['X1','Y1','Signal']) is row['data']
    alternate=grid_for_export(row,['X2','Y1','Other'])
    np.testing.assert_array_equal(alternate.x,row['data'].x+100)
    np.testing.assert_array_equal(alternate.values,[[0,2,4],[6,8,10]])
    with pytest.raises(ValueError,match='три разных'):
        grid_for_export(row,['X1','X1','Other'])
    with pytest.raises(ValueError,match='числа'):
        table_for_export(row,['X1','Y1','Label'])


def test_export_controls_use_selected_item(tmp_path,monkeypatch,qt_application_lifetime):
    window=MainWindow()
    try:
        assert set(window.map_export_buttons)==set(FORMATS)
        for fmt, button in window.map_export_buttons.items():
            assert not button.icon().isNull()
            assert button.iconSize()==QSize(48,22)
            assert button.accessibleName()==FORMATS[fmt]
        assert set(FORMAT_COLORS)==set(FORMATS)
        icons=[format_icon(fmt).pixmap(48,22).toImage() for fmt in FORMATS]
        assert all(a!=b for i,a in enumerate(icons) for b in icons[i+1:])
        assert not any(button.isEnabled() for button in window.map_export_buttons.values())
        row=source_row()
        window.add_grid(row['data'],xyz=row['xyz'])
        assert all(button.isEnabled() for button in window.map_export_buttons.values())
        window.layers.currentItem().setCheckState(Qt.Unchecked)
        assert all(button.isEnabled() for button in window.map_export_buttons.values())
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(tmp_path/'full.csv'),''))
        monkeypatch.setattr(QInputDialog,'getItem',lambda *a:('Вся исходная таблица',True))
        window.export_map('csv')
        pd.testing.assert_frame_equal(read_xyz_table(tmp_path/'full.csv'),row['xyz']['table'],check_dtype=False)

        def choose(dialog):
            for box in dialog.findChildren(QComboBox):
                box.setCurrentText(dict(X='X2',Y='Y1',Z='Other')[box.accessibleName()])
            return QDialog.Accepted
        monkeypatch.setattr(QDialog,'exec_',choose)
        monkeypatch.setattr(QInputDialog,'getItem',lambda *a:('Только East, North, Z',True))
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(tmp_path/'xyz.txt'),''))
        window.export_map('txt')
        assert list(read_xyz_table(tmp_path/'xyz.txt').columns)==['East','North','Z']
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(tmp_path/'alternate.grd'),''))
        window.export_map('grd')
        np.testing.assert_array_equal(load_grid(tmp_path/'alternate.grd').values,[[0,2,4],[6,8,10]])

        monkeypatch.setattr(QDialog,'exec_',lambda *a:QDialog.Rejected)
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:pytest.fail('Cancelled export opened a file dialog'))
        window.export_map('grd')
        window.add_map_layer(dict(kind='vector',name='line',visible=True,style={},
                                  data=Vector([dict(type='LineString',coordinates=[[1000,2000],[1020,2010]])],row['data'].crs)))
        assert not any(button.isEnabled() for button in window.map_export_buttons.values())
        window.layers.setCurrentRow(1)
        assert all(button.isEnabled() for button in window.map_export_buttons.values())
    finally:
        window.close()


@pytest.mark.parametrize('fmt',['xlsx','csv','txt'])
def test_table_coordinate_precision(tmp_path,fmt):
    table=pd.DataFrame(dict(East=[1234.123456789,-23.987654321,np.nan],
                            North=[-99.876543219,4.1,0.],
                            Z=[1.23456789012345,-0.000000000123456789,np.nan],
                            Other=[123.123456789123,2.,3.]))
    original=table.copy(deep=True)
    path=tmp_path/('precise.'+fmt)
    write_table(table,path,fmt)
    result=read_xyz_table(path)
    np.testing.assert_allclose(result.East,[1234.123457,-23.987654,np.nan],rtol=0,atol=1e-12,equal_nan=True)
    np.testing.assert_allclose(result.North,[-99.876543,4.1,0.],rtol=0,atol=1e-12)
    np.testing.assert_allclose(result.Z,table.Z,rtol=1e-14,atol=0,equal_nan=True)
    np.testing.assert_allclose(result.Other,table.Other,rtol=1e-14,atol=0)
    pd.testing.assert_frame_equal(table,original)
    if fmt=='xlsx':
        from openpyxl import load_workbook
        sheet=load_workbook(path).active
        assert sheet['A2'].number_format=='0.######'
        assert sheet['C2'].number_format=='General'
    else:
        raw=path.read_text(encoding='utf-8-sig')
        assert '1234.123457' in raw and '1234.123456789' not in raw


def test_full_table_rounds_only_active_coordinate_pair(tmp_path):
    table=pd.DataFrame(dict(X1=[1.123456789],Y1=[2.987654321],
                            X2=[3.123456789],Signal=[4.123456789],Label=['test']))
    path=tmp_path/'full.csv'
    write_table(table,path,'csv',coordinate_columns=('X1','Y1'))
    result=read_xyz_table(path)
    assert list(result.columns)==list(table.columns)
    assert result.X1[0]==1.123457 and result.Y1[0]==2.987654
    assert result.X2[0]==table.X2[0] and result.Signal[0]==table.Signal[0]


@pytest.mark.parametrize('fmt',['npz','grd','zmap','flt'])
def test_grid_coordinate_precision_and_z(tmp_path,fmt):
    step=10.123456789
    x=1000.123456789+step*np.arange(3)
    y=2000.987654321+step*np.arange(2)
    values=np.array([[1.2345678901234567,-0.000000000123456789,np.nan],
                     [4.567890123456789,-2.1234567891234567,6.]])
    grid=Grid(x,y,values,crs='EPSG:3857')
    path=tmp_path/('precise.'+fmt)
    write_grid(grid,path,fmt)
    loaded=load_grid(path)
    np.testing.assert_allclose(loaded.x,1000.123457+10.123457*np.arange(3),rtol=0,atol=1e-10)
    np.testing.assert_allclose(loaded.y,2000.987654+10.123457*np.arange(2),rtol=0,atol=1e-10)
    expected=values.astype(np.float32).astype(float) if fmt=='flt' else values
    np.testing.assert_array_equal(loaded.values,expected)
    np.testing.assert_array_equal(grid.x,x)
    np.testing.assert_array_equal(grid.y,y)
    np.testing.assert_array_equal(grid.values,values)
    if fmt=='flt':
        hdr=path.with_suffix('.hdr').read_text()
        assert 'XLLCENTER 1000.123457' in hdr and 'CELLSIZE 10.123457' in hdr
    if fmt=='zmap':
        raw=path.read_text()
        assert '1000.123457' in raw and '1.2345678901234567' in raw


def test_export_rejects_step_below_coordinate_precision(tmp_path):
    grid=Grid([0,1e-7],[0,1e-7],[[1.,2.],[3.,4.]])
    path=tmp_path/'fine.npz'
    with pytest.raises(ValueError,match='Шаг сетки слишком мал'):
        write_grid(grid,path,'npz')
    assert not path.exists()
