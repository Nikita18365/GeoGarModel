import json
import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication,QDialog,QFileDialog,QMenu
from geogarmodel.app import MainWindow
from geogarmodel.grids import Grid,save_grid
from geogarmodel.layer_properties import describe_layer,NO_PATH
from geogarmodel.layer_properties_ui import LayerPropertiesDialog
from geogarmodel.map_project import Vector,load_project
from geogarmodel.raster_map import RasterLayer


def properties(row):
    return dict(pair for _,section in describe_layer(row) for pair in section)


def grid():
    return Grid([100,110,120],[200,220],[[1,2,np.nan],[4,5,8]],'field','nT','EPSG:3857','m')


def test_grid_statistics_exclude_nan_and_keep_source():
    value=grid()
    original=value.values.copy()
    row=dict(kind='grid',name='renamed',data=value,visible=False,
             source_path=r'D:\survey\field.grd',source_crs='EPSG:4326')
    p=properties(row)
    assert p['Полный путь к исходному файлу']==row['source_path']
    assert p['Формат исходного файла']=='GRD'
    assert p['Название']=='renamed'
    assert [p[k] for k in ('xmin','xmax','ymin','ymax','x_space','y_space')]==['100','120','200','220','10','20']
    assert p['Количество узлов (всего)']=='6'
    assert p['Количество значений (конечных)']=='5'
    assert p['Пропуски (NaN)']=='1'
    assert float(p['Среднее значение'])==4
    assert float(p['Медианное значение'])==4
    assert float(p['Стандартное отклонение (σ)'])==pytest.approx(np.std([1,2,4,5,8]))
    assert 'EPSG:3857' in p['Система координат (текущая)']
    assert 'EPSG:4326' in p['Исходная система координат']
    np.testing.assert_array_equal(value.values,original)


def test_raster_properties():
    image=np.zeros((2,3,4),dtype=np.uint8)
    image[...,3]=[[0,255,255],[128,0,255]]
    p=properties(dict(kind='raster',name='image',data=RasterLayer(image,(100,200,130,240),'EPSG:3857')))
    assert p['Полный путь к исходному файлу']==NO_PATH
    assert p['Ширина, пикселей']=='3' and p['Высота, пикселей']=='2'
    assert p['Размер пикселя X']=='10' and p['Размер пикселя Y']=='20'
    assert p['Видимые пиксели (alpha > 0)']=='4'
    assert p['Полностью прозрачные (alpha = 0)']=='2'
    assert p['Частично прозрачные']=='1'
    assert p['Покрытие']=='66.67 %'


def test_vector_nested_geometry_statistics():
    vector=Vector([None,dict(type='Point',coordinates=[1,2,3]),
        dict(type='MultiLineString',coordinates=[[[0,0],[3,4]],[[5,6],[7,8]]]),
        dict(type='GeometryCollection',geometries=[
            dict(type='Polygon',coordinates=[[[0,0],[10,0],[10,10],[0,0]]]),
            dict(type='MultiPoint',coordinates=[[11,12],[13,14]])])],'EPSG:3857')
    p=properties(dict(kind='vector',name='vector',data=vector))
    assert p['Количество объектов в проекте']=='4'
    assert p['Объектов без координат']=='1'
    assert p['Точечных частей']=='3'
    assert p['Линейных частей']=='2'
    assert p['Полигональных частей']=='1'
    assert p['Количество записей вершин']=='11'
    assert p['xmax']=='13' and p['ymax']=='14'
    empty=properties(dict(kind='vector',name='empty',data=Vector([None],'EPSG:3857')))
    assert empty['xmin']=='Не указано' and empty['Количество записей вершин']=='0'


def test_import_path_context_menu_and_project_roundtrip(tmp_path,monkeypatch,qt_application_lifetime):
    window=MainWindow()
    path=tmp_path/'source.npz'
    vector_path=tmp_path/'line.geojson'
    save_grid(grid(),path)
    vector_path.write_text(json.dumps(dict(type='LineString',coordinates=[[0,0],[.001,.001]])),encoding='utf-8')
    try:
        assert window.apply_project_crs('EPSG:3857')
        monkeypatch.setattr(QFileDialog,'getOpenFileNames',lambda *a:([str(path)],''))
        window.open_grid()
        item=window.layers.currentItem()
        assert item.data(Qt.UserRole)['source_path']==str(path.resolve())
        item.setText('renamed')
        captured=[]
        def show(dialog):
            captured.append(dialog)
            dialog.copy_all()
            assert str(path.resolve()) in QApplication.clipboard().text()
            return QDialog.Rejected
        monkeypatch.setattr(LayerPropertiesDialog,'exec_',show)
        def context(menu,*args):
            action=next(a for a in menu.actions() if a.text()=='Свойства…')
            action.trigger()
        monkeypatch.setattr(QMenu,'exec_',context)
        window.map_context(window.layers.visualItemRect(item).center())
        assert len(captured)==1
        assert captured[0].windowTitle()=='Свойства слоя — renamed'
        monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a:(str(vector_path),''))
        window.open_vector()
        assert window.layers.currentItem().data(Qt.UserRole)['source_path']==str(vector_path.resolve())
        window.add_grid(grid().derived(grid().values.copy(),'SUM_calculated'))
        assert properties(window.layers.currentItem().data(Qt.UserRole))['Полный путь к исходному файлу']==NO_PATH
        target=tmp_path/'saved.ggproject'
        window.save_project_to(target)
        path.unlink()
        vector_path.unlink()
        assert window.open_project_from(target)
        rows=window.map_layer_rows()
        assert not rows[0].get('source_path')
        assert rows[1]['source_path']==str(vector_path.resolve())
        assert rows[1]['source_crs']=='EPSG:4326'
        assert rows[2]['source_path']==str(path.resolve())
        assert properties(rows[2])['Количество значений (конечных)']=='5'
    finally:
        window.close()
