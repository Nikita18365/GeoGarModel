import json
import numpy as np
import pytest
from PyQt5.QtCore import Qt,QPoint
from PyQt5.QtWidgets import QFileDialog
from geogarmodel.app import MainWindow
from geogarmodel.grids import Grid
from geogarmodel.map_project import read_vector,load_project


def grid(name='Карта'):
    x=np.arange(8)*100.; y=np.arange(6)*100.
    return Grid(x,y,np.arange(48.).reshape(6,8),name,'nT','EPSG:20918','m')


def test_parameters_visibility_and_view(qt_application_lifetime):
    window=MainWindow()
    for operation,visible in [('dz',()),('trend',('degree',)),('upward',('height',)),('lowpass',('wave',))]:
        window.operation.setCurrentIndex(window.operation.findData(operation))
        for key in ('height','wave','degree'): assert getattr(window,key).isHidden()==(key not in visible)
    window.add_grid(grid()); ax=window.figure.axes[0]; ax.set_xlim(100,400); ax.set_ylim(100,300)
    window.contours.setChecked(True); window.contour_step.setValue(5)
    window.contour_color.setText('red'); window.redraw(); window.palette.setCurrentText('terrain')
    ax=window.figure.axes[0]
    np.testing.assert_allclose(ax.get_xlim(),[100,400]); np.testing.assert_allclose(ax.get_ylim(),[100,300])
    assert window.contours.isChecked() and len(ax.collections)>0
    assert ax.get_xlabel()=='X, м'
    window.scale_on.setChecked(True); window.scale_step.setValue(20); window.scale_count.setValue(3)
    assert len(window.figure.axes[0].patches)==3
    window.figure.axes[0].set_xlim(200,500)
    assert len(window.figure.axes[0].patches)==3
    window.close()


def test_project_layers_styles_and_roundtrip(tmp_path,monkeypatch,qt_application_lifetime):
    window=MainWindow(); window.map_project_name='Магнитное поле'
    window.add_grid(grid('Нижняя')); lower=window.layers.currentItem()
    window.palette.setCurrentText('terrain'); window.contours.setChecked(True)
    window.add_grid(grid('Верхняя')); upper=window.layers.currentItem()
    window.palette.setCurrentText('coolwarm'); window.map_alpha.setValue(.5)
    assert len(window.figure.axes[0].images)==2
    window.layers.setCurrentItem(lower)
    assert window.palette.currentText()=='terrain' and window.contours.isChecked()
    window.layers.setCurrentItem(upper); assert window.palette.currentText()=='coolwarm'
    from types import SimpleNamespace
    position=window.layers.visualItemRect(lower).center()
    window.layers.dropEvent(SimpleNamespace(pos=lambda:position,accept=lambda:None))
    assert window.layers.item(0) is lower and window.grids[0].name=='Нижняя'
    position=window.layers.visualItemRect(lower).center()
    window.layers.dropEvent(SimpleNamespace(pos=lambda:position,accept=lambda:None))
    assert window.layers.item(0) is upper
    upper.setText('Новое имя'); lower.setCheckState(Qt.Unchecked)
    assert window.grids[0].name=='Новое имя' and len(window.figure.axes[0].images)==1
    path=tmp_path/'магнитное.ggmaps'
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(path),''))
    assert window.save_map_project()
    manifest,rows=load_project(path)
    assert manifest['name']=='Магнитное поле' and rows[0]['name']=='Новое имя'
    assert rows[1]['visible'] is False and rows[0]['style']['alpha']==.5
    np.testing.assert_array_equal(rows[0]['data'].values,grid().values)
    window.reset_map_project('Другой'); assert window.layers.count()==0
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a:(str(path),''))
    window.open_map_project(); assert window.layers.count()==2
    assert window.map_project_name=='Магнитное поле'
    window.layers.setCurrentRow(0); window.delete_map_layer()
    assert len(window.grids)==1
    window.close()


def test_geojson_and_shp_without_geopandas(tmp_path):
    from matplotlib.figure import Figure
    from pyproj import CRS
    import shapefile
    data=dict(type='FeatureCollection',features=[dict(type='Feature',properties={},geometry=dict(type='LineString',coordinates=[[0,0],[1,1]]))])
    path=tmp_path/'контур.geojson'; path.write_text(json.dumps(data))
    vector=read_vector(path); ax=Figure().add_subplot(); vector.draw(ax,'EPSG:3857')
    assert ax.lines[0].get_xdata()[-1]>100000
    shp=tmp_path/'contour.shp'
    with shapefile.Writer(str(shp)) as writer:
        writer.field('name','C'); writer.line([[[0,0],[1,1]]]); writer.record('test')
    shp.with_suffix('.prj').write_text(CRS.from_epsg(4326).to_wkt())
    imported=read_vector(shp); assert len(imported.features)==1
    imported.draw(ax,'EPSG:3857'); assert len(ax.lines)==2
