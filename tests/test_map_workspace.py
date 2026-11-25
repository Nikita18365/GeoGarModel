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


def test_missing_pyshp_reports_the_running_python(monkeypatch,tmp_path):
    import geogarmodel.map_project as project
    path=tmp_path/'contour.shp'; path.touch()
    missing=ModuleNotFoundError("No module named 'shapefile'",name='shapefile')
    monkeypatch.setattr(project.importlib,'import_module',lambda name: (_ for _ in ()).throw(missing))
    with pytest.raises(ValueError) as error:
        read_vector(path,'EPSG:4326')
    message=str(error.value)
    assert project.sys.executable in message
    assert '-m pip install "pyshp>=2.3,<3"' in message


def test_home_wheel_heading_scale_and_name_editor(qt_application_lifetime):
    from types import SimpleNamespace
    from PyQt5.QtWidgets import QStyleOptionViewItem,QLineEdit
    window=MainWindow(); window.add_grid(grid('ГФО_Grav'))
    window.show(); qt_application_lifetime.processEvents()
    assert 'ГФО_Grav' in window.map_style_heading.text()
    item=window.layers.currentItem(); window.layers.editItem(item)
    qt_application_lifetime.processEvents()
    editor=next(e for e in window.layers.findChildren(QLineEdit) if e.isVisible())
    assert editor.height()>=28 and editor.text()=='ГФО_Grav'
    editor.setText('Новое имя карты'); window.layers.itemDelegate().commitData.emit(editor)
    window.layers.itemDelegate().closeEditor.emit(editor)
    assert 'Новое имя карты' in window.map_style_heading.text()
    ax=window.figure.axes[0]; full=(ax.get_xlim(),ax.get_ylim())
    window.map_scroll(SimpleNamespace(inaxes=ax,xdata=200.,ydata=150.,step=1))
    assert np.ptp(ax.get_xlim())<np.ptp(full[0])
    # Publication view zoom is centered rather than anchored at the cursor.
    assert np.mean(ax.get_xlim())==pytest.approx(np.mean(full[0]))
    assert np.mean(ax.get_ylim())==pytest.approx(np.mean(full[1]))
    window.map_toolbar.home(); ax=window.figure.axes[0]
    np.testing.assert_allclose(ax.get_xlim(),full[0]); np.testing.assert_allclose(ax.get_ylim(),full[1])
    window.contours.setChecked(True); assert window.contour_color.isHidden()
    assert window.contour_color_button.isHidden() is False
    window.scale_on.setChecked(True); ax=window.figure.axes[0]; window.canvas.draw()
    assert all(p.get_window_extent().y1<ax.xaxis.label.get_window_extent().y0 and not p.get_clip_on() for p in ax.patches)
    first=min(t for t in ax.get_xticks() if min(ax.get_xlim())<=t<=max(ax.get_xlim()))
    assert ax.patches[0].get_x()==pytest.approx(first)
    window.close()


def test_geographic_north_uses_crs():
    from geogarmodel.map_navigation import north_direction
    from pyproj import Transformer
    project=Transformer.from_crs(4326,20918,always_xy=True)
    central=north_direction('EPSG:20918',*project.transform(105,70))
    west=north_direction('EPSG:20918',*project.transform(103,70))
    east=north_direction('EPSG:20918',*project.transform(107,70))
    np.testing.assert_allclose(central,[0,1],atol=1e-5)
    assert west[0]>0 and east[0]<0 and west[1]>0 and east[1]>0


def test_drag_lifecycle_never_removes_layers(monkeypatch,qt_application_lifetime):
    from types import SimpleNamespace
    import geogarmodel.maps_ui as ui
    window=MainWindow()
    for name in ('A','B','C'): window.add_grid(grid(name))
    original={id(window.layers.item(i)):window.layers.item(i).data(Qt.UserRole)['data'] for i in range(3)}
    class Drag:
        def __init__(self,parent): self.parent=parent
        def setMimeData(self,mime): assert mime.hasFormat('application/x-geogarmodel-layer')
        def exec_(self,action):
            assert action==Qt.MoveAction
            target=2 if self.parent.currentRow()==0 else 0
            pos=self.parent.visualItemRect(self.parent.item(target)).center()
            self.parent.dropEvent(SimpleNamespace(source=lambda:self.parent,pos=lambda:pos,accept=lambda:None))
            return Qt.MoveAction
    monkeypatch.setattr(ui,'QDrag',Drag)
    for _ in range(6):
        window.layers.startDrag(Qt.MoveAction)
        assert window.layers.count()==3
        assert {id(window.layers.item(i)):window.layers.item(i).data(Qt.UserRole)['data'] for i in range(3)}==original
    window.close()


def test_project_crs_reprojects_raster_vector_and_persists(tmp_path,monkeypatch,qt_application_lifetime):
    from geogarmodel.map_project import reproject_grid,Vector
    from pyproj import Transformer
    x=np.linspace(104.9,105.1,21); y=np.linspace(69.9,70.1,21); xx,yy=np.meshgrid(x,y)
    source=Grid(x,y,xx+2*yy,'WGS84 field','nT','EPSG:4326','degrees')
    target=reproject_grid(source,'EPSG:20918')
    assert target.crs=='EPSG:20918' and target.coordinate_unit=='m'
    tx,ty=np.meshgrid(target.x,target.y); lon,lat=Transformer.from_crs(20918,4326,always_xy=True).transform(tx,ty)
    valid=np.isfinite(target.values)
    np.testing.assert_allclose(target.values[valid],(lon+2*lat)[valid],atol=1e-8)
    assert np.isnan(target.values).any()  # Outside the transformed footprint is not filled.
    window=MainWindow(); assert window.apply_project_crs('EPSG:20918')
    window.add_grid(source)
    assert window.selected_grid().crs=='EPSG:20918'
    vector=Vector([dict(type='Point',coordinates=[105,70])],'EPSG:4326').reproject(window.map_project_crs)
    assert vector.crs=='EPSG:20918' and vector.features[0]['coordinates'][0]>18000000
    window.add_map_layer(dict(kind='vector',name='Точки',visible=True,style={},data=vector))
    window.scale_on.setChecked(True); window.scale_step.setValue(1000); window.scale_unit.setCurrentText('км')
    assert window.scale_step.value()==1
    window.scale_crs.setChecked(True); window.north_symbol.setCurrentIndex(0)
    path=tmp_path/'crs.ggmaps'; monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(path),''))
    assert window.save_map_project()
    manifest,rows=load_project(path)
    assert manifest['crs']=='EPSG:20918' and manifest['scale']['unit']=='км'
    assert manifest['scale']['north']=='arrow' and manifest['scale']['show_crs']
    window.reset_map_project(); monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a:(str(path),''))
    window.open_map_project()
    assert window.scale_unit.currentText()=='км' and window.scale_step.value()==1
    assert window.map_project_crs=='EPSG:20918' and window.north_symbol.currentData()=='arrow'
    window.close()


def test_cancel_crs_change_is_atomic(monkeypatch,qt_application_lifetime):
    known=grid('Известная СК'); known.x+=18500000; known.y+=7800000
    window=MainWindow(); window.add_grid(known)
    original=window.grids[0]
    unknown=grid('Неизвестная СК'); unknown.crs=''
    window.add_map_layer(dict(kind='grid',name=unknown.name,visible=False,style=window.read_map_style(),data=unknown))
    window.move_map_layer(1)  # Transform a known layer before encountering the cancelled source-CRS prompt.
    monkeypatch.setattr(window,'source_crs',lambda known,name:known or None)
    assert not window.apply_project_crs('EPSG:20919')
    assert window.map_project_crs=='EPSG:20918'
    assert window.layers.item(0).data(Qt.UserRole)['data'] is original
    assert window.layers.item(1).data(Qt.UserRole)['data'] is unknown
    window.close()


def test_xyz_import_uses_project_crs_independently_of_ml(tmp_path,monkeypatch,qt_application_lifetime):
    from PyQt5.QtWidgets import QComboBox,QDialog
    path=tmp_path/'survey.xyz'
    path.write_text('Signal East North Other East2\n1 18500000 7800000 4 18502000\n2 18501000 7800000 5 18503000\n3 18500000 7801000 6 18502000\n4 18501000 7801000 7 18503000\n')
    window=MainWindow(); window.apply_project_crs('EPSG:20918')
    asked=[]
    monkeypatch.setattr(window,'source_crs',lambda known,name:known or asked.append(name) or 'EPSG:4326')
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a:(str(path),''))
    def select(dialog):
        assert dialog.windowTitle()=='Столбцы таблицы XYZ'
        for box in dialog.findChildren(QComboBox):
            box.setCurrentText(dict(X='East',Y='North',Z='Signal')[box.accessibleName()])
        return QDialog.Accepted
    monkeypatch.setattr(QDialog,'exec_',select)
    window.open_xyz()
    assert asked==[] and window.selected_grid().crs=='EPSG:20918'
    assert window.layers.currentItem().text()=='TXT_survey · Signal'
    assert window.layers.currentItem().data(Qt.UserRole)['source_path']==str(path.resolve())
    assert not window.xyz_controls.isHidden()
    window.xyz_columns[0].setCurrentText('East2')
    window.xyz_columns[2].setCurrentText('Other')
    window.build_xyz_layer()
    assert window.layers.count()==2 and window.selected_grid().x.min()==18502000
    assert window.layers.currentItem().data(Qt.UserRole)['source_path']==str(path.resolve())
    np.testing.assert_allclose(window.selected_grid().values[0,0],4)
    project=tmp_path/'source.ggproject'
    window.save_project_to(project)
    path.unlink()
    assert window.open_project_from(project)
    assert window.layers.count()==2
    window.layers.setCurrentRow(1)
    assert [box.currentText() for box in window.xyz_columns]==['East','North','Signal']
    assert window.layers.currentItem().data(Qt.UserRole)['source_path']==str(path.resolve())
    window.xyz_columns[2].setCurrentText('Other')
    window.build_xyz_layer()
    assert window.layers.count()==3 and window.selected_grid().x.min()==18500000
    window.close()


@pytest.mark.parametrize(('order','dtype','origin'),[
    ('LSBFIRST','<f4','XLLCORNER 18500000\nYLLCORNER 7800000'),
    ('MSBFIRST','>f4','XLLCENTER 18500005\nYLLCENTER 7800005'),
])
def test_open_arcgis_float_map(tmp_path,monkeypatch,order,dtype,origin):
    from pyproj import CRS
    path=tmp_path/'поле.flt'
    np.array([[7,-9999,5],[2,3,4]],dtype=dtype).tofile(path)
    path.with_suffix('.hdr').write_text(
        f'NCOLS 3\nNROWS 2\n{origin}\nCELLSIZE 10\nNODATA_VALUE -9999\nBYTEORDER {order}\n',
        encoding='ascii')
    path.with_suffix('.prj').write_text(CRS('EPSG:20918').to_wkt('WKT1_ESRI'),encoding='utf-8')
    window=MainWindow()
    window.apply_project_crs('EPSG:20918')
    monkeypatch.setattr(QFileDialog,'getOpenFileNames',lambda *args:([str(path)],''))
    try:
        window.open_grid()
        raster=window.selected_grid()
        assert window.layers.currentItem().data(Qt.UserRole)['source_path']==str(path.resolve())
        assert CRS(raster.crs)==CRS('EPSG:20918')
        np.testing.assert_array_equal(raster.x,[18500005,18500015,18500025])
        np.testing.assert_array_equal(raster.y,[7800005,7800015])
        np.testing.assert_array_equal(raster.values,[[2,3,4],[7,np.nan,5]])
    finally:
        window.close()


def test_xyz_without_header_keeps_first_point(tmp_path):
    from geogarmodel.map_project import read_xyz_table
    path=tmp_path/'source.txt'
    path.write_text('# raw XYZ\n10 20 4 8\n11 21 5 9\n',encoding='utf-8')
    table=read_xyz_table(path)
    assert list(table.columns)==['Столбец 1','Столбец 2','Столбец 3','Столбец 4']
    np.testing.assert_array_equal(table.values,[[10,20,4,8],[11,21,5,9]])


def test_shared_title_layout_persistence_and_export(tmp_path,monkeypatch,qt_application_lifetime):
    from PyQt5.QtCore import QSettings
    import geogarmodel.maps_ui as ui
    real_settings=QSettings
    monkeypatch.setattr(ui,'QSettings',lambda *a:real_settings(str(tmp_path/'layout.ini'),real_settings.IniFormat))
    window=MainWindow(); window.add_grid(grid('A')); first=window.layers.currentItem()
    window.map_title.setText('Общий заголовок'); window.redraw()
    window.add_grid(grid('B')); second=window.layers.currentItem()
    assert window.figure.axes[0].get_title()=='Общий заголовок'
    window.layers.setCurrentItem(first); assert window.figure.axes[0].get_title()=='Общий заголовок'
    ax=window.figure.axes[0]; ax.set_xlim(100,400); ax.set_ylim(100,300)
    dialog=window.map_toolbar.configure_subplots()
    dialog._spinboxes['left'].setValue(.2); dialog._spinboxes['bottom'].setValue(.25)
    dialog.close(); window.layers.setCurrentItem(second); window.palette.setCurrentText('terrain')
    np.testing.assert_allclose(window.figure.axes[0].get_xlim(),[100,400])
    np.testing.assert_allclose(window.figure.axes[0].get_ylim(),[100,300])
    assert window.figure.subplotpars.left==pytest.approx(.2)
    assert window.figure.subplotpars.bottom==pytest.approx(.25)
    window.figure.savefig(tmp_path/'export.png')
    np.testing.assert_allclose(window.figure.axes[0].get_xlim(),[100,400])
    path=tmp_path/'title.ggmaps'; monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(path),''))
    window.save_map_project(); manifest,rows=load_project(path)
    assert manifest['presentation']['title']=='Общий заголовок'
    assert all('title' not in r['style'] for r in rows)
    other=MainWindow(); other.add_grid(grid()); assert other.auto_map_layout.isChecked()
    other.auto_map_layout.setChecked(False); assert other.figure.subplotpars.left==pytest.approx(.2)
    window.close(); other.close()


def test_sun_mouse_direction_and_independent_layers(qt_application_lifetime):
    from PyQt5.QtTest import QTest
    window=MainWindow(); window.add_grid(grid('A')); first=window.layers.currentItem()
    window.shade.setChecked(True); window.sun_azimuth.setValue(90); window.sun_intensity.setValue(75)
    original=window.selected_grid().values.copy()
    window.add_grid(grid('B')); second=window.layers.currentItem()
    window.sun_azimuth.setValue(180); window.sun_intensity.setValue(20)
    window.layers.setCurrentItem(first)
    assert window.sun_azimuth.value()==90 and window.sun_intensity.value()==75
    window.sun_picker.resize(200,180)
    center,radius=window.sun_picker.geometry_values()
    QTest.mouseClick(window.sun_picker,Qt.LeftButton,pos=QPoint(int(center.x()),int(center.y()+radius/2)))
    assert window.sun_azimuth.value()==pytest.approx(180)
    assert window.sun_altitude.value()==pytest.approx(45,abs=2)
    assert first.data(Qt.UserRole)['style']['sun_intensity']==.75
    assert second.data(Qt.UserRole)['style']['sun_intensity']==.2
    window.sun_intensity_slider.setValue(60)
    assert window.sun_intensity.value()==60 and first.data(Qt.UserRole)['style']['sun_intensity']==.6
    np.testing.assert_array_equal(window.selected_grid().values,original)
    window.close()


def test_hillshade_illumination_and_intensity():
    from matplotlib.figure import Figure
    from geogarmodel.plotting import draw_map
    model=grid(); model.values=np.broadcast_to(model.y[:,None],model.values.shape).copy()
    figure=Figure(); ax=draw_map(figure,model,hillshade=True,sun_azimuth=0,sun_altitude=45,sun_relief=1,sun_intensity=.7)
    north=np.asarray(ax.images[1].get_array()).mean()
    ax=draw_map(figure,model,hillshade=True,sun_azimuth=180,sun_altitude=45,sun_relief=1,sun_intensity=.2)
    south=np.asarray(ax.images[1].get_array()).mean()
    assert south>north  # A surface rising northward faces south (origin='lower').
    assert ax.images[1].get_alpha()==pytest.approx(.2)


def test_default_geophysical_palette_and_view_sections(tmp_path,qt_application_lifetime):
    from geogarmodel.palette_library import DEFAULT_MAP_PALETTE,bundled_palettes
    window=MainWindow()
    try:
        assert DEFAULT_MAP_PALETTE in bundled_palettes()
        assert window.palette.currentText()==DEFAULT_MAP_PALETTE
        assert window.palette.itemText(0)==DEFAULT_MAP_PALETTE
        assert all(window.palette.findText(name)>=0 for name in window.palette_library)
        sections=window.map_view_sections
        assert list(sections)==['colors','contours','relief','axes','scale','layout','fonts']
        assert sections['colors'].toggle.isChecked()
        assert all(not s.toggle.isChecked() for k,s in sections.items() if k!='colors')
        assert sections['colors'].content.isAncestorOf(window.palette)
        assert sections['contours'].content.isAncestorOf(window.contour_step)
        assert sections['relief'].content.isAncestorOf(window.sun_intensity)
        assert sections['axes'].content.isAncestorOf(window.geo_widgets['axes'])
        assert sections['scale'].content.isAncestorOf(window.scale_step)
        assert sections['layout'].content.isAncestorOf(window.map_title)
        window.add_grid(grid('first'))
        first=window.layers.currentItem()
        assert first.data(Qt.UserRole)['style']['cmap']==DEFAULT_MAP_PALETTE
        assert first.data(Qt.UserRole)['style']['palette_colors']==window._default_palette_entry['colors']
        window.palette.setCurrentText('terrain')
        window.map_reverse.setChecked(True)
        window.contours.setChecked(True)
        window.contour_step.setValue(5)
        section=sections['contours']
        section.toggle.click()
        assert not section.content.isHidden()
        assert not window.contour_step.isHidden()
        section.toggle.click()
        assert section.content.isHidden()
        assert window.contours.isChecked() and window.contour_step.value()==5
        section.toggle.click()
        assert not window.contour_color_button.isHidden()
        window.add_grid(grid('second'))
        assert window.palette.currentText()==DEFAULT_MAP_PALETTE
        assert not window.map_reverse.isChecked()
        assert first.data(Qt.UserRole)['style']['cmap']=='terrain'
        window.layers.setCurrentItem(first)
        assert window.palette.currentText()=='terrain' and window.map_reverse.isChecked()
        path=tmp_path/'styles.ggproject'
        window.save_project_to(path)
        assert window.open_project_from(path)
        rows=window.map_layer_rows()
        assert rows[0]['style']['cmap']==DEFAULT_MAP_PALETTE
        assert rows[1]['style']['cmap']=='terrain' and rows[1]['style']['reverse']
    finally:
        window.close()
