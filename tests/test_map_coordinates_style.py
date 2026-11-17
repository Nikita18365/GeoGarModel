import numpy as np
import pytest
from pyproj import CRS
from geogarmodel.crs_editor import with_false_easting,false_easting,CRSEditor
from geogarmodel.map_project import Vector,reproject_grid,save_project,load_project
from geogarmodel.grids import Grid


def test_zonal_prefix_conversion_and_project_roundtrip(tmp_path):
    source='EPSG:20917'; target=with_false_easting(source,500000)
    assert CRS(target).geodetic_crs.equals(CRS(source).geodetic_crs)
    assert false_easting(target)==500000
    vector=Vector([dict(type='Point',coordinates=[17100000,7800000])],source).reproject(target)
    np.testing.assert_allclose(vector.features[0]['coordinates'],[100000,7800000])
    grid=Grid(np.arange(3)*100.+17100000,np.arange(3)*100.+7800000,
              np.arange(9.).reshape(3,3),'Field','mGal',source,'m')
    result=reproject_grid(grid,target)
    np.testing.assert_allclose(result.x,grid.x-17000000,atol=1e-7)
    np.testing.assert_allclose(result.y,grid.y,atol=1e-7)
    np.testing.assert_allclose(result.values,grid.values,atol=1e-7)
    style=dict(line_color='#ff0000',line_width=3,fill=True,fill_color='#00ff00',fill_alpha=.4)
    path=tmp_path/'custom.ggmaps'
    save_project(path,'Test',[dict(kind='vector',name='SHP',visible=True,style=style,data=vector)],None,{},target)
    manifest,rows=load_project(path)
    assert CRS(manifest['crs']).equals(CRS(target)) and rows[0]['style']==style
    with pytest.raises(ValueError,match='метрах'): with_false_easting('EPSG:4326',500000)


def test_shp_with_prj_still_requests_source_crs(tmp_path,monkeypatch,qt_application_lifetime):
    import shapefile
    from PyQt5.QtWidgets import QFileDialog
    from geogarmodel.app import MainWindow
    path=tmp_path/'contour.shp'
    with shapefile.Writer(str(path)) as writer:
        writer.field('name','C'); writer.line([[[17100000,7800000],[17100100,7800100]]]); writer.record('test')
    path.with_suffix('.prj').write_text(CRS('EPSG:20917').to_wkt())
    window=MainWindow(); window.apply_project_crs(with_false_easting('EPSG:20917',500000))
    calls=[]
    def choose(known,name):
        calls.append(known); return 'EPSG:20917'
    monkeypatch.setattr(window,'choose_source_crs',choose)
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a:(str(path),''))
    window.open_vector()
    assert len(calls)==1 and CRS(calls[0]).equals(CRS('EPSG:20917'))
    np.testing.assert_allclose(window.layers.item(0).data(256)['data'].features[0]['coordinates'][0],[100000,7800000])
    editor=CRSEditor('EPSG:20917'); editor.fields[3].setValue(500000)
    assert false_easting(editor.value())==500000
    window.close()


ESRI_WKT='''PROJCS["GSK 9N",GEOGCS["GSK-2011",DATUM["Geodezicheskaya_Sistema_Koordinat_2011",SPHEROID["GSK-2011",6378136.5,298.2564151]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",63.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]'''


def test_arcgis_import_and_all_parameters(tmp_path,monkeypatch,qt_application_lifetime):
    from PyQt5.QtWidgets import QFileDialog
    from pyproj import Transformer
    from geogarmodel.crs_editor import read_prj,projection_parameters
    path=tmp_path/'GSK 11N.prj'; path.write_text(ESRI_WKT,encoding='utf-8-sig')
    original=CRS(read_prj(path))
    assert original.name=='GSK 9N'
    editor=CRSEditor('EPSG:4326'); assert not editor.fields
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a:(str(path),''))
    editor.import_button.click()
    params=projection_parameters(editor.value())
    assert len(editor.fields)==len(params)==5
    assert {p['name']:p['value'] for p in params}=={
        'Latitude of natural origin':0,'Longitude of natural origin':63,
        'Scale factor at natural origin':1,'False easting':500000,'False northing':0}
    values={'8801':5,'8802':69,'8805':.9996,'8806':400000,'8807':10000}
    for i,p in enumerate(params): editor.fields[i].setValue(values[str(p['id']['code'])])
    changed=CRS(editor.value())
    assert changed.geodetic_crs.equals(original.geodetic_crs)
    np.testing.assert_allclose(Transformer.from_crs(changed.geodetic_crs,changed,always_xy=True).transform(69,5),[400000,10000],atol=1e-7)
    # A list selection rebuilds the fields for that method (Lambert has 6, not 5).
    editor.crs.setCurrentText('EPSG:2154')
    assert len(editor.fields)==6
    assert CRS(editor.value()).equals(CRS('EPSG:2154'))
    editor.crs.setCurrentText('EPSG:4326'); assert not editor.fields
    assert CRS(editor.value()).is_geographic


def test_polygon_fill_preserves_holes_and_line_style():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure=Figure(figsize=(3,3),dpi=100); canvas=FigureCanvasAgg(figure); ax=figure.add_subplot()
    # Both rings intentionally have the same winding in the source file.
    polygon=dict(type='Polygon',coordinates=[[[0,0],[4,0],[4,4],[0,4],[0,0]],
                                             [[1,1],[3,1],[3,3],[1,3],[1,1]]])
    vector=Vector([polygon],'EPSG:3857')
    vector.draw(ax,'EPSG:3857',dict(line_color='blue',line_width=3,fill=True,fill_color='red',fill_alpha=1))
    ax.set(xlim=(-1,5),ylim=(-1,5)); canvas.draw()
    pixels=np.asarray(canvas.buffer_rgba())
    def pixel(x,y):
        px,py=ax.transData.transform((x,y)); return pixels[pixels.shape[0]-1-round(py),round(px),:3]
    np.testing.assert_array_equal(pixel(2,2),[255,255,255])
    np.testing.assert_array_equal(pixel(.5,.5),[255,0,0])
    assert all(line.get_linewidth()==3 and line.get_color()=='blue' for line in ax.lines)
