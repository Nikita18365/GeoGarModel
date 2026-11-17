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
    editor=CRSEditor('EPSG:20917'); editor.override.setChecked(True); editor.east.setValue(500000)
    assert false_easting(editor.value())==500000
    window.close()


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
