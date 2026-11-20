import numpy as np
import pytest
from pyproj import Transformer
from geogarmodel.graticule import degree_step,geographic_grid,draw_geography


def settings(**kwargs):
    return dict(axes='geographic',mode='solid',notation='decimal',longitude_step=0,latitude_step=0,width_microns=100,color='#222222',**kwargs)


def test_dms_and_geographic_intersections():
    assert degree_step(.3015,'dms')==pytest.approx(.5+15/3600)
    with pytest.raises(ValueError): degree_step(.6060,'dms')
    data=geographic_grid('EPSG:3857',(100000,600000,1000000,1500000),settings())
    inverse=Transformer.from_crs('EPSG:3857','EPSG:4326',always_xy=True)
    assert data['xticks'] and data['yticks']
    for x,longitude in data['xticks']: assert inverse.transform(x,1000000)[0]==pytest.approx(longitude,abs=1e-8)
    for y,latitude in data['yticks']: assert inverse.transform(100000,y)[1]==pytest.approx(latitude,abs=1e-5)


@pytest.mark.parametrize('mode',['none','solid','dotted','crosses','edges'])
def test_grid_styles_preserve_projection_limits(mode):
    from matplotlib.figure import Figure
    ax=Figure().add_subplot(); ax.set_xlim(100000,600000); ax.set_ylim(1000000,1500000)
    config=settings(); config['mode']=mode
    draw_geography(ax,'EPSG:3857',config)
    np.testing.assert_allclose(ax.get_xlim(),[100000,600000]); np.testing.assert_allclose(ax.get_ylim(),[1000000,1500000])
    assert ax.get_xlabel()=='Долгота'
    assert all('°' in label.get_text() for label in ax.get_xticklabels())


def test_map_geography_project_roundtrip_and_sheet_bounds(tmp_path,monkeypatch,qt_application_lifetime):
    from geogarmodel.app import MainWindow
    from geogarmodel.grids import Grid
    from PyQt5.QtWidgets import QFileDialog
    window=MainWindow(); window.resize(1200,850); window.show(); qt_application_lifetime.processEvents()
    x=np.linspace(18500000,18600000,30); y=np.linspace(7800000,7900000,30)
    window.add_grid(Grid(x,y,np.add.outer(np.arange(30),np.arange(30)),'Карта','nT','EPSG:20918','m'))
    window.geo_widgets['axes'].setCurrentIndex(1); window.geo_widgets['mode'].setCurrentIndex(3)
    window.scale_on.setChecked(True); window.scale_crs.setChecked(True)
    window.map_views.setCurrentIndex(1)
    for _ in range(4): window.canvas.draw(); qt_application_lifetime.processEvents()
    assert window.north_symbol.count()==1 and not window.north_symbol.isVisible()
    assert window.figure.axes[0].get_xlabel()=='Долгота'
    path=tmp_path/'geographic.ggmaps'; monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(path),''))
    assert window.save_map_project()
    from geogarmodel.map_project import load_project
    manifest,_=load_project(path); assert manifest['presentation']['geography']['axes']=='geographic'
    renderer=window.canvas.get_renderer(); bbox=window.figure.get_tightbbox(renderer).transformed(window.figure.dpi_scale_trans)
    assert bbox.x0>=-2 and bbox.y0>=-2
    assert bbox.x1<=window.figure.bbox.x1+2 and bbox.y1<=window.figure.bbox.y1+2
    window.close()
