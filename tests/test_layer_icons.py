"""Layer type decorations must not change the map stacking order."""
import numpy as np
from PyQt5.QtCore import QSize,Qt

from geogarmodel.app import MainWindow
from geogarmodel.grids import Grid
from geogarmodel.map_project import Vector
from geogarmodel.raster_map import RasterLayer
from geogarmodel.view_icons import view_icon


def test_layer_icons_and_mixed_project_roundtrip(tmp_path,qt_application_lifetime):
    crs='EPSG:3857'
    window=MainWindow()
    size=QSize(20,20)

    def decorations():
        rows=[]
        for index in range(window.layers.count()):
            item=window.layers.item(index)
            kind=item.data(Qt.UserRole)['kind']
            assert not item.icon().isNull()
            image=item.icon().pixmap(size).toImage()
            assert not image.isNull()
            expected=view_icon('layer_'+kind).pixmap(size).toImage()
            assert image==expected
            rows.append((kind,item.text(),item.checkState(),image))
        return rows

    try:
        window.add_grid(Grid(np.arange(8)*100.,np.arange(6)*100.,
                             np.arange(48.).reshape(6,8),'field.grd','nT',crs,'m'))
        window.add_map_layer(dict(kind='vector',name='faults.geojson',visible=True,style={},
                                  data=Vector([dict(type='LineString',coordinates=[[0,0],[700,500]])],crs)))
        window.add_map_layer(dict(kind='raster',name='background.tif',visible=True,style={},
                                  data=RasterLayer(np.full((5,7,4),255,dtype=np.uint8),(0,0,700,500),crs)))
        original=decorations()
        assert [row[0] for row in original]==['raster','vector','grid']
        assert all(a[3]!=b[3] for i,a in enumerate(original) for b in original[i+1:])
        window.move_map_layer(1)
        window.layers.currentItem().setCheckState(Qt.Unchecked)
        before=decorations()
        assert [row[0] for row in before]==['vector','raster','grid']
        assert before[1][2]==Qt.Unchecked
        path=tmp_path/'layers.ggproject'
        window.save_project_to(path)
        assert window.open_project_from(path)
        assert decorations()==before
    finally:
        window.close()
