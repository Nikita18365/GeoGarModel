import numpy as np
import pytest
from PyQt5.QtWidgets import QFileDialog,QMessageBox,QTableWidgetItem
from pyproj import CRS

from geogarmodel.app import MainWindow
from geogarmodel.raster_map import fit_affine, place_raster, read_raster
from geogarmodel.raster_points_ui import RasterPoints


def test_world_file_and_rotated_raster(tmp_path):
    from PIL import Image
    path = tmp_path/'map.png'
    rgba = np.zeros((5,6,4),dtype=np.uint8)
    rgba[:,:,:] = [70,90,110,255]
    Image.fromarray(rgba).save(path)
    path.with_suffix('.pgw').write_text('10\n0\n0\n-10\n105\n195\n')
    pixels,transform,_ = read_raster(path)
    np.testing.assert_array_equal(pixels,rgba)
    assert transform*(.5,.5)==(105,195)
    source='EPSG:20918'
    affine=fit_affine([[.5,.5],[5.5,.5],[.5,4.5]],[[105,195],[155,200],[109,155]])
    placed=place_raster(pixels,affine,source,source)
    assert placed.image[...,3].any()
    np.testing.assert_array_equal(placed.image[placed.image[...,3]>0][0],[70,90,110,255])
    with pytest.raises(ValueError,match='площадь'):
        fit_affine([[0,0],[1,1],[2,2]],[[0,0],[1,0],[1,1]])


@pytest.mark.parametrize('manual',[False,True])
def test_import_raster_project_and_both_map_views(tmp_path,monkeypatch,manual):
    import rasterio
    from rasterio.transform import from_origin
    from PIL import Image
    from PyQt5.QtCore import Qt
    from pyqtgraph import ImageItem

    path=tmp_path/('source.png' if manual else 'source.tif')
    image=np.full((6,8),98,dtype=np.uint8)
    if manual:
        Image.fromarray(image).save(path)
    else:
        with rasterio.open(path,'w',driver='GTiff',width=8,height=6,count=1,dtype='uint8',
                           crs='EPSG:20918',transform=from_origin(18500000,7800000,10,10)) as dst:
            dst.write(image,1)
    window=MainWindow()
    window.apply_project_crs('EPSG:20918')
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args:(str(path),''))
    def question(parent,title,*args):
        if title=='Привязка растра':
            return QMessageBox.No if manual else QMessageBox.Yes
        return QMessageBox.Discard
    monkeypatch.setattr(QMessageBox,'question',question)
    monkeypatch.setattr(window,'choose_source_crs',lambda known,name:CRS(known).to_string())
    if manual:
        def edit(dialog):
            for px,py,x,y in ((.5,.5,18500005,7799995),(7.5,.5,18500075,7799995),(.5,5.5,18500005,7799945)):
                dialog.add_point(px,py)
                row=dialog.table.rowCount()-1
                dialog.table.setItem(row,2,QTableWidgetItem(str(x)))
                dialog.table.setItem(row,3,QTableWidgetItem(str(y)))
            dialog.table.item(0,2).setText('18500004')
            dialog.table.item(0,2).setText('18500005')
            dialog.finish()
            return dialog.result()
        monkeypatch.setattr(RasterPoints,'exec_',edit)
    try:
        window.open_raster()
        row=window.layers.currentItem().data(Qt.UserRole)
        assert row['kind']=='raster'
        assert row['source_path']==str(path.resolve())
        assert CRS(row['source_crs'])==CRS('EPSG:20918')
        np.testing.assert_allclose(row['data'].bounds,[18500000,7799940,18500080,7800000],rtol=0,atol=1e-6)
        assert len(window.figure.axes[0].images)==1
        assert any(isinstance(item,ImageItem) for item in window.fast_map.getPlotItem().items)
        project=tmp_path/'test.ggproject'
        window.save_project_to(project)
        path.unlink()
        assert window.open_project_from(project)
        restored_row=window.layers.currentItem().data(Qt.UserRole)
        assert restored_row['source_path']==str(path.resolve())
        restored=restored_row['data']
        assert bool(restored.control_points)==manual
        np.testing.assert_array_equal(restored.image,row['data'].image)
    finally:
        window.close()
