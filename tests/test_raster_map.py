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


def test_raster_preview_wheel_zoom_preserves_source_pixels(qt_application_lifetime):
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtTest import QTest
    from geogarmodel.raster_points_ui import RasterPreview

    points=[]
    preview=RasterPreview(np.full((800,1200,4),255,dtype=np.uint8),
                          lambda x,y:points.append((x,y)))
    preview.resize(600,400)
    preview.show()
    QTest.qWait(20)
    try:
        preview.fit_image()
        initial=preview.transform().m11()
        pos=preview.viewport().rect().center()+QPoint(45,25)
        def wheel(delta):
            event=QWheelEvent(QPointF(pos),QPointF(preview.viewport().mapToGlobal(pos)),
                              QPoint(),QPoint(0,delta),Qt.NoButton,Qt.NoModifier,
                              Qt.NoScrollPhase,False)
            preview.wheelEvent(event)
            assert event.isAccepted()
        anchor=preview.mapToScene(pos)
        for _ in range(10): wheel(120)
        assert preview.transform().m11()>initial
        assert preview._zoom>1
        assert not points
        # Scrollbar integer rounding is at most a few display pixels.
        after=preview.mapToScene(pos)
        assert abs(after.x()-anchor.x())<5/initial
        assert abs(after.y()-anchor.y())<5/initial
        source=QPointF(after.x()+.173,after.y()+.219)
        click=preview.mapFromScene(source)
        expected=preview.scene_position(click)
        QTest.mouseClick(preview.viewport(),Qt.LeftButton,pos=click)
        np.testing.assert_allclose(points,[[expected.x(),expected.y()]],rtol=0,atol=1e-10)
        zoom=preview._zoom
        preview.resize(640,420)
        QTest.qWait(10)
        assert preview._zoom==pytest.approx(zoom)
        for _ in range(30): wheel(-120)
        assert preview._zoom==1.
        assert preview.transform().m11()==pytest.approx(preview._fit_scale)
        assert len(points)==1
        wheel(120)
        preview.fit_image()
        assert preview._zoom==1.
    finally:
        preview.close()


def test_raster_preview_max_zoom_and_corner_picking(qt_application_lifetime):
    from PyQt5.QtCore import QPoint,QPointF,Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtTest import QTest

    dialog=RasterPoints(np.full((800,1200,4),255,dtype=np.uint8),CRS('EPSG:20918'))
    dialog.show()
    QTest.qWait(20)
    preview=dialog.preview
    try:
        preview.fit_image()
        pos=preview.viewport().rect().center()
        for _ in range(50):
            event=QWheelEvent(QPointF(pos),QPointF(preview.viewport().mapToGlobal(pos)),
                              QPoint(),QPoint(0,120),Qt.NoButton,Qt.NoModifier,
                              Qt.NoScrollPhase,False)
            preview.wheelEvent(event)
        assert preview.transform().m11()==pytest.approx(100.)
        assert preview.transform().m11()>=5*16.
        bounds=preview.picture.boundingRect()
        corners=[QPointF(.13,.27),QPointF(bounds.right()-.19,.31),
                 QPointF(.23,bounds.bottom()-.37),
                 QPointF(bounds.right()-.41,bounds.bottom()-.17)]
        for index,source in enumerate(corners):
            preview.centerOn(source)
            click=preview.mapFromScene(source)
            assert (click-preview.viewport().rect().center()).manhattanLength()<=4
            expected=preview.scene_position(click)
            assert bounds.contains(expected)
            QTest.mouseClick(preview.viewport(),Qt.LeftButton,pos=click)
            assert len(dialog.pixels)==index+1
            actual=dialog.pixels[-1]
            np.testing.assert_allclose(actual,[expected.x(),expected.y()],rtol=0,atol=1e-10)
            marker,label=dialog.markers[-1]
            np.testing.assert_allclose([marker.pos().x(),marker.pos().y()],actual,rtol=0,atol=1e-10)
            assert (preview.mapFromScene(marker.scenePos())-click).manhattanLength()<=1
            assert label.parentItem() is marker
            assert float(dialog.table.item(index,0).text())==pytest.approx(actual[0],abs=5.1e-7)
            assert float(dialog.table.item(index,1).text())==pytest.approx(actual[1],abs=5.1e-7)
        # Navigation margins are not part of the raster, and clicks there
        # must never be added as control points.
        preview.centerOn(QPointF(0,0))
        outside=preview.mapFromScene(QPointF(-1,-1))
        assert preview.viewport().rect().contains(outside)
        QTest.mouseClick(preview.viewport(),Qt.LeftButton,pos=outside)
        assert len(dialog.pixels)==4
        original=list(dialog.pixels)
        preview.fit_image()
        assert dialog.pixels==original
        dialog.table.setCurrentCell(0,2)
        dialog.remove_point()
        assert len(dialog.markers)==3 and len(dialog.pixels)==3
        assert dialog.markers[0][1].text()=='1'
    finally:
        dialog.close()
