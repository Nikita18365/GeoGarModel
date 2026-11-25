import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from geogarmodel.grids import Grid
from geogarmodel.plotting import draw_map
from geogarmodel.map_presentation import centered_limits,apply_map_fonts,save_map_image,FONT_ELEMENTS
from geogarmodel.app import MainWindow


def grid():
    x=np.arange(40)*100.+18500000; y=np.arange(30)*100.+7800000
    return Grid(x,y,np.add.outer(np.arange(30),np.arange(40)), 'field','nT','EPSG:20918','m')


def test_centered_limits():
    assert centered_limits(((0,10),(0,20)),.5)==((2.5,7.5),(5.,15.))


def test_contour_frequency_and_colorbar():
    f=Figure(); FigureCanvasAgg(f)
    ax=draw_map(f,grid(),contours=True,contour_step=5,contour_label_every=2,colorbar_orientation='horizontal')
    assert f.axes[1]._colorbar.orientation=='horizontal'
    labels=[t for t in ax.texts if t.get_gid()=='map-contour-label']
    assert labels
    assert all(float(t.get_text())%10==0 for t in labels)
    styles={'contours':dict(size=15,color='#aa0000',bold=True,italic=True),
            'colorbar':dict(size=13,color='#0000aa')}
    apply_map_fonts(f,styles)
    assert all(t.get_fontsize()==15 and t.get_fontweight()=='bold' for t in labels)
    assert f.axes[1].xaxis.label.get_fontsize()==13
    assert ax.title.get_fontsize()!=15


def test_tight_export(tmp_path):
    from PIL import Image
    f=Figure(figsize=(8,6)); FigureCanvasAgg(f)
    ax=f.add_axes([.3,.3,.4,.4]); ax.plot([0,1],[0,1]); ax.set_title('Map')
    path=tmp_path/'map.png'
    save_map_image(f,path,dpi=100)
    with Image.open(path) as image:
        assert image.width<800 and image.height<600
    assert tuple(f.get_size_inches())==(8,6)


def test_presentation_project_roundtrip(tmp_path,qt_application_lifetime):
    w=MainWindow()
    try:
        w.add_grid(grid()); w.canvas.draw()
        home=w._map_home
        w.set_map_zoom(200)
        ax=w.figure.axes[0]
        assert np.ptp(ax.get_xlim())==pytest.approx(np.ptp(home[0])/2)
        assert np.mean(ax.get_xlim())==pytest.approx(np.mean(home[0]))
        w.colorbar_orientation.setCurrentIndex(w.colorbar_orientation.findData('horizontal'))
        assert w.figure.axes[1]._colorbar.orientation=='horizontal'
        w.contour_label_every.setValue(3)
        w.map_font_editor.element.setCurrentIndex(w.map_font_editor.element.findData('title'))
        w.map_font_editor.size.setValue(18); w.map_font_editor.bold.setChecked(True)
        w.map_font_editor.apply()
        assert w.figure.axes[0].title.get_fontsize()==18
        assert w.figure.axes[0].title.get_fontweight()=='bold'
        assert len(FONT_ELEMENTS)==8
        p=tmp_path/'map.ggproject'; w.save_project_to(p)
        assert w.open_project_from(p)
        assert w.colorbar_orientation.currentData()=='horizontal'
        assert w.contour_label_every.value()==3
        assert w.figure.axes[0].title.get_fontsize()==18
        w.map_toolbar.home()
        assert w.map_toolbar.zoom.value()==pytest.approx(100)
    finally: w.close()
