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
            'colorbar':dict(size=13,color='#0000aa'),
            'colorbar_label':dict(size=16,color='#aa0000',italic=True)}
    apply_map_fonts(f,styles)
    assert all(t.get_fontsize()==15 and t.get_fontweight()=='bold' for t in labels)
    assert f.axes[1].xaxis.label.get_fontsize()==16
    assert all(t.get_fontsize()==13 for t in f.axes[1].get_xticklabels())
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
        assert len(FONT_ELEMENTS)==9
        p=tmp_path/'map.ggproject'; w.save_project_to(p)
        assert w.open_project_from(p)
        assert w.colorbar_orientation.currentData()=='horizontal'
        assert w.contour_label_every.value()==3
        assert w.figure.axes[0].title.get_fontsize()==18
        w.map_toolbar.home()
        assert w.map_toolbar.zoom.value()==pytest.approx(100)
    finally: w.close()


def test_legacy_fonts_and_caption():
    from geogarmodel.map_layout import clean_crs_name
    from geogarmodel.map_presentation import migrate_map_fonts
    assert clean_crs_name('GSK-2011 (пользовательские параметры)')=='GSK-2011'
    style={'colorbar':dict(size=17,bold=True)}
    migrated=migrate_map_fonts(style)
    assert migrated['colorbar_label']==style['colorbar']
    assert migrated['colorbar_label'] is not migrated['colorbar']


def test_publication_colorbar_and_scale_geometry(tmp_path,qt_application_lifetime):
    from geogarmodel.map_layout import artist_bbox
    from matplotlib.backend_bases import MouseEvent
    w=MainWindow()
    try:
        w.figure.set_size_inches(10,7)
        w.add_grid(grid())
        w.scale_on.setChecked(True)
        w.scale_crs.setChecked(True)
        w.colorbar_orientation.setCurrentIndex(w.colorbar_orientation.findData('horizontal'))
        for _ in range(5): w.canvas.draw()
        ax,bar=w.figure.axes
        renderer=w.canvas.get_renderer()
        scale=ax._map_scale_artists
        unit,crs=scale[-2:]
        middle=(scale[0].get_x()+scale[w.scale_count.value()-1].get_x()+scale[w.scale_count.value()-1].get_width())/2
        assert unit.get_ha()==crs.get_ha()=='center'
        assert unit.get_position()[0]==pytest.approx(middle)
        assert crs.get_position()[0]==pytest.approx(middle)
        assert not artist_bbox(scale,renderer).overlaps(bar.get_tightbbox(renderer))
        assert bar.bbox.x1<=ax.bbox.x1+2
        original_view=(ax.get_xlim(),ax.get_ylim())
        for _ in range(3):
            w.colorbar_orientation.setCurrentIndex(w.colorbar_orientation.findData('vertical'))
            for _ in range(3): w.canvas.draw()
            ax,bar=w.figure.axes
            assert bar.bbox.x0-ax.bbox.x1==pytest.approx(w.figure.dpi*10/72,abs=1)
            np.testing.assert_allclose([bar.bbox.y0,bar.bbox.y1],[ax.bbox.y0,ax.bbox.y1],atol=1)
            w.colorbar_orientation.setCurrentIndex(w.colorbar_orientation.findData('horizontal'))
        w.canvas.draw()
        ax,bar=w.figure.axes
        np.testing.assert_allclose(ax.get_xlim(),original_view[0])
        np.testing.assert_allclose(ax.get_ylim(),original_view[1])
        w.map_toolbar.edit_layout.setChecked(True)
        drag=w.map_toolbar.decoration_drag
        title=ax.title.get_window_extent(w.canvas.get_renderer())
        x,y=title.x0+title.width/2,title.y0+title.height/2
        drag.press(MouseEvent('button_press_event',w.canvas,x,y,button=1))
        assert drag.active and drag.active[0]=='title'
        drag.move(MouseEvent('motion_notify_event',w.canvas,x+10,y-5,button=1))
        drag.release(MouseEvent('button_release_event',w.canvas,x+10,y-5,button=1))
        assert 'title' in w.map_offsets
        offsets=dict(w.map_offsets)
        w.canvas.draw()
        path=tmp_path/'layout.ggproject'
        w.save_project_to(path)
        assert w.open_project_from(path)
        assert w.map_offsets==offsets
        w.map_toolbar.reset_positions()
        assert w.map_offsets=={}
        save_map_image(w.figure,tmp_path/'map.png',dpi=120)
        assert (tmp_path/'map.png').stat().st_size>0
    finally: w.close()
