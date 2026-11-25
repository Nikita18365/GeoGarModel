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
        assert callable(w.map_toolbar.zoom)
        np.testing.assert_allclose(w.figure.axes[0].get_xlim(),w._map_home[0])
        np.testing.assert_allclose(w.figure.axes[0].get_ylim(),w._map_home[1])
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


@pytest.mark.parametrize('family',['MS Shell Dlg 2','MS Shell Dlg','GeoGarModel Missing Font 123',''])
def test_unavailable_font_resolves_before_render(family,caplog,tmp_path):
    from geogarmodel.map_presentation import available_map_fonts,resolve_map_font,text_options
    f=Figure(); FigureCanvasAgg(f)
    ax=draw_map(f,grid())
    style={'title':dict(family=family,size=15,bold=True,italic=True,color='#123456'),
           'colorbar':dict(family=family,size=12)}
    expected=resolve_map_font(family)
    assert expected in available_map_fonts()
    assert text_options(style,'title')['fontfamily']==expected
    with caplog.at_level('WARNING',logger='matplotlib.font_manager'):
        apply_map_fonts(f,style)
        for _ in range(2): f.canvas.draw()
        save_map_image(f,tmp_path/'font.png',dpi=100)
    assert not any('not found' in r.getMessage() and r.name=='matplotlib.font_manager'
                   for r in caplog.records)
    assert ax.title.get_fontfamily()==[expected]
    assert ax.title.get_fontweight()=='bold'
    assert ax.title.get_fontstyle()=='italic'
    assert ax.title.get_fontsize()==15
    assert ax.title.get_color()=='#123456'


def test_font_editor_uses_renderable_families_and_migrates_legacy(qt_application_lifetime):
    from geogarmodel.map_presentation import available_map_fonts,resolve_map_font,migrate_map_fonts
    from geogarmodel.map_presentation_ui import MapFontEditor
    w=MainWindow()
    try:
        w.add_grid(grid())
        original={'title':dict(family='MS Shell Dlg 2',size=19,bold=True)}
        w.map_fonts=migrate_map_fonts(original)
        editor=w.map_font_editor
        assert isinstance(editor,MapFontEditor)
        editor.element.setCurrentIndex(editor.element.findData('title'))
        editor.load_element()
        assert editor.family.currentText()==resolve_map_font('MS Shell Dlg 2')
        assert all(editor.family.itemText(i) in available_map_fonts()
                   for i in range(editor.family.count()))
        assert editor.size.value()==19 and editor.bold.isChecked()
        editor.apply()
        assert w.map_fonts['title']['family'] in available_map_fonts()
        assert original['title']['family']=='MS Shell Dlg 2'
        assert w.figure.axes[0].title.get_fontfamily()==[w.map_fonts['title']['family']]
    finally: w.close()


def test_publication_cursor_zoom_and_native_toolbar(qt_application_lifetime):
    from types import SimpleNamespace
    w=MainWindow()
    try:
        w.add_grid(grid())
        w.map_views.setCurrentWidget(w.canvas)
        w.canvas.draw()
        toolbar=w.map_toolbar
        assert callable(toolbar.zoom)
        assert not hasattr(toolbar,'reset_zoom')
        assert hasattr(toolbar,'nominal_scale')
        assert hasattr(toolbar,'edit_layout') and hasattr(toolbar,'reset_layout')
        # The standard rectangle-zoom mode must still be a callable action.
        toolbar.zoom()
        assert toolbar.mode.name=='ZOOM'
        toolbar.edit_layout.setChecked(True)
        assert not toolbar.mode
        toolbar.edit_layout.setChecked(False)
        # Off-centre pointers in opposite quadrants must produce distinct
        # views, with the pointed data coordinate fixed relative to the frame.
        centers=[]
        for fx,fy in ((.25,.2),(.75,.8)):
            toolbar.home()
            ax=w.figure.axes[0]
            old_x=np.array(ax.get_xlim()); old_y=np.array(ax.get_ylim())
            px=old_x[0]+fx*np.ptp(old_x)
            py=old_y[0]+fy*np.ptp(old_y)
            event=SimpleNamespace(inaxes=ax,xdata=px,ydata=py,step=1)
            w.map_scroll(event)
            new_x=np.array(ax.get_xlim()); new_y=np.array(ax.get_ylim())
            np.testing.assert_allclose(new_x,px+(old_x-px)/1.2,rtol=0,atol=1e-7)
            np.testing.assert_allclose(new_y,py+(old_y-py)/1.2,rtol=0,atol=1e-7)
            assert (px-new_x[0])/np.ptp(new_x)==pytest.approx(fx)
            assert (py-new_y[0])/np.ptp(new_y)==pytest.approx(fy)
            centers.append((new_x.mean(),new_y.mean()))
            assert w._map_view==(tuple(new_x),tuple(new_y))
            event.step=-1
            w.map_scroll(event)
            np.testing.assert_allclose(ax.get_xlim(),old_x,rtol=0,atol=1e-7)
            np.testing.assert_allclose(ax.get_ylim(),old_y,rtol=0,atol=1e-7)
        assert centers[0][0]<centers[1][0] and centers[0][1]<centers[1][1]
        toolbar.home()
        ax=w.figure.axes[0]
        np.testing.assert_allclose(ax.get_xlim(),w._map_home[0])
        np.testing.assert_allclose(ax.get_ylim(),w._map_home[1])
        before=ax.get_xlim(),ax.get_ylim()
        w.map_scroll(SimpleNamespace(inaxes=None,xdata=None,ydata=None,step=1))
        assert (ax.get_xlim(),ax.get_ylim())==before
    finally:
        w.close()
