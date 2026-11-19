import numpy as np
import pytest
from geogarmodel.inversion import Mesh,InversionResult
from geogarmodel.model_slices import select_depths,horizontal_slice,property_unit


def result(mode='gravity'):
    mesh=Mesh(np.arange(3)*100.+18000000,np.arange(3)*100.+7800000,np.array([0.,100.,300.,600.]))
    values=np.arange(mesh.size,dtype=float)
    if mode=='vector': values=np.column_stack((values,np.ones(mesh.size)*2,np.zeros(mesh.size)))
    return InversionResult(mesh,values,np.empty((0,3)),np.empty(0),np.empty(0),[],dict(mode=mode),'test')


def test_depth_selection_and_validation():
    r=result()
    np.testing.assert_allclose(select_depths(r.mesh,text='100; 250,5; 500'),[100,250.5,500])
    np.testing.assert_allclose(select_depths(r.mesh,'layers',top=40,bottom=500,every=2),[50,450])
    np.testing.assert_allclose(select_depths(r.mesh,'step',top=0,bottom=600,step=200),[0,200,400,600])
    for kwargs in [dict(text=''),dict(text='nan'),dict(text='-1'),dict(mode='step',step=0),dict(mode='step',step=.001),dict(mode='layers',every=0),dict(mode='step',top=500,bottom=100)]:
        with pytest.raises(ValueError): select_depths(r.mesh,**kwargs)
    np.testing.assert_allclose(select_depths(r.mesh,text='200',top=500,bottom=100),[200])


def test_slice_depth_boundaries_mask_and_partial_prisms():
    r=result(); original=r.model.copy()
    np.testing.assert_array_equal(horizontal_slice(r,100).values,r.model.reshape(r.mesh.shape)[:,:,1].T)
    np.testing.assert_array_equal(horizontal_slice(r,600).values,r.model.reshape(r.mesh.shape)[:,:,-1].T)
    bounds=r.mesh.cells(); bounds[0,2]=75; bounds[0,0]+=20; bounds[0,3]-=20
    active=r.mesh.active.copy(); active[-3:]=False
    r.mesh=Mesh(r.mesh.east,r.mesh.north,r.mesh.depth,active,bounds)
    shallow=horizontal_slice(r,50); assert np.isnan(shallow.values[0,0]) and np.isnan(shallow.values[1,1])
    deeper=horizontal_slice(r,80)
    np.testing.assert_allclose(deeper.rectangles[0],[18000020,7800000,18000080,7800100])
    np.testing.assert_array_equal(r.model,original)
    v=result('vector'); expected=np.linalg.norm(v.model,axis=1).reshape(v.mesh.shape)[:,:,0].T
    np.testing.assert_allclose(horizontal_slice(v,50).values,expected)
    np.testing.assert_allclose(horizontal_slice(v,50,'North').values,2)


@pytest.mark.parametrize('mode,unit',[('gravity','г/см³'),('induced','SI'),('vector','А/м')])
def test_slice_gallery_has_axes_units_and_common_colour_range(mode,unit,qt_application_lifetime,tmp_path,monkeypatch):
    from geogarmodel.slice_gallery import SliceGallery
    from matplotlib.colors import Normalize
    from geogarmodel.palette_library import palette_map
    r=result(mode); norm=Normalize(0,20)
    gallery=SliceGallery(r,np.array([50.,150.,450.,500.]),palette_map('turbo'),norm)
    try:
        gallery.canvas.draw()
        assert len(gallery.figure.axes)==6
        for ax in gallery.figure.axes:
            if ax.get_title():
                assert ax.get_xlabel()=='East, м' and ax.get_ylabel()=='North, м'
                assert ax.images[0].norm is norm
            else: assert ax.get_ylabel()==unit
        gallery.change(1); assert '4–4' in gallery.page_label.text()
        assert property_unit(r)==unit
        if mode=='gravity':
            from PyQt5.QtWidgets import QFileDialog
            import re
            path=tmp_path/'slices.pdf'
            monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(path),''))
            gallery.export_pdf(); data=path.read_bytes()
            assert data.startswith(b'%PDF') and len(re.findall(rb'/Type /Page\b',data))==2
            assert gallery.page==1 and '4–4' in gallery.page_label.text()
    finally: gallery.close()


def test_lod_switching_cached_geometry_and_exact_slice_coordinates(qt_application_lifetime,monkeypatch):
    pvqt=pytest.importorskip('pyvistaqt')
    def unsupported_probe(*args,**kwargs):
        raise AssertionError('Do not create a second GL window to probe transparency.')
    monkeypatch.setattr(pvqt.QtInteractor,'enable_depth_peeling',unsupported_probe)
    from geogarmodel.inversion_view import ModelView
    r=result(); r.mesh=Mesh(np.arange(13)*100.+18000000,np.arange(13)*100.+7800000,np.arange(13)*100.+10)
    r.model=np.linspace(-1,1,r.mesh.size)
    original=r.model.copy(); view=ModelView()
    try:
        view.render(r); geometry=view.display_grid
        assert view.plotter.renderer.GetUseDepthPeeling()
        view.high_quality_transparency.setChecked(False)
        assert not view.plotter.renderer.GetUseDepthPeeling()
        view.high_quality_transparency.setChecked(True)
        assert len(view._lod_actors)==2
        full=view._lod_actors[0][1]; assert full.visibility
        view.interaction_started(); assert not full.visibility
        view.interaction_finished(); assert full.visibility
        view.palette.setCurrentText('plasma'); assert view.display_grid is geometry
        view.slice_coordinates[2].setValue(123.456)
        view.palette.setCurrentText('turbo'); assert view.slice_coordinates[2].value()==pytest.approx(123.456)
        view.single_slice(2); assert view.kind.currentIndex()==3
        assert [w.isChecked() for w in view.slice_enabled]==[False,False,True]
        np.testing.assert_array_equal(r.model,original)
        np.testing.assert_allclose(view.bounds_axes.x_axis_range,[18000000,18001200])
        view.clip_ranges[0][0].setValue(18000300)
        view.clip_ranges[0][1].setValue(18000900)
        view.clip_enabled.setChecked(True)
        np.testing.assert_allclose(view.bounds_axes.x_axis_range,[18000300,18000900])
        view.outline.setChecked(False)
        replacement=result(); view.result=replacement  # Gallery can select a new result before opening 3D.
        view.render(replacement)
        assert view.slice_coordinates[2].minimum()==0 and view.slice_coordinates[2].maximum()==600
        assert view.slice_coordinates[2].value()==0  # Always start at the top.
    finally: view.shutdown(); view.close()


def test_prism_view_percentage_and_cut_round_trip(qt_application_lifetime):
    pytest.importorskip('pyvistaqt')
    from geogarmodel.inversion_view import ModelView
    view=ModelView()
    try:
        r=result()
        view.kind.setCurrentText('Верхние призмы и оболочка')
        view.render(r)
        assert view.prism_percent.value()==2
        assert not view.prism_panel.isHidden()
        geometry=view.display_grid
        actors=view.plotter.renderer.actors
        assert actors['upper-prisms'].prop.show_edges
        assert not actors['lower-shell'].prop.show_edges
        for percent in (0,100,12.5):
            view.prism_percent.setValue(percent)
            assert view.display_grid is geometry
            actors=view.plotter.renderer.actors
            assert ('upper-prisms' in actors)==(percent>0)
            assert ('lower-shell' in actors)==(percent<100)
        view.prism_layer.setValue(2)
        assert view.display_grid.bounds[-1]==-100
        saved=view.capture_state()
        view.prism_axis.setCurrentText('X')
        assert view.prism_layer.value()==1
        view.prism_percent.setValue(0)
        view.restore_state(saved)
        assert view.prism_percent.value()==12.5
        assert view.prism_axis.currentText()=='Z' and view.prism_layer.value()==2
    finally:
        view.shutdown()
        view.close()
