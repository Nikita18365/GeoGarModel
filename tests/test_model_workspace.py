import numpy as np
import pytest
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QApplication
from geogarmodel.inversion import Mesh,InversionResult
from geogarmodel.section_rendering import sample_section


def model():
    mesh=Mesh(np.array([100.,110.,130.,160.]),np.array([200.,220.,250.,300.]),np.array([10.,30.,60.,100.]))
    values=mesh.centers()[:,0]+2*mesh.centers()[:,1]+3*mesh.centers()[:,2]
    return InversionResult(mesh,values,np.empty((0,3)),np.empty(0),np.empty(0),[],{'mode':'gravity'},'test')


def test_smooth_sections_nonuniform_edges_depth_and_partial_masks():
    r=model(); original=r.model.copy()
    for axis in range(3):
        axes=(r.mesh.east,r.mesh.north,r.mesh.depth); coordinate=(axes[axis][1]+axes[axis][2])/2
        image,extent=sample_section(r,axis,coordinate,width=60,height=50)
        free=[i for i in range(3) if i!=axis]; u0,u1,v0,v1=extent
        x=u0+(np.arange(60)+.5)*(u1-u0)/60; y=v0+(np.arange(50)+.5)*(v1-v0)/50
        xx,yy=np.meshgrid(x,y); coefficients=[1,2,3]
        expected=coefficients[free[0]]*xx+coefficients[free[1]]*yy+coefficients[axis]*coordinate
        inner=(xx>(axes[free[0]][0]+axes[free[0]][1])/2)&(xx<(axes[free[0]][-2]+axes[free[0]][-1])/2)&(yy>(axes[free[1]][0]+axes[free[1]][1])/2)&(yy<(axes[free[1]][-2]+axes[free[1]][-1])/2)
        np.testing.assert_allclose(image[inner],expected[inner],rtol=1e-6)
    cells=r.mesh.cells(); cells[0,0]=108; cells[0,2]=25; r.mesh.bounds=cells; r.mesh.active[12]=False
    image,_=sample_section(r,2,20,width=60,height=100)
    assert np.isnan(image[:20,:10]).all()  # Above the clipped first prism.
    image,_=sample_section(r,2,26,width=60,height=100)
    assert np.isnan(image[:20,:8]).all() and np.isfinite(image[:20,8:10]).all()
    np.testing.assert_array_equal(r.model,original)


def test_window_membership_paths_state_and_hover(qt_application_lifetime,monkeypatch,tmp_path):
    from geogarmodel.model_workspace import ModelWorkspace,ModelView
    # Window/catalogue behaviour must also work when optional VTK is not installed.
    monkeypatch.setattr(ModelView,'render',lambda self,result,**kwargs:setattr(self,'result',result))
    workspace=ModelWorkspace(); workspace.resize(1200,750); workspace.show()
    r=model(); path=tmp_path/'density.vtr'
    first=workspace.render(r,path=str(path)); entry=first.selected
    try:
        workspace.render(r,path=str(path)); assert len(workspace.registry)==1 and len(first.entries)==1
        first.view.palette.setCurrentText('plasma')
        second_entry=workspace.register(model(),name='Second')
        first.add_entry(second_entry); first.view.palette.setCurrentText('turbo')
        first.select(entry); assert first.view.palette.currentText()=='plasma'
        first.select(second_entry); assert first.view.palette.currentText()=='turbo'
        second=workspace.new_window(entry); assert len(second.entries)==1 and len(first.entries)==2
        assert workspace.source_path.text()==str(path.resolve())
        assert workspace.catalogue.count()==2
        QApplication.sendEvent(first.view,QEvent(QEvent.Enter)); assert workspace.active_document() is first
        assert workspace.models.count()==2
        second.showMinimized(); workspace.refresh()
        index=next(i for i in range(workspace.window_selector.count()) if workspace.window_selector.itemData(i)==second.number)
        workspace.restore_window(index); assert not second.isMinimized() and workspace.active_document() is second
        second.remove_entry(entry); assert first.entries[0] is entry and len(workspace.registry)==2
    finally: workspace.shutdown(); workspace.close(); QApplication.processEvents()


def test_depth_clipping_is_downward_and_expanded_state_matches(qt_application_lifetime):
    pytest.importorskip('pyvistaqt')
    from geogarmodel.inversion_view import ModelView
    r=model(); view=ModelView()
    try:
        view.render(r); assert view.depth_value.value()==10 and view.depth_slider.value()==0
        view.kind.setCurrentIndex(4)
        assert view.clip_normal.currentText()=='Z'
        assert view.plotter.plane_widgets[-1].GetNormal()[2]<0
        view.depth_value.setValue(45)
        clipped=view.plotter.plane_clipped_meshes[-1]
        assert clipped.bounds[5]<=-45+1e-6 and clipped.bounds[4]==pytest.approx(-100)
        widget=view.plotter.plane_widgets[-1]; center=list(widget.GetOrigin()); center[2]=-70; widget.SetOrigin(center); widget.InvokeEvent('EndInteractionEvent')
        assert view.depth_value.value()==70
        view.palette.setCurrentText('plasma'); assert view.plotter.plane_widgets[-1].GetOrigin()[2]==pytest.approx(-70)
        view.expanded_view(); expanded=view._expanded_dialog.findChild(ModelView)
        assert expanded.depth_value.value()==70 and expanded.plotter.plane_widgets[-1].GetNormal()[2]<0
        expanded.depth_value.setValue(85); view._expanded_dialog.accept()
        assert view.depth_value.value()==85
        view.single_slice(2)
        textured=[a for a in view.plotter.renderer.actors.values() if getattr(a,'texture',None) is not None]
        assert len(textured)==1 and textured[0].mapper.dataset.n_points==4
    finally: view.shutdown(); view.close()
