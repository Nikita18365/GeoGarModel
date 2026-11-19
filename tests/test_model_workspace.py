import numpy as np
import pytest
from PyQt5.QtCore import QEvent,Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication,QInputDialog
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


def test_window_membership_paths_state_and_click(qt_application_lifetime,monkeypatch,tmp_path):
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
        QApplication.sendEvent(first.view,QEvent(QEvent.Enter)); assert workspace.active_document() is second
        QTest.mouseClick(first.view,Qt.LeftButton); assert workspace.active_document() is first
        assert workspace.models.count()==2
        second.showMinimized(); workspace.refresh()
        index=next(i for i in range(workspace.window_selector.count()) if workspace.window_selector.itemData(i)==second.number)
        workspace.restore_window(index); assert not second.isMinimized() and workspace.active_document() is second
        second.remove_entry(entry); assert first.entries[0] is entry and len(workspace.registry)==2
        incoming=model()
        assert workspace.finish_import(incoming,first)
        assert first.selected.result is incoming and not second.entries and workspace.active_document() is second
        monkeypatch.setattr(QInputDialog,'getText',lambda *a,**k:(' Фундамент ',True))
        second.systemMenu().actions()[-1].trigger()
        assert second.windowTitle()==workspace.window_selector.currentText()=='Фундамент'
        assert workspace.snapshot()[-1]['title']=='Фундамент'
        first.close()
        delayed=model()
        assert not workspace.finish_import(delayed,first)
        assert workspace.register(delayed).result is delayed and not second.entries
    finally: workspace.shutdown(); workspace.close(); QApplication.processEvents()


def test_import_keeps_original_destination_during_background_read(qt_application_lifetime,monkeypatch):
    from types import SimpleNamespace
    from geogarmodel.app import MainWindow
    from geogarmodel.model_workspace import ModelView
    monkeypatch.setattr(ModelView,'render',lambda self,result:setattr(self,'result',result))
    window=MainWindow()
    window.tabs.setCurrentIndex(2)
    window.show()
    QApplication.processEvents()
    tab=window.inversion
    pending=[]
    incoming=model()
    monkeypatch.setattr(QInputDialog,'getItem',lambda parent,title,label,items,*args:(items[0],True))
    monkeypatch.setattr(tab.initial_panel,'ask_load',lambda mode:lambda:SimpleNamespace(result=lambda:incoming))
    monkeypatch.setattr(tab,'model_task',lambda read,done:pending.append((read,done)))
    try:
        tab.scene.load_action.trigger()
        target=tab.scene.active_document()
        other=tab.scene.new_window()
        read,done=pending.pop()
        done(read())
        assert target.selected.result is incoming
        assert tab.scene.active_document() is other and not other.entries
    finally: window.close(); QApplication.processEvents()


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
        cap=view.plotter.renderer.actors['cut-lower-shell'].mapper.dataset
        np.testing.assert_allclose(cap.points[:,2],-45)
        density=r.model.reshape(r.mesh.shape)[:,:,1]
        np.testing.assert_allclose([cap['property'].min(),cap['property'].max()],[density.min(),density.max()])
        plane=view.plotter.renderer.actors['lower-shell'].mapper.GetClippingPlanes().GetItem(0)
        assert plane.EvaluateFunction(0,0,-100)>0>plane.EvaluateFunction(0,0,-44)
        assert view.plotter.plane_widgets[-1].GetPlaneProperty().GetOpacity()==0
        widget=view.plotter.plane_widgets[-1]; center=list(widget.GetOrigin()); center[2]=-70; widget.SetOrigin(center); widget.InvokeEvent('EndInteractionEvent')
        assert view.depth_value.value()==70
        view.palette.setCurrentText('plasma'); assert view.plotter.plane_widgets[-1].GetOrigin()[2]==pytest.approx(-70)
        view.expanded_view(); expanded=view._expanded_dialog.findChild(ModelView)
        assert expanded.depth_value.value()==70 and expanded.plotter.plane_widgets[-1].GetNormal()[2]<0
        expanded.depth_value.setValue(85); view._expanded_dialog.accept()
        assert view.depth_value.value()==85
        view.single_slice(2)
        section=view.plotter.renderer.actors['section-2-lower-shell'].mapper.dataset
        np.testing.assert_allclose(section.points[:,2],-85)
        assert section.n_cells==r.mesh.shape[0]*r.mesh.shape[1]
        assert not view.plotter.renderer.actors['section-2-lower-shell'].prop.show_edges
    finally: view.shutdown(); view.close()
