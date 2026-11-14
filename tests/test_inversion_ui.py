import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
import numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox,QPushButton,QDialog,QInputDialog
from PyQt5.QtCore import QEventLoop,QTimer
from geogarmodel.app import MainWindow


@pytest.fixture(scope='module',autouse=True)
def app_session():
    app=QApplication.instance() or QApplication([])
    yield app


@pytest.mark.parametrize('mode',[0,1,2])
def test_inversion_tab_background_workflow(mode,monkeypatch):
    errors=[]
    monkeypatch.setattr(QMessageBox,'warning',lambda *a: errors.append(a[-1]))
    window=MainWindow(); tab=window.inversion
    assert window.tabs.count()==3
    tab.mode.setCurrentIndex(mode); tab.demo()
    tab.iterations.setValue(4); tab.target.setValue(0)
    loop=QEventLoop(); timer=QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit)
    tab.run(); tab.worker.finished.connect(loop.quit); timer.start(30000); loop.exec_()
    assert tab.worker is None
    assert not errors,errors
    assert tab.result is not None,tab.status.text()
    assert tab.result.settings['resources']['policy']=='balanced'
    assert 'Свободно при старте' in tab.memory_info.text()
    assert len(tab.result.predicted)==225
    assert len(tab.figure.axes)==11
    assert tab.result.history[-1]['rmse']<tab.result.history[0]['rmse']
    tab.initial_panel.previous()
    initial=tab.initial_panel.current()
    np.testing.assert_allclose(initial.values,tab.result.model)
    assert initial.mesh.shape==tab.result.mesh.shape
    window.close(); QApplication.processEvents()


def test_vtk_coordinates_and_vector_order(tmp_path):
    pytest.importorskip('pyvista')
    from geogarmodel.inversion import Mesh,InversionResult
    from geogarmodel.inversion_view import vtk_grid
    mesh=Mesh(np.arange(3.)+100,np.arange(4.)+200,np.arange(5.)+10)
    model=np.arange(mesh.size*3).reshape(-1,3).astype(float)
    result=InversionResult(mesh,model,np.zeros((1,3)),np.zeros(1),np.zeros(1),[],{'mode':'vector'},'test')
    grid=vtk_grid(result)
    expected=mesh.centers(); expected[:,2]*=-1
    centers=grid.cell_centers().points
    for point,vector in zip(expected,model):
        index=np.where(np.all(np.isclose(centers,point),axis=1))[0][0]
        np.testing.assert_allclose(grid['M_END'][index],vector)
        assert grid['property'][index]==np.linalg.norm(vector)
    grid.save(tmp_path/'model.vtr')


def test_display_scale_keeps_physical_axes_and_model():
    pytest.importorskip('pyvistaqt')
    from geogarmodel.inversion import Mesh,InversionResult
    from geogarmodel.inversion_view import ModelView,vtk_grid
    mesh=Mesh(np.linspace(18000000,18010000,4),np.linspace(6000000,6020000,4),np.linspace(100,30000,4))
    result=InversionResult(mesh,np.ones(mesh.size),np.zeros((1,3)),np.zeros(1),np.zeros(1),[],{'mode':'gravity'},'test')
    original=vtk_grid(result).points.copy()
    view=ModelView(); view.resize(1100,850)
    try:
        view.render(result)
        view.axis_scales[2].setValue(3)
        np.testing.assert_allclose(view.bounds_axes.bounds,[0,10000,0,20000,-90000,-300])
        np.testing.assert_allclose(view.bounds_axes.z_axis_range,[-30000,-100])
        view.axis_unit.setCurrentText('км')
        np.testing.assert_allclose(view.bounds_axes.z_axis_range,[-30,-.1])
        view.plotter.camera.zoom(1.2)
        zoom=view.plotter.camera.parallel_scale
        view.axis_font.setValue(22)
        assert view.plotter.camera.parallel_scale==pytest.approx(zoom)
        for mode in range(5):
            view.kind.setCurrentIndex(mode)
            np.testing.assert_allclose(view.bounds_axes.z_axis_range,[-30,-.1])
        view.reset_scales()
        np.testing.assert_allclose(view.bounds_axes.bounds,[0,10000,0,20000,-30000,-100])
        np.testing.assert_allclose(vtk_grid(result).points,original)
        result.model[:]=0
        view.kind.setCurrentIndex(0); view.render(result)
        assert any(getattr(getattr(actor,'mapper',None),'dataset',None) is not None
                   and actor.mapper.dataset.n_cells==mesh.size for actor in view.plotter.renderer.actors.values())
        view.expanded_view()
        assert view._expanded_dialog.isVisible()
        view._expanded_dialog.accept()
        assert view._expanded_dialog is None
    finally:
        view.shutdown(); view.close()


def test_workflow_and_separate_editor(monkeypatch):
    window=MainWindow(); tab=window.inversion; tab.demo()
    tab.memory_policy.setCurrentIndex(2)
    tab.cache.setValue(1048576)
    assert tab.cache.value()==1048576 and tab.cache.isEnabled()
    tab.memory_policy.setCurrentIndex(0)
    assert not tab.cache.isEnabled()
    assert [tab.mode.itemText(i) for i in range(3)]==[
        'Плотностное моделирование · Δρ, g/cm³',
        'Моделирование магнитной восприимчивости · χ, SI',
        'Моделирование векторной намагниченности · J, A/m']
    assert tab.workflow.count()==3
    for i in range(2):
        tab.workflow.setCurrentIndex(i)
        button=next(b for b in tab.workflow.widget(i).findChildren(QPushButton) if b.text().startswith('Далее'))
        button.click(); assert tab.workflow.currentIndex()==i+1
    assert not any(b.text()=='Операции со свойствами…' for b in tab.initial_panel.findChildren(QPushButton))
    monkeypatch.setattr(QInputDialog,'getItem',lambda *args,**kwargs:('Заполнить константой (a)',True))
    monkeypatch.setattr(QInputDialog,'getText',lambda *args,**kwargs:('0.25, 1, 42',True))
    observed=[]
    def edit_and_close():
        dialog=QApplication.activeModalWidget()
        try:
            assert isinstance(dialog,QDialog)
            button=next(b for b in dialog.findChildren(QPushButton) if b.text()=='Операции со свойствами…')
            button.click(); observed.append(tab.initial_panel.current().values.copy())
        finally: dialog.accept()
    QTimer.singleShot(0,edit_and_close); tab.initial_panel.open_editor()
    assert len(observed)==1
    np.testing.assert_allclose(observed[0],.25)
    window.close(); QApplication.processEvents()
