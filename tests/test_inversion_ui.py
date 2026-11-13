import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
import numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox
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
