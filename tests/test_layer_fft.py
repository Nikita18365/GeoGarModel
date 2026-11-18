import os
import numpy as np
import pytest

from geogarmodel.inversion import FieldOperator, Mesh, host
from geogarmodel.layer_fft import LayerFFT


@pytest.mark.parametrize('device',['cpu','gpu'])
def test_fft_prisms_and_adjoint(device):
    if device=='gpu' and not os.environ.get('GEOGARMODEL_GPU_TESTS'):
        pytest.skip('optional CUDA test')
    mesh=Mesh(18e6+np.arange(7)*160, 7e6+np.arange(5)*210, np.array([50,130,410,810]))
    mesh.active[::7]=False
    cells=mesh.cells()
    cells[4,2]+=33
    cells[10,3]-=15
    mesh=Mesh(mesh.east,mesh.north,mesh.depth,mesh.active,cells)
    x,y=np.meshgrid(mesh.east[0]+45+np.arange(8)*160,mesh.north[0]-30+np.arange(5)*210)
    obs=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,-120.)])
    obs=obs[np.r_[1,1,3:37:2]]
    rng=np.random.default_rng(22)
    model=rng.normal(size=mesh.size)
    data=rng.normal(size=len(obs))
    sigma=rng.uniform(.2,2,len(obs))
    direct=FieldOperator(mesh,obs)
    fft=LayerFFT(mesh,obs,device=device)
    np.testing.assert_allclose(host(fft.field(model)),direct.field(model),rtol=1e-9,atol=1e-10)
    expected=direct.value_gradient(model,data,sigma,True)
    actual=fft.value_gradient(fft.xp.asarray(model),fft.xp.asarray(data),fft.xp.asarray(sigma),True)
    for a,b in zip(actual,expected): np.testing.assert_allclose(host(a),b,atol=1e-9)
    assert len(fft.partial)==2
    obs[-1,2]+=1
    with pytest.raises(ValueError,match='постоянной'): LayerFFT(mesh,obs)
    obs[-1,2]-=1
    obs[-1,0]+=.1
    with pytest.raises(ValueError,match='регулярной'): LayerFFT(mesh,obs)


def test_fft_workflow(qt_application_lifetime,tmp_path):
    from PyQt5.QtCore import QEventLoop,QTimer
    from geogarmodel.app import MainWindow
    window=MainWindow()
    tab=window.inversion
    try:
        tab.mode.setCurrentIndex(tab.fft_index)
        tab.demo()
        assert tab.nx.isHidden() and tab.stride.isHidden()
        tab.table.loc[0,'height']=10
        with pytest.raises(ValueError,match='постоянной'): tab.run()
        tab.table.loc[0,'height']=0
        tab.run()
        loop=QEventLoop()
        tab.worker.finished.connect(loop.quit)
        QTimer.singleShot(30000,loop.quit)
        loop.exec_()
        assert tab.result.settings['resources']['method']=='layer_fft'
        assert tab.result.mesh.shape==(15,15,6)
        assert tab.result.history[-1]['rmse']<tab.result.history[0]['rmse']
        path=tmp_path/'fft.ggproject'
        window.save_project_to(path)
        tab.mode.setCurrentIndex(0)
        assert not tab.stride.isHidden()
        assert window.open_project_from(path)
        assert tab.is_fft() and tab.stride.isHidden()
        assert tab.result.settings['resources']['method']=='layer_fft'
    finally: window.close()
