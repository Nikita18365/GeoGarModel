import os
import numpy as np
import pytest

from geogarmodel.inversion import FieldOperator, Mesh, host
from geogarmodel.layer_fft import LayerFFT, model_axis, spectrum_budget


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
    from geogarmodel.inversion import prism_kernel
    matrix=prism_kernel(obs,mesh.cells())
    matrix[:,~mesh.active]=0
    for fit_offset in (False,True):
        centered=matrix-np.average(matrix,axis=0,weights=1/sigma**2) if fit_offset else matrix
        np.testing.assert_allclose(host(fft.sensitivity_norm(sigma,fit_offset)),
                                   np.linalg.norm(centered/sigma[:,None],axis=0),atol=1e-9)
    obs[-1,2]+=1
    with pytest.raises(ValueError,match='постоянной'): LayerFFT(mesh,obs)
    obs[-1,2]-=1
    obs[-1,0]+=.1
    with pytest.raises(ValueError,match='регулярной'): LayerFFT(mesh,obs)


def test_fft_workflow(qt_application_lifetime,tmp_path,monkeypatch):
    from PyQt5.QtCore import QEventLoop,QTimer
    from geogarmodel.app import MainWindow
    window=MainWindow()
    tab=window.inversion
    # This test exercises the solver dialog and project archive. VTK has separate rendering tests.
    monkeypatch.setattr(tab.scene,'render',lambda *args:None)
    from geogarmodel.inversion_help import ParameterHelp
    from PyQt5.QtWidgets import QMessageBox
    shown=[]
    monkeypatch.setattr(QMessageBox,'information',lambda parent,title,body:shown.append(body))
    next(button for button in tab.findChildren(ParameterHelp) if button.field is tab.depth_index).click()
    assert 'от 0 до 4' in shown[0]
    try:
        tab.method.setCurrentIndex(1)
        tab.demo()
        assert tab.nx.isHidden() and tab.stride.isHidden() and not tab.memory_policy.isHidden()
        tab.fft_grid.setCurrentIndex(1)
        tab.nx.setValue(21)
        tab.ny.setValue(11)
        tab.nz.setValue(4)
        tab.memory_policy.setCurrentIndex(2)
        tab.cache.setValue(0)
        assert not tab.nx.isHidden() and tab.cache.isEnabled()
        tab.table.loc[0,'height']=10
        with pytest.raises(ValueError,match='постоянной'): tab.run()
        tab.table.loc[0,'height']=0
        ref=tab.prepared_model()
        ref.values[::2]=.15
        tab.set_reference(ref)
        tab.structural.setChecked(True)
        tab.penalty_weight.setCurrentIndex(2)
        tab.penalty_norm.setCurrentIndex(2)
        tab.vertical_factor.setValue(.3)
        tab.target.setValue(.2)
        tab.early_stop.setChecked(True)
        tab.early_tolerance.setValue(99)
        tab.early_patience.setValue(2)
        tab.run()
        loop=QEventLoop()
        tab.worker.finished.connect(loop.quit)
        QTimer.singleShot(30000,loop.quit)
        loop.exec_()
        assert tab.result.settings['resources']['method']=='layer_fft'
        assert tab.result.mesh.shape==(21,11,4)
        assert tab.result.settings['resources']['cached_layers']==0
        assert tab.result.status=='early_stop'
        assert tab.result.settings['vertical_factor']==.3 and tab.result.settings['target_rms']==.2
        assert tab.result.settings['norm']=='tv' and tab.result.settings['weighting']=='sensitivity'
        assert tab.result.history[-1]['rmse']<tab.result.history[0]['rmse']
        path=tmp_path/'fft.ggproject'
        window.save_project_to(path)
        tab.method.setCurrentIndex(0)
        assert not tab.stride.isHidden()
        assert window.open_project_from(path)
        assert tab.is_fft() and tab.stride.isHidden()
        assert tab.fft_grid.currentIndex()==1 and tab.nx.value()==21 and tab.cache.isEnabled()
        assert tab.result.settings['resources']['method']=='layer_fft'
        np.testing.assert_array_equal(tab.reference_model.values,ref.values)
        assert tab.structural.isChecked() and tab.early_stop.isChecked()
    finally: window.close()


@pytest.mark.parametrize('norm',['compact','tv','l2'])
def test_prior_regularizer_gradient(norm):
    from geogarmodel.inversion import regularization
    mesh=Mesh([0,2,4,6],[0,3,6],[1,2,4,8])
    mesh.active[4]=False
    rng=np.random.default_rng(57)
    model,delta=rng.normal(size=(2,mesh.size))
    options=dict(reference=np.arange(mesh.size)%2*.2,penalty_weights=rng.uniform(.01,1,mesh.size),
                 norm=norm,epsilon=.03,vertical_factor=.4,boundary_jump=.1)
    def evaluate(m): return regularization(m,mesh,.5,.7,**options)
    step=1e-5
    numeric=(evaluate(model+step*delta)[0]-evaluate(model-step*delta)[0])/(2*step)
    np.testing.assert_allclose(evaluate(model)[1] @ delta,numeric,atol=1e-8)


def test_fft_fixed_density_contributes_to_field():
    from geogarmodel.inversion import invert
    mesh=Mesh([0,1,2],[0,1,2],[1,2,3])
    op=LayerFFT(mesh,np.array([[.5,.5,0],[1.5,1.5,0]]))
    ref=np.linspace(.1,.8,mesh.size)
    fixed=np.arange(mesh.size)%2==0
    result=invert(op,np.array([.1,.2]),iterations=3,target_rms=0,reference=ref,fixed_mask=fixed)
    np.testing.assert_array_equal(result.model[fixed],ref[fixed])
    np.testing.assert_allclose(result.predicted,op.field(result.model))


@pytest.mark.parametrize('device',['cpu','gpu'])
def test_fft_different_grid_steps_with_padding(device):
    if device=='gpu' and not os.environ.get('GEOGARMODEL_GPU_TESTS'):
        pytest.skip('optional CUDA test')
    east = 18e6 + np.arange(11) * 500.
    north = 7e6 + np.arange(9) * 400.
    model_east, model_north = [origin + np.arange(count+1)*step for origin,step,count in
                             (model_axis(east,24,.1),model_axis(north,7,.2))]
    assert len(model_east)==25 and len(model_north)==8
    assert np.diff(model_east).mean() < 500 < np.diff(model_north).mean()
    assert model_east[0] < east[0]-500 and model_east[-1] > east[-1]+500
    mesh = Mesh(model_east, model_north, [50, 250, 700])
    x, y = np.meshgrid(east,north)
    obs = np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    obs = obs[np.r_[2, 2, 4:91]]
    rng = np.random.default_rng(41)
    model, residual = rng.normal(size=mesh.size), rng.normal(size=len(obs))
    direct = FieldOperator(mesh,obs)
    fft = LayerFFT(mesh,obs,device=device)
    field = host(fft.field(model))
    grad = host(fft.data_gradient(fft.xp.asarray(residual),None))
    np.testing.assert_allclose(field,direct.field(model),atol=1e-9)
    np.testing.assert_allclose(grad,direct.data_gradient(residual,None),atol=1e-9)
    np.testing.assert_allclose(field @ residual, model @ grad, atol=1e-10)


def test_fft_padding_preserves_imported_partial_cells():
    from geogarmodel.voxel import VoxelModel, extend_density_model

    mesh = Mesh([0,1,2], [0,1,2], [1,2,3])
    bounds = mesh.cells()
    bounds[0,2] += .1
    mesh = Mesh(mesh.east, mesh.north, mesh.depth, mesh.active, bounds)
    mesh.active[1] = False
    model = VoxelModel(mesh,np.full(mesh.size,.2))
    padded = extend_density_model(model,np.array([.1,1.9]),np.array([.1,1.9]),.25,.05)
    assert padded.mesh.shape == (4,4,2)
    np.testing.assert_array_equal(padded.mesh.bounds.reshape(4,4,2,6)[1:3,1:3],bounds.reshape(2,2,2,6))
    np.testing.assert_array_equal(padded.mesh.active.reshape(4,4,2)[1:3,1:3].ravel(),mesh.active)
    assert padded.values[0] == .05
    assert extend_density_model(padded,np.array([.1,1.9]),np.array([.1,1.9]),.25,.05) is padded


def test_fft_cache_modes():
    from geogarmodel.inversion_resources import MIB

    assert spectrum_budget(100*MIB,10*MIB,MIB,100,'balanced',0)==(50,50*MIB)
    assert spectrum_budget(100*MIB,10*MIB,MIB,100,'performance',0)==(70,70*MIB)
    assert spectrum_budget(100*MIB,10*MIB,MIB,100,'manual',0)==(0,0)
    with pytest.raises(ValueError,match='кэша FFT'):
        spectrum_budget(100*MIB,10*MIB,MIB,100,'manual',90)


@pytest.mark.parametrize('mode',['induced','vector'])
@pytest.mark.parametrize('device',['cpu','gpu'])
def test_magnetic_fft_projected_prisms(mode,device):
    from geogarmodel.inversion import invert
    if device=='gpu' and not os.environ.get('GEOGARMODEL_GPU_TESTS'):
        pytest.skip('optional CUDA test')
    mesh=Mesh(np.arange(5)*120.,np.arange(4)*90.,np.array([20.,60.,150.]))
    bounds=mesh.cells()
    bounds[0,2]+=10
    mesh=Mesh(mesh.east,mesh.north,mesh.depth,mesh.active,bounds)
    mesh.active[3]=False
    x,y=np.meshgrid(mesh.east+20,mesh.north+40)
    obs=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    kwargs=dict(mode=mode,inclination=-35,declination=28)
    op=LayerFFT(mesh,obs,device=device,**kwargs)
    direct=FieldOperator(mesh,obs,**kwargs)
    rng=np.random.default_rng(7)
    model=rng.uniform(.001,.1,op.parameter_shape)
    predicted=direct.field(model)@direct.f
    np.testing.assert_allclose(host(op.field(model)),predicted,atol=1e-9)
    r=rng.normal(size=len(obs))
    np.testing.assert_allclose(np.sum(model*host(op.data_gradient(op.xp.asarray(r),None))),predicted@r,atol=1e-9)
    result=invert(op,predicted,iterations=3,lower=0,upper=.3,weighting='sensitivity',target_rms=.0001)
    assert result.history[-1]['rmse']<result.history[0]['rmse']
    if mode=='induced':
        converted=result.as_magnetization()
        vec=LayerFFT(mesh,obs,mode='vector',inclination=-35,declination=28)
        np.testing.assert_allclose(vec.field(converted.model),host(op.field(result.model)),atol=1e-8)
