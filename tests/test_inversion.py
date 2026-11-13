import os
from threading import Event
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from geogarmodel.inversion import Mesh, FieldOperator, prism_kernel, direction, invert, regularization, InversionResult
from geogarmodel.compute import ComputationCancelled


def mesh():
    return Mesh(np.linspace(-300,300,4),np.linspace(-300,300,4),np.linspace(100,700,4))


def observations():
    x,y=np.meshgrid(np.linspace(-500,500,7),np.linspace(-500,500,7))
    return np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))


def test_prism_against_independent_volume_quadrature():
    cell=np.array([[-130.,-80.,120.,70.,140.,350.]])
    obs=np.array([[80.,-210.,-50.]])
    nodes,weights=leggauss(16)
    points=np.stack(np.meshgrid(*[.5*((b-a)*nodes+b+a) for a,b in zip(cell[0,:3],cell[0,3:])],indexing='ij'),axis=-1).reshape(-1,3)
    w=np.prod(np.stack(np.meshgrid(weights,weights,weights,indexing='ij'),axis=-1),axis=-1).ravel()*np.prod(cell[0,3:]-cell[0,:3])/8
    r=points-obs[0]; distance=np.linalg.norm(r,axis=1)
    gravity=np.sum(w*r[:,2]/distance**3)*6.6743e-3
    np.testing.assert_allclose(prism_kernel(obs,cell)[0,0],gravity,rtol=1e-10)
    tensor=np.sum(w[:,None,None]*(3*r[:,:,None]*r[:,None,:]/distance[:,None,None]**5-np.eye(3)/distance[:,None,None]**3),axis=0)*100
    actual=prism_kernel(obs,cell,True)[0,0]
    np.testing.assert_allclose(actual,[tensor[0,0],tensor[1,1],tensor[2,2],tensor[0,1],tensor[0,2],tensor[1,2]],rtol=1e-9,atol=1e-10)


@pytest.mark.parametrize('mode',['gravity','induced','vector'])
def test_exact_objective_gradient(mode):
    op=FieldOperator(mesh(),observations(),mode,cache_mb=0,declination=23,inclination=-35)
    rng=np.random.default_rng(3)
    m=rng.normal(size=op.parameter_shape)*(.001 if mode=='induced' else .1)
    d=rng.normal(size=49)
    v,g,_=op.value_gradient(m,d,np.ones(49)*2)
    direction_=rng.normal(size=m.shape)
    epsilon=1e-5 if mode!='induced' else 1e-7
    numeric=(op.value_gradient(m+epsilon*direction_,d,np.ones(49)*2)[0]-op.value_gradient(m-epsilon*direction_,d,np.ones(49)*2)[0])/(2*epsilon)
    np.testing.assert_allclose(numeric,np.sum(g*direction_),rtol=2e-5,atol=1e-6)


@pytest.mark.parametrize('mode',['gravity','induced','vector'])
def test_invert_reduces_misfit_bounds_and_save(mode,tmp_path):
    op=FieldOperator(mesh(),observations(),mode)
    truth=np.zeros(op.parameter_shape)
    truth[13]=[.2,-.4,.7] if mode=='vector' else (.03 if mode=='induced' else .5)
    d=op.predict(truth)
    bounds=(0,.1) if mode=='induced' else (-1,1)
    result=invert(op,d,sigma=max(np.std(d)*.02,1e-5),iterations=60,lower=bounds[0],upper=bounds[1],target_rms=0)
    assert result.history[-1]['rmse']<result.history[0]['rmse']*.15
    assert np.all(result.model>=bounds[0]) and np.all(result.model<=bounds[1])
    assert np.all(np.diff([r['objective'] for r in result.history])<=1e-9)
    result.save(tmp_path/'model.npz')
    np.testing.assert_allclose(InversionResult.load(tmp_path/'model.npz').predicted,result.predicted)
    with np.load(tmp_path/'model.npz',allow_pickle=False) as saved:
        np.testing.assert_allclose(saved['model'],result.model)
    event=Event(); event.set()
    with pytest.raises(ComputationCancelled):
        invert(op,d,lower=bounds[0],upper=bounds[1],cancel_event=event)


def test_regularizer_gradient_and_validation():
    rng=np.random.default_rng(4); m=rng.normal(size=(27,3)); v=rng.normal(size=m.shape)
    val,g=regularization(m,mesh(),.1,2.)
    numeric=(regularization(m+1e-6*v,mesh(),.1,2.)[0]-regularization(m-1e-6*v,mesh(),.1,2.)[0])/2e-6
    np.testing.assert_allclose(numeric,np.sum(g*v),rtol=1e-7)
    with pytest.raises(ValueError,match='выше'):
        FieldOperator(mesh(),[[0,0,100]])
    np.testing.assert_allclose(direction(90,0),[0,0,1],atol=1e-15)


def test_weighted_offset_and_gradient():
    op=FieldOperator(mesh(),observations())
    rng=np.random.default_rng(6); m=rng.normal(size=27)*.1; vector=rng.normal(size=27)
    sigma=np.linspace(.1,2,49); data=rng.normal(size=49)+4
    val,g,pred=op.value_gradient(m,data,sigma,fit_offset=True)
    assert abs(np.sum((pred-data)/sigma**2))<1e-10
    eps=1e-6
    numeric=(op.value_gradient(m+eps*vector,data,sigma,True)[0]-op.value_gradient(m-eps*vector,data,sigma,True)[0])/(2*eps)
    np.testing.assert_allclose(numeric,g@vector,rtol=1e-6)


def test_induced_and_aligned_vector_have_same_exact_field():
    induced=FieldOperator(mesh(),observations(),'induced',inclination=41,declination=28)
    vector=FieldOperator(mesh(),observations(),'vector',inclination=41,declination=28)
    chi=np.random.default_rng(11).uniform(0,.02,27)
    np.testing.assert_allclose(induced.predict(chi),vector.predict(chi[:,None]*induced.h*induced.f),atol=1e-10)


@pytest.mark.skipif(os.environ.get('GEOGARMODEL_GPU_TESTS')!='1',reason='Opt-in CUDA hardware test')
@pytest.mark.parametrize('mode',['gravity','induced','vector'])
def test_gpu_cpu_equivalence(mode):
    from geogarmodel.inversion import host
    cpu=FieldOperator(mesh(),observations(),mode)
    gpu=FieldOperator(mesh(),observations(),mode,device='gpu')
    m=np.random.default_rng(3).normal(size=cpu.parameter_shape)*.01
    np.testing.assert_allclose(host(gpu.predict(m)),cpu.predict(m),rtol=1e-8,atol=1e-8)
    d=np.ones(49)
    bounds=dict(lower=0,upper=.1) if mode=='induced' else {}
    a=invert(cpu,d,iterations=3,target_rms=0,**bounds)
    b=invert(gpu,d,iterations=3,target_rms=0,**bounds)
    np.testing.assert_allclose(a.model,b.model,rtol=1e-6,atol=1e-8)
