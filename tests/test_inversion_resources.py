import numpy as np
import pytest
from geogarmodel.inversion_resources import resource_plan, MIB


@pytest.mark.parametrize('gib',[2,4,16,80,1024])
@pytest.mark.parametrize('mode',['gravity','induced','vector'])
def test_budgets_scale_to_hardware(gib,mode):
    free=gib*1024*MIB
    balanced=resource_plan(free,108100,44288,mode,'gpu')
    fast=resource_plan(free,108100,44288,mode,'gpu','performance')
    assert 0<=balanced['cache_mib']<=fast['cache_mib']<=fast['safe_cache_mib']
    assert fast['cache_mib']*MIB<=free*.7
    assert fast['observation_block']*fast['cell_block']<=1048576
    assert fast['observation_block']*fast['cell_block']*512<free*.1


def test_manual_and_small_operator():
    plan=resource_plan(14*1024*MIB,108100,44288,'gravity','gpu','manual',10000)
    assert plan['cache_mib']==10000
    with pytest.raises(ValueError,match='бюджет'):
        resource_plan(2*1024*MIB,108100,44288,'gravity','gpu','manual',10000)
    small=resource_plan(2*1024*MIB,100,100,'gravity','cpu','manual',1048576)
    assert small['cache_mib']==1
    assert resource_plan(2*1024*MIB,100,100,'gravity','cpu','manual',0)['cache_mib']==0
    with pytest.raises(ValueError,match='Недостаточно'):
        resource_plan(100*MIB,108100,44288,'gravity','gpu')


@pytest.mark.parametrize('mode',['gravity','induced','vector'])
def test_new_block_shapes_preserve_fields_and_gradients(mode):
    from geogarmodel.inversion import Mesh,FieldOperator
    mesh=Mesh(np.arange(13.)*100,np.arange(11.)*100,np.arange(9.)*100+100)
    obs=np.column_stack((np.linspace(-50,1500,270),np.linspace(-50,1200,270),np.zeros(270)))
    a=FieldOperator(mesh,obs,mode,cache_mb=0)
    b=FieldOperator(mesh,obs,mode,memory_policy='balanced')
    b.resources.update(observation_block=256,cell_block=1024)
    model=np.random.default_rng(1).normal(size=a.parameter_shape)*.01
    va,ga,pa=a.value_gradient(model,np.ones(270),np.ones(270))
    vb,gb,pb=b.value_gradient(model,np.ones(270),np.ones(270))
    np.testing.assert_allclose(pb,pa,atol=1e-9)
    np.testing.assert_allclose(gb,ga,rtol=1e-8,atol=1e-8)
    assert vb==pytest.approx(va)
    assert b.cache_bytes<=b.cache_limit
