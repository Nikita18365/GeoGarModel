import numpy as np
import pytest
from geogarmodel.inversion import Mesh,FieldOperator,invert,regularization,InversionResult,prism_kernel
from geogarmodel.voxel import VoxelModel,read_ubc,read_centers,load_voxel,resample,clip_surface,edit,combine_components,read_vtk


def model(vector=False):
    mesh=Mesh(np.array([0.,1,3]),np.array([0.,2,5]),np.array([1.,3,7]))
    values=np.arange(mesh.size,dtype=float)/10
    return VoxelModel(mesh,np.column_stack((values,-values,values*2)) if vector else values,'vector' if vector else 'gravity')


def test_ubc_order_nonuniform_and_dummy(tmp_path):
    mesh=tmp_path/'model.msh'; prop=tmp_path/'model.mod'
    mesh.write_text('2 3 2\n100 200 50\n2*10\n20 30 40\n5 15\n')
    prop.write_text('\n'.join(map(str,range(12))))
    m=read_ubc(mesh,prop,dummy=0)
    assert m.mesh.shape==(2,3,2)
    np.testing.assert_array_equal(m.mesh.depth,[-50,-45,-30])
    assert m.values.reshape(m.mesh.shape)[1,2,1]==11
    assert m.values.reshape(m.mesh.shape)[0,1,0]==4
    assert not m.mesh.active[0]


@pytest.mark.parametrize('keep',['above','below'])
def test_surface_cut_exact_geometry_forward_and_roundtrip(tmp_path,keep):
    m=model(); original=m.copy()
    clipped=clip_surface(m,[0,3,0,3],[0,0,5,5],[2,2,2,2],keep)
    cells=clipped.mesh.cells()[clipped.mesh.active]
    assert np.all(cells[:,2]>=2) if keep=='below' else np.all(cells[:,5]<=2)
    np.testing.assert_array_equal(m.values,original.values)
    obs=np.array([[1,2,-1],[2,3,-2.]])
    op=FieldOperator(clipped.mesh,obs)
    expected=prism_kernel(obs,cells)@clipped.values[clipped.mesh.active]
    np.testing.assert_allclose(op.predict(clipped.values),expected)
    result=invert(op,expected,sigma=.1,initial=clipped.values,lower=.01,upper=1.,iterations=2,target_rms=0)
    assert np.all(result.model[~clipped.mesh.active]==0)
    path=tmp_path/'initial.npz'; clipped.save(path); reopened=load_voxel(path)
    np.testing.assert_array_equal(reopened.mesh.cells(),clipped.mesh.cells())
    path=tmp_path/'result.npz'; result.save(path); reopened=InversionResult.load(path)
    np.testing.assert_array_equal(reopened.mesh.active,clipped.mesh.active)
    np.testing.assert_allclose(FieldOperator(reopened.mesh,obs).predict(reopened.model),op.predict(result.model))


def test_surface_unknown_rejected():
    with pytest.raises(ValueError,match='покрывает'):
        clip_surface(model(),[0,.1,0],[0,0,.1],[1,1,1])


def test_inactive_cells_above_observers_do_not_affect_depth_weights():
    mesh=Mesh([0,1,2],[0,1,2],[-20,-10,1,10])
    mesh.active=mesh.centers()[:,2]>1
    op=FieldOperator(mesh,[[.5,.5,0],[1.5,1.5,0]])
    result=invert(op,[.01,.02],depth_index=1.5,iterations=2,target_rms=0)
    assert np.isfinite(result.model).all()
    assert np.isfinite(result.history[-1]['objective'])


def test_regularizer_mask_and_nonuniform_gradient():
    m=model(True); m.mesh.active[[1,5]]=False
    rng=np.random.default_rng(4); x=rng.normal(size=m.values.shape); v=rng.normal(size=x.shape)
    val,g=regularization(x,m.mesh,.2,3.)
    eps=1e-6
    numeric=(regularization(x+eps*v,m.mesh,.2,3.)[0]-regularization(x-eps*v,m.mesh,.2,3.)[0])/(2*eps)
    np.testing.assert_allclose(numeric,np.sum(g*v),rtol=1e-7)
    assert np.all(g[~m.mesh.active]==0)


def test_resample_mask_edges_and_constant():
    m=edit(model(),'constant',3)
    target=Mesh(np.linspace(-1,4,6),np.linspace(0,5,6),np.linspace(1,7,5))
    for method in ('nearest','linear'):
        result=resample(m,target,method)
        assert 0<result.mesh.active.sum()<result.mesh.size
        np.testing.assert_allclose(result.values[result.mesh.active],3)
        assert np.all(result.values[~result.mesh.active]==0)


def test_edit_and_components():
    m=model(); n=edit(m,'multiply',2)
    np.testing.assert_array_equal(n.values,2*m.values)
    a=edit(m,'normal',.3,.1,7); b=edit(m,'normal',.3,.1,7)
    np.testing.assert_array_equal(a.values,b.values)
    vector=combine_components([m,n,a])
    np.testing.assert_allclose(vector.values[:,1],n.values)
    changed=edit(vector,'constant',2,component=1)
    np.testing.assert_allclose(changed.values[:,0],vector.values[:,0]); assert np.all(changed.values[:,1]==2)


def test_vtk_roundtrip_and_clipped_volume(tmp_path):
    pytest.importorskip('pyvista')
    from geogarmodel.inversion_view import vtk_grid
    m=model(True); g=vtk_grid(m.result()); path=tmp_path/'model.vtr'; g.save(path)
    n=read_vtk(path,'vector')
    np.testing.assert_allclose(n.values,m.values)
    clipped=clip_surface(m,[0,3,0,3],[0,0,5,5],[2]*4)
    g=vtk_grid(clipped.result())
    assert g.n_cells==clipped.mesh.active.sum()
    expected=np.prod(clipped.mesh.cells()[clipped.mesh.active,3:]-clipped.mesh.cells()[clipped.mesh.active,:3],axis=1).sum()
    np.testing.assert_allclose(g.volume,expected)
    g.save(tmp_path/'clipped.vtu')


def test_centers_order_and_missing(tmp_path):
    path=tmp_path/'centers.csv'
    path.write_text('East,North,Down,Property\n0,0,1,.1\n1,0,1,.2\n0,1,2,.3\n1,1,2,.4\n')
    m=read_centers(path)
    assert m.mesh.shape==(2,2,2); assert m.mesh.active.sum()==4
    assert m.values.reshape(m.mesh.shape)[1,1,1]==.4
