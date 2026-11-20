import numpy as np
import pytest
from geogarmodel.inversion import Mesh, InversionResult
from geogarmodel.inversion_export import export_ubc, export_zmap, field_table
from geogarmodel.voxel import read_ubc
from geogarmodel.grids import read_zmap


def result(mode='gravity'):
    mesh=Mesh([100,200,400],[300,400,500,700],[10,30,80])
    mesh.active[2]=False
    values=np.arange(mesh.size*(3 if mode=='vector' else 1),dtype=float)
    if mode=='vector': values=values.reshape(-1,3)
    values[~mesh.active]=0
    xy=np.array([[x,y,0] for x in [1,2,3] for y in [10,20]],dtype=float)[:-1]
    predicted=np.arange(len(xy),dtype=float)-2
    return InversionResult(mesh,values,xy,predicted+1,predicted,[],dict(mode=mode,crs='test'),'cancelled')


@pytest.mark.parametrize('mode',['gravity','induced','vector'])
def test_ubc_round_trip(mode,tmp_path):
    r=result(mode); path=tmp_path/'model.msh'; export_ubc(r,path)
    names=[''] if mode!='vector' else ['_JE','_JN','_JD']
    for i,name in enumerate(names):
        loaded=read_ubc(path,tmp_path/('model'+name+'.mod'),dummy=-1e32)
        np.testing.assert_array_equal(loaded.mesh.active,r.mesh.active)
        np.testing.assert_allclose(loaded.values,r.model if mode!='vector' else r.model[:,i])
        for axis in ('east','north','depth'):
            np.testing.assert_array_equal(getattr(loaded.mesh,axis),getattr(r.mesh,axis))


def test_zmap_preserves_samples_and_holes(tmp_path):
    r=result(); path=tmp_path/'field.zmap'; export_zmap(r,path)
    grid=read_zmap(path)
    np.testing.assert_allclose(grid.values[[0,1,0,1,0],[0,0,1,1,2]],r.predicted)
    assert np.isnan(grid.values[1,2])
    np.testing.assert_array_equal(field_table(r).residual,-1)
    r.observations[0,0]=1.2
    with pytest.raises(ValueError,match='равномерную'):
        export_zmap(r,path)


def test_exports_reject_geometry_loss(tmp_path):
    r=result(); r.mesh.bounds=r.mesh.cells(); r.mesh.bounds[0,2]+=1
    with pytest.raises(ValueError,match='Обрезанные'):
        export_ubc(r,tmp_path/'bad.msh')
    r.observations[1]=r.observations[0]
    with pytest.raises(ValueError,match='повторяющиеся'):
        export_zmap(r,tmp_path/'bad.zmap')


@pytest.mark.parametrize('quantity',['J_parallel','J_magnitude','components'])
def test_magnetization_products_in_both_formats(tmp_path,quantity):
    from geogarmodel.inversion_export import magnetization_product
    from geogarmodel.gocad_voxet import write_voxet,read_voxet
    r=result('vector')
    r.mesh=Mesh([0,1,2],[0,1,2,3],[0,1,2],r.mesh.active)
    product=magnetization_product(r,quantity,[0,-1,0],.25)
    expected={'J_parallel':-r.model[:,1]-.25,'J_magnitude':np.linalg.norm(r.model,axis=1),'components':r.model}[quantity]
    export_ubc(product,tmp_path/'mag.msh')
    write_voxet(product,tmp_path/'mag.vo')
    loaded=read_voxet(tmp_path/'mag.vo')
    active=r.mesh.active
    np.testing.assert_allclose(loaded.values[active],expected[active],rtol=1e-6)
    for i,suffix in enumerate(['_JE','_JN','_JD'] if quantity=='components' else ['']):
        loaded=read_ubc(tmp_path/'mag.msh',tmp_path/('mag'+suffix+'.mod'),dummy=-1e32)
        np.testing.assert_allclose(loaded.values[active],expected[active,i] if quantity=='components' else expected[active])


def test_vector_ubc_and_vtu_reload_with_arbitrary_names(tmp_path):
    from geogarmodel.voxel import load_voxel
    from geogarmodel.stream_vtu import write_vtu
    r=result('vector')
    original=tmp_path/'old.msh'
    export_ubc(r,original)
    mesh=original.rename(tmp_path/'сетка.msh')
    parts=[(tmp_path/('old_'+name+'.mod')).rename(tmp_path/(str(i)+'.mod'))
           for i,name in enumerate(('JE','JN','JD'))]
    loaded=load_voxel(mesh,'vector',component_paths=parts,dummy=-1e32)
    np.testing.assert_array_equal(loaded.values,r.model)
    np.testing.assert_array_equal(loaded.mesh.active,r.mesh.active)
    pytest.importorskip('pyvista')
    r.mesh.bounds=r.mesh.cells()
    r.mesh.bounds[0,2]+=5
    path=tmp_path/'магнитный результат.vtu'
    write_vtu(r,path)
    restored=load_voxel(path,'vector')
    np.testing.assert_array_equal(restored.mesh.depth,[10,15,30,80])
    np.testing.assert_array_equal(restored.values[:3],[np.zeros(3),r.model[0],r.model[1]])
    assert not restored.mesh.active[0] and restored.mesh.active[1]
