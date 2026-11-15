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
