import json
from zipfile import ZipFile
import numpy as np
import pytest
from geogarmodel.inversion import Mesh,InversionResult
from geogarmodel.model_archive import ModelArchive,write_archive
from geogarmodel.inversion_export import export_ubc,oasis_package
from geogarmodel.stream_vtu import write_vtu
from geogarmodel.voxel import load_voxel


def sample(vector=False):
    mesh=Mesh([10,11,13],[20,23,27],[10,20,40])
    mesh.bounds=mesh.cells()
    mesh.bounds[0,2]=15
    mesh.active[-1]=False
    m=np.arange(1,25,dtype=float).reshape(-1,3) if vector else np.arange(1,9,dtype=float)
    return InversionResult(mesh,m,np.array([[10.,20.,0.]]),np.array([9.]),np.array([8.]),
                           [],{'mode':'vector' if vector else 'gravity','crs':'EPSG:20918'},'iteration_limit')


@pytest.mark.parametrize('vector',[True,False])
def test_archive_and_lazy_layer(tmp_path,vector):
    r=sample(vector)
    path=tmp_path/'result.ggmodel'
    write_archive(path,r)
    with np.load(path) as f:
        assert f['ids0'].tolist()==[0]
    archive=ModelArchive(path)
    m,active=archive.layer(1)
    np.testing.assert_equal(m,r.model.reshape(r.mesh.shape+r.model.shape[1:])[:,:,1])
    np.testing.assert_equal(active,r.mesh.active.reshape(r.mesh.shape)[:,:,1])
    loaded=archive.load()
    for field in ('model','observed','predicted','observations'):
        np.testing.assert_equal(getattr(loaded,field),getattr(r,field))
    np.testing.assert_equal(loaded.mesh.bounds,r.mesh.bounds)
    np.testing.assert_equal(load_voxel(path).mesh.active,r.mesh.active)


def test_mass_and_oasis(tmp_path):
    r=sample()
    folder=tmp_path/'проект.oasis'
    oasis_package(r,folder,'mass')
    meta=json.loads((folder/'model.json').read_text())
    assert meta['clipped_cells']==1
    values=np.loadtxt(folder/'model.mod')
    assert values[0]==.5 and values[-1]==-1e32
    assert 'model.geosoft_voxel' in (folder/'convert.gs').read_text()
    r.mesh.bounds[0,2]=10
    export_ubc(r,tmp_path/'exact.msh')


def test_stream_vtu(tmp_path):
    pv=pytest.importorskip('pyvista')
    r=sample(True)
    path=tmp_path/'exact.vtu'
    write_vtu(r,path)
    grid=pv.read(path)
    assert grid.n_cells==7
    assert grid.get_cell(0).bounds[-2:]==(-20.,-15.)
    np.testing.assert_equal(grid.cell_data['M_END'],r.model[:-1])
    np.testing.assert_allclose(grid.cell_data['property'],np.linalg.norm(r.model[:-1],axis=1))
