import numpy as np
import pytest
from geogarmodel.inversion import Mesh
from geogarmodel.voxel import VoxelModel
from geogarmodel.voxel_surface import exterior


def test_full_surface_uses_every_cell_property_and_threshold():
    pytest.importorskip('pyvista')
    mesh=Mesh(np.arange(6.),np.arange(5.),np.arange(4.))
    vals=np.arange(mesh.size,dtype=float)
    r=VoxelModel(mesh,vals).result()
    grid=exterior(r,'amplitude',0)
    assert grid.n_cells==2*(5*4+5*3+4*3)
    assert grid.bounds==(0,5,0,4,-3,0)
    assert grid['property'].max()==vals.max()
    chosen=exterior(r,'amplitude',99)
    assert chosen.n_cells==6
    np.testing.assert_equal(chosen['property'],np.full(6,59.))


def test_partial_prism_exposes_neighbour_wall():
    pytest.importorskip('pyvista')
    mesh=Mesh([0.,1.,2.],[0.,1.],[0.,1.])
    mesh.bounds=mesh.cells()
    mesh.bounds[0,2]=.5
    grid=exterior(VoxelModel(mesh,np.array([1.,2.])).result(),'amplitude',0)
    assert grid.n_cells==11
    assert -.5 in grid.points[:,2]
