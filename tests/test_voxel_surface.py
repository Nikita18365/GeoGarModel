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


def test_continuous_cap_joins_walls_and_preserves_column_properties():
    pytest.importorskip('pyvista')
    mesh=Mesh(np.arange(5.),np.arange(5.),np.arange(7.))
    top=np.array([[0,0,1,1],[0,1,1,2],[1,1,2,2],[1,2,2,3]])
    mesh.active[:]=(np.arange(6)>=top[:,:,None]).ravel()
    m=VoxelModel(mesh,np.arange(mesh.size,dtype=float))
    result=m.result()
    grid=exterior(result,'amplitude',0,smooth_top=True)
    center=grid.points[(grid.points[:,0]==2)&(grid.points[:,1]==2)]
    assert center[:,2].max()==-top[1:3,1:3].mean()
    assert grid.n_open_edges==0
    exaggerated=exterior(result,'amplitude',0,smooth_top=True,relief_gain=5.)
    assert exaggerated.bounds[-2]==grid.bounds[-2]
    assert exaggerated.n_open_edges==0
    np.testing.assert_allclose(exaggerated.bounds[-1]-grid.bounds[-1],4*(top.max()-top.min()))
    assert grid.bounds[-2]==-6.
    np.testing.assert_equal(mesh.active.reshape(mesh.shape),np.arange(6)>=top[:,:,None])
    for i,j in np.ndindex(top.shape):
        assert result.model[np.ravel_multi_index((i,j,top[i,j]),mesh.shape)] in grid['property']


def test_terrain_protects_thin_columns_and_does_not_invent_relief():
    from geogarmodel.voxel_surface import terrain_nodes
    active=np.ones((2,2,6),bool)
    assert terrain_nodes(active,active.shape,np.arange(7.)) is None
    active[0,0,:3]=False
    active[1,1,1:]=False
    _,last,*_=terrain_nodes(active,active.shape,np.arange(7.))
    assert last[1,1]==-1
