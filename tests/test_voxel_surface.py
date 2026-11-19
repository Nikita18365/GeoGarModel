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


def test_upper_percentage_follows_local_roof_and_skips_voids():
    from geogarmodel.voxel_surface import upper_prism_limits
    mask=np.zeros((2,2,100),bool)
    mask[0,0,20:]=True
    mask[1,0,[40,60,80]]=True
    mask[1,1,5]=True
    for percent,expected in [(0,[0,0,0,0]),(2,[22,0,41,6]),(50,[60,0,61,6])]:
        np.testing.assert_array_equal(upper_prism_limits(mask,mask.shape,percent),expected)
    end=upper_prism_limits(mask,mask.shape,100)
    np.testing.assert_array_equal(mask & (np.arange(100)<end.reshape(2,2,1)),mask)


def test_prism_layer_cut_reveals_exact_properties():
    pytest.importorskip('pyvista')
    mesh=Mesh(np.arange(4.),np.arange(4.),np.arange(8.))
    mesh.active[:3]=False
    original=mesh.active.copy()
    result=VoxelModel(mesh,np.arange(mesh.size,dtype=float)).result()
    interior=np.ravel_multi_index((1,1,3),mesh.shape)
    full=exterior(result,'amplitude',0,prism_shell=True)
    cut=exterior(result,'amplitude',0,prism_shell=True,first_layer=3)
    assert interior not in full['voxel_id'] and interior in cut['voxel_id']
    np.testing.assert_array_equal(cut['property'],result.model[cut['voxel_id']])
    assert cut.bounds[-1]==-3
    np.testing.assert_array_equal(mesh.active,original)
