from pathlib import Path
import numpy as np
import pytest
from geogarmodel.inversion import Mesh,InversionResult
from geogarmodel.voxel import load_voxel
from geogarmodel.gocad_voxet import read_voxet,write_voxet


@pytest.mark.parametrize('mode',['gravity','induced','vector'])
def test_gocad_export_import(tmp_path,mode):
    mesh=Mesh([10,20,30],[100,103,106,109],[20,40,60])
    m=np.arange(mesh.size,dtype=float)+.123
    if mode=='vector': m=np.column_stack((m,-m,m*.5))
    mesh.bounds=mesh.cells()
    mesh.bounds[0,2]=30
    mesh.active[-1]=False
    r=InversionResult(mesh,m,[],[],[],[],dict(mode=mode,crs='EPSG:20918'),'test')
    path=tmp_path/'my result.vo'
    with pytest.raises(ValueError,match='Обрезанные'): write_voxet(r,path)
    policy='mass' if mode=='gravity' else 'keep'
    write_voxet(r,path,policy)
    other=load_voxel(path)
    assert other.crs=='EPSG:20918' and other.mode==mode
    for key in ('east','north','depth','active'):
        np.testing.assert_equal(getattr(other.mesh,key),getattr(mesh,key))
    expected=m.copy()
    if mode=='gravity': expected[0]*=.5
    expected[-1]=0
    np.testing.assert_allclose(other.values,expected,rtol=1e-7)
    binary=np.fromfile(path.with_suffix('.bin'),'>f4')
    assert binary[1]==pytest.approx(expected[6,0] if mode=='vector' else expected[6])


def test_independent_permuted_axes_and_offset(tmp_path):
    path=tmp_path/'foreign.vo'
    path.write_text('''GOCAD Voxet 0.01
AXIS_O 120 35 0
AXIS_U 0 0 10
AXIS_V 0 -2 0
AXIS_W 3 0 0
AXIS_MIN 1 0 0
AXIS_MAX 3 1 1
AXIS_N 3 2 2
PROPERTY 7 "Density"
PROP_FILE 7 "external data.bin"
PROP_OFFSET 7 4
PROP_ETYPE 7 IEEE
PROP_ESIZE 7 4
PROP_NO_DATA_VALUE 7 -9999
END
''')
    data=np.arange(12,dtype='>f4')
    data[2]=-9999
    (tmp_path/'external data.bin').write_bytes(b'head'+data.tobytes())
    m=read_voxet(path)
    np.testing.assert_equal(m.mesh.depth,[-35,-25,-15,-5])
    np.testing.assert_equal(m.mesh.north,[32,34,36])
    assert m.values.reshape(m.mesh.shape)[1,0,0]==11
    assert not m.mesh.active.reshape(m.mesh.shape)[0,1,0]
    path.write_text(path.read_text().replace('AXIS_W 3 0 0','AXIS_W 3 1 0'))
    with pytest.raises(ValueError,match='Поворот'): read_voxet(path)


def test_vector_resample_and_rename_voxet_bundle(tmp_path):
    from geogarmodel.gocad_voxet import resample_for_voxet
    mesh=Mesh([0,2,6],[0,3],[10,20,55])
    values=np.arange(mesh.size*3,dtype=float).reshape(-1,3)
    r=InversionResult(mesh,values,[],[],[],[],{'mode':'vector'},'test')
    uniform=resample_for_voxet(r)
    np.testing.assert_array_equal(uniform.mesh.depth,[10,32.5,55])
    np.testing.assert_array_equal(uniform.model,values[[1,1,3,3]])
    original=tmp_path/'original.vo'
    write_voxet(uniform,original)
    renamed=original.rename(tmp_path/'другое имя.vo')
    binary=original.with_suffix('.bin').rename(tmp_path/'свойства 123.bin')
    loaded=load_voxel(renamed,binary_paths={'original.bin':binary})
    assert loaded.mode=='vector'
    np.testing.assert_array_equal(loaded.values,uniform.model)
