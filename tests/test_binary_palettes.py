import struct
import zlib
import numpy as np
import pytest
from geogarmodel.grids import load_grid,Grid
from geogarmodel.palette_library import parse_palette


@pytest.mark.parametrize('compressed',[False,True])
def test_grd_scale_dummy_and_compression(tmp_path,compressed):
    header=bytearray(512); struct.pack_into('<5i7d',header,0,1028 if compressed else 4,2,3,2,1,100,200,1000,2000,0,10,2)
    raw=np.array([0,2,-1e32,4,6,8],dtype='<f4').tobytes()
    if compressed:
        block=zlib.compress(raw); raw=struct.pack('<4iqi',0,0,1,2,540,len(block)+16)+bytes(16)+block
    path=tmp_path/'test.grd'; path.write_bytes(header+raw); grid=load_grid(path)
    np.testing.assert_allclose(grid.x,[1000,1100,1200]); np.testing.assert_allclose(grid.y,[2000,2200])
    np.testing.assert_allclose(grid.values,[[10,11,np.nan],[12,13,14]],equal_nan=True)
    with pytest.raises(ValueError):
        (tmp_path/'bad.grd').write_bytes(bytes(512)); load_grid(tmp_path/'bad.grd')


def test_colour_models():
    cmyk=parse_palette('{blk cyn mag yel}\n0 255 0 0\n128 0 0 0','test.tbl')['colors']
    np.testing.assert_allclose(cmyk,[[0,1,1],[127/255]*3])
    zone=parse_palette('{zone cyn mag yel}\n1.2 255 255 0\n1.4 0 0 0','test.zon')
    assert zone['zones']==[1.2,1.4]; np.testing.assert_allclose(zone['colors'],[[0,0,1],[1,1,1]])
    lut=parse_palette('LookUpTable Begin\n0 65535 0 0\n1 0 65535 0','test.lut')
    np.testing.assert_allclose(lut['colors'],[[1,0,0],[0,1,0]])


def test_bundled_library_available_without_user_import(monkeypatch):
    import geogarmodel.palette_library as palettes
    class EmptySettings:
        def value(self,*args): return '{}'
    monkeypatch.setattr(palettes,'QSettings',lambda *args:EmptySettings())
    library=palettes.local_library(); bundled=palettes.bundled_palettes()
    assert len(bundled)==144
    assert {name.split('/')[0] for name in bundled}=={'heat map','geophysics','topography','monochromatic','miscellaneous'}
    assert library['geophysics/#GRAV.TBL']['colors']
    assert all(name in library and len(value['colors'])>=2 for name,value in bundled.items())


def test_zone_thresholds_and_portable_palette(tmp_path):
    from matplotlib.figure import Figure
    from geogarmodel.plotting import draw_map
    from geogarmodel.map_project import save_project,load_project
    values=np.array([[.5,1.5],[2.5,np.nan]])
    grid=Grid([0,1],[0,1],values,'zones')
    style=dict(cmap='external/custom.zon',palette_colors=[[0,0,1],[0,1,0],[1,0,0]],palette_zones=[1,2,3],zone_values=True)
    ax=draw_map(Figure(),grid,**style)
    np.testing.assert_array_equal(ax.images[0].norm(values)[:,:1],[[0],[2]])
    path=tmp_path/'colors.ggmaps'
    save_project(path,'Colors',[dict(kind='grid',name='zones',visible=True,style=style,data=grid)],None,{},'')
    _,rows=load_project(path); assert rows[0]['style']==style
    draw_map(Figure(),rows[0]['data'],**rows[0]['style'])


def test_fast_view_only_samples_visible_screen(qt_application_lifetime):
    from geogarmodel.app import MainWindow
    window=MainWindow(); window.resize(1100,800); window.show(); qt_application_lifetime.processEvents()
    x=np.arange(2200,dtype=float); grid=Grid(x,x,np.add.outer(x,x),'large','nT','EPSG:20918','m')
    window.add_grid(grid)
    image=window.fast_map.rasters[0][0]
    assert image.image.dtype==np.float32 and image.image.size<grid.values.size
    window.fast_map.setRange(xRange=(500,550),yRange=(600,650),padding=0)
    window.fast_map.update_rasters()
    assert image.image.size<=1600**2 and np.isfinite(image.image).all()
    # Display sampling must not modify the input grid.
    assert grid.values[600,500]==1100
    window.close()
