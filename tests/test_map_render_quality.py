import numpy as np
from geogarmodel.grids import Grid
from geogarmodel.map_sampling import sample_display


def test_bilinear_values_and_preserved_holes():
    axis=np.arange(5.)
    grid=Grid(axis,axis,axis[None,:]*10+axis[:,None])
    original=grid.values.copy()
    image,bounds=sample_display(grid,(.5,3.5,.5,3.5),6,6)
    expected=np.arange(.75,3.5,.5)
    np.testing.assert_allclose(image,10*expected[None,:]+expected[:,None])
    assert bounds==(.5,3.5,.5,3.5)
    np.testing.assert_array_equal(grid.values,original)
    grid.values[2,2]=np.nan
    image,_=sample_display(grid,(-.5,4.5,-.5,4.5),50,50)
    assert np.isnan(image[20:30,20:30]).all()
    assert np.isfinite(image[10:20,10:20]).all()


def test_minification_suppresses_checkerboard_aliasing():
    axis=np.arange(128.)
    checker=(np.indices((128,128)).sum(axis=0)%2).astype(float)
    grid=Grid(axis,axis,checker)
    image,_=sample_display(grid,(-.5,127.5,-.5,127.5),32,32)
    np.testing.assert_allclose(image,.5)
    assert sample_display(grid,(200,300,200,300),32,32) is None


def test_white_sheet_export_and_cached_interpolation(tmp_path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgba
    from geogarmodel.plotting import draw_map
    figure=Figure(figsize=(5,4),dpi=120); canvas=FigureCanvasAgg(figure)
    grid=Grid([0,1],[0,1],[[0,1],[2,3]])
    ax=draw_map(figure,grid)
    assert figure.get_facecolor()==to_rgba('white') and ax.get_facecolor()==to_rgba('white')
    assert ax.xaxis.label.get_color()=='#222222'
    assert ax.images[0].get_interpolation()=='bilinear'
    assert ax.images[0].get_interpolation_stage()=='data'
    draw_map(figure,grid,smooth=False)
    assert ax.images[0].get_interpolation()=='nearest'
    draw_map(figure,grid,smooth=True)
    canvas.draw(); np.testing.assert_array_equal(np.asarray(canvas.buffer_rgba())[0,0],[255,255,255,255])
    figure.savefig(tmp_path/'map.png'); figure.savefig(tmp_path/'map.pdf')
