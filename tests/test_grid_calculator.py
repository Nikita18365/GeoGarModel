import numpy as np
import pytest
from PyQt5.QtWidgets import QDialog

from geogarmodel.app import MainWindow
from geogarmodel.grids import Grid
from geogarmodel.grid_calculator import calculate_grids, GridFormula, operand_name
from geogarmodel.grid_calculator_ui import GridCalculatorDialog
from geogarmodel.map_project import Vector


def grid(name,values,x=(0,1,2),y=(0,1),crs='EPSG:3857',unit='nT'):
    return Grid(x,y,values,name,unit,crs,'m')


def inputs():
    a=grid('a',[[1,2,np.nan],[4,0,6]])
    b=grid('b',[[2,3,5],[0,2,3]])
    c=grid('c',[[1,1,1],[2,1,2]])
    return a,b,c


@pytest.mark.parametrize('operation,expected',[
    ('SUM',[[4,6,np.nan],[6,3,11]]),
    ('SUB',[[-2,-2,np.nan],[2,-3,1]]),
    ('MUL',[[2,6,np.nan],[0,0,36]]),
    ('DIV',[[.5,2/3,np.nan],[np.nan,0,1]]),
    ('MEAN',[[4/3,2,np.nan],[2,1,11/3]]),
    ('MIN',[[1,1,np.nan],[0,0,2]]),
    ('MAX',[[2,3,np.nan],[4,2,6]]),
    ('AND',[[1,1,np.nan],[0,0,1]]),
    ('OR',[[1,1,np.nan],[1,1,1]]),
    ('XOR',[[1,1,np.nan],[0,0,1]]),
])
def test_presets(operation,expected):
    a,b,c=inputs()
    original=a.values.copy()
    result=calculate_grids([a,b,c],operation)
    np.testing.assert_allclose(result.values,expected,equal_nan=True)
    np.testing.assert_array_equal(a.values,original)
    np.testing.assert_array_equal(result.x,a.x)
    assert result.name.startswith(operation+'_')
    assert result.history[-1]['operands']=={'A':'a','B':'b','C':'c'}
    assert result.crs==a.crs
    assert not np.shares_memory(result.values,a.values)


def test_unary_and_formula():
    a,b,c=inputs()
    np.testing.assert_allclose(calculate_grids([a],'NOT').values,[[0,0,np.nan],[0,1,0]],equal_nan=True)
    np.testing.assert_allclose(calculate_grids([a],'SQRT').values,np.sqrt(a.values),equal_nan=True)
    result=calculate_grids([a,b,c],'FORMULA',expression='where((A>1) & (B!=0), A/B, C)')
    np.testing.assert_allclose(result.values,[[1,2/3,np.nan],[2,1,2]],equal_nan=True)
    result=calculate_grids([a,b],'FORMULA',expression='(A/B)>1')
    assert np.isnan(result.values[1,0])
    assert np.isnan(result.values[0,2])
    assert result.values[1,2]==1
    result=calculate_grids([a,b,c],'FORMULA',expression='sum(A,B,C)/3')
    np.testing.assert_allclose(result.values,calculate_grids([a,b,c],'MEAN').values,equal_nan=True)


@pytest.mark.parametrize('expression',['__import__("os")','A.__class__','A[0]','lambda: A',
                                      'open("file")','D+1','A+1e999','42','where(A, B)'])
def test_formula_rejects_unsafe_or_unknown(expression):
    with pytest.raises(ValueError):
        GridFormula(expression,['A','B','C'])


def test_no_fixed_23_operand_limit():
    grids=[grid(str(i),np.full((2,3),i+1)) for i in range(30)]
    assert operand_name(25)=='Z' and operand_name(26)=='AA'
    result=calculate_grids(grids,'SUM')
    np.testing.assert_array_equal(result.values,np.full((2,3),465))
    result=calculate_grids(grids,'FORMULA',expression='A+AA+AD')
    np.testing.assert_array_equal(result.values,np.full((2,3),58))


def test_alignment_and_no_extrapolation():
    a=grid('a',np.ones((3,4)),x=(0,1,2,3),y=(0,1,2))
    b=grid('b',[[0,2],[2,4]],x=(0,2),y=(0,2))
    with pytest.raises(ValueError,match='различаются'):
        calculate_grids([a,b],'SUM')
    result=calculate_grids([a,b],'SUM',alignment='linear')
    np.testing.assert_allclose(result.values,[[1,2,3,np.nan],[2,3,4,np.nan],[3,4,5,np.nan]],equal_nan=True)
    np.testing.assert_array_equal(result.x,a.x)
    b.crs='EPSG:4326'
    with pytest.raises(ValueError,match='СК'):
        calculate_grids([a,b],'SUM',alignment='linear')


def test_exact_node_not_contaminated_by_zero_weight_nan():
    a=grid('a',np.ones((3,3)),x=(0,1,2),y=(0,1,2))
    b=grid('b',[[5,np.nan],[7,9]],x=(0,2),y=(0,2))
    result=calculate_grids([a,b],'SUM',alignment='linear')
    np.testing.assert_allclose(result.values,[[6,np.nan,np.nan],[7,np.nan,np.nan],[8,9,10]],equal_nan=True)


def test_units_and_all_invalid():
    a,b,_=inputs()
    assert calculate_grids([a,b],'SUM').unit=='nT'
    assert calculate_grids([a,b],'AND').unit=='1'
    b.unit='mGal'
    assert calculate_grids([a,b],'SUM').unit==''
    assert calculate_grids([a,b],'MUL',unit='custom').unit=='custom'
    with pytest.raises(ValueError,match='конечных узлов'):
        calculate_grids([a],'FORMULA',expression='A/0')
    with pytest.raises(ValueError):
        calculate_grids([a],'SUM')


def test_calculator_ui_result_and_project_roundtrip(tmp_path,monkeypatch,qt_application_lifetime):
    window=MainWindow()
    try:
        a,b,c=inputs()
        window.add_grid(c); window.add_grid(b); window.add_grid(a)
        window.add_map_layer(dict(kind='vector',name='line.shp',visible=True,style={},
                                  data=Vector([dict(type='LineString',coordinates=[[0,0],[2,1]])],a.crs)))
        window.layers.setCurrentRow(1)
        def select(dialog):
            assert isinstance(dialog,GridCalculatorDialog)
            assert dialog.grids[0].name=='a'
            assert len(dialog.grids)==3
            dialog.count.setValue(3)
            dialog.operation.setCurrentIndex(dialog.operation.findData('SUM'))
            dialog.suffix.setText('test')
            return QDialog.Accepted
        monkeypatch.setattr(GridCalculatorDialog,'exec_',select)
        monkeypatch.setattr(window,'run_job',lambda fn,callback,**kw:callback(fn()))
        window.open_grid_calculator()
        result=window.selected_grid()
        assert result.name=='SUM_test'
        np.testing.assert_allclose(result.values,[[4,6,np.nan],[6,3,11]],equal_nan=True)
        assert window.layers.count()==5
        path=tmp_path/'calc.ggproject'
        window.save_project_to(path)
        assert window.open_project_from(path)
        result=window.layers.item(0).data(256)['data']
        assert result.name=='SUM_test'
        assert result.history[-1]['preset']=='SUM'
        np.testing.assert_allclose(result.values,[[4,6,np.nan],[6,3,11]],equal_nan=True)
    finally:
        window.close()
