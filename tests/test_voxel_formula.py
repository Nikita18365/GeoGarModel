import numpy as np
import pytest

from geogarmodel.inversion import Mesh
from geogarmodel.voxel import VoxelModel
from geogarmodel.voxel_formula import Formula, calculate


@pytest.mark.parametrize('mode',['gravity','vector'])
def test_formula_mask_depth_and_two_operands(mode):
    grid = Mesh([0,1,2],[0,1],[0,10,20],active=[1,1,0,1])
    values = np.array([1.,-2,0,4])
    if mode=='vector':
        values = np.column_stack([values,2*values,-values])
    a = VoxelModel(grid,values,mode)
    b = a.copy()
    b.values[1] = 0
    out = calculate(a,Formula('where(Z >= 5 and B != 0, A / B + A * B, 0)'),b)
    expected = np.where(b.values!=0,1+values*b.values,0)
    np.testing.assert_array_equal(out.values,expected)
    np.testing.assert_array_equal(a.values,values)
    np.testing.assert_array_equal(out.mesh.active,grid.active)


def test_bad_formula_and_incompatible_geometry():
    a = VoxelModel(Mesh([0,1],[0,1],[0,10]),[2])
    b = a.copy()
    b.mesh.bounds = b.mesh.cells()
    b.mesh.bounds[0,2] = 5
    assert calculate(b,Formula('Z')).values[0]==7.5
    with pytest.raises(ValueError,match='границы'):
        calculate(a,Formula('A+B'),b)
    with pytest.raises(ValueError):
        calculate(a,Formula('A / 0'))
    with pytest.raises(ValueError):
        Formula('A.mean()')
    assert calculate(a,Formula('1 <= A < 3')).values[0]==1


def test_editor_formula_loads_b_and_supports_undo(qt_application_lifetime,monkeypatch):
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication,QDialogButtonBox,QComboBox
    from geogarmodel.app import MainWindow
    window=MainWindow()
    panel=window.inversion.initial_panel
    a=VoxelModel(Mesh([0,1],[0,1],[0,1,2]),[1,3])
    b=VoxelModel(a.mesh,[2,5])
    panel.assign(a)
    monkeypatch.setattr(panel,'ask_load',lambda *args: lambda:b)
    monkeypatch.setattr(panel,'preview',lambda:None)
    def submit():
        dialog=QApplication.activeModalWidget()
        dialog.findChild(QComboBox).setEditText('where(B > 2, A + B, A)')
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
    try:
        QTimer.singleShot(0,submit)
        panel.formula()
        np.testing.assert_array_equal(panel.model.values,[1,8])
        panel.undo()
        np.testing.assert_array_equal(panel.model.values,[1,3])
    finally:
        window.close()
