import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from geogarmodel.app import MainWindow
from geogarmodel.ui_layout import ParameterSplitter


def test_parameter_panels_drag_collapse_and_reopen(qt_application_lifetime):
    window=MainWindow()
    window.resize(1400,900)
    window.show()
    try:
        for index in range(3):
            window.tabs.setCurrentIndex(index)
            qt_application_lifetime.processEvents()
            splitter=window.tabs.currentWidget().findChild(ParameterSplitter)
            toggle=splitter.handle(1).toggle
            for fraction in (.08,.90):
                splitter.moveSplitter(round(splitter.width()*fraction),1)
                left,right=splitter.sizes()
                assert left/(left+right)==pytest.approx(fraction,abs=.03)
            splitter.moveSplitter(splitter.width(),1)
            assert splitter.sizes()[1]==0 and toggle.isVisible()
            QTest.mouseClick(toggle,Qt.LeftButton)
            assert splitter.sizes()[0]==0 and toggle.isVisible()
            QTest.mouseClick(toggle,Qt.LeftButton)
            left,right=splitter.sizes()
            assert left/(left+right)==pytest.approx(.35,abs=.015)
    finally:
        window.close()
