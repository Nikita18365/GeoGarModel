import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PyQt5.QtWidgets import QApplication
from geogarmodel.app import MainWindow
from geogarmodel.grids import Grid
from geogarmodel.coordinates import gsk2011_zones, validate_crs


def test_gsk_zones():
    zones = dict(gsk2011_zones())
    for zone in range(10, 26):
        assert f"EPSG:{20900 + zone}" in zones
        assert validate_crs(f"EPSG:{20900 + zone}", "m")


def test_qt_map_styles_and_layers(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    xx, yy = np.meshgrid(np.arange(30) * 100., np.arange(25) * 100.)
    grid = Grid(xx[0], yy[:, 0], np.sin(xx / 300) + np.cos(yy / 500), "Synthetic", "nT", "EPSG:20910", "m")
    window.add_grid(grid)
    window.norm.setCurrentIndex(2)
    window.contours.setChecked(True)
    window.shade.setChecked(True)
    window.redraw()
    assert len(window.figure.axes) == 2  # map and exactly one colourbar
    window.redraw()
    assert len(window.figure.axes) == 2
    assert len(window.grids) == 1
    window.figure.savefig(tmp_path / "map.png")
    window.close()
    app.processEvents()


def test_background_training_and_prediction():
    from PyQt5.QtCore import QEventLoop, QTimer
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.load_demo()
    window.model_choice.setCurrentText("Ridge")
    def finish_job():
        loop = QEventLoop()
        window.worker.finished.connect(loop.quit)
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(30000)
        loop.exec_()
        assert window.worker is None
    window.fit()
    finish_job()
    assert window.result.metrics["RMSE"] < 20
    window.predict()
    finish_job()
    assert len(window.output) == 1280
    window.prediction_map()
    assert window.selected_grid().crs == "EPSG:20910"
    assert window.selected_grid().x.min() > 10000000  # easting, not northing
    window.close()
    app.processEvents()
