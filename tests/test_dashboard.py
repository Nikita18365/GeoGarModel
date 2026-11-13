import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PyQt5.QtWidgets import QApplication
from matplotlib.figure import Figure
from geogarmodel.dashboard import TrainingDashboard
from geogarmodel.demo import demo_survey
from geogarmodel.models import train_horizon
from geogarmodel.plotting import draw_map


def test_live_trials_diagnostics_and_exports(tmp_path):
    app = QApplication.instance() or QApplication([])
    dashboard = TrainingDashboard()
    _, table, _ = demo_survey()
    result = train_horizon(table, "Ridge", block_size=3000, optimize=True, n_trials=3,
                           search={"alpha": [.01, .1, 1]}, progress=dashboard.update_trial)
    assert dashboard.trials_table.rowCount() == 3
    dashboard.show_result(result, north_first=True)
    assert dashboard.parameter_table.rowCount() == 1
    assert len(dashboard.map_items) == 4
    points = dashboard.map_items[0][1].points()
    assert points[0].pos().x() > 10000000
    assert points[0].pos().y() < 10000000
    # Replacing results removes colourbars and their layout items.
    dashboard.show_result(result, north_first=True)
    assert len(dashboard.map_items) == 4
    path = tmp_path / "history.png"
    dashboard.export_plot("История Optuna", str(path))
    assert path.stat().st_size > 1000
    dashboard.reset()
    assert dashboard.trials_table.rowCount() == 0
    assert not dashboard.map_items
    dashboard.close()
    app.processEvents()


def test_map_restyles_without_recreating_raster():
    grid, _, _ = demo_survey()
    figure = Figure()
    ax = draw_map(figure, grid)
    raster = ax.images[0]
    limits = ax.get_xlim()
    second = draw_map(figure, grid, cmap="cividis", mode="symmetric")
    assert second is ax
    assert second.images[0] is raster
    assert len(figure.axes) == 2
    np.testing.assert_allclose(second.get_xlim(), limits)
