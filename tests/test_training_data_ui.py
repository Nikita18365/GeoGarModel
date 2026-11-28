import csv
import numpy as np
import pandas as pd
from pyproj import CRS
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem
from geogarmodel.training_data_ui import FeaturePanel, TableCRSSelector
from geogarmodel.training_statistics import StatisticsDialog, summarize_table, format_stat
from geogarmodel.crs_editor import CRSEditor, false_easting
from geogarmodel.project_ui import controls_state, apply_controls


def test_statistics_numeric_values_precision_and_sources(qt_application_lifetime, tmp_path, monkeypatch):
    frame = pd.DataFrame({"X": [10., 20., np.nan, np.inf],
                          "feature": ["0.001", "0.003", None, "bad"],
                          "line": ["A", "B", "C", "D"]})
    original = frame.copy(deep=True)
    rows = summarize_table(frame)
    assert rows[0][:4] == ("X", 2, 1, 1)
    assert rows[0][4] == 15.
    assert rows[1][:4] == ("feature", 2, 1, 1)
    assert rows[2][1:4] == (0, 0, 4)
    assert rows[2][4] is None
    assert format_stat(1.2, 4) == "1.2000"
    assert format_stat(0.00001, 4, True) == "1.0000e-05"
    dialog = StatisticsDialog({"Обучение": frame, "Прогноз": pd.DataFrame({"z": [3.5, 4.5]})})
    try:
        dialog.search.setText("feature")
        row = next(i for i in range(dialog.table.rowCount()) if not dialog.table.isRowHidden(i))
        assert dialog.table.item(row, 4).text() == "0.0020"
        dialog.precision.setValue(6)
        row = next(i for i in range(dialog.table.rowCount()) if not dialog.table.isRowHidden(i))
        assert dialog.table.item(row, 4).text() == "0.002000"
        assert dialog.histogram.listDataItems()
        destination = tmp_path / "statistics.csv"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(destination), "CSV"))
        dialog.export_csv()
        with destination.open(encoding="utf-8-sig", newline="") as stream:
            exported = list(csv.reader(stream))
        assert len(exported) == 2 and exported[1][0] == "feature"
        assert float(exported[1][4]) == 0.002
        dialog.search.clear()
        dialog.table.sortItems(1, Qt.DescendingOrder)
        assert dialog.table.item(0, 1).text() == "2"
        dialog.source.setCurrentText("Прогноз")
        assert dialog.table.rowCount() == 1
        assert dialog.table.item(0, 4).text() == "4.000000"
        pd.testing.assert_frame_equal(frame, original)
    finally:
        dialog.close()


def test_feature_shift_range_filter_and_batch_actions(qt_application_lifetime):
    panel = FeaturePanel()
    for name in ("Mag 1", "Grav 1", "Mag 2", "Mag 3"):
        item = QListWidgetItem(name, panel.items)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
    panel.resize(480, 470)
    panel.show()
    qt_application_lifetime.processEvents()
    items = panel.items
    try:
        items.item(0).setCheckState(Qt.Unchecked)
        first = items.visualItemRect(items.item(0)).center()
        last = items.visualItemRect(items.item(2)).center()
        QTest.mouseClick(items.viewport(), Qt.LeftButton, Qt.NoModifier, first)
        QTest.mouseClick(items.viewport(), Qt.LeftButton, Qt.ShiftModifier, last)
        assert [items.item(i).checkState() for i in range(4)] == [
            Qt.Unchecked, Qt.Unchecked, Qt.Unchecked, Qt.Checked]
        panel.search.setText("Mag")
        panel.set_checked(Qt.Checked)
        assert items.item(1).isHidden() and items.item(1).checkState() == Qt.Unchecked
        assert "3 / 4" in panel.counter.text()
        panel.invert_visible()
        assert all(items.item(i).checkState() == Qt.Unchecked for i in range(4))
        panel.search.clear()
        assert not any(items.item(i).isHidden() for i in range(4))
        items.clear()
        assert "0 / 0" in panel.counter.text()
    finally:
        panel.close()


def test_table_crs_shared_editor_import_and_legacy_project(qt_application_lifetime, tmp_path, monkeypatch):
    path = tmp_path / "local.prj"
    path.write_text(CRS("EPSG:20917").to_wkt(), encoding="utf-8")
    editor = CRSEditor("EPSG:20910")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), "PRJ"))
    editor.import_prj()
    assert CRS(editor.value()) == CRS("EPSG:20917")
    editor.fields[3].setValue(500000)
    custom = editor.value()
    assert false_easting(custom) == 500000
    class Owner:
        pass
    owner = Owner()
    owner.table_crs = TableCRSSelector()
    owner.table_crs.setValue(custom)
    saved = controls_state(owner, ("table_crs",))
    assert false_easting(saved["table_crs"]) == 500000
    owner.table_crs.setValue("EPSG:20910")
    apply_controls(owner, saved)
    assert CRS(owner.table_crs.value()) == CRS(custom)
    apply_controls(owner, {"table_crs": "EPSG:20917 | GSK-2011 / Gauss-Kruger zone 17"})
    assert owner.table_crs.value() == "EPSG:20917"
