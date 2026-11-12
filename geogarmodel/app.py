"""PyQt5 desktop workbench. Long computations run in a worker thread."""
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QFormLayout, QPushButton, QComboBox, QLineEdit, QCheckBox, QLabel,
    QFileDialog, QMessageBox, QListWidget, QSplitter, QPlainTextEdit, QDoubleSpinBox,
    QSpinBox, QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from .grids import load_grid, save_grid, points_to_grid
from .fields import TRANSFORMS, transform
from .models import read_table, available_models, train_horizon
from .plotting import draw_map
from .coordinates import gsk2011_zones, validate_crs


class Worker(QThread):
    result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.result.emit(self.function())
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoGarModel 2 · горизонты и потенциальные поля")
        self.resize(1350, 850)
        self.grids, self.overlays = [], []
        self.training = self.prediction = self.result = self.output = None
        self.worker = None
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.make_maps()
        self.make_training()
        menu = self.menuBar().addMenu("Данные")
        action = menu.addAction("Открыть синтетический пример")
        action.triggered.connect(lambda: self.guard(self.load_demo))
        self.statusBar().showMessage("Загрузите карту или обучающую таблицу.")

    def load_demo(self):
        if self.worker:
            raise ValueError("Дождитесь текущего расчёта.")
        from .demo import demo_survey
        grid, training, prediction = demo_survey()
        self.training, self.prediction = training, prediction
        self.result = self.output = None
        self.configure_columns()
        self.block.setValue(3000)
        self.table_crs.setCurrentText("EPSG:20910 | GSK-2011 / Gauss-Kruger zone 10")
        self.north_first.setChecked(True)
        self.table_info.setText("DEMO: 320 обучающих точек, 1280 прогнозных; глубина в метрах")
        self.add_grid(grid)

    def button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(lambda: self.guard(callback))
        layout.addWidget(button)
        return button

    def guard(self, function):
        try:
            function()
        except Exception as exc:
            QMessageBox.warning(self, "GeoGarModel", str(exc))

    def make_maps(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        plot_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        plot_layout.addWidget(self.canvas)
        controls = QWidget()
        panel = QVBoxLayout(controls)
        self.button("Открыть ZMAP / DAT / NPZ", self.open_grid, panel)
        self.button("Таблица XYZ → регулярная сетка", self.open_xyz, panel)
        self.layers = QListWidget()
        self.layers.currentRowChanged.connect(lambda _: self.guard(self.redraw))
        panel.addWidget(self.layers)
        self.button("CRS и единицы карты…", self.metadata, panel)
        self.button("Добавить SHP / GeoJSON", self.open_vector, panel)
        self.button("Убрать векторные наложения", self.clear_overlays, panel)
        self.button("Обрезать по текущему виду карты", self.crop_view, panel)
        form = QFormLayout()
        panel.addLayout(form)
        self.palette = QComboBox()
        self.palette.addItems(["viridis", "cividis", "RdBu_r", "coolwarm", "terrain", "seismic", "rainbow"])
        self.norm = QComboBox()
        for text, value in [("Линейная", "linear"), ("2–98 процентили", "percentile"), ("Симметрично около 0", "symmetric")]:
            self.norm.addItem(text, value)
        self.minimum, self.maximum, self.map_title = QLineEdit(), QLineEdit(), QLineEdit()
        self.contours, self.shade = QCheckBox("Изолинии"), QCheckBox("Отмывка")
        for label, widget in [("Палитра", self.palette), ("Нормировка", self.norm), ("Минимум (авто: пусто)", self.minimum),
                              ("Максимум", self.maximum), ("Название (авто: пусто)", self.map_title),
                              ("", self.contours), ("", self.shade)]:
            form.addRow(label, widget)
        self.button("Применить оформление", self.redraw, panel)
        self.operation = QComboBox()
        for key, text in TRANSFORMS.items():
            self.operation.addItem(text, key)
        self.height, self.wave = QDoubleSpinBox(), QDoubleSpinBox()
        for widget, value in [(self.height, 1000), (self.wave, 5000)]:
            widget.setRange(0, 1e8)
            widget.setValue(value)
        self.degree = QSpinBox()
        self.degree.setRange(1, 3)
        form.addRow("Трансформанта", self.operation)
        form.addRow("Высота, м", self.height)
        form.addRow("Длина волны фильтра, м", self.wave)
        form.addRow("Степень тренда", self.degree)
        self.transform_button = self.button("Рассчитать → новый слой", self.apply_transform, panel)
        self.button("Сохранить сетку NPZ", self.export_grid, panel)
        self.button("Экспорт XYZ CSV", self.export_xyz, panel)
        controls.setMaximumWidth(420)
        splitter = QSplitter()
        splitter.addWidget(plot)
        splitter.addWidget(controls)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Карты и трансформанты")

    def make_training(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.train_load = self.button("Обучающая таблица XLSX / CSV", self.open_training, row)
        self.pred_load = self.button("Прогнозная таблица XLSX / CSV", self.open_prediction, row)
        self.button("Просмотр / редактирование обучения", self.edit_training, row)
        self.button("Статистика обучения", self.show_statistics, row)
        self.table_info = QLabel("Формат по умолчанию: X, Y, признаки…, глубина")
        layout.addWidget(self.table_info)
        form = QFormLayout()
        layout.addLayout(form)
        self.x_column, self.y_column, self.target = QComboBox(), QComboBox(), QComboBox()
        self.feature_list = QListWidget()
        self.feature_list.setMaximumHeight(140)
        self.target.currentTextChanged.connect(self.update_features)
        self.x_column.currentTextChanged.connect(self.update_features)
        self.y_column.currentTextChanged.connect(self.update_features)
        self.model_choice = QComboBox()
        self.model_choice.addItems(available_models())
        self.model_choice.setCurrentText("Histogram Boosting")
        self.spatial = QCheckBox("Пространственные блоки (X/Y в метрах)")
        self.spatial.setChecked(True)
        self.block = QDoubleSpinBox()
        self.block.setRange(.001, 1e9)
        self.block.setValue(10000)
        self.test_fraction = QDoubleSpinBox()
        self.test_fraction.setRange(.05, .8)
        self.test_fraction.setSingleStep(.05)
        self.test_fraction.setValue(.2)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        self.parameters, self.search = QLineEdit("{}"), QLineEdit("{}")
        self.table_crs = QComboBox()
        self.table_crs.setEditable(True)
        for code, name in gsk2011_zones():
            self.table_crs.addItem(f"{code} | {name}")
        self.table_crs.setCurrentText("EPSG:20910 | GSK-2011 / Gauss-Kruger zone 10")
        self.north_first = QCheckBox("Таблица: X = север, Y = восток (геодезический порядок)")
        self.north_first.setChecked(True)
        self.parameters.setPlaceholderText('{"max_iter": 300}')
        self.search.setPlaceholderText('{"max_leaf_nodes": [15, 31]}')
        for label, widget in [("X", self.x_column), ("Y", self.y_column), ("Глубина / горизонт", self.target),
                              ("Признаки (отметьте нужные)", self.feature_list), ("CRS таблиц", self.table_crs),
                              ("Порядок координат", self.north_first), ("Метод", self.model_choice),
                              ("Валидация", self.spatial), ("Размер блока, м", self.block), ("Доля контроля", self.test_fraction),
                              ("Seed", self.seed), ("Параметры модели (JSON)", self.parameters), ("Сетка подбора (JSON, необязательно)", self.search)]:
            form.addRow(label, widget)
        actions = QHBoxLayout()
        layout.addLayout(actions)
        self.fit_button = self.button("Обучить и оценить", self.fit, actions)
        self.predict_button = self.button("Прогнозировать", self.predict, actions)
        self.button("Экспорт прогноза CSV", self.export_prediction, actions)
        self.button("Экспорт контроля CSV", self.export_validation, actions)
        self.button("Прогноз → карта", self.prediction_map, actions)
        self.button("Отчёт JSON", self.export_report, actions)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        self.tabs.addTab(tab, "Прогноз глубин горизонтов")

    def selected_grid(self):
        row = self.layers.currentRow()
        if row < 0:
            raise ValueError("Сначала загрузите карту.")
        return self.grids[row]

    def add_grid(self, grid):
        self.grids.append(grid)
        self.layers.addItem(grid.name)
        self.layers.setCurrentRow(len(self.grids) - 1)

    def open_grid(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Сетки", "", "Grid (*.zmap *.dat *.npz);;Все файлы (*)")
        for path in paths:
            self.add_grid(load_grid(path))

    def open_xyz(self):
        path, _ = QFileDialog.getOpenFileName(self, "Таблица: первые 3 столбца X Y поле", "", "Таблицы (*.xlsx *.csv *.txt)")
        if path:
            data = read_table(path)
            # Share the explicit table CRS and axis-order controls with horizon data.
            north_first = self.north_first.isChecked()
            crs = validate_crs(self.table_crs.currentText().split("|")[0].strip(), "m")
            self.add_grid(points_to_grid(data.iloc[:, 1 if north_first else 0],
                                        data.iloc[:, 0 if north_first else 1], data.iloc[:, 2],
                                        name=Path(path).stem, crs=crs, coordinate_unit="m"))

    def metadata(self):
        grid = self.selected_grid()
        dialog = QDialog(self)
        dialog.setWindowTitle("Метаданные: координаты должны быть в одной системе")
        form = QFormLayout(dialog)
        crs, unit = QLineEdit(grid.crs), QLineEdit(grid.unit)
        presets = QComboBox()
        presets.addItem("Выберите зону ГСК-2011…", "")
        for code, name in gsk2011_zones():
            presets.addItem(f"{code} | {name}", code)
        coord_unit = QComboBox()
        coord_unit.addItems(["unknown", "m", "degrees"])
        coord_unit.setCurrentText(grid.coordinate_unit)
        def use_preset():
            if presets.currentData():
                crs.setText(presets.currentData())
                coord_unit.setCurrentText("m")
        presets.currentIndexChanged.connect(use_preset)
        form.addRow("ГСК-2011 / Гаусса—Крюгера (6°)", presets)
        form.addRow("CRS, например EPSG:32647", crs)
        form.addRow("Единицы поля, например mGal / nT / m", unit)
        form.addRow("Единицы X и Y", coord_unit)
        form.addRow(QLabel("Изменение метаданных не пересчитывает координаты.\nДля FFT нужны плоские координаты в метрах."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_():
            checked = validate_crs(crs.text().strip(), coord_unit.currentText())
            grid.crs, grid.unit, grid.coordinate_unit = checked, unit.text().strip(), coord_unit.currentText()
            self.redraw()

    def open_vector(self):
        import geopandas as gpd
        grid = self.selected_grid()
        if not grid.crs:
            raise ValueError("Сначала задайте CRS карты.")
        path, _ = QFileDialog.getOpenFileName(self, "Векторный слой", "", "Vector (*.shp *.geojson)")
        if path:
            layer = gpd.read_file(path)
            if layer.crs is None:
                raise ValueError("У вектора отсутствует CRS.")
            layer.to_crs(grid.crs)  # validate before keeping it
            self.overlays.append(layer)
            self.redraw()

    def clear_overlays(self):
        self.overlays.clear()
        self.redraw()

    def crop_view(self):
        grid = self.selected_grid()
        if not self.figure.axes:
            raise ValueError("Сначала отобразите карту.")
        ax = self.figure.axes[0]
        self.add_grid(grid.crop(*ax.get_xlim(), *ax.get_ylim()))

    def redraw(self):
        if self.layers.currentRow() < 0:
            return
        draw_map(self.figure, self.selected_grid(), cmap=self.palette.currentText(), mode=self.norm.currentData(),
                 vmin=float(self.minimum.text()) if self.minimum.text().strip() else None,
                 vmax=float(self.maximum.text()) if self.maximum.text().strip() else None,
                 title=self.map_title.text() or None, contours=self.contours.isChecked(),
                 hillshade=self.shade.isChecked(), overlays=self.overlays)
        self.canvas.draw_idle()

    def run_job(self, function, callback):
        if self.worker is not None:
            raise ValueError("Дождитесь текущего расчёта.")
        self.worker = Worker(function, self)
        self.worker.result.connect(lambda value: self.guard(lambda: callback(value)))
        self.worker.failed.connect(lambda message: QMessageBox.warning(self, "Ошибка расчёта", message))
        self.worker.finished.connect(self.job_finished)
        for widget in (self.fit_button, self.predict_button, self.transform_button, self.train_load, self.pred_load):
            widget.setEnabled(False)
        self.statusBar().showMessage("Выполняется расчёт…")
        self.worker.start()

    def job_finished(self):
        self.worker.deleteLater()
        self.worker = None
        for widget in (self.fit_button, self.predict_button, self.transform_button, self.train_load, self.pred_load):
            widget.setEnabled(True)
        self.statusBar().showMessage("Расчёт завершён.")

    def apply_transform(self):
        grid, operation = self.selected_grid(), self.operation.currentData()
        args = dict(height=self.height.value(), wavelength=self.wave.value(), degree=self.degree.value())
        self.run_job(lambda: transform(grid, operation, **args), self.add_grid)

    def export_grid(self):
        grid = self.selected_grid()
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить сетку с метаданными", "", "NPZ (*.npz)")
        if path:
            save_grid(grid, path if path.lower().endswith(".npz") else path + ".npz")

    def export_xyz(self):
        import pandas as pd
        grid = self.selected_grid()
        xx, yy = np.meshgrid(grid.x, grid.y)
        self.save_csv(pd.DataFrame({"X": xx.ravel(), "Y": yy.ravel(), grid.name: grid.values.ravel()}), "Поле XYZ")

    def open_training(self):
        path, _ = QFileDialog.getOpenFileName(self, "Обучающая таблица", "", "Таблицы (*.xlsx *.csv *.txt)")
        if path:
            data = read_table(path)
            if len(data.columns) < 4:
                raise ValueError("Нужны X, Y, минимум один признак и глубина.")
            self.training, self.result, self.output = data, None, None
            self.configure_columns()
            self.table_info.setText(f"{Path(path).name}: {data.shape[0]} строк, {data.shape[1]} столбцов")

    def configure_columns(self):
        for combo in (self.x_column, self.y_column, self.target):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([str(c) for c in self.training.columns])
        self.x_column.setCurrentIndex(0)
        self.y_column.setCurrentIndex(1)
        self.target.setCurrentIndex(len(self.training.columns) - 1)
        for combo in (self.x_column, self.y_column, self.target):
            combo.blockSignals(False)
        self.update_features()

    def open_prediction(self):
        path, _ = QFileDialog.getOpenFileName(self, "Прогнозная таблица", "", "Таблицы (*.xlsx *.csv *.txt)")
        if path:
            self.prediction = read_table(path)
            self.output = None
            self.log.appendPlainText(f"Прогнозная таблица: {len(self.prediction)} строк.")

    def update_features(self):
        self.feature_list.clear()
        if self.training is None:
            return
        excluded = {self.x_column.currentText(), self.y_column.currentText(), self.target.currentText()}
        from PyQt5.QtWidgets import QListWidgetItem
        for column in self.training.columns:
            if column not in excluded:
                item = QListWidgetItem(str(column), self.feature_list)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)

    def edit_training(self):
        if self.training is None:
            raise ValueError("Загрузите обучающую таблицу.")
        if self.worker:
            raise ValueError("Редактирование доступно после расчёта.")
        dialog = QDialog(self)
        dialog.setWindowTitle("Обучающие данные — изменения в памяти")
        dialog.resize(900, 600)
        layout = QVBoxLayout(dialog)
        widget = QTableWidget(*self.training.shape)
        widget.setHorizontalHeaderLabels(list(self.training.columns))
        for i, row in enumerate(self.training.itertuples(index=False, name=None)):
            for j, value in enumerate(row):
                widget.setItem(i, j, QTableWidgetItem(str(value)))
        layout.addWidget(widget)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_():
            import pandas as pd
            edited = pd.DataFrame([[widget.item(i, j).text() for j in range(widget.columnCount())]
                                   for i in range(widget.rowCount())], columns=self.training.columns)
            self.training = edited.replace({"": np.nan, "nan": np.nan}).apply(pd.to_numeric, errors="raise")
            self.result = self.output = None

    def show_statistics(self):
        if self.training is None:
            raise ValueError("Загрузите обучающую таблицу.")
        self.log.setPlainText(self.training.describe().to_string())

    def fit(self):
        if self.training is None:
            raise ValueError("Загрузите обучающую таблицу.")
        features = [self.feature_list.item(i).text() for i in range(self.feature_list.count())
                    if self.feature_list.item(i).checkState() == Qt.Checked]
        params, search = json.loads(self.parameters.text()), json.loads(self.search.text())
        if not isinstance(params, dict) or not isinstance(search, dict):
            raise ValueError("Параметры и сетка подбора должны быть объектами JSON.")
        args = dict(target=self.target.currentText(), coordinates=[self.x_column.currentText(), self.y_column.currentText()],
                    features=features, test_size=self.test_fraction.value(), spatial=self.spatial.isChecked(),
                    block_size=self.block.value(), seed=self.seed.value(), parameters=params, search=search)
        table, name = self.training.copy(), self.model_choice.currentText()
        if self.spatial.isChecked():
            validate_crs(self.table_crs.currentText().split("|")[0].strip(), "m")
        self.fitted_crs = self.table_crs.currentText().split("|")[0].strip()
        self.fitted_north_first = self.north_first.isChecked()
        self.result = self.output = None
        self.run_job(lambda: train_horizon(table, name, **args), self.trained)

    def trained(self, result):
        self.result = result
        self.log.setPlainText(json.dumps({"model": result.model_name, "metrics": result.metrics,
                                         "best_parameters": result.settings["best_parameters"]}, ensure_ascii=False, indent=2))
        self.log.appendPlainText("Контроль: отдельные точки/блоки. Итоговая модель переобучена на всех размеченных данных.")

    def predict(self):
        if self.result is None or self.prediction is None:
            raise ValueError("Обучите модель и загрузите прогнозную таблицу.")
        result, table = self.result, self.prediction.copy()
        self.run_job(lambda: result.predict(table), self.predicted)

    def predicted(self, output):
        self.output = output
        self.log.appendPlainText(f"Получен прогноз: {len(output)} точек.\n{output.describe().to_string()}")

    def save_csv(self, data, title):
        if data is None:
            raise ValueError("Нет данных для экспорта.")
        path, _ = QFileDialog.getSaveFileName(self, title, "", "CSV (*.csv)")
        if path:
            data.to_csv(path, index=False, encoding="utf-8-sig")

    def export_prediction(self):
        self.save_csv(self.output, "Прогноз X Y глубина")

    def export_validation(self):
        self.save_csv(None if self.result is None else self.result.validation, "Контроль и невязки")

    def prediction_map(self):
        if self.output is None:
            raise ValueError("Сначала рассчитайте прогноз.")
        grid = points_to_grid(self.output.iloc[:, 1 if self.fitted_north_first else 0],
                              self.output.iloc[:, 0 if self.fitted_north_first else 1], self.output.iloc[:, 2],
                              name=self.output.columns[-1], unit="m", crs=self.fitted_crs, coordinate_unit="m")
        self.add_grid(grid)
        self.tabs.setCurrentIndex(0)

    def export_report(self):
        if self.result is None:
            raise ValueError("Сначала обучите модель.")
        path, _ = QFileDialog.getSaveFileName(self, "Отчёт обучения", "", "JSON (*.json)")
        if path:
            import sklearn
            report = dict(model=self.result.model_name, features=self.result.features, target=self.result.target,
                          coordinates=self.result.coordinates, metrics=self.result.metrics, settings=self.result.settings,
                          crs=self.fitted_crs, northing_first=self.fitted_north_first,
                          versions={"numpy": np.__version__, "sklearn": sklearn.__version__})
            Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def closeEvent(self, event):
        if self.worker is not None:
            QMessageBox.information(self, "Расчёт выполняется", "Дождитесь окончания расчёта перед закрытием.")
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
