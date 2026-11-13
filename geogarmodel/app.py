"""PyQt5 desktop workbench. Long computations run in a worker thread."""
import json
from pathlib import Path
import sys
import traceback
from threading import Event
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QFormLayout, QPushButton, QComboBox, QLineEdit, QCheckBox, QLabel,
    QFileDialog, QMessageBox, QListWidget, QSplitter, QPlainTextEdit, QDoubleSpinBox,
    QSpinBox, QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem,
    QScrollArea, QGroupBox, QProgressBar, QGridLayout, QSizePolicy)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from .grids import load_grid, save_grid, points_to_grid
from .fields import TRANSFORMS, transform
from .models import read_table, available_models, train_horizon, make_model
from .plotting import draw_map
from .coordinates import gsk2011_zones, validate_crs
from .dashboard import TrainingDashboard
from .theme import STYLE
from .compute import GPU_MODELS, ComputationCancelled
from .optimization import default_space
from .parameter_files import validate_parameters, read_parameters, best_snapshot
from .window_theme import apply_window_theme


class Worker(QThread):
    result = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.cancel_event = Event()

    def run(self):
        try:
            self.result.emit(self.function())
        except ComputationCancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoGarModel 2.1 · интерпретация и моделирование")
        self.resize(1480, 940)
        self.setStyleSheet(STYLE)
        self.grids, self.overlays = [], []
        self.training = self.prediction = self.result = self.output = None
        self.worker = None
        self.job_outcome = ""
        self.optimization_context = None
        self.tabs = QTabWidget()
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 0, 12, 10)
        header = QWidget()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        logo = QLabel("G")
        logo.setObjectName("logo")
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(46, 46)
        header_layout.addWidget(logo)
        brand_layout = QVBoxLayout()
        brand = QLabel("GeoGarModel")
        brand.setObjectName("brand")
        subtitle = QLabel("Геофизическая интерпретация · горизонты · потенциальные поля")
        subtitle.setObjectName("muted")
        brand_layout.addWidget(brand)
        brand_layout.addWidget(subtitle)
        header_layout.addLayout(brand_layout)
        header_layout.addStretch()
        self.compute_badge = QLabel("CPU  /  OPTUNA TPE")
        self.compute_badge.setObjectName("muted")
        header_layout.addWidget(self.compute_badge)
        root_layout.addWidget(header)
        root_layout.addWidget(self.tabs)
        self.setCentralWidget(root)
        self.make_maps()
        self.make_training()
        menu = self.menuBar().addMenu("Данные")
        action = menu.addAction("Открыть синтетический пример")
        action.triggered.connect(lambda: self.guard(self.load_demo))
        self.statusBar().showMessage("Загрузите карту или обучающую таблицу.")

    def showEvent(self, event):
        super().showEvent(event)
        apply_window_theme(self)

    def load_demo(self):
        if self.worker:
            raise ValueError("Дождитесь текущего расчёта.")
        from .demo import demo_survey
        grid, training, prediction = demo_survey()
        self.training, self.prediction = training, prediction
        self.result = self.output = None
        self.dashboard.reset("Синтетический пример загружен. Выберите модель и запустите расчёт.")
        self.progress_bar.setValue(0)
        self.configure_columns()
        self.block.setValue(3000)
        self.table_crs.setCurrentText("EPSG:20910 | GSK-2011 / Gauss-Kruger zone 10")
        self.table_crs.lineEdit().setCursorPosition(0)
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
        controls.setMinimumWidth(320)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(350)
        scroll.setMaximumWidth(470)
        splitter = QSplitter()
        splitter.addWidget(plot)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "01  ·  Карты и трансформанты")

    def make_training(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        controls = QWidget()
        panel = QVBoxLayout(controls)
        data_group = QGroupBox("01 · Данные и признаки")
        data_layout = QVBoxLayout(data_group)
        self.train_load = self.button("Загрузить обучение · XLSX / CSV", self.open_training, data_layout)
        self.pred_load = self.button("Загрузить прогноз · XLSX / CSV", self.open_prediction, data_layout)
        row = QHBoxLayout()
        data_layout.addLayout(row)
        self.button("Таблица", self.edit_training, row)
        self.button("Статистика", self.show_statistics, row)
        self.table_info = QLabel("Формат по умолчанию: X, Y, признаки…, глубина")
        self.table_info.setWordWrap(True)
        self.table_info.setObjectName("muted")
        data_layout.addWidget(self.table_info)
        form = QFormLayout()
        data_layout.addLayout(form)
        self.x_column, self.y_column, self.target = QComboBox(), QComboBox(), QComboBox()
        self.feature_list = QListWidget()
        self.feature_list.setFixedHeight(100)
        self.target.currentTextChanged.connect(self.update_features)
        self.x_column.currentTextChanged.connect(self.update_features)
        self.y_column.currentTextChanged.connect(self.update_features)
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
        self.north_first.setText("X — север, Y — восток")
        self.north_first.setToolTip("Геодезический порядок таблиц: X=Northing, Y=Easting. Снимите флажок для X=Easting, Y=Northing.")
        self.north_first.setChecked(True)
        for label, widget in [("X", self.x_column), ("Y", self.y_column), ("Глубина / горизонт", self.target),
                              ("Признаки", self.feature_list), ("CRS таблиц", self.table_crs),
                              ("Порядок координат", self.north_first)]:
            form.addRow(label, widget)
        panel.addWidget(data_group)
        model_group = QGroupBox("02 · Модель и контроль")
        model_form = QFormLayout(model_group)
        self.spatial.setText("Пространственные блоки")
        self.model_choice = QComboBox()
        self.model_choice.addItems(available_models())
        self.model_choice.setCurrentText("Histogram Boosting")
        self.device_choice = QComboBox()
        self.device_choice.addItem("CPU", "cpu")
        self.device_choice.addItem("GPU · CUDA / OpenCL", "gpu")
        self.gpu_id = QSpinBox()
        self.gpu_id.setRange(0, 31)
        self.device_hint = QLabel()
        self.device_hint.setWordWrap(True)
        self.device_hint.setObjectName("muted")
        self.parameters.setToolTip('Фиксированные параметры модели, например {"max_iter": 200}')
        for label, widget in [("Метод", self.model_choice), ("Вычисления", self.device_choice), ("GPU ID", self.gpu_id),
                              ("", self.device_hint), ("Валидация", self.spatial), ("Размер блока, м", self.block),
                              ("Доля контроля", self.test_fraction), ("Seed", self.seed), ("Параметры · JSON", self.parameters)]:
            model_form.addRow(label, widget)
        panel.addWidget(model_group)
        parameter_actions = QHBoxLayout()
        self.button("Редактор параметров…", self.edit_parameters, parameter_actions)
        self.button("Загрузить параметры JSON…", self.import_parameters, parameter_actions)
        model_form.addRow(parameter_actions)
        optimization = QGroupBox("03 · Optuna TPE")
        opt_form = QFormLayout(optimization)
        self.optimize_check = QCheckBox("Включить оптимизацию")
        self.trials = QSpinBox()
        self.trials.setRange(1, 10000)
        self.trials.setValue(30)
        self.folds = QSpinBox()
        self.folds.setRange(2, 10)
        self.folds.setValue(3)
        self.timeout = QSpinBox()
        self.timeout.setRange(0, 86400)
        self.timeout.setSpecialValueText("Без ограничения")
        self.search.setToolTip('Пустой объект {} = диапазоны для выбранной модели. Пример: {"alpha": {"type":"float","low":0.001,"high":10,"log":true}}')
        for label, widget in [("Поиск", self.optimize_check), ("Trials", self.trials), ("CV фолды", self.folds),
                              ("Лимит времени, с", self.timeout), ("Диапазоны · JSON", self.search)]:
            opt_form.addRow(label, widget)
        preset = QPushButton("Заполнить диапазоны для модели")
        preset.clicked.connect(lambda: self.search.setText(json.dumps(default_space(self.model_choice.currentText()))))
        opt_form.addRow(preset)
        self.best_export = QPushButton("Сохранить лучший trial · JSON…")
        self.best_export.clicked.connect(lambda: self.guard(self.export_best_parameters))
        self.best_export.setToolTip("Доступно после первого завершённого trial, в том числе во время поиска и после остановки.")
        opt_form.addRow(self.best_export)
        self.best_trial_label = QLabel("Лучший trial: ещё нет завершённых результатов")
        self.best_trial_label.setWordWrap(True)
        opt_form.addRow(self.best_trial_label)
        panel.addWidget(optimization)
        actions = QHBoxLayout()
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(0, 4, 0, 0)
        footer_layout.addLayout(actions)
        self.fit_button = self.button("Обучить и оценить", self.fit, actions)
        self.fit_button.setObjectName("primary")
        self.predict_button = self.button("Прогнозировать", self.predict, actions)
        self.stop_button = self.button("Остановить расчёт", self.cancel_job, footer_layout)
        self.stop_button.setObjectName("stop")
        self.stop_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        footer_layout.addWidget(self.progress_bar)
        export = QGroupBox("04 · Экспорт и результаты")
        export_layout = QGridLayout(export)
        for n, (text, function) in enumerate([
            ("Прогноз CSV", self.export_prediction), ("Контроль CSV", self.export_validation),
            ("Trials CSV", self.export_trials), ("Отчёт JSON", self.export_report),
            ("Прогноз → карта", self.prediction_map), ("Невязки → карта", self.validation_map)]):
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, fn=function: self.guard(fn))
            export_layout.addWidget(button, n // 2, n % 2)
        panel.addWidget(export)
        panel.addStretch()
        for combo in controls.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for item_form in (form, model_form, opt_form):
            item_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
            item_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(390)
        scroll.setMaximumWidth(560)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(scroll, 1)
        left_layout.addWidget(footer)
        left.setMinimumWidth(390)
        left.setMaximumWidth(560)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.dashboard = TrainingDashboard()
        right_layout.addWidget(self.dashboard, 1)
        export_row = QHBoxLayout()
        self.plot_export_choice = QComboBox()
        self.plot_export_choice.addItems(self.dashboard.plots)
        export_row.addWidget(self.plot_export_choice)
        self.button("Сохранить график PNG", self.export_chart, export_row)
        right_layout.addLayout(export_row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMaximumHeight(110)
        self.log.setPlaceholderText("Журнал расчётов и лучшие параметры…")
        right_layout.addWidget(self.log)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([450, 1000])
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.model_choice.currentTextChanged.connect(self.update_device_controls)
        self.device_choice.currentIndexChanged.connect(self.update_device_controls)
        self.update_device_controls()
        self.tabs.addTab(tab, "02  ·  Прогноз и оптимизация")

    def update_device_controls(self):
        name = self.model_choice.currentText()
        supports_gpu = name in GPU_MODELS
        self.device_choice.model().item(1).setEnabled(supports_gpu)
        if not supports_gpu:
            self.device_choice.setCurrentIndex(0)
        gpu = self.device_choice.currentData() == "gpu"
        self.gpu_id.setEnabled(gpu)
        self.compute_badge.setText(f"{'GPU' if gpu else 'CPU'}  /  OPTUNA TPE")
        if not supports_gpu:
            text = "Метод выполняется на CPU. GPU доступен для бустингов и PyTorch-сетей."
        elif name == "LightGBM":
            text = "GPU: OpenCL-сборка LightGBM и подходящий драйвер. CPU доступен в обычной установке."
        else:
            text = "GPU: CUDA-совместимая сборка библиотеки и NVIDIA GPU; ID нумеруется с 0."
        self.device_hint.setText(text)

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

    def run_job(self, function, callback, with_context=False):
        if self.worker is not None:
            raise ValueError("Дождитесь текущего расчёта.")
        worker = Worker(function, self)
        if with_context:
            worker.function = lambda: function(worker.progress.emit, worker.cancel_event)
        self.worker = worker
        self.job_outcome = "Расчёт завершён."
        self.worker.result.connect(lambda value: self.guard(lambda: callback(value)))
        self.worker.failed.connect(self.job_failed)
        self.worker.cancelled.connect(self.job_cancelled)
        self.worker.progress.connect(self.trial_progress)
        self.worker.finished.connect(self.job_finished)
        for widget in (self.fit_button, self.predict_button, self.transform_button, self.train_load, self.pred_load):
            widget.setEnabled(False)
        self.statusBar().showMessage("Выполняется расчёт…")
        self.stop_button.setEnabled(with_context)
        self.worker.start()

    def trial_progress(self, record):
        self.dashboard.update_trial(record)
        if self.optimization_context:
            try:
                snapshot = best_snapshot(self.optimization_context, self.dashboard.history)
                self.best_trial_label.setText(f"Лучший trial: {snapshot['best_trial'] + 1} · CV RMSE {snapshot['best_cv_rmse']:.6g}")
            except ValueError:
                pass
        self.progress_bar.setValue(record["trial"] + 1)
        self.log.appendPlainText(f"Trial {record['trial'] + 1} · {record['state']} · CV RMSE={record['rmse']}\n"
                                 + json.dumps(record["parameters"], ensure_ascii=False))

    def job_failed(self, message):
        self.job_outcome = "Ошибка расчёта; подробности в журнале."
        self.log.appendPlainText(message)
        self.dashboard.status.setText(self.job_outcome)
        QMessageBox.warning(self, "Ошибка расчёта", message.splitlines()[-1])

    def job_cancelled(self):
        self.job_outcome = "Расчёт отменён. Историю завершённых trials можно экспортировать."
        self.dashboard.status.setText(self.job_outcome)
        self.log.appendPlainText(self.job_outcome)

    def cancel_job(self):
        if self.worker:
            self.worker.cancel_event.set()
            self.stop_button.setEnabled(False)
            self.statusBar().showMessage("Остановка после текущего обучения / фолда…")

    def job_finished(self):
        self.worker.function = None
        self.worker.deleteLater()
        self.worker = None
        for widget in (self.fit_button, self.predict_button, self.transform_button, self.train_load, self.pred_load):
            widget.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage(self.job_outcome)

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
            self.dashboard.reset()
            self.progress_bar.setValue(0)
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
            self.dashboard.reset()
            self.progress_bar.setValue(0)

    def show_statistics(self):
        if self.training is None:
            raise ValueError("Загрузите обучающую таблицу.")
        self.log.setPlainText(self.training.describe().to_string())

    def fit(self):
        if self.worker is not None:
            raise ValueError("Дождитесь текущего расчёта.")
        if self.training is None:
            raise ValueError("Загрузите обучающую таблицу.")
        features = [self.feature_list.item(i).text() for i in range(self.feature_list.count())
                    if self.feature_list.item(i).checkState() == Qt.Checked]
        params, search = json.loads(self.parameters.text()), json.loads(self.search.text())
        if not isinstance(params, dict) or not isinstance(search, dict):
            raise ValueError("Параметры и диапазоны Optuna должны быть объектами JSON.")
        args = dict(target=self.target.currentText(), coordinates=[self.x_column.currentText(), self.y_column.currentText()],
                    features=features, test_size=self.test_fraction.value(), spatial=self.spatial.isChecked(),
                    block_size=self.block.value(), seed=self.seed.value(), parameters=params,
                    search=search if self.optimize_check.isChecked() else None,
                    optimize=self.optimize_check.isChecked(), n_trials=self.trials.value(), cv_folds=self.folds.value(),
                    timeout=self.timeout.value() or None, device=self.device_choice.currentData(), gpu_id=self.gpu_id.value())
        table, name = self.training.copy(), self.model_choice.currentText()
        validate_parameters(params)
        make_model(name, params, self.seed.value(), self.device_choice.currentData(), self.gpu_id.value())
        if self.spatial.isChecked():
            validate_crs(self.table_crs.currentText().split("|")[0].strip(), "m")
        self.fitted_crs = self.table_crs.currentText().split("|")[0].strip()
        self.fitted_north_first = self.north_first.isChecked()
        self.result = self.output = None
        self.dashboard.reset("Расчёт запущен. Контрольные точки не участвуют в подборе.")
        self.optimization_context = dict(model=name, parameters=dict(params), seed=args["seed"],
                                         device=args["device"], gpu_id=args["gpu_id"],
                                         search=search or default_space(name), cv_folds=args["cv_folds"],
                                         spatial=args["spatial"], block_size=args["block_size"],
                                         test_size=args["test_size"], n_trials=args["n_trials"],
                                         timeout_seconds=args["timeout"], features=features,
                                         target=args["target"], coordinates=args["coordinates"])
        self.best_trial_label.setText("Лучший trial: ещё нет завершённых результатов")
        self.progress_bar.setRange(0, self.trials.value() if self.optimize_check.isChecked() else 1)
        self.progress_bar.setValue(0)
        self.run_job(lambda progress, cancel: train_horizon(table, name, **args, progress=progress, cancel_event=cancel),
                     self.trained, with_context=True)

    def trained(self, result):
        self.result = result
        self.log.setPlainText(json.dumps({"model": result.model_name, "metrics": result.metrics,
                                         "best_parameters": result.settings["best_parameters"]}, ensure_ascii=False, indent=2))
        self.log.appendPlainText("Контроль: отдельные точки/блоки. Итоговая модель переобучена на всех размеченных данных.")
        self.dashboard.show_result(result, self.fitted_north_first)
        self.progress_bar.setValue(self.progress_bar.maximum())

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

    def export_trials(self):
        import pandas as pd
        history = self.result.optimization_history if self.result is not None else self.dashboard.history
        if not history:
            raise ValueError("Нет истории Optuna. Включите оптимизацию и выполните расчёт.")
        rows = [{**r, "parameters": json.dumps(r["parameters"], ensure_ascii=False),
                 "fold_rmse": json.dumps(r["fold_rmse"])} for r in history]
        self.save_csv(pd.DataFrame(rows), "История Optuna")

    def edit_parameters(self):
        if self.worker:
            raise ValueError("Редактирование доступно после окончания расчёта.")
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Гиперпараметры · {self.model_choice.currentText()}")
        dialog.resize(700, 520)
        layout = QVBoxLayout(dialog)
        hint = QLabel("JSON-объект параметров модели. Optuna заменяет значения, включённые в диапазоны поиска.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        editor = QPlainTextEdit()
        editor.setPlainText(json.dumps(json.loads(self.parameters.text()), ensure_ascii=False, indent=2))
        layout.addWidget(editor)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        def save():
            def checked():
                params = validate_parameters(json.loads(editor.toPlainText()))
                make_model(self.model_choice.currentText(), params, self.seed.value(),
                           self.device_choice.currentData(), self.gpu_id.value())
                self.parameters.setText(json.dumps(params, ensure_ascii=False))
                dialog.accept()
            self.guard(checked)
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        apply_window_theme(dialog)
        dialog.exec_()

    def import_parameters(self):
        if self.worker:
            raise ValueError("Загрузка параметров доступна после окончания расчёта.")
        path, _ = QFileDialog.getOpenFileName(self, "Гиперпараметры / лучший trial / отчёт", "", "JSON (*.json)")
        if not path:
            return
        model, params = read_parameters(json.loads(Path(path).read_text(encoding="utf-8-sig")))
        model = model or self.model_choice.currentText()
        if model not in available_models():
            raise ValueError(f"Модель {model} недоступна. Установите её зависимости.")
        make_model(model, params)
        self.model_choice.setCurrentText(model)
        self.parameters.setText(json.dumps(params, ensure_ascii=False))
        self.optimize_check.setChecked(False)
        self.statusBar().showMessage("Параметры загружены. Optuna выключена; CPU/GPU и разбиение проверьте перед обучением.")

    def export_best_parameters(self):
        if not self.optimization_context:
            raise ValueError("Сначала запустите Optuna и дождитесь первого завершённого trial.")
        snapshot = best_snapshot(self.optimization_context, self.dashboard.history)
        path, _ = QFileDialog.getSaveFileName(self, "Лучший завершённый trial Optuna", "optuna-best.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def export_chart(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить график", "", "PNG (*.png)")
        if path:
            self.dashboard.export_plot(self.plot_export_choice.currentText(), path)

    def validation_map(self):
        if self.result is None:
            raise ValueError("Сначала обучите модель.")
        data = self.result.validation
        grid = points_to_grid(data.iloc[:, 1 if self.fitted_north_first else 0],
                              data.iloc[:, 0 if self.fitted_north_first else 1],
                              data["residual_pred_minus_observed"], name="Контроль · прогноз − наблюдения", unit="m",
                              crs=self.fitted_crs, coordinate_unit="m")
        self.add_grid(grid)
        self.palette.setCurrentText("RdBu_r")
        self.norm.setCurrentIndex(2)
        self.redraw()
        self.tabs.setCurrentIndex(0)

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
            import optuna
            report = dict(model=self.result.model_name, features=self.result.features, target=self.result.target,
                          coordinates=self.result.coordinates, metrics=self.result.metrics, settings=self.result.settings,
                          crs=self.fitted_crs, northing_first=self.fitted_north_first,
                          optimization_history=self.result.optimization_history,
                          versions={"numpy": np.__version__, "sklearn": sklearn.__version__, "optuna": optuna.__version__})
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
