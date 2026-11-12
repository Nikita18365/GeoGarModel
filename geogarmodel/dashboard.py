"""Fast Qt-native optimization and validation charts; full data remain in exports."""
import json
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QTabWidget, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView)

BACKGROUND = "#101c2f"
FOREGROUND = "#dbe7f6"
ACCENT = "#38bdf8"
GREEN = "#2dd4bf"


def plot_widget(title, xlabel="", ylabel=""):
    widget = pg.PlotWidget(background=BACKGROUND)
    widget.setTitle(title, color=FOREGROUND, size="11pt")
    widget.setLabel("bottom", xlabel, color=FOREGROUND)
    widget.setLabel("left", ylabel, color=FOREGROUND)
    widget.showGrid(x=True, y=True, alpha=.12)
    for axis in ("left", "bottom"):
        widget.getAxis(axis).setPen("#52647c")
        widget.getAxis(axis).setTextPen("#a9bbd3")
    widget.setMinimumSize(220, 180)
    return widget


def table_widget(labels):
    table = QTableWidget(0, len(labels))
    table.setHorizontalHeaderLabels(labels)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setAlternatingRowColors(True)
    table.verticalHeader().hide()
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    return table


class TrainingDashboard(QWidget):
    """One dashboard instance reuses curve objects during streaming updates."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.map_items = []
        self.plots = {}
        layout = QVBoxLayout(self)
        metrics = QHBoxLayout()
        self.metric_labels = {}
        for name in ("CV RMSE", "RMSE", "MAE", "R²"):
            card = QLabel(f"{name}\n—")
            card.setObjectName("metricCard")
            card.setMinimumHeight(65)
            metrics.addWidget(card)
            self.metric_labels[name] = card
        layout.addLayout(metrics)
        self.status = QLabel("Загрузите данные и запустите обучение или Optuna.")
        self.status.setWordWrap(True)
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        optimization = QWidget()
        opt_layout = QVBoxLayout(optimization)
        row = QHBoxLayout()
        self.history_plot = plot_widget("Optuna · ошибка по trials", "Trial", "Внутренняя CV RMSE")
        self.history_plot.addLegend(offset=(10, 10), labelTextColor=FOREGROUND)
        self.trial_curve = self.history_plot.plot(pen=None, symbol="o", symbolSize=6,
                                                  symbolBrush=ACCENT, symbolPen=None, name="Trial")
        self.best_curve = self.history_plot.plot(pen=pg.mkPen(GREEN, width=2), name="Лучшее значение")
        self.best_curve.setDownsampling(auto=True, method="peak")
        self.best_curve.setClipToView(True)
        self.duration_plot = plot_widget("Стоимость подбора", "Trial", "Секунды")
        self.duration_curve = self.duration_plot.plot(pen=pg.mkPen("#a78bfa", width=1), symbol="o", symbolSize=4)
        row.addWidget(self.history_plot)
        row.addWidget(self.duration_plot)
        opt_layout.addLayout(row, 2)
        self.trials_table = table_widget(["Trial", "Состояние", "CV RMSE", "Лучшее RMSE", "Секунды", "Параметры"])
        opt_layout.addWidget(self.trials_table, 1)
        self.tabs.addTab(optimization, "Оптимизация")
        diagnostics = QWidget()
        grid = QGridLayout(diagnostics)
        self.crossplot = plot_widget("Прогноз / наблюдения", "Наблюдения", "Прогноз")
        self.histogram = plot_widget("Распределение невязок", "Прогноз − наблюдения", "Число точек")
        self.residual_plot = plot_widget("Смещение прогноза", "Наблюдения", "Прогноз − наблюдения")
        self.loss_plot = plot_widget("Обучение итоговой сети", "Эпоха", "MSE · нормированная цель")
        for n, widget in enumerate((self.crossplot, self.histogram, self.residual_plot, self.loss_plot)):
            grid.addWidget(widget, n // 2, n % 2)
        self.tabs.addTab(diagnostics, "Диагностика")
        maps = QWidget()
        map_layout = QGridLayout(maps)
        self.map_plots = []
        for n, title in enumerate(("Наблюдения · контроль", "Прогноз · контроль", "Невязка: прогноз − наблюдения", "Абсолютная ошибка")):
            widget = plot_widget(title, "Восток, м", "Север, м")
            widget.setAspectLocked(True)
            self.map_plots.append(widget)
            map_layout.addWidget(widget, n // 2, n % 2)
        self.tabs.addTab(maps, "Карты контроля")
        parameters = QWidget()
        parameter_layout = QVBoxLayout(parameters)
        self.parameter_table = table_widget(["Параметр", "Лучшее значение"])
        parameter_layout.addWidget(self.parameter_table)
        self.parameter_choice = QComboBox()
        self.parameter_choice.currentTextChanged.connect(self.draw_parameter)
        parameter_layout.addWidget(self.parameter_choice)
        self.parameter_plot = plot_widget("Параметр / ошибка", "Параметр", "Внутренняя CV RMSE")
        parameter_layout.addWidget(self.parameter_plot)
        self.tabs.addTab(parameters, "Параметры")
        self.plots = {"История Optuna": self.history_plot, "Длительность trials": self.duration_plot,
                      "Прогноз / наблюдения": self.crossplot, "Гистограмма невязок": self.histogram,
                      "Смещение": self.residual_plot, "Обучение сети": self.loss_plot,
                      "Параметр / ошибка": self.parameter_plot}
        self.plots.update({f"Карта {i + 1}": p for i, p in enumerate(self.map_plots)})

    def reset(self):
        self.history.clear()
        self.trials_table.setRowCount(0)
        self.parameter_table.setRowCount(0)
        self.parameter_choice.clear()
        self.clear_map_items()
        for plot in (self.crossplot, self.histogram, self.residual_plot, self.loss_plot, self.parameter_plot, *self.map_plots):
            plot.clear()
        self.trial_curve.setData([], [])
        self.best_curve.setData([], [])
        self.duration_curve.setData([], [])
        for name, widget in self.metric_labels.items():
            widget.setText(f"{name}\n—")
        self.status.setText("Расчёт запущен. Контрольные точки не участвуют в подборе.")

    def update_trial(self, record):
        self.history.append(record)
        row = self.trials_table.rowCount()
        self.trials_table.insertRow(row)
        values = (record["trial"], record["state"], record["rmse"], record["best_rmse"],
                  round(record["seconds"], 2), json.dumps(record["parameters"], ensure_ascii=False))
        for column, value in enumerate(values):
            self.trials_table.setItem(row, column, QTableWidgetItem("—" if value is None else str(value)))
        complete = [r for r in self.history if r["state"] == "COMPLETE"]
        self.trial_curve.setData([r["trial"] for r in complete], [r["rmse"] for r in complete])
        self.best_curve.setData([r["trial"] for r in complete], [r["best_rmse"] for r in complete])
        self.duration_curve.setData([r["trial"] for r in self.history], [r["seconds"] for r in self.history])
        if record["best_rmse"] is not None:
            self.metric_labels["CV RMSE"].setText(f"CV RMSE\n{record['best_rmse']:.4g}")
        self.status.setText(f"Trial {record['trial'] + 1}: {record['state']} · {record['elapsed_seconds']:.0f} с")

    def show_result(self, result, north_first=False):
        for key in ("RMSE", "MAE", "R²"):
            metric = result.metrics["R2" if key == "R²" else key]
            self.metric_labels[key].setText(f"{key}\n{metric:.4g}")
        self.history = list(result.optimization_history)
        best = result.settings.get("best_parameters", {})
        self.parameter_table.setRowCount(len(best))
        for row, (key, value) in enumerate(best.items()):
            self.parameter_table.setItem(row, 0, QTableWidgetItem(key))
            self.parameter_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.parameter_choice.clear()
        self.parameter_choice.addItems(list(best))
        data = result.validation
        # Limit ONLY rendering; predictions, metrics and CSV exports remain full resolution.
        index = np.linspace(0, len(data) - 1, min(len(data), 25000), dtype=int)
        shown = data.iloc[index]
        observed, predicted = shown["observed"].to_numpy(), shown["predicted"].to_numpy()
        residual = predicted - observed
        self.crossplot.clear()
        self.crossplot.plot(observed, predicted, pen=None, symbol="o", symbolSize=5, symbolPen=None, symbolBrush=ACCENT)
        low, high = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
        self.crossplot.plot([low, high], [low, high], pen=pg.mkPen(GREEN, width=1))
        self.residual_plot.clear()
        self.residual_plot.plot(observed, residual, pen=None, symbol="o", symbolSize=5, symbolPen=None, symbolBrush="#a78bfa")
        self.residual_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#52647c")))
        counts, bins = np.histogram(data["residual_pred_minus_observed"], bins="auto")
        self.histogram.clear()
        self.histogram.addItem(pg.BarGraphItem(x=(bins[:-1] + bins[1:]) / 2, height=counts,
                                              width=np.diff(bins) * .9, brush=ACCENT, pen=None))
        self.loss_plot.clear()
        estimator = result.model.regressor_.steps[-1][1]
        losses = getattr(estimator, "loss_curve_", [])
        if len(losses):
            self.loss_plot.plot(np.arange(1, len(losses) + 1), losses, pen=pg.mkPen(GREEN, width=1))
        self.clear_map_items()
        limits = (float(min(observed.min(), predicted.min())), float(max(observed.max(), predicted.max())))
        error_limit = max(float(np.max(np.abs(residual))), 1e-9)
        east, north = shown.iloc[:, 1 if north_first else 0].to_numpy(), shown.iloc[:, 0 if north_first else 1].to_numpy()
        for plot, z, levels, palette in zip(self.map_plots,
                (observed, predicted, residual, np.abs(residual)),
                (limits, limits, (-error_limit, error_limit), (0, error_limit)),
                ("viridis", "viridis", "CET-D1", "viridis")):
            plot.clear()
            cmap = pg.colormap.get(palette)
            denominator = max(levels[1] - levels[0], 1e-12)
            colors = cmap.mapToQColor(np.clip((z - levels[0]) / denominator, 0, 1))
            brushes = [pg.mkBrush(c) for c in colors]
            scatter = pg.ScatterPlotItem(x=east, y=north, data=z, size=6, pen=None, brush=brushes,
                                         hoverable=True, tip=lambda x, y, data: f"E={x:.2f}\nN={y:.2f}\nZ={data:.4g}")
            plot.addItem(scatter)
            bar = pg.ColorBarItem(values=levels if levels[0] < levels[1] else (levels[0], levels[0] + 1),
                                   colorMap=cmap, interactive=False, label="м")
            plot.plotItem.layout.addItem(bar, 2, 5)
            self.map_items.append((plot, scatter, bar))
        self.status.setText(f"Контроль: {len(data)} точек · на графиках до 25 000 точек · "
                            f"устройство: {result.settings.get('device', 'cpu').upper()}. Метрики и экспорт — по всем точкам.")
        self.draw_parameter()
        self.tabs.setCurrentIndex(1)

    def clear_map_items(self):
        for plot, _, bar in self.map_items:
            plot.plotItem.layout.removeItem(bar)
            if bar.scene() is not None:
                bar.scene().removeItem(bar)
        self.map_items.clear()

    def draw_parameter(self):
        name = self.parameter_choice.currentText()
        self.parameter_plot.clear()
        rows = [r for r in self.history if r["state"] == "COMPLETE" and name in r["parameters"]]
        if not rows:
            return
        values = [r["parameters"][name] for r in rows]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        if numeric:
            x = values
            self.parameter_plot.getAxis("bottom").setTicks(None)
        else:
            categories = list(dict.fromkeys(str(v) for v in values))
            x = [categories.index(str(v)) for v in values]
            self.parameter_plot.getAxis("bottom").setTicks([list(enumerate(categories))])
        self.parameter_plot.setLabel("bottom", name)
        self.parameter_plot.plot(x, [r["rmse"] for r in rows], pen=None, symbol="o", symbolSize=7,
                                 symbolBrush=ACCENT, symbolPen=None)

    def export_plot(self, name, path):
        from pyqtgraph.exporters import ImageExporter
        exporter = ImageExporter(self.plots[name].plotItem)
        exporter.parameters()["width"] = 1800
        exporter.export(path)
