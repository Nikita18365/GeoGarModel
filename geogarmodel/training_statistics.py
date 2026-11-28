"""Non-destructive descriptive statistics and interactive distributions."""
import math
import csv
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSplitter, QWidget, QPushButton, QFileDialog,
    QMessageBox)
from .theme import current_theme


HEADERS = ("Признак / столбец", "Чисел", "Пропусков", "Нечисл. / ∞",
           "Среднее", "σ", "Min", "Q25", "Медиана", "Q75", "Max")


def numeric_values(series):
    if (pd.api.types.is_datetime64_any_dtype(series.dtype) or
            pd.api.types.is_timedelta64_dtype(series.dtype) or
            pd.api.types.is_complex_dtype(series.dtype)):
        return np.full(len(series), np.nan)
    converted = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_complex_dtype(converted.dtype):
        return np.full(len(series), np.nan)
    return converted.to_numpy(dtype=float, na_value=np.nan)


def summarize_table(frame):
    rows = []
    for name, series in frame.items():
        values = numeric_values(series)
        finite = values[np.isfinite(values)]
        missing = int(series.isna().sum())
        count = len(finite)
        invalid = len(series) - missing - count
        if count:
            q25, median, q75 = np.quantile(finite, [.25, .5, .75])
            moments = (float(finite.mean()), float(finite.std(ddof=1)) if count > 1 else None,
                       float(finite.min()), float(q25), float(median), float(q75), float(finite.max()))
        else:
            moments = (None,) * 7
        rows.append((str(name), count, missing, invalid, *moments))
    return rows


def format_stat(value, digits, scientific=False):
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}{'e' if scientific else 'f'}}"


class NumericItem(QTableWidgetItem):
    def __init__(self, value, display):
        super().__init__(display)
        self.number = value
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        if value is not None:
            self.setToolTip(format(value, ".17g"))

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            a = self.number if self.number is not None and math.isfinite(self.number) else math.inf
            b = other.number if other.number is not None and math.isfinite(other.number) else math.inf
            return a < b
        return super().__lt__(other)


class StatisticsDialog(QDialog):
    def __init__(self, sources, parent=None):
        super().__init__(parent)
        self.sources = dict(sources)
        self.summaries = {}
        self.setWindowTitle("Статистика геофизических данных")
        self.resize(1240, 790)
        self.setMinimumSize(680, 520)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Набор данных"))
        self.source = QComboBox()
        self.source.addItems(self.sources)
        controls.addWidget(self.source, 2)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск столбца…")
        self.search.setClearButtonEnabled(True)
        controls.addWidget(self.search, 2)
        controls.addWidget(QLabel("Точность"))
        self.precision = QSpinBox()
        self.precision.setRange(0, 12)
        self.precision.setValue(4)
        self.precision.setToolTip("Количество знаков после запятой")
        controls.addWidget(self.precision)
        self.notation = QComboBox()
        self.notation.addItems(["Десятичный", "Научный"])
        controls.addWidget(self.notation)
        layout.addLayout(controls)
        cards = QHBoxLayout()
        self.cards = {}
        for title in ("Строки", "Столбцы", "Пропуски", "Нечисловые / ∞"):
            box = QWidget()
            box_layout = QVBoxLayout(box)
            label = QLabel(title)
            label.setObjectName("muted")
            value = QLabel("—")
            font = value.font()
            font.setPointSizeF(max(10., font.pointSizeF()) + 7)
            font.setBold(True)
            value.setFont(font)
            box_layout.addWidget(label)
            box_layout.addWidget(value)
            cards.addWidget(box, 1)
            self.cards[title] = value
        layout.addLayout(cards)
        splitter = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(65)
        self.table.setColumnWidth(0, 235)
        for i in range(1, len(HEADERS)):
            self.table.setColumnWidth(i, 97 if i >= 4 else 100)
        header.setSortIndicator(0, Qt.AscendingOrder)
        splitter.addWidget(self.table)
        chart = QWidget()
        chart_layout = QVBoxLayout(chart)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self.chart_title = QLabel("Распределение выбранного столбца")
        self.chart_title.setTextFormat(Qt.PlainText)
        self.chart_title.setWordWrap(True)
        chart_layout.addWidget(self.chart_title)
        theme = current_theme()
        self.histogram = pg.PlotWidget(background=theme.panel)
        self.histogram.setMinimumHeight(150)
        self.histogram.setLabel("left", "Количество")
        self.histogram.setLabel("bottom", "Значение")
        for axis in ("left", "bottom"):
            self.histogram.getAxis(axis).setPen(theme.border)
            self.histogram.getAxis(axis).setTextPen(theme.text)
        self.histogram.showGrid(x=True, y=True, alpha=.2)
        chart_layout.addWidget(self.histogram, 1)
        splitter.addWidget(chart)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([450, 240])
        layout.addWidget(splitter, 1)
        foot = QLabel("σ — выборочное стандартное отклонение (ddof=1). Квартили: 25%, 50%, 75%. "
                      "Пропуски и нечисловые/бесконечные значения исключены из статистики. "
                      "Числа округляются только при отображении; исходная точность доступна в подсказке.")
        foot.setWordWrap(True)
        foot.setObjectName("muted")
        layout.addWidget(foot)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.export = buttons.addButton("Экспорт CSV…", QDialogButtonBox.ActionRole)
        self.export.clicked.connect(self.export_csv)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.source.currentTextChanged.connect(self.change_source)
        self.search.textChanged.connect(self.apply_filter)
        self.precision.valueChanged.connect(self.populate)
        self.notation.currentIndexChanged.connect(self.populate)
        self.table.currentCellChanged.connect(self.draw_histogram)
        self.change_source()

    def change_source(self, *args):
        name = self.source.currentText()
        frame = self.sources[name]
        if name not in self.summaries:
            self.summaries[name] = summarize_table(frame)
        self.rows = self.summaries[name]
        for title, value in zip(self.cards, (len(frame), len(frame.columns),
                    sum(r[2] for r in self.rows), sum(r[3] for r in self.rows))):
            self.cards[title].setText(f"{value:,}".replace(",", " "))
        self.populate()

    def populate(self, *args):
        previous = self.table.item(self.table.currentRow(), 0) if self.table.currentRow() >= 0 else None
        key = previous.data(Qt.UserRole) if previous is not None else None
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(len(self.rows))
        digits = self.precision.value()
        scientific = self.notation.currentIndex() == 1
        for r, row in enumerate(self.rows):
            item = QTableWidgetItem(row[0])
            item.setData(Qt.UserRole, r)
            item.setToolTip(row[0])
            self.table.setItem(r, 0, item)
            for c, value in enumerate(row[1:], 1):
                display = str(value) if c <= 3 else format_stat(value, digits, scientific)
                self.table.setItem(r, c, NumericItem(value, display))
        self.table.setSortingEnabled(True)
        self.table.blockSignals(False)
        self.apply_filter()
        if key is not None:
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0).data(Qt.UserRole) == key and not self.table.isRowHidden(r):
                    self.table.setCurrentCell(r, 0)
                    break
        self.draw_histogram()

    def apply_filter(self, *args):
        query = self.search.text().strip().casefold()
        visible = []
        for r in range(self.table.rowCount()):
            hide = query not in self.table.item(r, 0).text().casefold()
            self.table.setRowHidden(r, hide)
            if not hide:
                visible.append(r)
        if self.table.currentRow() not in visible:
            self.table.setCurrentCell(visible[0], 0) if visible else self.table.setCurrentCell(-1, -1)
        self.draw_histogram()

    def draw_histogram(self, *args):
        self.histogram.clear()
        row = self.table.currentRow()
        if row < 0 or self.table.isRowHidden(row):
            self.chart_title.setText("Нет видимых столбцов")
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        index = item.data(Qt.UserRole)
        values = numeric_values(self.sources[self.source.currentText()].iloc[:, index])
        values = values[np.isfinite(values)]
        self.chart_title.setText(f"{item.text()} · {len(values)} конечных чисел")
        if not len(values):
            return
        try:
            counts, edges = np.histogram(values, bins=min(60, max(1, math.ceil(math.sqrt(len(values))))))
            if not np.isfinite(edges).all():
                raise ValueError("non-finite bin edges")
            theme = current_theme()
            self.histogram.plot(edges, counts, stepMode="center", fillLevel=0,
                                pen=pg.mkPen(theme.accent, width=1.5), brush=pg.mkBrush(theme.primary))
            mean = self.rows[index][4]
            if mean is not None and math.isfinite(mean):
                self.histogram.addItem(pg.InfiniteLine(pos=mean, angle=90,
                    pen=pg.mkPen(theme.muted, width=2, style=Qt.DashLine)))
            self.histogram.enableAutoRange()
        except (ValueError, OverflowError, FloatingPointError):
            self.chart_title.setText(f"{item.text()} · диапазон значений не позволяет построить гистограмму")

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт статистики", "statistics.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(HEADERS)
                for row in range(self.table.rowCount()):
                    if not self.table.isRowHidden(row):
                        index = self.table.item(row, 0).data(Qt.UserRole)
                        writer.writerow(self.rows[index])
        except OSError as error:
            QMessageBox.warning(self, "Экспорт статистики", str(error))
