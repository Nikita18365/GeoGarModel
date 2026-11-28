"""Feature selection and shared projection editor for horizon modelling."""
from PyQt5.QtCore import Qt, QItemSelectionModel, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QListWidget, QAbstractItemView, QDialog, QDialogButtonBox,
    QMessageBox, QSizePolicy)
from pyproj import CRS
from .crs_editor import CRSEditor
from .ui_layout import SummaryLabel
from .window_theme import apply_window_theme


class FeatureList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setUniformItemSizes(True)
        self.setMinimumHeight(210)
        self._anchor = None
        self._range_press = False

    def clear(self):
        self._anchor = None
        self._range_press = False
        super().clear()

    def check_range(self, first, last):
        if not (0 <= first < self.count() and 0 <= last < self.count()):
            return
        anchor = self.item(first)
        if anchor.isHidden():
            return
        state = anchor.checkState()
        self.clearSelection()
        self.setCurrentItem(self.item(last), QItemSelectionModel.NoUpdate)
        for index in range(min(first, last), max(first, last) + 1):
            item = self.item(index)
            if not item.isHidden():
                item.setSelected(True)
                item.setCheckState(state)

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        self._range_press = False
        if item is not None and event.button() == Qt.LeftButton:
            row = self.row(item)
            if event.modifiers() & Qt.ShiftModifier and self._anchor is not None:
                if 0 <= self._anchor < self.count() and not self.item(self._anchor).isHidden():
                    self.check_range(self._anchor, row)
                    self._range_press = True
                    event.accept()
                    return
            self._anchor = row
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._range_press and event.button() == Qt.LeftButton:
            self._range_press = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            items = [item for item in self.selectedItems() if not item.isHidden()]
            if items:
                state = Qt.Unchecked if all(i.checkState() == Qt.Checked for i in items) else Qt.Checked
                for item in items:
                    item.setCheckState(state)
                event.accept()
                return
        super().keyPressEvent(event)


class FeaturePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск по названию признака…")
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        self.items = FeatureList()
        self.items.setAccessibleName("Признаки для обучения")
        layout.addWidget(self.items)
        self.counter = QLabel()
        self.counter.setWordWrap(True)
        layout.addWidget(self.counter)
        for actions in (
            (("Все видимые", lambda: self.set_checked(Qt.Checked)),
             ("Снять видимые", lambda: self.set_checked(Qt.Unchecked)),
             ("Инверсия", self.invert_visible)),
            (("Отметить выделенные", lambda: self.set_checked(Qt.Checked, True)),
             ("Снять выделенные", lambda: self.set_checked(Qt.Unchecked, True)))):
            row = QHBoxLayout()
            for title, callback in actions:
                button = QPushButton(title)
                button.setToolTip(title)
                button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
                button.clicked.connect(callback)
                row.addWidget(button)
            layout.addLayout(row)
        hint = QLabel("Щёлкните по флажку, затем Shift + щелчок — применить его состояние к диапазону. "
                      "Ctrl — выделить отдельные строки; пробел — переключить выделенные.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.search.textChanged.connect(self.apply_filter)
        self.items.itemChanged.connect(self.update_counter)
        self.items.itemSelectionChanged.connect(self.update_counter)
        model = self.items.model()
        model.rowsInserted.connect(self.apply_filter)
        model.rowsRemoved.connect(self.update_counter)
        model.modelReset.connect(self.update_counter)
        self.update_counter()

    def apply_filter(self, *args):
        query = self.search.text().strip().casefold()
        for index in range(self.items.count()):
            item = self.items.item(index)
            hidden = query not in item.text().casefold()
            if hidden:
                item.setSelected(False)
            item.setHidden(hidden)
        self.update_counter()

    def set_checked(self, state, selected_only=False):
        for index in range(self.items.count()):
            item = self.items.item(index)
            if not item.isHidden() and (not selected_only or item.isSelected()):
                item.setCheckState(state)
        self.update_counter()

    def invert_visible(self):
        for index in range(self.items.count()):
            item = self.items.item(index)
            if not item.isHidden():
                item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)

    def update_counter(self, *args):
        items = [self.items.item(i) for i in range(self.items.count())]
        checked = sum(i.checkState() == Qt.Checked for i in items)
        visible = sum(not i.isHidden() for i in items)
        selected = sum(i.isSelected() and not i.isHidden() for i in items)
        self.counter.setText(f"В модели: {checked} / {len(items)}  ·  Видно: {visible}  ·  Выделено: {selected}")


class TableCRSSelector(QWidget):
    """Store complete CRS definitions, including custom projection parameters."""
    valueChanged = pyqtSignal(str)

    def __init__(self, known="EPSG:20910", parent=None):
        super().__init__(parent)
        self._value = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = SummaryLabel()
        layout.addWidget(self.summary)
        self.edit_button = QPushButton("Выбрать / изменить СК…")
        self.edit_button.clicked.connect(self.edit)
        layout.addWidget(self.edit_button)
        hint = QLabel("СК исходных X/Y; координаты не пересчитываются. Для пространственных блоков нужна метрическая проекция.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.setValue(known)

    def value(self):
        return self._value

    def setValue(self, value):
        value = str(value or "").strip()
        # Compatibility with the previous project schema.
        if value.upper().startswith("EPSG:") and " | " in value:
            value = value.split(" | ", 1)[0].strip()
        if value:
            crs = CRS.from_user_input(value)
            value = crs.to_string()
            caption = crs.name
            if value.upper().startswith("EPSG:"):
                caption = value + " · " + caption
        else:
            caption = "СК не задана"
        changed = value != self._value
        self._value = value
        self.summary.setText(caption)
        if changed:
            self.valueChanged.emit(value)

    def edit(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Система координат обучающей и прогнозной таблиц")
        dialog.resize(660, 530)
        layout = QVBoxLayout(dialog)
        editor = CRSEditor(self.value(), dialog)
        layout.addWidget(editor, 1)
        note = QLabel("Укажите СК, в которой уже заданы координаты обеих таблиц. "
                      "Импорт .prj и изменение параметров проекции не изменяют сами X/Y.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        def accept():
            try:
                self.setValue(editor.value())
            except Exception as error:
                QMessageBox.warning(dialog, "Система координат", str(error))
                return
            dialog.accept()
        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        apply_window_theme(dialog)
        dialog.exec_()
