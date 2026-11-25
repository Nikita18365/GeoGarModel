"""Map calculator dialog and integration with the existing calculation worker."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox,QDialog,QDialogButtonBox,QFormLayout,QLabel,
                             QLineEdit,QMessageBox,QScrollArea,QSpinBox,QVBoxLayout,QWidget)

from .grid_calculator import (OPERATIONS, UNARY_OPERATIONS, operand_name, suggested_unit,
                              validate_operation, calculate_grids)


class GridCalculatorDialog(QDialog):
    def __init__(self, grids, parent=None):
        super().__init__(parent)
        self.grids = tuple(grids)
        if not self.grids:
            raise ValueError('Загрузите хотя бы одну числовую карту.')
        self.setWindowTitle('Калькулятор гридов')
        self.resize(650,580)
        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)
        self.operation = QComboBox()
        for key,label in OPERATIONS.items():
            self.operation.addItem(f'{key} · {label}',key)
        if len(grids)==1:
            self.operation.setCurrentIndex(self.operation.findData('FORMULA'))
        self.count = QSpinBox()
        self.count.setRange(1,len(grids)); self.count.setValue(min(2,len(grids)))
        form.addRow('Операция',self.operation)
        form.addRow('Количество гридов',self.count)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        panel = QWidget(); self.operand_form = QFormLayout(panel)
        scroll.setWidget(panel); scroll.setMinimumHeight(120)
        root.addWidget(scroll,1)
        self.selectors = []
        self.expression = QLineEdit('A+B' if len(grids)>1 else 'A')
        self.expression.setPlaceholderText('Например: (A+B)/2 или where(A>0,B,C)')
        self.expression.setMaxLength(4096)
        form2 = QFormLayout(); root.addLayout(form2)
        form2.addRow('Формула',self.expression)
        self.hint = QLabel('Формулы: + − * / ** %; сравнения; &, |, ^, ~; '
                           'where, abs, sqrt, log, log10, exp, sin, cos, min, max, sum, mean, clip. '
                           'Логика: 1 — истина, 0 — ложь. Пропуски и деление на ноль → NaN.')
        self.hint.setWordWrap(True); root.addWidget(self.hint)
        self.alignment = QComboBox()
        self.alignment.addItem('Только совпадающие сетки','strict')
        self.alignment.addItem('Линейная интерполяция на A','linear')
        self.alignment.setToolTip('Без экстраполяции, в одной СК. Геометрия результата — сетка A.')
        self.suffix = QLineEdit()
        self.suffix.setPlaceholderText('Автоматически из названий карт')
        self.unit = QLineEdit()
        self.unit.setPlaceholderText('Автоматически, если единицы совпадают')
        self.unit_note = QLabel(); self.unit_note.setWordWrap(True)
        options = QFormLayout(); root.addLayout(options)
        options.addRow('Согласование сеток',self.alignment)
        options.addRow('Имя после префикса',self.suffix)
        options.addRow('Единицы результата',self.unit)
        root.addWidget(self.unit_note)
        self.error = QLabel(); self.error.setWordWrap(True)
        self.error.setObjectName('gridCalculatorError')
        root.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.addButton('Рассчитать → новый грид',QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.confirm)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.count.valueChanged.connect(self.rebuild)
        self.operation.currentIndexChanged.connect(self.update_operation)
        self.rebuild()
        self.update_operation()

    def selected_grids(self):
        return [self.grids[box.currentIndex()] for box in self.selectors]

    def rebuild(self,*args):
        saved = [box.currentIndex() for box in self.selectors]
        # Keep removed choices for subsequent expansion only if still present.
        while self.operand_form.rowCount():
            self.operand_form.removeRow(0)
        self.selectors = []
        for i in range(self.count.value()):
            combo = QComboBox()
            for grid in self.grids:
                combo.addItem(grid.name+(f' [{grid.unit}]' if grid.unit else ''))
            combo.setCurrentIndex(saved[i] if i<len(saved) else i)
            combo.setToolTip('Сетка '+operand_name(i))
            self.operand_form.addRow(operand_name(i),combo)
            self.selectors.append(combo)
            combo.currentIndexChanged.connect(self.update_unit)
        self.update_unit()

    def update_operation(self,*args):
        operation = self.operation.currentData()
        unary = operation in UNARY_OPERATIONS
        if unary:
            self.count.setValue(1)
        elif operation!='FORMULA' and self.count.value()<2 and len(self.grids)>1:
            self.count.setValue(2)
        self.count.setEnabled(not unary)
        custom = operation=='FORMULA'
        self.expression.setEnabled(custom)
        self.hint.setVisible(custom)
        self.error.clear()
        self.update_unit()

    def update_unit(self,*args):
        if not self.selectors:
            return
        grids = self.selected_grids()
        operation = self.operation.currentData()
        auto = suggested_unit(grids,operation)
        self.unit.setPlaceholderText(auto or 'Не задано — можно указать вручную')
        units = {grid.unit for grid in grids}
        if len(units)>1:
            self.unit_note.setText('Единицы исходных полей различаются. Автоматического преобразования нет; проверьте физический смысл и укажите единицы результата.')
        else:
            self.unit_note.setText('Геометрия результата — A. СК должна совпадать; исходные карты не меняются.')

    def request(self):
        grids = self.selected_grids()
        operation = self.operation.currentData()
        expression = self.expression.text() if operation=='FORMULA' else ''
        validate_operation(operation,len(grids),expression)
        suffix = self.suffix.text().strip()
        if not suffix:
            suffix = '_'.join(grid.name for grid in grids[:2])
            if len(grids)>2:
                suffix += f'_and_{len(grids)-2}'
        name = operation+'_'+suffix
        unit = self.unit.text().strip() or None
        return grids,dict(operation=operation,expression=expression,
                          alignment=self.alignment.currentData(),name=name,unit=unit)

    def confirm(self):
        try:
            self.request()
        except ValueError as exc:
            self.error.setText(str(exc)); return
        self.accept()


class GridCalculatorMixin:
    def open_grid_calculator(self):
        if self.worker is not None or self.inversion.worker is not None:
            raise ValueError('Дождитесь завершения текущего расчёта.')
        items = [self.layers.item(i) for i in range(self.layers.count())]
        items = [item for item in items if item.data(Qt.UserRole)['kind']=='grid']
        if not items:
            raise ValueError('Сначала загрузите числовые карты в проект.')
        current = self.layers.currentItem()
        if current in items:
            items.remove(current); items.insert(0,current)
        grids = [item.data(Qt.UserRole)['data'] for item in items]
        dialog = GridCalculatorDialog(grids,self)
        if dialog.exec_()!=QDialog.Accepted:
            return
        grids,options = dialog.request()

        def finished(grid):
            existing = {self.layers.item(i).text() for i in range(self.layers.count())}
            original = grid.name
            number = 2
            while grid.name in existing:
                grid.name = f'{original} ({number})'
                number += 1
            self.add_grid(grid)

        self.run_job(lambda: calculate_grids(grids,**options),finished,title='Калькулятор гридов')
