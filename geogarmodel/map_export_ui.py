"""Map export controls. The selected project item is the only export source."""
from pathlib import Path
import re

from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QFont, QColor
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QFileDialog, QLabel, QInputDialog, QMessageBox

from .map_export import (FORMATS, TABLE_FORMATS, companion_paths, grid_for_export,
                         table_for_export, write_table, write_grid, square_cells, square_grid)


FORMAT_COLORS = dict(npz='#7dd3fc', xlsx='#86d6a0', csv='#86d6a0', txt='#86d6a0',
                     grd='#edc775', zmap='#edc775', flt='#edc775')
FORMAT_LABELS = dict(npz='Graticule NPZ', xlsx='Excel file (East, North, Z)',
                     csv='CSV file (East, North, Z)', txt='TXT file (East, North, Z)',
                     grd='Geosoft Binary Grid file', zmap='Petrel file format', flt='ESRI ArcGIS')


def format_icon(fmt):
    """Fixed-width coloured extension at the left of the action."""
    pixmap = QPixmap(48, 22)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.TextAntialiasing)
    font = QFont()
    font.setPixelSize(13)
    font.setBold(True)
    painter.setFont(font)
    from .theme import current_theme,theme_colors
    t=current_theme(); c=theme_colors(t)
    color=FORMAT_COLORS[fmt] if not t.light else (t.accent if fmt=='npz' else c['success'] if fmt in ('xlsx','csv','txt') else c['gold'])
    painter.setPen(QColor(color))
    painter.drawText(QRect(0, 0, 48, 22), Qt.AlignLeft|Qt.AlignVCenter, fmt.upper())
    painter.end()
    return QIcon(pixmap)


class MapExportMixin:
    def setup_map_exports(self, layout):
        self.map_export_hint = QLabel()
        self.map_export_hint.setWordWrap(True)
        layout.addWidget(self.map_export_hint)
        self.map_export_buttons = {}
        for fmt, label in FORMATS.items():
            button = self.button(FORMAT_LABELS[fmt], lambda fmt=fmt: self.export_map(fmt), layout)
            button.setAccessibleName(label)
            button.setIcon(format_icon(fmt))
            button.setIconSize(QSize(48, 22))
            button.setStyleSheet('QPushButton { text-align: left; padding-left: 8px; }')
            self.map_export_buttons[fmt] = button
        self.layers.currentItemChanged.connect(self.update_map_exports)
        self.update_map_exports()

    def update_map_exports(self, *args):
        item = self.layers.currentItem()
        row = item.data(Qt.UserRole) if item else None
        enabled = bool(row and row['kind'] == 'grid')
        for button in self.map_export_buttons.values():
            button.setEnabled(enabled)
        if enabled:
            hint = ('Исходная таблица: можно выбрать все столбцы или X, Y, Z.'
                    if row.get('xyz') is not None else 'Числовая карта: экспорт текущей сетки.')
        else:
            hint = 'Выберите числовую карту или слой XYZ в дереве «1 · Карта».'
        self.map_export_hint.setText(hint)

    def choose_export_columns(self, source, fmt):
        dialog = QDialog(self)
        dialog.setWindowTitle('Столбцы экспорта · '+fmt.upper())
        form = QFormLayout(dialog)
        note = QLabel('Выберите East (X), North (Y) и Z из исходной таблицы. СК исходной таблицы: '+str(source['crs']))
        note.setWordWrap(True)
        form.addRow(note)
        boxes = self.make_xyz_columns(form, source['table'].columns, source['columns'])
        buttons = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        def validate():
            if len({box.currentText() for box in boxes}) != 3:
                QMessageBox.warning(dialog, 'Экспорт XYZ', 'Выберите три разных столбца.')
            else:
                dialog.accept()
        buttons.accepted.connect(validate)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return None
        return [box.currentText() for box in boxes]

    def export_map(self, fmt):
        item = self.layers.currentItem()
        row = item.data(Qt.UserRole) if item else None
        if not row or row['kind'] != 'grid':
            raise ValueError('Выберите числовую карту или слой XYZ в дереве проекта.')
        source = row.get('xyz')
        columns = None
        tabular = fmt in TABLE_FORMATS
        if source is not None:
            if tabular:
                modes = ['Вся исходная таблица', 'Только East, North, Z']
                mode, ok = QInputDialog.getItem(self, 'Экспорт таблицы · '+fmt.upper(),
                                               'Какие столбцы сохранить?', modes, 0, False)
                if not ok:
                    return
                if mode == modes[1]:
                    columns = self.choose_export_columns(source, fmt)
                    if columns is None:
                        return
            else:
                columns = self.choose_export_columns(source, fmt)
                if columns is None:
                    return

        label = str(row['name'])
        if columns is not None and not tabular:
            label = str(source['stem'])+'_'+columns[2]
        label = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', label).strip(' .') or 'export'
        if Path(label).suffix.lower() in ('.grd','.npz','.flt','.zmap','.dat','.xyz','.txt','.csv','.xlsx','.tsv'):
            label = Path(label).stem
        path, _ = QFileDialog.getSaveFileName(self, 'Экспорт · '+fmt.upper(), label+'.'+fmt, FORMATS[fmt])
        if not path:
            return
        if not path.lower().endswith('.'+fmt):
            path += '.'+fmt

        if tabular:
            grid = row['data']
            if fmt == 'xlsx' and source is None and grid.values.size > 1048575:
                raise ValueError('Карта превышает лимит строк XLSX. Используйте CSV или TXT.')
            def export_table():
                table = table_for_export(row, columns)
                coordinates = source['columns'][:2] if source is not None and columns is None else ('East', 'North')
                write_table(table, path, fmt, coordinate_columns=coordinates)
            self.activity.run('Экспорт '+fmt.upper(), export_table)
        else:
            grid = self.activity.run('Подготовка сетки', grid_for_export, row, columns)
            if fmt == 'flt' and not square_cells(grid):
                answer = QMessageBox.question(self, 'Квадратная сетка FLT',
                    f'FLT требует одинаковый шаг X и Y.\nТекущий шаг: {grid.dx:.8g} × {grid.dy:.8g}.\n'
                    f'Пересчитать линейной интерполяцией с шагом {min(grid.dx, grid.dy):.8g} '
                    'и оставить за границами исходной карты пропуски?',
                    QMessageBox.Yes|QMessageBox.Cancel, QMessageBox.Cancel)
                if answer != QMessageBox.Yes:
                    return
                grid = self.activity.run('Пересчёт сетки FLT', square_grid, grid)
            existing = [p.name for p in companion_paths(path, fmt) if p.exists()]
            if existing:
                answer = QMessageBox.question(self, 'Экспорт · сопутствующие файлы',
                    'Заменить существующие файлы?\n'+ '\n'.join(existing),
                    QMessageBox.Yes|QMessageBox.Cancel, QMessageBox.Cancel)
                if answer != QMessageBox.Yes:
                    return
            self.activity.run('Экспорт '+fmt.upper(), write_grid, grid, path, fmt)
        self.statusBar().showMessage('Экспорт завершён: '+str(path))

    # Preserve the existing entry points for callers of the old buttons.
    def export_grid(self):
        return self.export_map('npz')

    def export_xyz(self):
        return self.export_map('csv')
