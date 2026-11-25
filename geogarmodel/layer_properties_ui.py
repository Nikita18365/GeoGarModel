"""Selectable, read-only property dialog shared by all map layer types."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication,QDialog,QDialogButtonBox,QFormLayout,
                             QGroupBox,QLabel,QScrollArea,QVBoxLayout,QWidget)


class LayerPropertiesDialog(QDialog):
    def __init__(self,name,sections,parent=None):
        super().__init__(parent)
        self.setWindowTitle('Свойства слоя — '+name)
        self.resize(720,650)
        self.sections = sections
        layout = QVBoxLayout(self)
        note = QLabel('Сведения о текущем слое проекта. Исходный путь — на момент импорта.')
        note.setWordWrap(True)
        layout.addWidget(note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        groups = QVBoxLayout(content)
        for title,rows in sections:
            group = QGroupBox(title)
            form = QFormLayout(group)
            form.setRowWrapPolicy(QFormLayout.WrapLongRows)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            for key,value in rows:
                label = QLabel(str(value))
                label.setTextFormat(Qt.PlainText)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse|Qt.TextSelectableByKeyboard)
                label.setWordWrap(True)
                label.setMinimumWidth(100)
                label.setAccessibleName(key)
                form.addRow(key,label)
            groups.addWidget(group)
        groups.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll,1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        copy = buttons.addButton('Копировать всё',QDialogButtonBox.ActionRole)
        copy.clicked.connect(self.copy_all)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def copy_all(self):
        text = '\n\n'.join(title+'\n'+'\n'.join(f'{key}: {value}' for key,value in rows)
                           for title,rows in self.sections)
        QApplication.clipboard().setText(text)
