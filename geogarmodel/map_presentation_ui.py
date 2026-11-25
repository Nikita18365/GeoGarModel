"""Compact font editor and publication image export options."""
from pathlib import Path
from PyQt5.QtGui import QColor,QFont
from PyQt5.QtWidgets import (QWidget,QFormLayout,QComboBox,QFontComboBox,QSpinBox,
    QDoubleSpinBox,QCheckBox,QPushButton,QLabel,QColorDialog,QHBoxLayout,QDialog,
    QDialogButtonBox,QFileDialog,QVBoxLayout)
from .map_presentation import FONT_ELEMENTS,text_options
from .ui_layout import quiet_button


class MapFontEditor(QWidget):
    def __init__(self,window):
        super().__init__(window)
        self.window=window
        form=QFormLayout(self); form.setContentsMargins(0,0,0,0)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.element=QComboBox()
        for key,(name,_,_) in FONT_ELEMENTS.items(): self.element.addItem(name,key)
        self.family=QFontComboBox()
        self.size=QSpinBox(); self.size.setRange(4,72); self.size.setSuffix(' pt')
        self.bold=QCheckBox('Жирный'); self.italic=QCheckBox('Курсив')
        self.color_button=QPushButton('Цвет текста…')
        self.preview=QLabel('Пример · 123,45'); self.preview.setWordWrap(True)
        self.apply_button=QPushButton('Применить к элементу')
        self.reset_button=QPushButton('Сбросить элемент')
        for label,widget in [('Элемент',self.element),('Шрифт',self.family),('Размер',self.size),
                             ('',self.bold),('',self.italic),('Цвет',self.color_button),('Предпросмотр',self.preview),
                             ('',self.apply_button),('',self.reset_button)]: form.addRow(label,widget)
        for button in (self.color_button,self.apply_button,self.reset_button): quiet_button(button)
        self.element.currentIndexChanged.connect(self.load_element)
        self.family.currentFontChanged.connect(self.update_preview)
        self.size.valueChanged.connect(self.update_preview)
        self.bold.toggled.connect(self.update_preview); self.italic.toggled.connect(self.update_preview)
        self.color_button.clicked.connect(self.choose_color)
        self.apply_button.clicked.connect(self.apply)
        self.reset_button.clicked.connect(self.reset)
        self.color=None
        self.load_element()

    def load_element(self,*args):
        key=self.element.currentData()
        options=text_options(self.window.map_fonts,key)
        self.color=options.get('color')
        self.family.setCurrentFont(QFont(options['fontfamily']))
        self.size.setValue(int(options['fontsize']))
        self.bold.setChecked(options['fontweight']=='bold')
        self.italic.setChecked(options['fontstyle']=='italic')
        self.update_preview()

    def update_preview(self,*args):
        font=self.family.currentFont(); font.setPointSize(self.size.value())
        font.setBold(self.bold.isChecked()); font.setItalic(self.italic.isChecked())
        self.preview.setFont(font)
        color=self.color or '#777777'
        self.preview.setStyleSheet('background: white; color: '+color+'; padding: 8px;')
        self.color_button.setStyleSheet('QPushButton { border-left: 6px solid '+color+'; }')

    def choose_color(self):
        value=QColorDialog.getColor(QColor(self.color or '#222222'),self,'Цвет текста')
        if value.isValid(): self.color=value.name(); self.update_preview()

    def apply(self):
        self.window.map_fonts[self.element.currentData()]=dict(family=self.family.currentFont().family(),
            size=self.size.value(),bold=self.bold.isChecked(),italic=self.italic.isChecked(),color=self.color)
        self.window.guard(self.window.redraw)

    def reset(self):
        self.window.map_fonts.pop(self.element.currentData(),None)
        self.load_element()
        self.window.guard(self.window.redraw)


class MapImageExportDialog(QDialog):
    def __init__(self,parent):
        super().__init__(parent)
        self.setWindowTitle('Экспорт оформления карты')
        root=QVBoxLayout(self); form=QFormLayout(); root.addLayout(form)
        self.format=QComboBox()
        for name,ext in [('PNG · без потерь','png'),('JPEG · фото','jpg'),('TIFF · без потерь','tif'),
                         ('PDF · векторный лист','pdf'),('SVG · векторный лист','svg')]:
            self.format.addItem(name,ext)
        self.dpi=QSpinBox(); self.dpi.setRange(72,1200); self.dpi.setValue(300); self.dpi.setSuffix(' dpi')
        self.padding=QDoubleSpinBox(); self.padding.setRange(0,30); self.padding.setDecimals(2); self.padding.setSuffix(' мм')
        self.transparent=QCheckBox('Прозрачный фон')
        self.quality=QSpinBox(); self.quality.setRange(1,100); self.quality.setValue(95); self.quality.setSuffix(' %')
        form.addRow('Формат',self.format); form.addRow('Разрешение',self.dpi)
        form.addRow('Внешний отступ',self.padding); form.addRow(self.transparent)
        form.addRow('Качество JPEG',self.quality)
        note=QLabel('Экспортируется текущий вид с подписями, шкалами и легендой. '
                    'Лишние внешние поля обрезаются. Для PDF/SVG DPI относится к растровым частям.')
        note.setWordWrap(True); root.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.choose_path); buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.format.currentIndexChanged.connect(self.update_format); self.update_format()

    def update_format(self,*args):
        jpeg=self.format.currentData()=='jpg'
        self.quality.setEnabled(jpeg); self.transparent.setEnabled(not jpeg)

    def choose_path(self):
        ext=self.format.currentData()
        path,_=QFileDialog.getSaveFileName(self,'Сохранить изображение','map.'+ext,ext.upper()+' (*.'+ext+')')
        if not path: return
        if Path(path).suffix.lower() not in ('.'+ext, '.jpeg' if ext=='jpg' else '.tiff' if ext=='tif' else '.'+ext):
            path+='.'+ext
        self.result_options=(path,dict(dpi=self.dpi.value(),padding_mm=self.padding.value(),
                          transparent=self.transparent.isChecked(),jpeg_quality=self.quality.value()))
        self.accept()
