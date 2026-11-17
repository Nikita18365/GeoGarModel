"""Local colour library with CMYK TBL, colour zones and ER Mapper LUT."""
import json
import re
import zipfile
from pathlib import Path
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from PyQt5.QtCore import Qt,QSettings
from PyQt5.QtGui import QImage,QPixmap,QIcon
from PyQt5.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QComboBox,QLineEdit,QTreeWidget,
    QTreeWidgetItem,QPushButton,QDialogButtonBox,QFileDialog,QMessageBox)


def parse_palette(text,name):
    header=text.lower().split('}')[0]
    rows=[]
    for line in text.splitlines():
        if re.fullmatch(r'\s*[-+\d.eE]+(?:[\s,]+[-+\d.eE]+){2,5}\s*',line):
            rows.append([float(v) for v in re.split(r'[\s,]+',line.strip())])
    if not rows or len({len(r) for r in rows})!=1: raise ValueError('Не найдена таблица цветов: '+name)
    a=np.array(rows); zones=None
    if name.lower().endswith('.lut'):
        a=a[:,-3:]/(65535 if a[:,-3:].max()>255 else 255)
    else:
        if 'zone' in header: zones=a[:,0].tolist(); a=a[:,1:]
        if 'cyn' in header:
            if a.shape[1]==4: black=a[:,0:1]/255; cmy=a[:,1:]/255
            else: black=0; cmy=a/255
            a=(1-cmy)*(1-black)
        else: a=a[:,-3:]/255
    if not np.isfinite(a).all() or (a<0).any() or (a>1).any(): raise ValueError('Цвет вне диапазона: '+name)
    return dict(colors=a.tolist(),zones=zones)


def load_palettes(path):
    path=Path(path); result={}
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if Path(member.filename).suffix.lower() not in ('.tbl','.zon','.lut'): continue
                if member.file_size>2_000_000: raise ValueError('Слишком большой файл палитры.')
                key=member.filename.removeprefix('tbl/')
                result[key]=parse_palette(archive.read(member).decode('utf-8-sig',errors='replace'),key)
    else: result['Пользовательские/'+path.name]=parse_palette(path.read_text(encoding='utf-8-sig'),path.name)
    return result


def palette_map(name,colors=None,discrete=False,bins=32,reverse=False):
    cm=(LinearSegmentedColormap.from_list('custom',colors,N=256) if colors else colormaps[name])
    if reverse: cm=cm.reversed()
    return ListedColormap(cm(np.linspace(0,1,bins))) if discrete else cm


def preview(colors):
    a=(np.clip(np.array(colors),0,1)*255).astype(np.uint8)
    image=QImage(a.data,len(a),1,len(a)*3,QImage.Format_RGB888).copy()
    return QIcon(QPixmap.fromImage(image).scaled(280,24,Qt.IgnoreAspectRatio,Qt.FastTransformation))


class PaletteDialog(QDialog):
    def __init__(self,library,parent=None):
        super().__init__(parent); self.setWindowTitle('Выбор цветовой палитры'); self.resize(820,560)
        self.library=library; self.selected=None; root=QVBoxLayout(self); filters=QHBoxLayout()
        self.category=QComboBox(); self.search=QLineEdit(); self.search.setPlaceholderText('Поиск по названию')
        self.file_type=QComboBox(); self.file_type.addItems(['Все типы','.tbl','.zon','.lut'])
        filters.addWidget(self.category); filters.addWidget(self.file_type); filters.addWidget(self.search); root.addLayout(filters)
        self.tree=QTreeWidget(); self.tree.setHeaderLabels(['Раскраска','Название','Цветов'])
        self.tree.setIconSize(__import__('PyQt5.QtCore',fromlist=['QSize']).QSize(280,24))
        self.tree.setColumnWidth(0,295); self.tree.setColumnWidth(1,350); root.addWidget(self.tree)
        button=QPushButton('Импорт TBL / ZON / LUT / ZIP…'); button.clicked.connect(self.import_files); root.addWidget(button)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept_selection); buttons.rejected.connect(self.reject); root.addWidget(buttons)
        self.tree.itemDoubleClicked.connect(lambda *a:self.accept_selection())
        self.category.currentTextChanged.connect(self.populate); self.search.textChanged.connect(self.populate)
        self.file_type.currentTextChanged.connect(self.populate)
        self.refresh()

    def refresh(self):
        self.category.blockSignals(True); self.category.clear(); self.category.addItem('Все категории')
        self.category.addItems(sorted({k.split('/')[0] if '/' in k else 'Встроенные' for k in self.library}))
        self.category.blockSignals(False); self.populate()

    def populate(self):
        self.tree.clear()
        for name,p in sorted(self.library.items()):
            category=name.split('/')[0] if '/' in name else 'Встроенные'
            if self.category.currentIndex()>0 and category!=self.category.currentText(): continue
            if self.search.text().lower() not in name.lower(): continue
            if self.file_type.currentIndex()>0 and not name.lower().endswith(self.file_type.currentText()): continue
            item=QTreeWidgetItem(['',name.rsplit('/',1)[-1],str(len(p['colors']))])
            item.setIcon(0,preview(p['colors'])); item.setData(0,Qt.UserRole,name); self.tree.addTopLevelItem(item)

    def import_files(self):
        paths,_=QFileDialog.getOpenFileNames(self,'Импорт библиотеки палитр','','Палитры (*.zip *.tbl *.zon *.lut)')
        try:
            imported={}
            for path in paths: imported.update(load_palettes(path))
            self.library.update(imported); QSettings('GeoGarModel','Palettes').setValue('library',json.dumps(self.library))
            self.refresh()
        except (ValueError,OSError,zipfile.BadZipFile) as error: QMessageBox.warning(self,'Палитры',str(error))

    def accept_selection(self):
        item=self.tree.currentItem()
        if item: self.selected=item.data(0,Qt.UserRole); self.accept()


def local_library():
    try: library=json.loads(QSettings('GeoGarModel','Palettes').value('library','{}'))
    except (TypeError,ValueError): library={}
    for name in ('viridis','cividis','RdBu_r','coolwarm','terrain','seismic','rainbow','turbo'):
        library.setdefault(name,dict(colors=colormaps[name](np.linspace(0,1,256))[:,:3].tolist(),zones=None))
    return library
