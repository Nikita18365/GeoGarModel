"""Paged maps of horizontal model slices, with a common physical colour scale."""
import numpy as np
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QComboBox,
    QLineEdit,QDoubleSpinBox,QSpinBox,QDialogButtonBox,QPushButton,QLabel,QMessageBox,QFileDialog,QProgressDialog,QApplication)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .model_slices import select_depths,horizontal_slice,property_unit
from .theme import style_map_figure


class SliceDepthDialog(QDialog):
    def __init__(self,mesh,parent=None):
        super().__init__(parent); self.setWindowTitle('Горизонтальные срезы модели'); self.mesh=mesh; self.depths=None
        form=QFormLayout(self); self.mode=QComboBox()
        for label,key in [('Свои глубины','manual'),('Каждый N-й слой','layers'),('Диапазон с шагом, м','step')]: self.mode.addItem(label,key)
        self.manual=QLineEdit('; '.join(f'{v:g}' for v in np.linspace(mesh.depth[0],mesh.depth[-1],5)[1:-1]))
        self.top,self.bottom,self.step=QDoubleSpinBox(),QDoubleSpinBox(),QDoubleSpinBox()
        for w in (self.top,self.bottom): w.setRange(mesh.depth[0],mesh.depth[-1]); w.setDecimals(3); w.setKeyboardTracking(False)
        self.top.setValue(mesh.depth[0]); self.bottom.setValue(mesh.depth[-1])
        self.step.setRange(.001,1e9); self.step.setDecimals(3); self.step.setValue(np.median(np.diff(mesh.depth)))
        self.every=QSpinBox(); self.every.setRange(1,1000000)
        form.addRow('Способ выбора',self.mode); form.addRow('Глубины Down, м (через ;)',self.manual)
        form.addRow('Верхняя граница Down, м',self.top); form.addRow('Нижняя граница Down, м',self.bottom)
        form.addRow('Шаг глубины, м',self.step); form.addRow('Каждый N-й слой',self.every)
        note=QLabel('Down — глубина в метрах, положительная вниз. Карты строятся по пересечению активных ячеек, с учётом обрезки поверхности.'); note.setWordWrap(True); form.addRow(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel); buttons.accepted.connect(self.validate); buttons.rejected.connect(self.reject); form.addRow(buttons)
        self.mode.currentIndexChanged.connect(self.visibility); self.visibility()

    def visibility(self):
        mode=self.mode.currentData()
        for w,visible in [(self.manual,mode=='manual'),(self.top,mode!='manual'),(self.bottom,mode!='manual'),(self.step,mode=='step'),(self.every,mode=='layers')]:
            w.setVisible(visible); self.layout().labelForField(w).setVisible(visible)

    def validate(self):
        try: self.depths=select_depths(self.mesh,self.mode.currentData(),self.manual.text(),self.top.value(),self.bottom.value(),self.step.value(),self.every.value())
        except ValueError as error: QMessageBox.warning(self,'Глубины срезов',str(error)); return
        self.accept()


class SliceGallery(QDialog):
    def __init__(self,result,depths,cmap,norm,component='amplitude',parent=None):
        super().__init__(parent); self.setWindowTitle('Карты горизонтальных срезов'); self.resize(1300,800)
        self.setWindowFlags(self.windowFlags()|Qt.WindowMaximizeButtonHint)
        self.result=result; self.depths=depths; self.cmap=cmap; self.norm=norm; self.component=component; self.page=0
        root=QVBoxLayout(self); toolbar=QHBoxLayout(); root.addLayout(toolbar)
        previous=QPushButton('← Назад'); following=QPushButton('Далее →'); self.page_label=QLabel()
        self.count=QSpinBox(); self.count.setRange(1,3); self.count.setValue(min(3,len(depths)))
        toolbar.addWidget(previous); toolbar.addWidget(following); toolbar.addWidget(self.page_label,1); toolbar.addWidget(QLabel('Карт на странице')); toolbar.addWidget(self.count)
        self.figure=Figure(facecolor='white'); self.canvas=FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect('resize_event',lambda event:self.relayout())
        self.toolbar=NavigationToolbar2QT(self.canvas,self); root.addWidget(self.toolbar); root.addWidget(self.canvas,1)
        previous.clicked.connect(lambda:self.change(-1)); following.clicked.connect(lambda:self.change(1)); self.count.valueChanged.connect(lambda *a:self.change(0,reset=True))
        export=QPushButton('Сохранить все карты в PDF…'); toolbar.addWidget(export); export.clicked.connect(self.export_pdf)
        self.render_page()
        QTimer.singleShot(0,self.relayout)

    def relayout(self):
        if self.figure.axes:
            self.figure.tight_layout(pad=1.6,w_pad=1.4); self.canvas.draw_idle()

    def change(self,delta,reset=False):
        maximum=(len(self.depths)-1)//self.count.value()
        self.page=0 if reset else min(maximum,max(0,self.page+delta)); self.render_page()

    def render_page(self):
        self.figure.clear(); start=self.page*self.count.value(); selected=self.depths[start:start+self.count.value()]
        self.toolbar.update()
        mesh=self.result.mesh; unit=property_unit(self.result)
        for i,depth in enumerate(selected):
            section=horizontal_slice(self.result,float(depth),self.component); ax=self.figure.add_subplot(1,len(selected),i+1)
            if section.rectangles is None:
                image=ax.pcolormesh(mesh.east,mesh.north,section.values,cmap=self.cmap,norm=self.norm,shading='flat',rasterized=True)
            else:
                r=section.rectangles; polygons=np.stack([r[:,[0,1]],r[:,[2,1]],r[:,[2,3]],r[:,[0,3]]],axis=1)
                image=PolyCollection(polygons,array=section.rectangle_values,cmap=self.cmap,norm=self.norm,rasterized=True,edgecolors='none'); ax.add_collection(image)
                ax.set_xlim(mesh.east[0],mesh.east[-1]); ax.set_ylim(mesh.north[0],mesh.north[-1])
            divider=make_axes_locatable(ax); color_ax=divider.append_axes('right',size='5%',pad=.08)
            self.figure.colorbar(image,cax=color_ax,label=unit)
            ax.set(xlabel='East, м',ylabel='North, м',title=f'Глубина Down = {depth:g} м'); ax.set_aspect('equal')
            ax.ticklabel_format(style='plain',useOffset=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3)); ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(labelsize=9)
            if not np.isfinite(section.values).any(): ax.text(.5,.5,'Нет активных ячеек',transform=ax.transAxes,ha='center')
        style_map_figure(self.figure); self.figure.tight_layout(pad=1.8)
        self.page_label.setText(f'Срезы {start+1}–{start+len(selected)} из {len(self.depths)} · {self.component}')
        self.canvas.draw_idle()

    def export_pdf(self):
        path,_=QFileDialog.getSaveFileName(self,'Все срезы в PDF','','PDF (*.pdf)')
        if not path: return
        from matplotlib.backends.backend_pdf import PdfPages
        page=self.page
        pages=(len(self.depths)+self.count.value()-1)//self.count.value()
        progress=QProgressDialog('Сохранение карт срезов…','Остановить',0,pages,self)
        progress.setWindowModality(Qt.WindowModal); progress.setMinimumDuration(500)
        try:
            with PdfPages(path if path.lower().endswith('.pdf') else path+'.pdf') as pdf:
                for self.page in range(pages):
                    progress.setValue(self.page); QApplication.processEvents()
                    if progress.wasCanceled(): break
                    self.render_page(); pdf.savefig(self.figure)
        except (OSError,ValueError) as error: QMessageBox.warning(self,'Экспорт срезов',str(error))
        finally: progress.close(); self.page=page; self.render_page()
