"""Independent, non-modal CPU preview of one completed inversion."""
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from .figure_toolbar import FigureToolbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMenu, QPushButton,
                             QSlider, QTabWidget, QVBoxLayout, QWidget)
from .result_preview import preview_volume, surface_faces


class ResultWindow(QDialog):
    def __init__(self, tab, result, number):
        super().__init__(tab, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(f'Результат 3D-моделирования · расчёт {number}')
        self.resize(900, 680)
        self.result = result
        from .model_archive import ModelArchive
        archived=isinstance(result,ModelArchive)
        self.axes, self.values = result.preview if archived else tab.workbench.activity.run('Обзор результата',preview_volume,result)
        self.unit = {'gravity': 'г/см³', 'induced': 'ед. СИ', 'vector': 'А/м'}[result.settings['mode']]
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(projection='3d')
        self.ax.set(xlabel='East, м', ylabel='North, м', zlabel='Глубина, м',
                    xlim=self.axes[0][[0, -1]], ylim=self.axes[1][[0, -1]],
                    zlim=self.axes[2][[-1, 0]])
        self.actor = Poly3DCollection([], linewidths=0, antialiaseds=False, cmap='coolwarm')
        finite = self.values[np.isfinite(self.values)]
        if finite.size:
            self.actor.set_norm(Normalize(finite.min(), finite.max()))
        self.ax.add_collection3d(self.actor)
        self.figure.colorbar(self.actor, ax=self.ax, shrink=.7, label=self.unit)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        model_page = QWidget()
        model_layout = QVBoxLayout(model_page)
        model_layout.addWidget(self.canvas)
        model_layout.addWidget(FigureToolbar(self.canvas, self))
        row = QHBoxLayout()
        self.depth = QLabel()
        row.addWidget(self.depth)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, self.values.shape[2] - 1)
        slider.setTracking(False)
        slider.valueChanged.connect(self.show_layer)
        row.addWidget(slider)
        model_layout.addLayout(row)
        self.tabs.addTab(model_page, '3D-модель')
        self.field_page = QWidget()
        self.tabs.addTab(self.field_page, 'Рассчитанное поле')
        self.tabs.currentChanged.connect(self.show_field)
        footer = QHBoxLayout()
        counts = ' × '.join(map(str, result.shape if archived else result.mesh.shape))
        footer.addWidget(QLabel(f'Сетка результата {counts}. В окне показан обзор.'))
        footer.addStretch()
        export = QPushButton('Сохранить…')
        menu = QMenu(export)
        def full_result():
            return tab.workbench.activity.run('Чтение полной модели',self.result.load) if isinstance(self.result,ModelArchive) else self.result
        if archived:
            load_button=QPushButton('Загрузить полностью')
            def load_full():
                if tab.worker or tab.workbench.worker:
                    raise ValueError('Дождитесь завершения текущего расчёта.')
                def loaded(r):
                    r.settings['source_path']=str(result.path.resolve())
                    tab.finished_result(r,offer_export=False)
                    tab.curve.setData([v['iteration'] for v in r.history],[v['rmse'] for v in r.history])
                    from PyQt5 import sip
                    if not sip.isdeleted(load_button):
                        self.result=r
                        load_button.setEnabled(False)
                tab.model_task(result.load,loaded)
            load_button.clicked.connect(lambda:tab.workbench.guard(load_full))
            footer.addWidget(load_button)
            layers=QPushButton('Точный слой…')
            layers.clicked.connect(lambda:tab.workbench.guard(lambda:self.open_layer(result)))
            footer.addWidget(layers)
        for name, save in [('Модель и поле · GeoGarModel / NPZ', tab.save), ('Модель · UBC / Oasis montaj', tab.save_ubc),
                           ('Модель · VTK', tab.save_vtk), ('Модель · GOCAD Voxet',tab.save_voxet),
                           ('Поле · CSV / ZMAP', tab.save_fields)]:
            menu.addAction(name, lambda checked=False, fn=save: tab.workbench.guard(
                lambda: fn(self.result if fn==tab.save_fields else full_result())))
        if result.settings.get('mode') in ('induced','vector'):
            tab.magnetic_export_menu(menu,full_result)
        export.setMenu(menu)
        footer.addWidget(export)
        layout.addLayout(footer)
        self.show_layer(0)

    def open_layer(self,archive):
        from PyQt5.QtWidgets import QInputDialog
        k,ok=QInputDialog.getInt(self,'Горизонтальный слой','Номер слоя от поверхности',1,1,archive.shape[2])
        if not ok: return
        values,active=self.parentWidget().workbench.activity.run('Чтение слоя модели',archive.layer,k-1)
        if values.ndim==3: values=np.linalg.norm(values,axis=-1)
        dialog=QDialog(self,Qt.Window)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.setWindowTitle(f'Слой {k} · {archive.axes[2][k-1]:g}…{archive.axes[2][k]:g} м')
        layout=QVBoxLayout(dialog)
        figure=Figure(constrained_layout=True)
        canvas=FigureCanvasQTAgg(figure)
        layout.addWidget(canvas)
        layout.addWidget(FigureToolbar(canvas,dialog))
        ax=figure.add_subplot()
        im=ax.pcolormesh(*archive.axes[:2],np.ma.array(values,mask=~active).T,cmap='coolwarm')
        ax.set(xlabel='East, м',ylabel='North, м',aspect='equal')
        figure.colorbar(im,ax=ax,label=self.unit)
        dialog.resize(800,600)
        dialog.show()

    def show_layer(self, index):
        faces, colors = surface_faces(self.axes, self.values, index)
        self.actor.set_verts(faces)
        self.actor.set_array(colors)
        self.depth.setText(f'Верх среза  {self.axes[2][index]:g} м')
        self.canvas.draw_idle()

    def show_field(self, index):
        if index != 1 or self.field_page.layout() is not None:
            return
        layout = QVBoxLayout(self.field_page)
        figure = Figure(constrained_layout=True)
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(canvas)
        layout.addWidget(FigureToolbar(canvas, self))
        ax = figure.add_subplot()
        r = self.result
        stride = max(1, (len(r.predicted) + 99999) // 100000)
        obs = r.observations[::stride]
        image = ax.scatter(obs[:, 0], obs[:, 1], c=r.predicted[::stride], s=8, cmap='coolwarm')
        ax.set(xlabel='East, м', ylabel='North, м', aspect='equal')
        figure.colorbar(image, ax=ax, label='мГал' if r.settings['mode'] == 'gravity' else 'нТл')
        canvas.draw_idle()
