"""Independent, non-modal CPU preview of one completed inversion."""
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
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
        self.axes, self.values = preview_volume(result)
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
        model_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
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
        counts = ' × '.join(map(str, result.mesh.shape))
        footer.addWidget(QLabel(f'Сетка результата {counts}. В окне показан обзор.'))
        footer.addStretch()
        export = QPushButton('Сохранить…')
        menu = QMenu(export)
        for name, save in [('Модель и поле · NPZ', tab.save), ('Модель · UBC', tab.save_ubc),
                           ('Модель · VTK', tab.save_vtk), ('Поле · CSV / ZMAP', tab.save_fields)]:
            menu.addAction(name, lambda checked=False, fn=save: tab.workbench.guard(lambda: fn(result)))
        export.setMenu(menu)
        footer.addWidget(export)
        layout.addLayout(footer)
        self.show_layer(0)

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
        layout.addWidget(NavigationToolbar2QT(canvas, self))
        ax = figure.add_subplot()
        r = self.result
        stride = max(1, (len(r.predicted) + 99999) // 100000)
        obs = r.observations[::stride]
        image = ax.scatter(obs[:, 0], obs[:, 1], c=r.predicted[::stride], s=8, cmap='coolwarm')
        ax.set(xlabel='East, м', ylabel='North, м', aspect='equal')
        figure.colorbar(image, ax=ax, label='мГал' if r.settings['mode'] == 'gravity' else 'нТл')
        canvas.draw_idle()
