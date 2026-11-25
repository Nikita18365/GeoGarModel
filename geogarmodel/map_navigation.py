"""Map navigation and geographic north in display coordinates."""
import numpy as np
from PyQt5.QtWidgets import QDoubleSpinBox,QLabel,QPushButton
from .map_presentation_ui import MapImageExportDialog
from .map_presentation import save_map_image
from .map_layout import DecorationDrag
from pyproj import CRS,Transformer
from .figure_toolbar import FigureToolbar
from matplotlib.backends.backend_qt import SubplotToolQt


class MapSubplotTool(SubplotToolQt):
    def __init__(self,workbench):
        self.workbench=workbench
        super().__init__(workbench.figure,workbench)

    def _on_value_changed(self):
        views=[(ax,ax.get_xlim(),ax.get_ylim()) for ax in self._figure.axes]
        super()._on_value_changed()
        for ax,x,y in views: ax.set_xlim(x); ax.set_ylim(y)
        self.workbench.remember_map_layout()

    def _tight_layout(self):
        views=[(ax,ax.get_xlim(),ax.get_ylim()) for ax in self._figure.axes]
        super()._tight_layout()
        for ax,x,y in views: ax.set_xlim(x); ax.set_ylim(y)
        self.workbench.remember_map_layout()


class MapToolbar(FigureToolbar):
    def __init__(self,canvas,workbench):
        self.workbench=workbench
        super().__init__(canvas,workbench)
        self.addSeparator()
        self.addWidget(QLabel('Масштаб'))
        self.zoom=QDoubleSpinBox()
        self.zoom.setRange(.1,100000); self.zoom.setDecimals(1)
        self.zoom.setSuffix(' %'); self.zoom.setValue(100)
        self.zoom.setKeyboardTracking(False); self.zoom.setToolTip('100 % — полный охват; увеличение относительно центра карты')
        self.addWidget(self.zoom)
        self.zoom.valueChanged.connect(lambda value:workbench.guard(lambda:workbench.set_map_zoom(value)))
        self.reset_zoom=QPushButton('100 %')
        self.reset_zoom.setToolTip('Вписать все видимые слои по центру')
        self.reset_zoom.clicked.connect(lambda: self.home())
        self.addWidget(self.reset_zoom)
        self.nominal_scale=QLabel('1 : —')
        self.nominal_scale.setToolTip('Расчётный масштаб листа в проекции; не физический масштаб монитора')
        self.addWidget(self.nominal_scale)
        self.addSeparator()
        self.edit_layout=QPushButton('Перемещать элементы')
        self.edit_layout.setCheckable(True)
        self.edit_layout.setToolTip('Левой кнопкой мыши перетащите заголовок, подпись оси, линейку, цветовую шкалу или север')
        self.addWidget(self.edit_layout)
        self.edit_layout.toggled.connect(self.toggle_layout_edit)
        self.reset_layout=QPushButton('Сброс позиций')
        self.reset_layout.clicked.connect(self.reset_positions)
        self.addWidget(self.reset_layout)
        self.decoration_drag=DecorationDrag(self)

    def toggle_layout_edit(self,enabled):
        if enabled:
            if self.mode.name=='PAN': self.pan()
            elif self.mode.name=='ZOOM': self.zoom()
            self.workbench.map_views.setCurrentWidget(self.canvas)
            self.workbench.statusBar().showMessage('Перетащите элемент оформления левой кнопкой мыши')
        else:
            self.decoration_drag.active=None

    def reset_positions(self,*args):
        self.decoration_drag.active=None
        self.workbench.map_offsets.clear()
        self.workbench.guard(self.workbench.redraw)

    def sync_scale(self):
        home=getattr(self.workbench,'_map_home',None)
        axes=self.canvas.figure.axes
        self.zoom.setEnabled(bool(home and axes))
        self.reset_zoom.setEnabled(bool(home and axes))
        if not home or not axes:
            self.nominal_scale.setText('1 : —'); return
        ax=axes[0]; width=abs(np.ptp(ax.get_xlim()))
        if width<=0: return
        self.zoom.blockSignals(True)
        self.zoom.setValue(100*abs(np.ptp(home[0]))/width)
        self.zoom.blockSignals(False)
        # Scale denominator is meaningful only with projected metre coordinates.
        try:
            crs=CRS.from_user_input(self.workbench.map_project_crs)
            metres=crs.is_projected and all(abs(a.unit_conversion_factor-1)<1e-9 for a in crs.axis_info[:2])
        except Exception:
            metres=False
        inches=ax.get_window_extent().width/self.canvas.figure.dpi
        self.nominal_scale.setText(f'1 : {width/(inches*.0254):,.0f}'.replace(',',' ') if metres and inches>0 else '1 : —')

    def save_figure(self,*args):
        dialog=MapImageExportDialog(self.workbench)
        if dialog.exec_()!=dialog.Accepted: return
        path,options=dialog.result_options
        try:
            self.workbench._map_exporting=True
            self.canvas.draw()
            save_map_image(self.canvas.figure,path,**options)
        except Exception as error:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self,'Экспорт изображения',str(error))
            return
        finally:
            self.workbench._map_exporting=False
            self.canvas.draw_idle()
        self.workbench.statusBar().showMessage('Изображение сохранено: '+str(path))
        return str(path)

    def home(self,*args):
        self.workbench.guard(self.workbench.fit_maps)

    def configure_subplots(self):
        self.workbench.auto_map_layout.setChecked(False)
        if self._subplot_dialog is None: self._subplot_dialog=MapSubplotTool(self.workbench)
        self._subplot_dialog.update_from_current_subplotpars()
        self._subplot_dialog.setModal(True); self._subplot_dialog.show()
        return self._subplot_dialog


def north_direction(crs,x,y):
    """Unit direction of geographic north in map XY (always east/north order)."""
    source=CRS.from_user_input(crs)
    geodetic=source.geodetic_crs
    inverse=Transformer.from_crs(source,geodetic,always_xy=True)
    forward=Transformer.from_crs(geodetic,source,always_xy=True)
    lon,lat=inverse.transform(x,y)
    lon2,lat2,_=source.get_geod().fwd(lon,lat,0,100.)
    xx,yy=forward.transform(lon2,lat2)
    delta=np.array([xx-x,yy-y]); size=np.linalg.norm(delta)
    if not np.isfinite(delta).all() or size<=0: raise ValueError('Направление севера не определено в этой точке.')
    return delta/size
