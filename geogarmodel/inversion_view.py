"""VTK scene with cell thresholding, slices, volume and isosurfaces; lazy imports."""
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QDoubleSpinBox, QLabel, QSlider, QSpinBox, QPushButton, QDialog
from PyQt5.QtCore import Qt


def vtk_grid(result, component='amplitude'):
    import pyvista as pv
    mesh=result.mesh
    # VTK uses an upward vertical axis; preserve coordinates in exports.
    grid=pv.RectilinearGrid(mesh.east,mesh.north,-mesh.depth[::-1])
    m=result.model
    if m.ndim==2:
        values=np.linalg.norm(m,axis=1) if component=='amplitude' else m[:,{'East':0,'North':1,'Down':2}[component]]
        grid.cell_data['M_END']=m.reshape(mesh.shape+(3,))[:,:,::-1,:].reshape(-1,3,order='F')
    else:
        values=m
    grid.cell_data['property']=values.reshape(mesh.shape)[:,:,::-1].ravel(order='F')
    if mesh.bounds is not None or not mesh.active.all():
        # Exact vertically clipped prisms, independently represented as hexahedra.
        cells=mesh.cells()[mesh.active]
        corners=np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1],[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
        points=cells[:,None,:3]+corners[None,:,:]*(cells[:,None,3:]-cells[:,None,:3])
        points[:,:,2]*=-1
        connectivity=np.column_stack((np.full(len(cells),8),np.arange(len(cells)*8).reshape(-1,8)))
        grid=pv.UnstructuredGrid(connectivity.ravel(),np.full(len(cells),pv.CellType.HEXAHEDRON),points.reshape(-1,3))
        grid.cell_data['property']=values[mesh.active]
        if m.ndim==2: grid.cell_data['M_END']=m[mesh.active]
    return grid


class ModelView(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        layout=QVBoxLayout(self)
        self.info=QLabel('Для интерактивного 3D нажмите «Открыть / обновить 3D». Требуется geogarmodel[volume].')
        self.info.setWordWrap(True); layout.addWidget(self.info)
        self.controls=QHBoxLayout(); layout.addLayout(self.controls)
        self.kind=QComboBox(); self.kind.addItems(['Воксели','Объём','Изоповерхности','Срезы XYZ','Интерактивное отсечение'])
        self.component=QComboBox(); self.component.addItems(['amplitude','East','North','Down'])
        self.threshold=QDoubleSpinBox(); self.threshold.setRange(0,99); self.threshold.setValue(20); self.threshold.setSuffix('% от max |m|')
        for w in (self.kind,self.component,self.threshold): self.controls.addWidget(w)
        slices=QHBoxLayout(); layout.addLayout(slices); self.slices=[]
        for name in ('East','North','Up'):
            slices.addWidget(QLabel(name))
            slider=QSlider(Qt.Horizontal); slider.setRange(0,100); slider.setValue(50)
            slider.sliderReleased.connect(self.refresh_if_ready)
            slices.addWidget(slider); self.slices.append(slider)
        scale_row=QHBoxLayout(); layout.addLayout(scale_row); self.axis_scales=[]
        for name in ('X / East','Y / North','Z / Up'):
            scale_row.addWidget(QLabel(name))
            scale=QDoubleSpinBox(); scale.setRange(.01,1000); scale.setDecimals(2)
            scale.setValue(1); scale.setSuffix(' ×'); scale.setKeyboardTracking(False)
            scale.setToolTip('Визуальное растяжение оси. Координаты модели и экспорт сохраняются.')
            scale.valueChanged.connect(self.rescale)
            scale_row.addWidget(scale); self.axis_scales.append(scale)
        self.reset_scale=QPushButton('Масштаб 1:1'); scale_row.addWidget(self.reset_scale)
        self.reset_scale.clicked.connect(self.reset_scales)
        view_row=QHBoxLayout(); layout.addLayout(view_row)
        view_row.addWidget(QLabel('Подписи осей'))
        self.axis_font=QSpinBox(); self.axis_font.setRange(10,36); self.axis_font.setValue(18)
        self.axis_font.setSuffix(' px'); self.axis_font.setKeyboardTracking(False)
        view_row.addWidget(self.axis_font)
        self.axis_unit=QComboBox(); self.axis_unit.addItems(['м','км']); view_row.addWidget(self.axis_unit)
        self.camera_view=QComboBox(); self.camera_view.addItems(['Изометрия','Сверху XY','Спереди XZ','Сбоку YZ'])
        view_row.addWidget(self.camera_view)
        self.fit=QPushButton('Вписать модель'); view_row.addWidget(self.fit)
        self.expand=QPushButton('Большое окно'); view_row.addWidget(self.expand)
        self.expand.clicked.connect(self.expanded_view)
        self.fit.clicked.connect(self.fit_model)
        self.camera_view.activated.connect(self.fit_model)
        self.axis_font.valueChanged.connect(self.refresh_if_ready)
        self.axis_unit.currentTextChanged.connect(self.refresh_if_ready)
        self.plotter=None; self.result=None
        self._expanded_dialog=None
        self._fit_pending=False
        for w in (self.kind,self.component): w.currentTextChanged.connect(self.refresh_if_ready)
        self.threshold.valueChanged.connect(self.refresh_if_ready)

    def rescale(self):
        self._fit_pending=True
        self.refresh_if_ready()

    def reset_scales(self):
        for scale in self.axis_scales:
            scale.blockSignals(True); scale.setValue(1); scale.blockSignals(False)
        self.rescale()

    def fit_model(self):
        if self.plotter is None or self.result is None: return
        p=self.plotter
        [p.view_isometric,p.view_xy,p.view_xz,p.view_yz][self.camera_view.currentIndex()]()
        p.reset_camera(); p.camera.zoom(.82); p.render()

    def expanded_view(self):
        if self.result is None: return
        if self._expanded_dialog is not None:
            self._expanded_dialog.raise_(); self._expanded_dialog.activateWindow(); return
        dialog=QDialog(self); dialog.setWindowTitle('3D-модель · масштаб и координаты')
        self._expanded_dialog=dialog
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint)
        layout=QVBoxLayout(dialog); view=ModelView(dialog); layout.addWidget(view)
        view.expand.hide()
        for source,target in zip(self.axis_scales,view.axis_scales): target.setValue(source.value())
        for source,target in ((self.kind,view.kind),(self.component,view.component),
                              (self.axis_unit,view.axis_unit),(self.camera_view,view.camera_view)):
            target.setCurrentIndex(source.currentIndex())
        view.threshold.setValue(self.threshold.value()); view.axis_font.setValue(self.axis_font.value())
        for source,target in zip(self.slices,view.slices): target.setValue(source.value())
        def cleanup():
            view.shutdown(); self._expanded_dialog=None; dialog.deleteLater()
        dialog.finished.connect(cleanup)
        try:
            dialog.showMaximized(); view.render(self.result)
        except Exception:
            dialog.reject(); raise

    def refresh_if_ready(self):
        if self.result is not None and self.plotter is not None:
            self.render(self.result)

    def render(self,result):
        try:
            from pyvistaqt import QtInteractor
        except ImportError as exc:
            raise ValueError('Установите пакет визуализации: pip install -e ".[volume]"') from exc
        if self.plotter is None:
            self.plotter=QtInteractor(self)
            self.plotter.enable_parallel_projection()
            self.layout().addWidget(self.plotter.interactor,1)
        new_model=self.result is not result
        self.result=result
        p=self.plotter
        camera=p.camera_position if p.renderer.actors else None
        camera_scale=p.camera.parallel_scale
        p.clear(); p.clear_plane_widgets()
        p.set_background('#101c2f')
        grid=vtk_grid(result,self.component.currentText())
        # Use a local origin only in the display; exported VTK retains full CRS coordinates.
        origin=np.array([result.mesh.east[0],result.mesh.north[0],0.])
        grid.translate(-origin,inplace=True)
        # Transform display geometry only, so slices and clip widgets share its coordinates.
        # Axes below retain the unscaled physical coordinate ranges.
        physical_bounds=np.array(grid.bounds)
        scale=np.array([widget.value() for widget in self.axis_scales])
        grid.scale(scale,inplace=True)
        mode=result.settings['mode']
        unit='g/cm³' if mode=='gravity' else ('SI' if mode=='induced' else 'A/m')
        self.info.setText(f'{result.mesh.active.sum():,} активных ячеек · {unit} · X = East − E₀, Y = North − N₀; E₀={origin[0]:.1f}, N₀={origin[1]:.1f} м; Z = Up (вверх). Вращение: ЛКМ, масштаб: колесо.')
        self.component.setEnabled(result.model.ndim==2)
        values=grid['property']; amplitude=float(np.max(np.abs(values))); limit=max(amplitude,1e-15)
        cmap='viridis' if np.min(values)>=0 else 'coolwarm'
        clim=(0,limit) if np.min(values)>=0 else (-limit,limit)
        options=dict(scalars='property',cmap=cmap,clim=clim,scalar_bar_args={
            'title':unit,'color':'#f0f5ff','title_font_size':18,'label_font_size':16,
            'n_labels':4,'fmt':'%.3g','position_x':.88,'position_y':.18,'width':.07,'height':.65,'vertical':True})
        kind=self.kind.currentIndex()
        if kind==0:
            chosen=grid.extract_cells(np.flatnonzero(np.abs(values)>=amplitude*self.threshold.value()/100))
            if chosen.n_cells: p.add_mesh(chosen,show_edges=True,edge_color='#33465a',**options)
        elif kind==1:
            if grid.__class__.__name__=='UnstructuredGrid':
                p.add_mesh(grid,opacity=.3,**options)
                self.info.setText(self.info.text()+' · Обрезанные призмы: полупрозрачная геометрия.')
            else:
                p.add_volume(grid.cell_data_to_point_data(),opacity='sigmoid',**options)
        elif kind==2:
            points=grid.cell_data_to_point_data()
            lo,hi=float(values.min()),float(values.max())
            if hi>lo:
                surface=points.contour(np.linspace(lo,hi,7)[1:-1],scalars='property')
                if surface.n_points: p.add_mesh(surface,opacity=.65,**options)
        elif kind==3:
            centers=[grid.bounds[2*i]+(grid.bounds[2*i+1]-grid.bounds[2*i])*slider.value()/100 for i,slider in enumerate(self.slices)]
            p.add_mesh(grid.slice_orthogonal(x=centers[0],y=centers[1],z=centers[2]),**options)
            p.add_mesh(grid.outline(),color='#93a9c4')
        else:
            p.add_mesh_clip_plane(grid,normal='x',**options)
        p.add_axes(xlabel='X / East',ylabel='Y / North',zlabel='Z / Up',color='#f0f5ff')
        divisor=1000 if self.axis_unit.currentText()=='км' else 1
        unit_label='km' if divisor==1000 else 'm'
        self.bounds_axes=p.show_grid(bounds=grid.bounds,axes_ranges=physical_bounds/divisor,
                    color='#e3edff',font_size=self.axis_font.value(),bold=True,fmt='%.3g',
                    use_2d=False,use_3d_text=False,
                    xtitle=f'X ({unit_label})',ytitle=f'Y ({unit_label})',
                    ztitle=f'Z ({unit_label})',n_xlabels=4,n_ylabels=4,n_zlabels=4)
        self.bounds_axes.GetProperty().SetLineWidth(1.5)
        if camera is None or new_model or self._fit_pending: self.fit_model()
        else:
            p.camera_position=camera
            p.camera.parallel_scale=camera_scale
        self._fit_pending=False
        p.render()

    def shutdown(self):
        if self._expanded_dialog is not None:
            self._expanded_dialog.close()
        if self.plotter is not None:
            self.plotter.close()
