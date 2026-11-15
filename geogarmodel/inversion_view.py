"""VTK scene with cell thresholding, slices, volume and isosurfaces; lazy imports."""
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QDoubleSpinBox, QLabel, QSlider, QSpinBox, QPushButton, QDialog, QCheckBox, QScrollArea, QGroupBox, QFormLayout
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
            scale=QDoubleSpinBox(); scale.setRange(.001,1000); scale.setDecimals(3)
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
        colors=QHBoxLayout(); layout.addLayout(colors)
        colors.addWidget(QLabel('Палитра'))
        self.palette=QComboBox(); self.palette.addItems(['Авто','viridis','cividis','plasma','turbo','coolwarm','RdBu','terrain','gray'])
        colors.addWidget(self.palette)
        self.reverse_palette=QCheckBox('Развернуть'); colors.addWidget(self.reverse_palette)
        self.color_range=QComboBox(); self.color_range.addItems(['Авто диапазон','Симметрично ±','Вручную'])
        colors.addWidget(self.color_range)
        self.color_min,self.color_max=QDoubleSpinBox(),QDoubleSpinBox()
        for label,control in [('Мин.',self.color_min),('Макс.',self.color_max)]:
            colors.addWidget(QLabel(label)); colors.addWidget(control)
            control.setRange(-1e12,1e12); control.setDecimals(8); control.setKeyboardTracking(False)
            control.setEnabled(False)
        self.color_max.setValue(1)
        self.apply_colors=QPushButton('Применить цвета'); colors.addWidget(self.apply_colors)
        self.color_note=QLabel(''); layout.addWidget(self.color_note)
        self.palette.currentTextChanged.connect(self.refresh_if_ready)
        self.reverse_palette.toggled.connect(self.refresh_if_ready)
        self.color_range.currentIndexChanged.connect(self.range_changed)
        self.apply_colors.clicked.connect(self.refresh_if_ready)
        self.plotter=None; self.result=None
        self._expanded_dialog=None
        self._fit_pending=False
        self._reorient_pending=False
        self._display_scale=np.ones(3)
        self.display_grid=None
        for w in (self.kind,self.component): w.currentTextChanged.connect(self.refresh_if_ready)
        self.threshold.valueChanged.connect(self.refresh_if_ready)
        # All display controls remain live, but no longer compete with the 3D canvas.
        self.view_settings=QDialog(self); self.view_settings.setWindowTitle('Настроить 3D-вид')
        self.view_settings.resize(440,650)
        outer=QVBoxLayout(self.view_settings); scroll=QScrollArea(); scroll.setWidgetResizable(True)
        body=QWidget(); settings_rows=QVBoxLayout(body); scroll.setWidget(body); outer.addWidget(scroll)
        controls=[self.component,self.threshold,*self.slices,*self.axis_scales,self.reset_scale,
            self.axis_font,self.axis_unit,self.camera_view,self.fit,self.palette,self.reverse_palette,
            self.color_range,self.color_min,self.color_max,self.apply_colors,self.color_note,self.info]
        for w in controls: w.setParent(body)
        view_row.removeWidget(self.expand); self.expand.setParent(self)
        # Dispose only the old layout containers, never the connected input widgets.
        while layout.count():
            item=layout.takeAt(0)
            sub=item.layout()
            if sub:
                while sub.count():
                    child=sub.takeAt(0).widget()
                    if child and child not in controls and child not in (self.kind,self.expand): child.hide(); child.deleteLater()
            elif item.widget() and item.widget() not in controls: item.widget().setParent(None)
        toolbar=QHBoxLayout(); toolbar.addWidget(self.kind,1)
        self.configure_view=QPushButton('Настроить вид…'); toolbar.addWidget(self.configure_view); toolbar.addWidget(self.expand)
        self.configure_view.clicked.connect(self.show_view_settings)
        layout.addLayout(toolbar)
        layout.setAlignment(Qt.AlignTop)
        for title,fields in [
            ('Отображение',[('Компонента',self.component),('Порог вокселей',self.threshold)]),
            ('Сечения',[('East',self.slices[0]),('North',self.slices[1]),('Up',self.slices[2])]),
            ('Пропорции модели',[('Растянуть X',self.axis_scales[0]),('Растянуть Y',self.axis_scales[1]),('Растянуть Z',self.axis_scales[2]),('',self.reset_scale)]),
            ('Камера и оси',[('Ракурс',self.camera_view),('Единицы',self.axis_unit),('Размер подписей',self.axis_font),('',self.fit)]),
            ('Цвета',[('Палитра',self.palette),('',self.reverse_palette),('Диапазон',self.color_range),('Минимум',self.color_min),('Максимум',self.color_max),('',self.apply_colors),('',self.color_note)]),
            ('Координаты и управление',[('',self.info)])]:
            group=QGroupBox(title); form=QFormLayout(group)
            form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            for label,widget in fields:
                if label: form.addRow(label,widget)
                else: form.addRow(widget)
            settings_rows.addWidget(group)
        done=QPushButton('Готово'); done.clicked.connect(self.view_settings.close); outer.addWidget(done)

    def show_view_settings(self):
        self.view_settings.show(); self.view_settings.raise_(); self.view_settings.activateWindow()

    def rescale(self):
        self._fit_pending=True
        if self.plotter is not None:
            # A top view hides Z exaggeration entirely (similarly X/Y in side views).
            changed=np.array([w.value() for w in self.axis_scales])!=self._display_scale
            direction=np.abs(self.plotter.camera.direction)
            if np.any(changed & (direction>.999)):
                self._reorient_pending=True
                self.camera_view.setCurrentIndex(0)
        self.refresh_if_ready()

    def range_changed(self):
        manual=self.color_range.currentIndex()==2
        self.color_min.setEnabled(manual); self.color_max.setEnabled(manual)
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
                              (self.axis_unit,view.axis_unit),(self.camera_view,view.camera_view),
                              (self.palette,view.palette),(self.color_range,view.color_range)):
            target.setCurrentIndex(source.currentIndex())
        view.color_min.setValue(self.color_min.value()); view.color_max.setValue(self.color_max.value())
        view.reverse_palette.setChecked(self.reverse_palette.isChecked())
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
        if self.color_range.currentIndex()==2 and self.color_min.value()>=self.color_max.value():
            self.color_note.setText('Цветовой минимум должен быть меньше максимума. Исправьте значения и нажмите «Применить цвета».')
            return
        self.color_note.clear()
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
        self.display_grid=grid
        self._display_scale=scale.copy()
        mode=result.settings['mode']
        unit='g/cm³' if mode=='gravity' else ('SI' if mode=='induced' else 'A/m')
        self.info.setText(f'{result.mesh.active.sum():,} активных ячеек · {unit} · X = East − E₀, Y = North − N₀; E₀={origin[0]:.1f}, N₀={origin[1]:.1f} м; Z = Up (вверх). Вращение: ЛКМ, масштаб: колесо.')
        self.component.setEnabled(result.model.ndim==2)
        values=grid['property']; amplitude=float(np.max(np.abs(values))); limit=max(amplitude,1e-15)
        cmap='viridis' if np.min(values)>=0 else 'coolwarm'
        clim=(0,limit) if np.min(values)>=0 else (-limit,limit)
        if self.palette.currentIndex()>0: cmap=self.palette.currentText()
        if self.reverse_palette.isChecked(): cmap+='_r'
        if self.color_range.currentIndex()==1: clim=(-limit,limit)
        elif self.color_range.currentIndex()==2: clim=(self.color_min.value(),self.color_max.value())
        if self.color_range.currentIndex()!=2:
            self.color_min.setValue(clim[0]); self.color_max.setValue(clim[1])
        options=dict(scalars='property',cmap=cmap,clim=clim,scalar_bar_args={
            'title':unit,'color':'#f0f5ff','title_font_size':14,'label_font_size':12,
            'n_labels':4,'fmt':'%.3g','position_x':.80,'position_y':.18,'width':.05,'height':.65,'vertical':True})
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
            # One mapper keeps the same explicit color range on all three slices.
            p.add_mesh(grid.slice_orthogonal(x=centers[0],y=centers[1],z=centers[2]).combine(),**options)
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
        if camera is None or new_model or self._reorient_pending: self.fit_model()
        else:
            p.camera_position=camera
            if self._fit_pending:
                # Refit the stretched geometry while retaining the user's rotated view.
                p.reset_camera(bounds=grid.bounds); p.camera.zoom(.82)
            else: p.camera.parallel_scale=camera_scale
        self._fit_pending=False
        self._reorient_pending=False
        p.render()

    def shutdown(self):
        self.view_settings.close()
        if self._expanded_dialog is not None:
            self._expanded_dialog.close()
        if self.plotter is not None:
            self.plotter.close()
