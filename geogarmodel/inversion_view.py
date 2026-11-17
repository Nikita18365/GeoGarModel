"""VTK scene with cell thresholding, slices, volume and isosurfaces; lazy imports."""
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QDoubleSpinBox, QLabel, QSlider, QSpinBox, QPushButton, QDialog, QCheckBox, QScrollArea, QGroupBox, QFormLayout,QToolBar,QMessageBox,QToolBox
from PyQt5.QtCore import Qt,QSize
from .palette_library import local_library,PaletteDialog,preview
from .model_slices import property_values


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
        self.threshold=QDoubleSpinBox(); self.threshold.setRange(0,99); self.threshold.setValue(0); self.threshold.setSuffix('% от max |m|')
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
        self.camera_view=QComboBox(); self.camera_view.addItems(['Изометрия','Сверху XY','Спереди XZ','Сбоку YZ','Снизу XY','Сзади XZ','С другой стороны YZ'])
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
            ('Сечения',[('East',self.slices[0]),('North',self.slices[1]),('Глубина Down',self.slices[2])]),
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
        self.palette_library=local_library(); self.palette.setIconSize(QSize(120,18))
        name='geophysics/colour.tbl'; self.palette.addItem(preview(self.palette_library[name]['colors']),name)
        self.palette.setCurrentText(name)
        self.palette_button=QPushButton('Библиотека палитр…'); self.palette_button.clicked.connect(self.choose_palette)
        colors_group=QGroupBox('Палитры и оформление'); color_form=QFormLayout(colors_group)
        color_form.addRow(self.palette_button)
        self.color_distribution=QComboBox(); self.color_distribution.addItem('Линейное','linear'); self.color_distribution.addItem('Равная площадь (гистограмма)','equal_area')
        color_form.addRow('Распределение цветов',self.color_distribution)
        self.color_distribution.currentIndexChanged.connect(self.refresh_if_ready)
        self.discrete_colors=QCheckBox('Дискретные цветовые зоны'); self.color_bins=QSpinBox(); self.color_bins.setRange(2,255); self.color_bins.setValue(32)
        self.zone_values=QCheckBox('Использовать границы значений ZON')
        color_form.addRow(self.discrete_colors); color_form.addRow('Число зон',self.color_bins); color_form.addRow(self.zone_values)
        self.discrete_colors.toggled.connect(self.refresh_if_ready); self.zone_values.toggled.connect(self.refresh_if_ready); self.color_bins.valueChanged.connect(self.refresh_if_ready)
        self.opacity=QDoubleSpinBox(); self.opacity.setRange(0,1); self.opacity.setValue(1); self.opacity.setSingleStep(.05); self.opacity.setKeyboardTracking(False)
        self.opacity.setToolTip('0 — полностью прозрачно; 1 — непрозрачно.')
        self.edges=QCheckBox('Рёбра ячеек'); self.outline=QCheckBox('Рамка модели'); self.outline.setChecked(True)
        self.smooth_colors=QCheckBox('Плавная окраска поверхности'); self.smooth_colors.setChecked(True)
        self.smooth_colors.setToolTip('Интерполяция для изображения. Свойства ячеек модели и экспорт не изменяются.')
        color_form.addRow('Непрозрачность',self.opacity); color_form.addRow(self.edges); color_form.addRow(self.outline)
        color_form.addRow(self.smooth_colors)
        for control in (self.opacity,self.edges,self.outline,self.smooth_colors):
            (control.valueChanged if isinstance(control,QDoubleSpinBox) else control.toggled).connect(self.refresh_if_ready)
        settings_rows.insertWidget(settings_rows.count()-1,colors_group)
        performance=QGroupBox('Быстродействие'); performance_form=QFormLayout(performance)
        self.moving_detail,self.rest_detail=QSpinBox(),QSpinBox()
        for w,value in ((self.moving_detail,10),(self.rest_detail,100)):
            w.setRange(1,100); w.setValue(value); w.setSuffix(' %'); w.setKeyboardTracking(False); w.valueChanged.connect(self.refresh_if_ready)
        self.full_resolution=QCheckBox('Всегда полная детализация'); self.full_resolution.toggled.connect(self.refresh_if_ready)
        self.high_quality_transparency=QCheckBox('Качественная прозрачность'); self.high_quality_transparency.setChecked(True); self.high_quality_transparency.toggled.connect(self.refresh_if_ready)
        performance_form.addRow('Во время движения',self.moving_detail); performance_form.addRow('В неподвижном виде',self.rest_detail)
        performance_form.addRow(self.full_resolution); performance_form.addRow(self.high_quality_transparency)
        settings_rows.insertWidget(settings_rows.count()-1,performance)
        sections=QGroupBox('Координаты сечений'); sections_form=QFormLayout(sections)
        self.slice_coordinates=[]; self.slice_enabled=[]
        for i,label in enumerate(('X / East, м','Y / North, м','Глубина Down, м')):
            spin=QDoubleSpinBox(); spin.setRange(-1e12,1e12); spin.setDecimals(3); spin.setKeyboardTracking(False)
            enabled=QCheckBox('Показать'); enabled.setChecked(True); enabled.toggled.connect(self.refresh_if_ready)
            spin.valueChanged.connect(lambda value,i=i:self.coordinate_changed(i,value))
            self.slice_coordinates.append(spin); self.slice_enabled.append(enabled)
            sections_form.addRow(label,spin); sections_form.addRow(enabled)
        settings_rows.insertWidget(settings_rows.count()-1,sections)
        clipping=QGroupBox('Пространственное отсечение'); clip_form=QFormLayout(clipping)
        self.clip_enabled=QCheckBox('Ограничить область XYZ'); clip_form.addRow(self.clip_enabled); self.clip_ranges=[]
        for name in ('East','North','Глубина Down'):
            bounds=[]
            for end in ('мин.','макс.'):
                w=QDoubleSpinBox(); w.setRange(-1e12,1e12); w.setDecimals(3); w.setKeyboardTracking(False); clip_form.addRow(name+' '+end+', м',w); bounds.append(w)
            self.clip_ranges.append(bounds)
        apply_clip=QPushButton('Применить отсечение'); apply_clip.clicked.connect(self.refresh_if_ready); clip_form.addRow(apply_clip)
        self.clip_enabled.toggled.connect(self.refresh_if_ready)
        self.clip_normal=QComboBox(); self.clip_normal.addItems(['X','Y','Z']); self.clip_normal.currentIndexChanged.connect(self.refresh_if_ready); clip_form.addRow('Нормаль интерактивной плоскости',self.clip_normal)
        settings_rows.insertWidget(settings_rows.count()-1,clipping)
        self.navigation=QToolBar(); self.navigation.setToolButtonStyle(Qt.ToolButtonTextOnly); layout.insertWidget(1,self.navigation)
        for text,index in [('XYZ',0),('Сверху',1),('Спереди',2),('Справа',3),('Снизу',4),('Сзади',5),('Слева',6)]:
            self.navigation.addAction(text).triggered.connect(lambda checked=False,index=index:self.set_camera(index))
        self.navigation.addAction('Вписать').triggered.connect(self.fit_model)
        for text,index in [('Срез X',0),('Срез Y',1),('Срез Z',2)]: self.navigation.addAction(text).triggered.connect(lambda checked=False,index=index:self.single_slice(index))
        self.navigation.addAction('Карты глубин…').triggered.connect(self.open_slice_maps)
        self.navigation.addAction('Палитры…').triggered.connect(self.choose_palette)
        self._geometry_key=None; self._geometry=None; self._lod_actors=[]; self._active_actor=None; self._moving=False
        self._slice_dialogs=[]
        self._explicit_slices=set()
        for i,slider in enumerate(self.slices):
            slider.sliderReleased.disconnect(); slider.sliderReleased.connect(lambda i=i:self.slider_changed(i))
        sections_box=QToolBox()
        while settings_rows.count():
            group=settings_rows.takeAt(0).widget()
            title=group.title(); group.setTitle(''); sections_box.addItem(group,title)
        settings_rows.addWidget(sections_box)

    def show_view_settings(self):
        self.view_settings.show(); self.view_settings.raise_(); self.view_settings.activateWindow()

    def choose_palette(self):
        dialog=PaletteDialog(self.palette_library,self)
        if dialog.exec_()!=QDialog.Accepted: return
        name=dialog.selected
        if self.palette.findText(name)<0: self.palette.addItem(preview(self.palette_library[name]['colors']),name)
        self.palette.setCurrentText(name)

    def colors_for(self,result,values=None):
        from types import SimpleNamespace
        from .plotting import map_colors
        values=property_values(result,self.component.currentText())[result.mesh.active] if values is None else np.asarray(values)
        limit=max(float(np.max(np.abs(values))),1e-15)
        clim=(0,limit) if np.min(values)>=0 else (-limit,limit)
        if self.color_range.currentIndex()==1: clim=(-limit,limit)
        elif self.color_range.currentIndex()==2:
            clim=(self.color_min.value(),self.color_max.value())
            if clim[0]>=clim[1]: raise ValueError('Цветовой минимум должен быть меньше максимума.')
        name=self.palette.currentText() if self.palette.currentIndex()>0 else ('viridis' if np.min(values)>=0 else 'coolwarm')
        entry=self.palette_library.get(name)
        cmap,norm=map_colors(SimpleNamespace(values=values),cmap=name,mode=self.color_distribution.currentData(),vmin=clim[0],vmax=clim[1],
            palette_colors=entry['colors'] if entry else None,reverse=self.reverse_palette.isChecked(),
            discrete=self.discrete_colors.isChecked(),bins=self.color_bins.value(),
            palette_zones=entry.get('zones') if entry else None,zone_values=self.zone_values.isChecked())
        return cmap,norm,(norm.vmin,norm.vmax)

    def set_camera(self,index):
        self.camera_view.setCurrentIndex(index); self.fit_model()

    def single_slice(self,index):
        for i,w in enumerate(self.slice_enabled): w.blockSignals(True); w.setChecked(i==index); w.blockSignals(False)
        if self.kind.currentIndex()==3: self.refresh_if_ready()
        else: self.kind.setCurrentIndex(3)

    def coordinate_changed(self,index,value):
        if self.result is None: return
        axis=(self.result.mesh.east,self.result.mesh.north,self.result.mesh.depth)[index]
        slider=self.slices[index]; slider.blockSignals(True); slider.setValue(round((value-axis[0])/(axis[-1]-axis[0])*100)); slider.blockSignals(False)
        self._explicit_slices.add(index); self.refresh_if_ready()

    def slider_changed(self,index):
        self._explicit_slices.discard(index); self.refresh_if_ready()

    def open_slice_maps(self):
        if self.result is None:
            QMessageBox.information(self,'Срезы','Сначала загрузите модель или результат моделирования.'); return
        from .slice_gallery import SliceDepthDialog,SliceGallery
        dialog=SliceDepthDialog(self.result.mesh,self)
        if dialog.exec_()!=QDialog.Accepted: return
        try: cmap,norm,_=self.colors_for(self.result)
        except ValueError as error: QMessageBox.warning(self,'Цветовая шкала',str(error)); return
        gallery=SliceGallery(self.result,dialog.depths,cmap,norm,self.component.currentText(),self)
        self._slice_dialogs.append(gallery)
        gallery.finished.connect(lambda:self._slice_dialogs.remove(gallery) if gallery in self._slice_dialogs else None)
        gallery.show()

    def interaction_started(self,*args):
        self._moving=True; self.apply_detail()

    def interaction_finished(self,*args):
        self._moving=False; self.apply_detail()

    def apply_detail(self):
        if not self._lod_actors: return
        desired=100 if self.full_resolution.isChecked() else (self.moving_detail.value() if self._moving else self.rest_detail.value())
        chosen=min(self._lod_actors,key=lambda pair:abs(pair[0]-desired))[1]
        for _,actor in self._lod_actors: actor.visibility=actor is chosen
        if self.plotter is not None: self.plotter.render()

    def prepare_lod(self,grid,actor,options,key):
        self._lod_actors=[(100,actor)]
        if self.full_resolution.isChecked() or grid.n_cells<500: return
        requested={self.moving_detail.value(),self.rest_detail.value()}-{100}
        cache_key=(key,tuple(sorted(requested)))
        if cache_key!=getattr(self,'_lod_cache_key',None):
            surface=grid.extract_surface().triangulate().cell_data_to_point_data()
            self._lod_cache={level:surface.decimate_pro(1-level/100,preserve_topology=True,splitting=False) for level in requested}
            self._lod_cache_key=cache_key
        for level,mesh in self._lod_cache.items():
            low=self.plotter.add_mesh(mesh,show_scalar_bar=False,opacity=self.opacity.value(),**options)
            self._lod_actors.append((level,low))
        self.apply_detail()

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
        index=self.camera_view.currentIndex()
        if index<4: [p.view_isometric,p.view_xy,p.view_xz,p.view_yz][index]()
        else: [p.view_xy,p.view_xz,p.view_yz][index-4](negative=True)
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
                              (self.color_range,view.color_range),(self.color_distribution,view.color_distribution)):
            target.setCurrentIndex(source.currentIndex())
        name=self.palette.currentText()
        if view.palette.findText(name)<0: view.palette.addItem(preview(self.palette_library[name]['colors']),name)
        view.palette.setCurrentText(name)
        view.color_min.setValue(self.color_min.value()); view.color_max.setValue(self.color_max.value())
        view.reverse_palette.setChecked(self.reverse_palette.isChecked())
        view.threshold.setValue(self.threshold.value()); view.axis_font.setValue(self.axis_font.value())
        for source,target in zip(self.slices,view.slices): target.setValue(source.value())
        for source,target in ((self.opacity,view.opacity),(self.color_bins,view.color_bins),(self.moving_detail,view.moving_detail),(self.rest_detail,view.rest_detail)): target.setValue(source.value())
        for source,target in ((self.discrete_colors,view.discrete_colors),(self.zone_values,view.zone_values),(self.edges,view.edges),(self.outline,view.outline),(self.smooth_colors,view.smooth_colors),(self.full_resolution,view.full_resolution),(self.high_quality_transparency,view.high_quality_transparency)):
            target.setChecked(source.isChecked())
        view.clip_normal.setCurrentIndex(self.clip_normal.currentIndex())
        for source,target in zip(self.slice_enabled,view.slice_enabled): target.setChecked(source.isChecked())
        def cleanup():
            view.shutdown(); self._expanded_dialog=None; dialog.deleteLater()
        dialog.finished.connect(cleanup)
        try:
            dialog.showMaximized(); view.render(self.result)
            for source,target in zip(self.slices,view.slices): target.setValue(source.value())
            for source,target in zip(self.slice_coordinates,view.slice_coordinates): target.blockSignals(True); target.setValue(source.value()); target.blockSignals(False)
            view._explicit_slices=set(range(3))
            for source_pair,target_pair in zip(self.clip_ranges,view.clip_ranges):
                for source,target in zip(source_pair,target_pair): target.setValue(source.value())
            view.clip_enabled.blockSignals(True); view.clip_enabled.setChecked(self.clip_enabled.isChecked()); view.clip_enabled.blockSignals(False)
            view.refresh_if_ready()
        except Exception:
            dialog.reject(); raise

    def refresh_if_ready(self):
        if self.result is not None and self.plotter is not None:
            self.render(self.result,reuse_geometry=True)

    def render(self,result,reuse_geometry=False):
        try:
            from pyvistaqt import QtInteractor
        except ImportError as exc:
            raise ValueError('Установите пакет визуализации: pip install -e ".[volume]"') from exc
        if self.plotter is None:
            self.plotter=QtInteractor(self)
            self.plotter.enable_parallel_projection()
            self.plotter.iren.add_observer('StartInteractionEvent',self.interaction_started)
            self.plotter.iren.add_observer('EndInteractionEvent',self.interaction_finished)
            self.layout().addWidget(self.plotter.interactor,1)
        new_model=self._geometry_key is None or self._geometry_key[0]!=id(result)
        self.result=result
        p=self.plotter
        if self.color_range.currentIndex()==2 and self.color_min.value()>=self.color_max.value():
            self.color_note.setText('Цветовой минимум должен быть меньше максимума. Исправьте значения и нажмите «Применить цвета».')
            return
        self.color_note.clear()
        camera=p.camera_position if p.renderer.actors else None
        camera_scale=p.camera.parallel_scale
        p.clear(); p.clear_plane_widgets()
        p.set_background('white'); self._lod_actors=[]
        # Configure the current Qt OpenGL context. PyVista's convenience method
        # probes a separate temporary window, which can crash software drivers.
        # VTK falls back to standard blending when peeling is unavailable here.
        p.renderer.SetUseDepthPeeling(self.high_quality_transparency.isChecked())
        p.renderer.SetMaximumNumberOfPeels(4)
        p.renderer.SetOcclusionRatio(.05)
        p.render_window.SetAlphaBitPlanes(1)
        p.render_window.SetMultiSamples(0)
        # Use a local origin only in the display; exported VTK retains full CRS coordinates.
        origin=np.array([result.mesh.east[0],result.mesh.north[0],0.])
        # Transform display geometry only, so slices and clip widgets share its coordinates.
        # Axes below retain the unscaled physical coordinate ranges.
        scale=np.array([widget.value() for widget in self.axis_scales])
        geometry_key=(id(result),self.component.currentText(),tuple(scale))
        if not reuse_geometry or geometry_key!=self._geometry_key:
            grid=vtk_grid(result,self.component.currentText()); grid.translate(-origin,inplace=True)
            if grid.__class__.__name__=='UnstructuredGrid': grid=grid.clean(produce_merge_map=False)
            self._physical_bounds=np.array(grid.bounds); grid.scale(scale,inplace=True)
            self._geometry=grid; self._geometry_key=geometry_key; self._lod_cache_key=None
        grid=self._geometry; physical_bounds=self._physical_bounds
        self.display_grid=grid
        self._display_scale=scale.copy()
        mode=result.settings['mode']
        unit='g/cm³' if mode=='gravity' else ('SI' if mode=='induced' else 'A/m')
        if new_model or not getattr(self,'_coordinate_ready',False):
            self._explicit_slices.clear()
            self.clip_enabled.blockSignals(True); self.clip_enabled.setChecked(False); self.clip_enabled.blockSignals(False)
            for i,axis in enumerate((result.mesh.east,result.mesh.north,result.mesh.depth)):
                self.slices[i].setValue(50)
                w=self.slice_coordinates[i]; w.blockSignals(True); w.setRange(axis[0],axis[-1]); w.setValue((axis[0]+axis[-1])/2); w.blockSignals(False)
                for control,value in zip(self.clip_ranges[i],(axis[0],axis[-1])): control.setValue(value)
            self._coordinate_ready=True
        for i,axis in enumerate((result.mesh.east,result.mesh.north,result.mesh.depth)):
            if i in self._explicit_slices: continue
            w=self.slice_coordinates[i]; w.blockSignals(True); w.setValue(axis[0]+(axis[-1]-axis[0])*self.slices[i].value()/100); w.blockSignals(False)
        if self.clip_enabled.isChecked():
            limits=[(a.value(),b.value()) for a,b in self.clip_ranges]
            if any(a>=b for a,b in limits): self.color_note.setText('Минимальные границы отсечения должны быть меньше максимальных.'); return
            bounds=[(limits[0][0]-origin[0])*scale[0],(limits[0][1]-origin[0])*scale[0],(limits[1][0]-origin[1])*scale[1],(limits[1][1]-origin[1])*scale[1],-limits[2][1]*scale[2],-limits[2][0]*scale[2]]
            grid=grid.clip_box(bounds,invert=False)
            if grid.n_cells==0: self.color_note.setText('В выбранной области нет активных ячеек.'); p.render(); return
        self.info.setText(f'{result.mesh.active.sum():,} активных ячеек · {unit} · Оси: исходные East / North, Z = Up (вверх). Локальное начало геометрии: E₀={origin[0]:.1f}, N₀={origin[1]:.1f} м. Вращение: ЛКМ; перемещение: средняя кнопка; масштаб: колесо.')
        self.component.setEnabled(result.model.ndim==2)
        values=grid['property']; amplitude=float(np.max(np.abs(values))); limit=max(amplitude,1e-15)
        cmap,norm,clim=self.colors_for(result)
        from matplotlib.colors import ListedColormap
        cmap=ListedColormap(cmap(norm(np.linspace(clim[0],clim[1],1024))))
        if self.color_range.currentIndex()!=2:
            self.color_min.setValue(clim[0]); self.color_max.setValue(clim[1])
        options=dict(scalars='property',cmap=cmap,clim=clim,scalar_bar_args={
            'title':unit,'color':'#222222','title_font_size':self.axis_font.value(),'label_font_size':max(10,self.axis_font.value()-4),
            'n_labels':4,'fmt':'%.3g','position_x':.80,'position_y':.18,'width':.05,'height':.65,'vertical':True})
        kind=self.kind.currentIndex()
        if kind==0:
            chosen=grid.extract_cells(np.flatnonzero(np.abs(values)>=amplitude*self.threshold.value()/100))
            if chosen.n_cells:
                if self.smooth_colors.isChecked(): chosen=chosen.cell_data_to_point_data(pass_cell_data=True)
                actor=p.add_mesh(chosen,show_edges=self.edges.isChecked(),edge_color='#333333',opacity=self.opacity.value(),interpolate_before_map=True,**options)
                lod_options={k:v for k,v in options.items() if k!='scalar_bar_args'}
                key=(self._geometry_key,self.threshold.value(),self.clip_enabled.isChecked(),tuple(w.value() for pair in self.clip_ranges for w in pair))
                self.prepare_lod(chosen,actor,lod_options,key)
        elif kind==1:
            if grid.__class__.__name__=='UnstructuredGrid':
                p.add_mesh(grid,opacity=self.opacity.value(),**options)
                self.info.setText(self.info.text()+' · Обрезанные призмы: геометрия ячеек.')
            else:
                actor=p.add_volume(grid.cell_data_to_point_data(),opacity=self.opacity.value()/(1+np.exp(-np.linspace(-6,6,256))),**options)
                if hasattr(actor.mapper,'SetInteractiveAdjustSampleDistances'): actor.mapper.SetInteractiveAdjustSampleDistances(not self.full_resolution.isChecked())
        elif kind==2:
            points=grid.cell_data_to_point_data()
            lo,hi=float(values.min()),float(values.max())
            if hi>lo:
                surface=points.contour(np.linspace(lo,hi,7)[1:-1],scalars='property')
                if surface.n_points: p.add_mesh(surface,opacity=self.opacity.value(),**options)
        elif kind==3:
            centers=[(self.slice_coordinates[i].value()-origin[i])*scale[i] if i<2 else -self.slice_coordinates[i].value()*scale[i] for i in range(3)]
            # One mapper keeps the same explicit color range on all three slices.
            parts=[grid.slice(normal=axis,origin=centers) for axis,enabled in zip(('x','y','z'),self.slice_enabled) if enabled.isChecked()]
            import pyvista as pv
            combined=pv.MultiBlock(parts).combine() if parts else None
            if combined is not None and combined.n_cells:
                if self.smooth_colors.isChecked(): combined=combined.cell_data_to_point_data(pass_cell_data=True)
                p.add_mesh(combined,opacity=self.opacity.value(),**options)
        else:
            if self.smooth_colors.isChecked(): grid=grid.cell_data_to_point_data(pass_cell_data=True)
            p.add_mesh_clip_plane(grid,normal=self.clip_normal.currentText().lower(),opacity=self.opacity.value(),**options)
        if self.outline.isChecked(): p.add_mesh(grid.outline(),color='#333333')
        p.add_axes(xlabel='X / East',ylabel='Y / North',zlabel='Z / Up',color='#222222')
        divisor=1000 if self.axis_unit.currentText()=='км' else 1
        unit_label='km' if divisor==1000 else 'm'
        physical_bounds=np.array(grid.bounds)/np.repeat(scale,2)
        self.bounds_axes=p.show_bounds(bounds=grid.bounds,axes_ranges=(physical_bounds+np.repeat(origin,2))/divisor,
                    grid='back' if self.outline.isChecked() else False,all_edges=self.outline.isChecked(),location='outer',
                    color='#222222',font_size=self.axis_font.value(),bold=True,fmt='%.3f' if divisor==1000 else '%.0f',
                    use_2d=False,use_3d_text=False,
                    xtitle=f'East ({unit_label})',ytitle=f'North ({unit_label})',
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
        for dialog in list(self._slice_dialogs): dialog.close()
        self.view_settings.close()
        if self._expanded_dialog is not None:
            self._expanded_dialog.close()
        if self.plotter is not None:
            self._lod_actors=[]; self.plotter.close(); self.plotter=None
