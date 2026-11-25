"""Layer tree, independent layer styling and portable map workspaces."""
from pathlib import Path
import json
import numpy as np
from PyQt5.QtCore import Qt,QMimeData,QSettings
from PyQt5.QtGui import QColor,QKeySequence,QDrag
from PyQt5.QtWidgets import (QListWidgetItem,QAbstractItemView,QMenu,QShortcut,QInputDialog,
    QFileDialog,QMessageBox,QDoubleSpinBox,QSpinBox,QLineEdit,QCheckBox,QColorDialog,QStyledItemDelegate,QPushButton,QComboBox,QDialog,QDialogButtonBox,QFormLayout,QSlider,QWidget,QHBoxLayout)
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtWidgets import QLabel
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory,IdentityTransform
from .map_project import read_vector,save_project,load_project,reproject_grid
from .plotting import draw_map
from .map_navigation import north_direction
from .sun_picker import SunPicker
from .crs_editor import CRSEditor,read_prj
from .palette_library import local_library,PaletteDialog,preview
from .ui_layout import Disclosure


class LayerNameDelegate(QStyledItemDelegate):
    def sizeHint(self,option,index):
        size=super().sizeHint(option,index); size.setHeight(max(size.height(),34)); return size

    def createEditor(self,parent,option,index):
        editor=QLineEdit(parent)
        editor.setStyleSheet('QLineEdit {padding: 2px 4px; border: 1px solid #38bdf8; color: #f1f7ff; background: #101c2f; selection-color: white;}')
        return editor

    def updateEditorGeometry(self,editor,option,index):
        editor.setGeometry(option.rect.adjusted(0,0,0,0))


class LayerItem(QListWidgetItem):
    def data(self,role):
        return self.payload if role==Qt.UserRole else super().data(role)

    def setData(self,role,value):
        if role==Qt.UserRole: self.payload=value
        else: super().setData(role,value)


class LayerList(QListWidget):
    def startDrag(self,actions):
        # Own the move completely. QAbstractItemView.startDrag would remove the
        # source again after our custom dropEvent has already moved the item.
        drag=QDrag(self); mime=QMimeData(); mime.setData('application/x-geogarmodel-layer',b'layer')
        drag.setMimeData(mime); drag.exec_(Qt.MoveAction)

    def dragEnterEvent(self,event):
        if event.source() is self: event.acceptProposedAction()
        else: event.ignore()

    def dragMoveEvent(self,event):
        if event.source() is self: event.acceptProposedAction()
        else: event.ignore()

    def dropEvent(self,event):
        if hasattr(event,'source') and event.source() is not self: event.ignore(); return
        source=self.currentRow(); target=self.indexAt(event.pos()).row()
        if source<0: return
        if target<0: target=self.count()-1
        window=self.window(); window._map_loading=True
        item=self.takeItem(source); self.insertItem(target,item); self.setCurrentItem(item)
        window._map_loading=False; window.guard(window.map_reordered); event.accept()


class MapWorkspace:
    def setup_map_workspace(self,style_form,transform_form,project_layout):
        self.map_project_name='Без названия'; self.map_project_path=None; self.map_dirty=False
        self.map_project_crs=''
        self.map_layout_settings=QSettings('GeoGarModel','MapLayout')
        try: self._map_layout=json.loads(self.map_layout_settings.value('subplot_layout','null'))
        except (ValueError,TypeError): self._map_layout=None
        self._map_selected=None; self._map_loading=False; self._map_fit=True
        self.map_style_group.setTitle('')
        self.map_style_heading=QLabel('Оформление карты — выберите карту')
        self.map_style_heading.setWordWrap(True)
        self.map_style_group.layout().insertWidget(0,self.map_style_heading)
        self.project_label=QLabel('Проект: Без названия'); self.project_label.setWordWrap(True)
        project_layout.insertWidget(0,self.project_label)
        help_label=QLabel('✓ — показать слой · верхний слой впереди\nПеретащите для порядка · F2 — имя · Delete — удалить')
        help_label.setWordWrap(True); project_layout.insertWidget(1,help_label)
        self.button('СК проекта…',self.project_crs_dialog,project_layout)
        self.ask_source_crs=QCheckBox('Уточнять исходную СК при импорте')
        self.ask_source_crs.setToolTip('По умолчанию СК читается из файла; если её нет, используется СК проекта.')
        project_layout.addWidget(self.ask_source_crs)
        self.vector_style_button=self.button('Оформление SHP / GeoJSON…',self.vector_style_dialog,project_layout)
        self.vector_style_button.setEnabled(False)
        self.layers.currentRowChanged.disconnect()
        self.layers.setSelectionMode(QAbstractItemView.SingleSelection)
        self.layers.setDragDropMode(QAbstractItemView.InternalMove)
        self.layers.setItemDelegate(LayerNameDelegate(self.layers))
        self.layers.setContextMenuPolicy(Qt.CustomContextMenu)
        self.layers.customContextMenuRequested.connect(self.map_context)
        self.layers.currentItemChanged.connect(self.map_selection)
        self.layers.itemChanged.connect(lambda _:self.guard(self.map_item_changed))
        self.layers.model().rowsMoved.connect(lambda *a:self.guard(self.map_reordered))
        self.map_delete=QShortcut(QKeySequence.Delete,self.layers)
        self.map_delete.setContext(Qt.WidgetShortcut); self.map_delete.activated.connect(lambda:self.guard(self.delete_map_layer))
        self.map_style_widgets={'cmap':self.palette,'mode':self.norm,'vmin':self.minimum,'vmax':self.maximum,
            'contours':self.contours,'hillshade':self.shade}
        self.palette_library=local_library()
        self.palette.setIconSize(__import__('PyQt5.QtCore',fromlist=['QSize']).QSize(130,18))
        for i in range(self.palette.count()):
            name=self.palette.itemText(i); self.palette.setItemIcon(i,preview(self.palette_library[name]['colors']))
        palette_button=QPushButton('Библиотека палитр…'); palette_button.clicked.connect(lambda:self.guard(self.choose_palette))
        style_form.insertRow(1,palette_button)
        self.map_discrete=QCheckBox('Дискретные цветовые зоны')
        self.map_bins=QSpinBox(); self.map_bins.setRange(2,255); self.map_bins.setValue(32)
        self.map_reverse=QCheckBox('Обратить палитру'); self.map_smooth=QCheckBox('Плавное изображение (рекомендуется)')
        self.map_smooth.setChecked(True)
        self.map_smooth.setToolTip('Интерполяция значений перед раскраской; исходные данные не меняются. Выключите, чтобы видеть ячейки сетки.')
        self.map_zone_values=QCheckBox('Использовать границы значений из ZON')
        for key,label,w in [('discrete','',self.map_discrete),('bins','Число зон',self.map_bins),('reverse','',self.map_reverse),('smooth','',self.map_smooth)]:
            style_form.addRow(label,w); self.map_style_widgets[key]=w
        style_form.addRow(self.map_zone_values); self.map_style_widgets['zone_values']=self.map_zone_values
        style_form.labelForField(self.map_title).setText('Общий заголовок (пусто — имя проекта)')
        self.shade.setText('Теневой рельеф')
        self.map_title.editingFinished.connect(lambda:self.guard(self.redraw))
        def number(minimum,maximum,value,decimals=2):
            w=QDoubleSpinBox(); w.setDecimals(decimals); w.setRange(minimum,maximum); w.setValue(value); return w
        self.contour_step=number(0,1e12,0,6); self.contour_step.setSpecialValueText('Авто')
        # Keep the serialized colour value for compatibility; only the palette button is shown.
        self.contour_color=QLineEdit('#dbe7f6',self.layers); self.contour_color.hide()
        self.contour_width=number(.1,10,.7)
        self.contour_labels=QCheckBox('Подписывать значения'); self.contour_labels.setChecked(True)
        self.map_alpha=number(0,1,1); self.map_alpha.setSingleStep(.1)
        for key,label,w in [('contour_step','Сечение изолиний (единицы поля)',self.contour_step),
                            ('contour_width','Толщина изолиний',self.contour_width),
                            ('contour_labels','',self.contour_labels),('alpha','Непрозрачность слоя',self.map_alpha)]:
            style_form.addRow(label,w); self.map_style_widgets[key]=w
        self.map_style_widgets['contour_color']=self.contour_color
        self.sun_azimuth=number(0,360,315,1); self.sun_azimuth.setWrapping(True)
        self.sun_altitude=number(0,90,45,1); self.sun_intensity=number(0,100,25,1)
        self.sun_relief=number(.001,1e6,1,3)
        self.sun_picker=SunPicker(); self.sun_picker.changed.connect(self.sun_from_mouse)
        self.sun_picker.setAccessibleName('Положение солнца: азимут и высота')
        self.sun_intensity_slider=QSlider(Qt.Horizontal); self.sun_intensity_slider.setRange(0,100)
        self.sun_intensity_slider.setValue(25); self.sun_intensity_slider.setTracking(False)
        self.sun_intensity_slider.valueChanged.connect(self.sun_intensity.setValue)
        def sync_intensity(value):
            self.sun_intensity_slider.blockSignals(True); self.sun_intensity_slider.setValue(round(value)); self.sun_intensity_slider.blockSignals(False)
        self.sun_intensity.valueChanged.connect(sync_intensity)
        self.sun_intensity_row=QWidget(); intensity_layout=QHBoxLayout(self.sun_intensity_row); intensity_layout.setContentsMargins(0,0,0,0)
        intensity_layout.addWidget(self.sun_intensity_slider,1); intensity_layout.addWidget(self.sun_intensity)
        style_form.addRow('Солнце: перетащите мышью',self.sun_picker)
        for key,label,w in [('sun_azimuth','Азимут света, °',self.sun_azimuth),('sun_altitude','Высота солнца, °',self.sun_altitude),
                            ('sun_intensity','Интенсивность тени, %',self.sun_intensity),('sun_relief','Усиление рельефа',self.sun_relief)]:
            style_form.addRow(label,self.sun_intensity_row if key=='sun_intensity' else w); self.map_style_widgets[key]=w
        color_button=QPushButton('Выбрать цвет…'); self.contour_color_button=color_button
        color_button.clicked.connect(lambda:self.guard(self.choose_contour_color))
        style_form.insertRow(style_form.getWidgetPosition(self.contour_width)[0],'Цвет изолиний',color_button)
        self.scale_on=QCheckBox('Масштабная линейка'); style_form.addRow(self.scale_on)
        self.scale_step=number(0,1e9,0); self.scale_step.setSpecialValueText('Авто')
        self.scale_count=QSpinBox(); self.scale_count.setRange(1,20); self.scale_count.setValue(4)
        self.scale_unit=QComboBox(); self.scale_unit.addItems(['м','км'])
        self.scale_crs=QCheckBox('Подписывать СК под линейкой')
        self.north_symbol=QComboBox(self.layers); self.north_symbol.addItem('Север N','arrow'); self.north_symbol.hide()
        self.auto_map_layout=QCheckBox('Автокомпоновка белого листа'); self.auto_map_layout.setChecked(True)
        style_form.addRow(self.auto_map_layout); self.auto_map_layout.toggled.connect(lambda *a:self.guard(self.redraw))
        geo_panel=QWidget(); geo_form=QFormLayout(geo_panel); self.geo_widgets={}
        axes=QComboBox(); axes.addItem('X / Y, метры','projected'); axes.addItem('Долгота / широта','geographic')
        mode=QComboBox()
        for label,key in [('Нет','none'),('Кресты','crosses'),('Пунктирные линии','dotted'),('Сплошные линии','solid'),('Только засечки по рамке','edges')]: mode.addItem(label,key)
        notation=QComboBox(); notation.addItem('Десятичные градусы','decimal'); notation.addItem('ГГГ.ММСС / подписи Г°М′С″','dms')
        lon=number(0,360,0,4); lat=number(0,180,0,4)
        for step in (lon,lat): step.setSpecialValueText('Авто')
        thickness=number(1,2000,100,1)
        for key,label,w in [('axes','Подписи осей',axes),('mode','Географическая сетка',mode),('notation','Формат углов',notation),
                            ('longitude_step','Шаг долготы (0 — авто)',lon),('latitude_step','Шаг широты (0 — авто)',lat),('width_microns','Толщина линий, мкм',thickness)]:
            geo_form.addRow(label,w); self.geo_widgets[key]=w
            signal=w.currentIndexChanged if isinstance(w,QComboBox) else w.valueChanged
            signal.connect(lambda *a:self.guard(self.redraw))
        self.geo_color='#333333'; geo_color=QPushButton('Выбрать цвет…')
        def choose_geo_color():
            selected=QColorDialog.getColor(QColor(self.geo_color),self,'Цвет географической сетки')
            if selected.isValid(): self.geo_color=selected.name(); self.redraw()
        geo_color.clicked.connect(choose_geo_color); geo_form.addRow('Цвет линий',geo_color)
        style_form.addRow(Disclosure('Оси и географическая сетка',geo_panel))
        style_form.addRow('Единицы линейки',self.scale_unit)
        style_form.addRow('Шаг линейки',self.scale_step); style_form.addRow('Число делений',self.scale_count)
        style_form.addRow(self.scale_crs)
        def show_details():
            for widget in (self.sun_picker,self.sun_azimuth,self.sun_altitude,self.sun_intensity_row,self.sun_relief):
                widget.setVisible(self.shade.isChecked())
                label=style_form.labelForField(widget)
                if label: label.setVisible(self.shade.isChecked())
            for widget in (self.contour_step,self.contour_width,self.contour_labels):
                widget.setVisible(self.contours.isChecked())
                label=style_form.labelForField(widget)
                if label: label.setVisible(self.contours.isChecked())
            color_button.setVisible(self.contours.isChecked())
            style_form.labelForField(color_button).setVisible(self.contours.isChecked())
            for widget in (self.scale_step,self.scale_count,self.scale_unit,self.scale_crs):
                widget.setVisible(self.scale_on.isChecked())
                label=style_form.labelForField(widget)
                if label: label.setVisible(self.scale_on.isChecked())
        self.contours.toggled.connect(show_details); self.scale_on.toggled.connect(show_details); self.shade.toggled.connect(show_details); show_details()
        self.scale_unit.currentIndexChanged.connect(self.change_scale_unit)
        for w in [*self.map_style_widgets.values(),self.scale_on,self.scale_step,self.scale_count,self.scale_unit,self.scale_crs,self.north_symbol]:
            signal=(w.editingFinished if isinstance(w,QLineEdit) else w.toggled if isinstance(w,QCheckBox)
                    else w.currentIndexChanged if hasattr(w,'currentIndexChanged') else w.valueChanged)
            signal.connect(lambda *a:self.guard(self.redraw))
        self.transform_rows={w:transform_form.labelForField(w) for w in (self.height,self.wave,self.degree)}
        self.operation.currentIndexChanged.connect(self.update_transform_parameters); self.update_transform_parameters()
        self.canvas.mpl_connect('scroll_event',self.map_scroll)
        self.canvas.mpl_connect('draw_event',self.map_drawn)
        menu=self.menuBar().addMenu('Проект')
        for title,fn in [('Создать…',self.new_map_project),('Открыть…',self.open_map_project),
                         ('Сохранить',self.save_map_project),('Сохранить как…',lambda:self.save_map_project(True)),
                         ('Система координат…',self.project_crs_dialog),
                         ('Закрыть',self.close_map_project),('Вписать все видимые карты',self.fit_maps)]:
            action=menu.addAction(title); action.triggered.connect(lambda checked=False,fn=fn:self.guard(fn))
            if title in ('Создать…','Открыть…','Сохранить'): action.setShortcut(QKeySequence({'Создать…':'Ctrl+N','Открыть…':'Ctrl+O','Сохранить':'Ctrl+S'}[title]))

    def update_transform_parameters(self):
        operation=self.operation.currentData()
        shown={self.height:operation in ('upward','residual'),self.wave:operation in ('lowpass','highpass'),
               self.degree:operation in ('trend','detrend')}
        for widget,visible in shown.items():
            widget.setVisible(visible); self.transform_rows[widget].setVisible(visible)

    def remember_map_layout(self):
        keys=('left','right','top','bottom','wspace','hspace')
        self._map_layout={key:float(getattr(self.figure.subplotpars,key)) for key in keys}
        self.map_layout_settings.setValue('subplot_layout',json.dumps(self._map_layout)); self.map_layout_settings.sync()

    def read_geography(self):
        return {**{k:w.currentData() if isinstance(w,QComboBox) else w.value() for k,w in self.geo_widgets.items()},'color':self.geo_color}

    def restore_geography(self,settings=None):
        settings=settings or dict(axes='projected',mode='none',notation='decimal',longitude_step=0,latitude_step=0,width_microns=100,color='#333333')
        for key,w in self.geo_widgets.items():
            w.blockSignals(True)
            if key in settings:
                if isinstance(w,QComboBox): w.setCurrentIndex(max(0,w.findData(settings[key])))
                else: w.setValue(settings[key])
            w.blockSignals(False)
        self.geo_color=settings.get('color','#333333'); self._geo_key=None

    def refresh_map_geography(self,update_fast=True,strict=True):
        if not self.figure.axes or not hasattr(self,'geo_widgets'): return False
        ax=self.figure.axes[0]; settings=self.read_geography()
        key=(self.map_project_crs,*ax.get_xlim(),*ax.get_ylim(),tuple(settings.items()))
        if key==getattr(self,'_geo_key',None): return False
        from .graticule import draw_geography
        try: self._map_geo_data=draw_geography(ax,self.map_project_crs,settings)
        except ValueError as error:
            if strict: raise
            self._map_geo_data=None
            if settings['axes']=='geographic': ax.set_xticks([]); ax.set_yticks([])
            self.statusBar().showMessage(str(error),10000)
        self._geo_key=key
        if update_fast and hasattr(self,'fast_map'): self.fast_map.draw_geography(self._map_geo_data,settings)
        return True

    def sun_from_mouse(self,azimuth,altitude):
        for widget,value in ((self.sun_azimuth,azimuth),(self.sun_altitude,altitude)):
            widget.blockSignals(True); widget.setValue(value); widget.blockSignals(False)
        self.guard(self.redraw)

    def change_scale_unit(self):
        if self._map_loading: return
        # Keep physical length when changing m ↔ km.
        factor=.001 if self.scale_unit.currentText()=='км' else 1000
        self.scale_step.blockSignals(True)
        self.scale_step.setDecimals(6 if self.scale_unit.currentText()=='км' else 2)
        self.scale_step.setValue(self.scale_step.value()*factor); self.scale_step.blockSignals(False)

    def crs_caption(self):
        from pyproj import CRS
        crs=self.map_project_crs
        if not crs:
            rows=self.map_layer_rows(); crs=next((r['data'].crs for r in rows if r['kind']=='grid'),'')
        if not crs: return 'СК не задана'
        source=CRS.from_user_input(crs); authority=source.to_authority()
        return f'{source.name} ({authority[0]}:{authority[1]})' if authority else source.name

    def update_project_label(self):
        caption=self.crs_caption() if self.map_project_crs else ''
        self.project_label.setText('Проект: '+self.map_project_name+('\nСК: '+caption if caption else ''))

    def source_crs(self,known,name):
        from pyproj import CRS
        if known: return CRS.from_user_input(known).to_string()
        if self.map_project_crs: return self.map_project_crs
        return self.choose_source_crs(known,name)

    def ensure_map_project(self):
        if not self.map_project_crs: self.project_crs_dialog()
        return bool(self.map_project_crs)

    def import_source_crs(self,known,name):
        if self.ask_source_crs.isChecked(): return self.choose_source_crs(known or self.map_project_crs,name)
        if not known: self.statusBar().showMessage('СК файла не найдена: используется СК проекта. Изменить: «СК и параметры слоя…».',12000)
        return known or self.map_project_crs

    def choose_palette(self):
        dialog=PaletteDialog(self.palette_library,self)
        if dialog.exec_()!=QDialog.Accepted: return
        name=dialog.selected
        if self.palette.findText(name)<0: self.palette.addItem(preview(self.palette_library[name]['colors']),name)
        self.palette.setCurrentText(name)

    def choose_source_crs(self,known,name):
        dialog=QDialog(self); dialog.setWindowTitle('Исходная СК — '+name); dialog.resize(620,320)
        form=QFormLayout(dialog); editor=CRSEditor(known,dialog); form.addRow(editor)
        note=QLabel('Укажите СК координат файла до преобразования в СК проекта.\n'
            'Параметры берутся из выбранной СК или импортированного .prj.\n'
            'Смещения — полные значения параметров проекции, а не величины вычитания.')
        note.setWordWrap(True); form.addRow(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); form.addRow(buttons)
        return editor.value() if dialog.exec_()==QDialog.Accepted else None

    def layer_coordinates(self):
        item=self.layers.currentItem()
        if item is None: raise ValueError('Выберите слой.')
        row=item.data(Qt.UserRole)
        if row['kind']=='grid': return self.metadata()
        source=self.choose_source_crs(row['data'].crs,item.text()+' (текущие координаты слоя)')
        if source is None: return
        if row['kind']=='raster':
            from dataclasses import replace
            revised=replace(row['data'],crs=source)
        else:
            from .map_project import Vector
            revised=Vector(row['data'].features,source)
        row['data']=self.activity.run('Пересчёт слоя',revised.reproject,self.map_project_crs) if self.map_project_crs else revised
        self._map_fit=True; self.map_reordered()

    def apply_project_crs(self,crs):
        from pyproj import CRS
        target=CRS.from_user_input(crs)
        if not target.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in target.axis_info[:2]):
            raise ValueError('Выберите проекцию в метрах, например GSK-2011 / Gauss-Kruger zone 18.')
        if self.worker is not None: raise ValueError('Дождитесь завершения расчёта перед сменой СК проекта.')
        converted=[]
        for row in self.map_layer_rows():
            source=self.source_crs(row['data'].crs,row['name'])
            if source is None: return False
            converted.append(self.activity.run('Пересчёт карты',reproject_grid,row['data'],target,source) if row['kind']=='grid' else self.activity.run('Пересчёт слоя',row['data'].reproject,target))
        # Commit only after every layer has been transformed successfully.
        for i,value in enumerate(converted): self.layers.item(i).data(Qt.UserRole)['data']=value
        self.map_project_crs=target.to_string(); self.update_project_label()
        self._map_fit=True; self.map_reordered(); return True

    def project_crs_dialog(self,new=False):
        dialog=QDialog(self); dialog.setWindowTitle('Новый проект GeoGarModel' if new else 'Система координат проекта')
        dialog.resize(540,260); form=QFormLayout(dialog); form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        name=QLineEdit('' if new else self.map_project_name)
        crs=CRSEditor('EPSG:20918' if new else self.map_project_crs or 'EPSG:20918',dialog)
        form.addRow('Название проекта',name); form.addRow('СК проекта (EPSG / PROJ / WKT)',crs)
        note=QLabel('Все карты и контуры приводятся к этой СК. Растры интерполируются на регулярную сетку.\nДля файлов без СК потребуется указать исходную систему.')
        note.setWordWrap(True); form.addRow(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); form.addRow(buttons)
        if dialog.exec_()!=QDialog.Accepted: return
        value=crs.value()
        if not name.text().strip(): raise ValueError('Введите название проекта.')
        if new:
            from pyproj import CRS
            target=CRS.from_user_input(value)
            if not target.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in target.axis_info[:2]):
                raise ValueError('Выберите СК в метрах.')
            if not self.confirm_map_close(): return
            self.reset_map_project(name.text().strip()); self.map_project_crs=target.to_string(); self.update_project_label(); self.map_dirty=True
        elif self.apply_project_crs(value):
            self.map_project_name=name.text().strip(); self.update_project_label()

    def choose_contour_color(self):
        color=QColorDialog.getColor(QColor(self.contour_color.text()),self,'Цвет изолиний')
        if color.isValid(): self.contour_color.setText(color.name()); self.redraw()

    def read_map_style(self):
        result={}
        for key,w in self.map_style_widgets.items():
            result[key]=(w.text() if isinstance(w,QLineEdit) else w.isChecked() if isinstance(w,QCheckBox)
                         else w.currentText() if key=='cmap' else w.currentData() if key=='mode' else w.value())
            if key=='sun_intensity': result[key]/=100
        entry=self.palette_library.get(result['cmap'])
        if entry: result.update(palette_colors=entry['colors'],palette_zones=entry.get('zones'))
        result['render_quality']=2
        return result

    def map_selection(self,item,previous):
        if self._map_loading: return
        self.vector_style_button.setEnabled(bool(item and item.data(Qt.UserRole)['kind']=='vector'))
        if previous and previous.data(Qt.UserRole)['kind']=='grid': previous.data(Qt.UserRole)['style'].update(self.read_map_style())
        self._map_selected=item
        if item and item.data(Qt.UserRole)['kind']=='grid':
            style=item.data(Qt.UserRole)['style']; self._map_loading=True
            if style.get('palette_colors'):
                self.palette_library[style['cmap']]=dict(colors=style['palette_colors'],zones=style.get('palette_zones'))
                if self.palette.findText(style['cmap'])<0: self.palette.addItem(preview(style['palette_colors']),style['cmap'])
            for key,w in self.map_style_widgets.items():
                value=style.get(key)
                if value is None: continue
                if isinstance(w,QLineEdit): w.setText(str(value))
                elif isinstance(w,QCheckBox): w.setChecked(value)
                elif key=='cmap': w.setCurrentText(value)
                elif key=='mode': w.setCurrentIndex(w.findData(value))
                elif key=='sun_intensity': w.setValue(value*100)
                else: w.setValue(value)
            self._map_loading=False
        self.guard(self.redraw)

    def map_layer_rows(self):
        rows=[]
        for i in range(self.layers.count()):
            item=self.layers.item(i); row=item.data(Qt.UserRole)
            row['style'].pop('title',None)
            row.update(name=item.text(),visible=item.checkState()==Qt.Checked)
            if row['kind']=='grid': row['data'].name=item.text()
            rows.append(row)
        return rows

    def add_map_layer(self,row):
        if row['kind']=='grid':
            if row['style'].get('render_quality',0)<2:
                row['style'].update(smooth=True,render_quality=2)
            for key,value in dict(sun_azimuth=315.,sun_altitude=45.,sun_intensity=.25,sun_relief=1.).items():
                row['style'].setdefault(key,value)
        self._map_loading=True
        item=LayerItem(row['name']); item.setData(Qt.UserRole,row)
        item.setForeground(QColor('#dbe7f6'))
        item.setFlags(item.flags()|Qt.ItemIsUserCheckable|Qt.ItemIsEditable|Qt.ItemIsDragEnabled)
        item.setCheckState(Qt.Checked if row['visible'] else Qt.Unchecked)
        self.layers.insertItem(0,item); self._map_loading=False
        self.layers.setCurrentItem(item); self.map_reordered()

    def selected_grid(self):
        item=self.layers.currentItem()
        if item is None or item.data(Qt.UserRole)['kind']!='grid': raise ValueError('Выберите карту в списке слоёв.')
        return item.data(Qt.UserRole)['data']

    def add_grid(self,grid,xyz=None):
        if self.map_project_crs:
            source=self.source_crs(grid.crs,grid.name)
            if not source: return
            grid=self.activity.run('Пересчёт карты',reproject_grid,grid,self.map_project_crs,source)
        elif grid.crs:
            if self.layers.count():
                if not self.apply_project_crs(grid.crs): return
            else:
                self.map_project_crs=grid.crs; self.update_project_label()
        style=self.read_map_style(); style.update(vmin='',vmax='')
        row=dict(kind='grid',name=grid.name,visible=True,style=style,data=grid)
        if xyz is not None: row['xyz']=xyz
        self.add_map_layer(row)

    def map_reordered(self):
        rows=self.map_layer_rows(); self.grids=[r['data'] for r in rows if r['kind']=='grid']
        self.overlays=[r['data'] for r in rows if r['kind']=='vector' and r['visible']]
        self.redraw()

    def map_item_changed(self):
        if not self._map_loading: self.map_reordered()

    def map_context(self,position):
        item=self.layers.itemAt(position)
        if item is None: return
        self.layers.setCurrentItem(item); menu=QMenu(self)
        menu.addAction('Переименовать',lambda:self.layers.editItem(item))
        menu.addAction('СК и параметры проекции…',lambda:self.guard(self.layer_coordinates))
        if item.data(Qt.UserRole)['kind']=='vector':
            menu.addAction('Оформление контура…',lambda:self.guard(self.vector_style_dialog))
        menu.addAction('Выше',lambda:self.move_map_layer(-1))
        menu.addAction('Ниже',lambda:self.move_map_layer(1))
        menu.addAction('Удалить',lambda:self.guard(self.delete_map_layer))
        menu.exec_(self.layers.mapToGlobal(position))

    def move_map_layer(self,direction):
        source=self.layers.currentRow(); target=source+direction
        if source<0 or not 0<=target<self.layers.count(): return
        self._map_loading=True; item=self.layers.takeItem(source)
        self.layers.insertItem(target,item); self.layers.setCurrentItem(item); self._map_loading=False
        self.map_reordered()

    def delete_map_layer(self):
        row=self.layers.currentRow()
        if row<0: return
        self._map_loading=True; self._map_selected=None
        self.layers.takeItem(row); self._map_loading=False
        self.map_selection(self.layers.currentItem(),None)
        self.map_reordered()

    def open_vector(self):
        if not self.ensure_map_project(): return
        crs=self.map_project_crs
        if not crs: raise ValueError('Сначала задайте СК проекта.')
        path,_=QFileDialog.getOpenFileName(self,'Добавить векторный слой','','Vector (*.shp *.geojson *.json)')
        if path:
            source=None
            if Path(path).suffix.lower()=='.shp':
                prj=Path(path).with_suffix('.prj')
                source=self.import_source_crs(read_prj(prj) if prj.exists() else '',Path(path).stem)
                if source is None: return
            vector=self.activity.run('Загрузка векторного слоя',lambda: read_vector(path,source).reproject(crs))
            self.add_map_layer(dict(kind='vector',name=Path(path).name,visible=True,style={},data=vector))

    def clear_overlays(self):
        self._map_loading=True
        for i in reversed(range(self.layers.count())):
            if self.layers.item(i).data(Qt.UserRole)['kind']=='vector': self.layers.takeItem(i)
        self._map_loading=False; self.map_selection(self.layers.currentItem(),None); self.map_reordered()

    def vector_style_dialog(self):
        item=self.layers.currentItem()
        if item is None or item.data(Qt.UserRole)['kind']!='vector': return
        row=item.data(Qt.UserRole); style=row['style']
        dialog=QDialog(self); dialog.setWindowTitle('Оформление контура — '+item.text())
        form=QFormLayout(dialog)
        def color_button(value):
            button=QPushButton('Выбрать цвет…'); button.setProperty('color',value)
            def update(): button.setStyleSheet('QPushButton {border: 3px solid '+button.property('color')+';}')
            def choose():
                color=QColorDialog.getColor(QColor(button.property('color')),dialog,'Выберите цвет')
                if color.isValid(): button.setProperty('color',color.name()); update()
            button.clicked.connect(choose); update(); return button
        line=color_button(style.get('line_color','#fbbf24'))
        width=QDoubleSpinBox(); width.setRange(.1,20); width.setValue(style.get('line_width',1)); width.setSuffix(' pt')
        fill=QCheckBox('Заливка (FILL)'); fill.setChecked(style.get('fill',False))
        face=color_button(style.get('fill_color','#fbbf24'))
        opacity=QSpinBox(); opacity.setRange(0,100); opacity.setValue(round(style.get('fill_alpha',.3)*100)); opacity.setSuffix(' %')
        def polygon(g):
            return bool(g) and (g['type'] in ('Polygon','MultiPolygon') or
                g['type']=='GeometryCollection' and any(polygon(c) for c in g['geometries']))
        has_polygon=any(polygon(g) for g in row['data'].features)
        form.addRow('Цвет линии',line); form.addRow('Толщина линии',width)
        if has_polygon:
            form.addRow(fill); form.addRow('Цвет заливки',face); form.addRow('Непрозрачность заливки',opacity)
            fill.toggled.connect(face.setEnabled); fill.toggled.connect(opacity.setEnabled)
            face.setEnabled(fill.isChecked()); opacity.setEnabled(fill.isChecked())
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); form.addRow(buttons)
        if dialog.exec_()==QDialog.Accepted:
            style.update(line_color=line.property('color'),line_width=width.value(),fill=has_polygon and fill.isChecked(),
                         fill_color=face.property('color'),fill_alpha=opacity.value()/100)
            self.redraw()

    def redraw(self):
        if not hasattr(self,'_map_loading') or self._map_loading: return
        self._auto_layout_key=None
        item=self.layers.currentItem()
        self.map_style_heading.setText('Оформление карты — '+item.text() if item and item.data(Qt.UserRole)['kind']=='grid' else 'Оформление карты — выберите карту')
        self.map_style_group.setToolTip(self.map_style_heading.text())
        self.contour_color_button.setIcon(self.color_icon())
        self.sun_picker.set_light(self.sun_azimuth.value(),self.sun_altitude.value())
        if item and item.data(Qt.UserRole)['kind']=='grid': item.data(Qt.UserRole)['style'].update(self.read_map_style())
        rows=self.map_layer_rows(); grids=[r for r in rows if r['visible'] and r['kind']=='grid']
        self._map_decorations=[]
        old=self.figure.axes[0] if self.figure.axes else None
        if old is not None and old.images: self._map_view=(old.get_xlim(),old.get_ylim())
        view=getattr(self,'_map_view',None) if not self._map_fit else None
        self.figure._map_cache=None; self.figure.clear(); ax=None
        if grids:
            crs=grids[0]['data'].crs
            if any(r['data'].crs!=crs or r['data'].coordinate_unit!=grids[0]['data'].coordinate_unit for r in grids):
                raise ValueError('Для совместного отображения задайте всем картам одинаковые CRS и единицы координат.')
            for row in reversed(grids):
                before=set(ax.get_children()) if ax is not None else set()
                options=dict(row['style']); options['title']=self.map_title.text().strip() or self.map_project_name
                for k in ('vmin','vmax'): options[k]=float(options[k]) if str(options.get(k,'')).strip() else None
                if not QColor(options.get('contour_color','#dbe7f6')).isValid(): raise ValueError('Выберите допустимый цвет изолиний.')
                ax=draw_map(self.figure,row['data'],target_ax=ax,auto_layout=False,**options)
                order=(len(rows)-next(i for i,r in enumerate(rows) if r is row))*10
                for artist in set(ax.get_children())-before:
                    if artist in ax.images or artist in ax.collections or artist in ax.texts:
                        artist.set_zorder(order+min(artist.get_zorder(),5))
            # The one colourbar describes the frontmost visible map.
            self.figure.axes[1]._colorbar.update_normal([im for im in ax.images if im.get_gid()=='map-field'][-1])
            self.figure.axes[1]._colorbar.set_label(grids[0]['data'].unit)
        else:
            ax=self.figure.add_subplot(111)
            ax.set_xlabel('X, м'); ax.set_ylabel('Y, м')
            ax.set_aspect('equal',adjustable='box')
            ax.set_title(self.map_title.text().strip() or self.map_project_name)
            from .theme import style_map_figure
            style_map_figure(self.figure)
        visible=[r for r in rows if r['visible']]
        crs=grids[0]['data'].crs if grids else self.map_project_crs
        for i,row in enumerate(rows):
            if not row['visible']: continue
            order=(len(rows)-i)*10
            if row['kind']=='vector':
                before=set(ax.get_children())
                row['data'].draw(ax,crs,row['style'])
                for artist in set(ax.get_children())-before:
                    artist.set_zorder(order+1 if isinstance(artist,Rectangle) or artist in ax.patches else order+2)
            elif row['kind']=='raster':
                west,south,east,north=row['data'].bounds
                ax.imshow(row['data'].image,origin='upper',extent=(west,east,south,north),
                          zorder=order,interpolation='nearest')
        if visible:
            ax.relim(); ax.autoscale_view()
            self._map_home=(ax.get_xlim(),ax.get_ylim())
            if view: ax.set_xlim(view[0]); ax.set_ylim(view[1])
            self._geo_key=None; self.refresh_map_geography(update_fast=False)
            if self._map_layout and not self.auto_map_layout.isChecked(): self.figure.subplots_adjust(**self._map_layout)
            else: self.figure.subplots_adjust(left=.14,right=.9,bottom=(.23 if self.scale_crs.isChecked() else .20) if self.scale_on.isChecked() else .13,top=.84)
            if grids: self.draw_north(ax,self.figure.axes[1],crs)
            if self.scale_on.isChecked():
                if grids and grids[0]['data'].coordinate_unit=='degrees': raise ValueError('Масштабная линейка в метрах требует метрических координат.')
                self.draw_scale(ax)
        self._map_fit=False; self.map_dirty=True; self.canvas.draw_idle()
        if hasattr(self,'fast_map'): self.fast_map.show_layers(rows,(ax.get_xlim(),ax.get_ylim()) if ax.images else None)

    def color_icon(self):
        from PyQt5.QtGui import QPixmap,QIcon
        pixmap=QPixmap(24,16); pixmap.fill(QColor(self.contour_color.text())); return QIcon(pixmap)

    def map_scroll(self,event):
        if not self.figure.axes: return
        ax=self.figure.axes[0]
        if event.inaxes is not ax or event.xdata is None or event.ydata is None: return
        factor=1.2**(-event.step)
        ax.set_xlim(*[event.xdata+(v-event.xdata)*factor for v in ax.get_xlim()])
        ax.set_ylim(*[event.ydata+(v-event.ydata)*factor for v in ax.get_ylim()])
        self.map_dirty=True; self.canvas.draw_idle()

    def map_drawn(self,event):
        # Decorations depend on final axes positions after aspect adjustment and resize.
        changed=self.refresh_map_geography(strict=False)
        for update in getattr(self,'_map_decorations',[]): changed=update() or changed
        if hasattr(self,'auto_map_layout') and self.auto_map_layout.isChecked() and len(self.figure.axes)>1:
            ax=self.figure.axes[0]
            key=(*self.figure.bbox.bounds,*ax.get_xlim(),*ax.get_ylim())
            if key!=getattr(self,'_auto_layout_key',None):
                self._auto_layout_key=key
                self.figure.tight_layout(pad=1.4,rect=(.01,.02,.99,.98)); changed=True
        if changed: self.canvas.draw_idle()

    def draw_north(self,ax,bar,crs):
        self._map_decorations=[]
        arrow=bar.annotate('',xy=(0,0),xytext=(0,0),xycoords='figure pixels',textcoords='figure pixels',
                           arrowprops=dict(arrowstyle='-|>',color='black',lw=2.5,mutation_scale=18,shrinkA=0,shrinkB=0),annotation_clip=False)
        label=bar.text(0,0,'N',transform=IdentityTransform(),ha='left',va='center',fontsize=14,fontweight='bold',color='black',clip_on=False)
        def update(_=None):
            x=np.mean(ax.get_xlim()); y=np.mean(ax.get_ylim())
            try:
                if not crs: raise ValueError('СК не задана')
                direction=north_direction(crs,x,y)
                points=ax.transData.transform([[x,y],[x+direction[0],y+direction[1]]])
                delta=points[1]-points[0]; delta/=np.linalg.norm(delta)
                center=bar.transAxes.transform((.15,1))+np.array([0,22*self.figure.dpi/100])
                base=center-delta*15*self.figure.dpi/100; tip=center+delta*15*self.figure.dpi/100
                changed=not np.allclose(arrow.xy,tip) or not np.allclose(arrow.get_position(),base)
                arrow.xy=tip; arrow.set_position(base); arrow.set_visible(True)
                label.set_position((max(base[0],tip[0])+10*self.figure.dpi/100,center[1])); label.set_text('N'); label.set_visible(True)
                return changed
            except (ValueError,RuntimeError):
                arrow.set_visible(False); label.set_visible(False)
                return False
        original_draw=arrow.draw
        def draw_arrow(renderer):
            update(); original_draw(renderer)
        arrow.draw=draw_arrow
        update(); self._map_decorations.append(update)
        ax.callbacks.connect('xlim_changed',update); ax.callbacks.connect('ylim_changed',update)

    def draw_scale(self,ax):
        artists=[]; state=[]
        def update(_=None):
            lo,hi=ax.get_xlim(); count=self.scale_count.value()
            factor=1000 if self.scale_unit.currentText()=='км' else 1
            step=self.scale_step.value()*factor
            if step==0:
                desired=abs(hi-lo)*.25/count; base=10**np.floor(np.log10(max(desired,1e-12)))
                step=next(v*base for v in (1,2,5,10) if v*base>=desired)
            if self.auto_map_layout.isChecked() and step*count>abs(hi-lo)*.75:
                desired=abs(hi-lo)*.75/count; base=10**np.floor(np.log10(max(desired,1e-12)))
                step=max(v*base for v in (.1,.2,.5,1,2,5,10) if v*base<=desired)
            ticks=[v for v in ax.get_xticks() if min(lo,hi)<=v<=max(lo,hi)]
            x=(min(ticks) if lo<hi else max(ticks)) if ticks else min(lo,hi)
            pixels=self.figure.dpi/72
            renderer=_ if hasattr(_,'points_to_pixels') else self.canvas.get_renderer()
            bottom=ax.xaxis.label.get_window_extent(renderer).y0
            base=bottom-23*pixels
            current=(lo,hi,x,step,base)
            if state and np.allclose(state[0],current,rtol=0,atol=.01): return False
            state[:]=[current]
            for artist in artists: artist.remove()
            artists.clear()
            trans=blended_transform_factory(ax.transData,IdentityTransform())
            for i in range(count):
                patch=Rectangle((x+i*step,base),step,5*pixels,transform=trans,clip_on=False,
                                facecolor='white' if i%2 else 'black',edgecolor='black',linewidth=.7,zorder=100000)
                ax.add_patch(patch); artists.append(patch)
            for i in range(count+1):
                artists.append(ax.text(x+i*step,base+9*pixels,f'{i*step/factor:g}',transform=trans,ha='center',fontsize=8,color='#222222',zorder=100000,clip_on=False))
            artists.append(ax.text(x+count*step,base-11*pixels,self.scale_unit.currentText(),transform=trans,ha='right',fontsize=9,color='#222222',zorder=100000,clip_on=False))
            if self.scale_crs.isChecked():
                artists.append(ax.text(x,base-24*pixels,self.crs_caption(),transform=trans,ha='left',fontsize=8,color='#222222',zorder=100000,clip_on=False))
            return True
        original_draw=ax.draw
        def draw_with_scale(renderer):
            ax.xaxis.get_tightbbox(renderer); update(renderer); original_draw(renderer)
        ax.draw=draw_with_scale
        update(); self._map_decorations.append(update); ax.callbacks.connect('xlim_changed',update)

    def fit_maps(self): self._map_fit=True; self.redraw()

    def confirm_map_close(self):
        if self.worker is not None:
            raise ValueError('Дождитесь завершения текущего расчёта перед переключением проекта карт.')
        if not self.map_dirty or not self.layers.count(): return True
        answer=QMessageBox.question(self,'Проект карт','Сохранить изменения проекта карт?',
                                   QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel)
        if answer==QMessageBox.Cancel: return False
        return self.save_map_project() if answer==QMessageBox.Save else True

    def reset_map_project(self,name='Без названия'):
        self._map_loading=True; self._map_selected=None; self.layers.clear(); self._map_loading=False
        self.map_title.clear()
        self.restore_geography()
        self.auto_map_layout.blockSignals(True); self.auto_map_layout.setChecked(True); self.auto_map_layout.blockSignals(False)
        self.grids=[]; self.overlays=[]; self.map_project_name=name; self.map_project_path=None; self.map_project_crs=''
        self._map_loading=True; self.scale_on.setChecked(False); self.scale_step.setValue(0); self.scale_count.setValue(4); self._map_loading=False
        self._map_loading=True; self.scale_unit.setCurrentIndex(0); self.scale_crs.setChecked(False); self.north_symbol.setCurrentIndex(0); self._map_loading=False
        self.project_label.setText('Проект: '+name); self._map_view=None; self._map_fit=True; self.redraw()
        self._map_view=None; self._map_fit=True; self.map_dirty=False

    def new_map_project(self):
        self.project_crs_dialog(new=True)

    def close_map_project(self):
        if self.confirm_map_close(): self.reset_map_project()

    def save_map_project(self,save_as=False):
        self.map_layer_rows()
        if self._map_selected and self._map_selected.data(Qt.UserRole)['kind']=='grid':
            self._map_selected.data(Qt.UserRole)['style'].update(self.read_map_style())
        path=self.map_project_path
        if save_as or not path:
            path,_=QFileDialog.getSaveFileName(self,'Сохранить проект карт',self.map_project_name+'.ggmaps','GeoGarModel карты (*.ggmaps)')
        if not path: return False
        if not Path(path).suffix: path+='.ggmaps'
        ax=self.figure.axes[0] if self.figure.axes else None
        view=[list(ax.get_xlim()),list(ax.get_ylim())] if ax else None
        save_project(path,self.map_project_name,self.map_layer_rows(),view,
                     dict(enabled=self.scale_on.isChecked(),step=self.scale_step.value(),count=self.scale_count.value(),
                          unit=self.scale_unit.currentText(),show_crs=self.scale_crs.isChecked(),north=self.north_symbol.currentData()),
                     self.map_project_crs,presentation=dict(title=self.map_title.text(),geography=self.read_geography(),auto_layout=self.auto_map_layout.isChecked()))
        self.map_project_path=path; self.map_dirty=False; return True

    def open_map_project(self):
        path,_=QFileDialog.getOpenFileName(self,'Открыть проект карт','','GeoGarModel карты (*.ggmaps)')
        if not path: return
        manifest,rows=load_project(path)
        if not self.confirm_map_close(): return
        self.restore_map_project(manifest,rows,path)

    def restore_map_project(self,manifest,rows,path=None):
        project_crs=manifest.get('crs','') or next((r['data'].crs for r in rows if r['kind']=='grid' and r['data'].crs),'')
        if project_crs:
            for row in rows:
                if row['kind']=='vector': row['data']=row['data'].reproject(project_crs)
        MapWorkspace.reset_map_project(self,manifest['name'])
        self.map_title.setText(manifest.get('presentation',{}).get('title',''))
        self.restore_geography(manifest.get('presentation',{}).get('geography'))
        self.auto_map_layout.blockSignals(True); self.auto_map_layout.setChecked(manifest.get('presentation',{}).get('auto_layout',True)); self.auto_map_layout.blockSignals(False)
        self.map_project_crs=project_crs; self.update_project_label()
        for row in reversed(rows): self.add_map_layer(row)
        self._map_loading=True
        scale=manifest['scale']; self.scale_on.setChecked(scale['enabled'])
        self.scale_unit.setCurrentText(scale.get('unit','м'))
        self.scale_step.setDecimals(6 if self.scale_unit.currentText()=='км' else 2)
        self.scale_step.setValue(scale['step']); self.scale_count.setValue(scale['count'])
        self.scale_crs.setChecked(scale.get('show_crs',False))
        self.north_symbol.setCurrentIndex(max(0,self.north_symbol.findData(scale.get('north','arrow')))); self._map_loading=False
        self.redraw()
        if manifest['view'] and self.figure.axes:
            ax=self.figure.axes[0]; ax.set_xlim(manifest['view'][0]); ax.set_ylim(manifest['view'][1])
        self.map_project_path=path; self.map_dirty=False; self.canvas.draw_idle()
