"""Layer tree, independent layer styling and portable map workspaces."""
from pathlib import Path
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor,QKeySequence
from PyQt5.QtWidgets import (QListWidgetItem,QAbstractItemView,QMenu,QShortcut,QInputDialog,
    QFileDialog,QMessageBox,QDoubleSpinBox,QSpinBox,QLineEdit,QCheckBox,QColorDialog)
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtWidgets import QLabel
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from .map_project import read_vector,save_project,load_project
from .plotting import draw_map


class LayerItem(QListWidgetItem):
    def data(self,role):
        return self.payload if role==Qt.UserRole else super().data(role)

    def setData(self,role,value):
        if role==Qt.UserRole: self.payload=value
        else: super().setData(role,value)


class LayerList(QListWidget):
    def dropEvent(self,event):
        source=self.currentRow(); target=self.indexAt(event.pos()).row()
        if source<0: return
        if target<0: target=self.count()-1
        window=self.window(); window._map_loading=True
        item=self.takeItem(source); self.insertItem(target,item); self.setCurrentItem(item)
        window._map_loading=False; window.guard(window.map_reordered); event.accept()


class MapWorkspace:
    def setup_map_workspace(self,style_form,transform_form,project_layout):
        self.map_project_name='Без названия'; self.map_project_path=None; self.map_dirty=False
        self._map_selected=None; self._map_loading=False; self._map_fit=True
        self.project_label=QLabel('Проект: Без названия'); self.project_label.setWordWrap(True)
        project_layout.insertWidget(0,self.project_label)
        help_label=QLabel('✓ — показать слой · верхний слой впереди\nПеретащите для порядка · F2 — имя · Delete — удалить')
        help_label.setWordWrap(True); project_layout.insertWidget(1,help_label)
        self.layers.currentRowChanged.disconnect()
        self.layers.setSelectionMode(QAbstractItemView.SingleSelection)
        self.layers.setDragDropMode(QAbstractItemView.InternalMove)
        self.layers.setContextMenuPolicy(Qt.CustomContextMenu)
        self.layers.customContextMenuRequested.connect(self.map_context)
        self.layers.currentItemChanged.connect(self.map_selection)
        self.layers.itemChanged.connect(lambda _:self.guard(self.map_item_changed))
        self.layers.model().rowsMoved.connect(lambda *a:self.guard(self.map_reordered))
        self.map_delete=QShortcut(QKeySequence.Delete,self.layers)
        self.map_delete.setContext(Qt.WidgetShortcut); self.map_delete.activated.connect(lambda:self.guard(self.delete_map_layer))
        self.map_style_widgets={'cmap':self.palette,'mode':self.norm,'vmin':self.minimum,'vmax':self.maximum,
            'title':self.map_title,'contours':self.contours,'hillshade':self.shade}
        def number(minimum,maximum,value,decimals=2):
            w=QDoubleSpinBox(); w.setDecimals(decimals); w.setRange(minimum,maximum); w.setValue(value); return w
        self.contour_step=number(0,1e12,0,6); self.contour_step.setSpecialValueText('Авто')
        self.contour_color=QLineEdit('#dbe7f6'); self.contour_width=number(.1,10,.7)
        self.contour_labels=QCheckBox('Подписывать значения'); self.contour_labels.setChecked(True)
        self.map_alpha=number(0,1,1); self.map_alpha.setSingleStep(.1)
        for key,label,w in [('contour_step','Сечение изолиний (единицы поля)',self.contour_step),
                            ('contour_color','Цвет изолиний (имя или #RRGGBB)',self.contour_color),
                            ('contour_width','Толщина изолиний',self.contour_width),
                            ('contour_labels','',self.contour_labels),('alpha','Непрозрачность слоя',self.map_alpha)]:
            style_form.addRow(label,w); self.map_style_widgets[key]=w
        color_button=self.button('Выбрать цвет изолиний…',self.choose_contour_color,style_form)
        self.scale_on=QCheckBox('Масштабная линейка'); style_form.addRow(self.scale_on)
        self.scale_step=number(0,1e9,0); self.scale_step.setSpecialValueText('Авто')
        self.scale_count=QSpinBox(); self.scale_count.setRange(1,20); self.scale_count.setValue(4)
        style_form.addRow('Шаг линейки, м',self.scale_step); style_form.addRow('Число делений',self.scale_count)
        def show_details():
            for widget in (self.contour_step,self.contour_color,self.contour_width,self.contour_labels):
                widget.setVisible(self.contours.isChecked())
                label=style_form.labelForField(widget)
                if label: label.setVisible(self.contours.isChecked())
            color_button.setVisible(self.contours.isChecked())
            for widget in (self.scale_step,self.scale_count):
                widget.setVisible(self.scale_on.isChecked()); style_form.labelForField(widget).setVisible(self.scale_on.isChecked())
        self.contours.toggled.connect(show_details); self.scale_on.toggled.connect(show_details); show_details()
        for w in [*self.map_style_widgets.values(),self.scale_on,self.scale_step,self.scale_count]:
            signal=(w.editingFinished if isinstance(w,QLineEdit) else w.toggled if isinstance(w,QCheckBox)
                    else w.currentIndexChanged if hasattr(w,'currentIndexChanged') else w.valueChanged)
            signal.connect(lambda *a:self.guard(self.redraw))
        self.transform_rows={w:transform_form.labelForField(w) for w in (self.height,self.wave,self.degree)}
        self.operation.currentIndexChanged.connect(self.update_transform_parameters); self.update_transform_parameters()
        menu=self.menuBar().addMenu('Проект карт')
        for title,fn in [('Создать…',self.new_map_project),('Открыть…',self.open_map_project),
                         ('Сохранить',self.save_map_project),('Сохранить как…',lambda:self.save_map_project(True)),
                         ('Закрыть',self.close_map_project),('Вписать все видимые карты',self.fit_maps)]:
            menu.addAction(title).triggered.connect(lambda checked=False,fn=fn:self.guard(fn))

    def update_transform_parameters(self):
        operation=self.operation.currentData()
        shown={self.height:operation in ('upward','residual'),self.wave:operation in ('lowpass','highpass'),
               self.degree:operation in ('trend','detrend')}
        for widget,visible in shown.items():
            widget.setVisible(visible); self.transform_rows[widget].setVisible(visible)

    def choose_contour_color(self):
        color=QColorDialog.getColor(QColor(self.contour_color.text()),self,'Цвет изолиний')
        if color.isValid(): self.contour_color.setText(color.name()); self.redraw()

    def read_map_style(self):
        result={}
        for key,w in self.map_style_widgets.items():
            result[key]=(w.text() if isinstance(w,QLineEdit) else w.isChecked() if isinstance(w,QCheckBox)
                         else w.currentText() if key=='cmap' else w.currentData() if key=='mode' else w.value())
        return result

    def map_selection(self,item,previous):
        if self._map_loading: return
        if previous and previous.data(Qt.UserRole)['kind']=='grid': previous.data(Qt.UserRole)['style'].update(self.read_map_style())
        self._map_selected=item
        if item and item.data(Qt.UserRole)['kind']=='grid':
            style=item.data(Qt.UserRole)['style']; self._map_loading=True
            for key,w in self.map_style_widgets.items():
                value=style.get(key)
                if value is None: continue
                if isinstance(w,QLineEdit): w.setText(str(value))
                elif isinstance(w,QCheckBox): w.setChecked(value)
                elif key=='cmap': w.setCurrentText(value)
                elif key=='mode': w.setCurrentIndex(w.findData(value))
                else: w.setValue(value)
            self._map_loading=False
        self.guard(self.redraw)

    def map_layer_rows(self):
        rows=[]
        for i in range(self.layers.count()):
            item=self.layers.item(i); row=item.data(Qt.UserRole)
            row.update(name=item.text(),visible=item.checkState()==Qt.Checked)
            if row['kind']=='grid': row['data'].name=item.text()
            rows.append(row)
        return rows

    def add_map_layer(self,row):
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

    def add_grid(self,grid):
        style=self.read_map_style(); style.update(vmin='',vmax='',title='')
        self.add_map_layer(dict(kind='grid',name=grid.name,visible=True,style=style,data=grid))

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
        menu.addAction('Удалить',lambda:self.guard(self.delete_map_layer))
        menu.exec_(self.layers.mapToGlobal(position))

    def delete_map_layer(self):
        row=self.layers.currentRow()
        if row<0: return
        self._map_loading=True; self._map_selected=None
        self.layers.takeItem(row); self._map_loading=False
        self.map_selection(self.layers.currentItem(),None)
        self.map_reordered()

    def open_vector(self):
        grid=self.selected_grid()
        if not grid.crs: raise ValueError('Сначала задайте CRS карты: «CRS и единицы карты…».')
        path,_=QFileDialog.getOpenFileName(self,'Добавить векторный слой','','Vector (*.shp *.geojson *.json)')
        if path:
            vector=read_vector(path)
            from pyproj import Transformer
            Transformer.from_crs(vector.crs,grid.crs,always_xy=True)
            self.add_map_layer(dict(kind='vector',name=Path(path).stem,visible=True,style={},data=vector))

    def clear_overlays(self):
        self._map_loading=True
        for i in reversed(range(self.layers.count())):
            if self.layers.item(i).data(Qt.UserRole)['kind']=='vector': self.layers.takeItem(i)
        self._map_loading=False; self.map_selection(self.layers.currentItem(),None); self.map_reordered()

    def redraw(self):
        if not hasattr(self,'_map_loading') or self._map_loading: return
        item=self.layers.currentItem()
        if item and item.data(Qt.UserRole)['kind']=='grid': item.data(Qt.UserRole)['style'].update(self.read_map_style())
        rows=self.map_layer_rows(); grids=[r for r in rows if r['visible'] and r['kind']=='grid']
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
                options=dict(row['style']); options['title']=options.get('title') or row['name']
                for k in ('vmin','vmax'): options[k]=float(options[k]) if str(options.get(k,'')).strip() else None
                if not QColor(options.get('contour_color','#dbe7f6')).isValid(): raise ValueError('Выберите допустимый цвет изолиний.')
                ax=draw_map(self.figure,row['data'],target_ax=ax,**options)
                order=(len(rows)-next(i for i,r in enumerate(rows) if r is row))*10
                for artist in set(ax.get_children())-before:
                    if artist in ax.images or artist in ax.collections or artist in ax.texts:
                        artist.set_zorder(order+min(artist.get_zorder(),5))
            # The one colourbar describes the frontmost visible map.
            self.figure.axes[1]._colorbar.update_normal([im for im in ax.images if im.get_gid()=='map-field'][-1])
            self.figure.axes[1]._colorbar.set_label(grids[0]['data'].unit)
            for row in rows:
                if row['visible'] and row['kind']=='vector':
                    before=set(ax.get_children()); row['data'].draw(ax,crs)
                    order=(len(rows)-next(i for i,r in enumerate(rows) if r is row))*10
                    for artist in set(ax.get_children())-before: artist.set_zorder(order+2)
            ax.relim(); ax.autoscale_view()
            if view: ax.set_xlim(view[0]); ax.set_ylim(view[1])
            if self.scale_on.isChecked():
                if grids[0]['data'].coordinate_unit=='degrees': raise ValueError('Масштабная линейка в метрах требует метрических координат.')
                self.draw_scale(ax)
        else:
            ax=self.figure.add_subplot(111); ax.set_xlabel('X, м'); ax.set_ylabel('Y, м')
            from .theme import style_figure
            style_figure(self.figure)
        self._map_fit=False; self.map_dirty=True; self.canvas.draw_idle()

    def draw_scale(self,ax):
        artists=[]
        def update(_=None):
            for artist in artists: artist.remove()
            artists.clear(); lo,hi=ax.get_xlim(); count=self.scale_count.value()
            step=self.scale_step.value()
            if step==0:
                desired=abs(hi-lo)*.25/count; base=10**np.floor(np.log10(max(desired,1e-12)))
                step=next(v*base for v in (1,2,5,10) if v*base>=desired)
            x=min(lo,hi)+abs(hi-lo)*.05
            trans=blended_transform_factory(ax.transData,ax.transAxes)
            for i in range(count):
                patch=Rectangle((x+i*step,.045),step,.015,transform=trans,
                                facecolor='white' if i%2 else 'black',edgecolor='white',linewidth=.7,zorder=200)
                ax.add_patch(patch); artists.append(patch)
            for i in range(count+1):
                artists.append(ax.text(x+i*step,.07,f'{i*step:g}',transform=trans,ha='center',fontsize=8,color='white',zorder=200))
            artists.append(ax.text(x+count*step,.025,'м',transform=trans,ha='right',fontsize=9,color='white',zorder=200))
        update(); ax.callbacks.connect('xlim_changed',update)

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
        self.grids=[]; self.overlays=[]; self.map_project_name=name; self.map_project_path=None
        self._map_loading=True; self.scale_on.setChecked(False); self.scale_step.setValue(0); self.scale_count.setValue(4); self._map_loading=False
        self.project_label.setText('Проект: '+name); self._map_view=None; self._map_fit=True; self.redraw()
        self._map_view=None; self._map_fit=True; self.map_dirty=False

    def new_map_project(self):
        name,ok=QInputDialog.getText(self,'Новый проект карт','Название проекта:')
        if ok and name.strip() and self.confirm_map_close(): self.reset_map_project(name.strip())

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
                     dict(enabled=self.scale_on.isChecked(),step=self.scale_step.value(),count=self.scale_count.value()))
        self.map_project_path=path; self.map_dirty=False; return True

    def open_map_project(self):
        path,_=QFileDialog.getOpenFileName(self,'Открыть проект карт','','GeoGarModel карты (*.ggmaps)')
        if not path: return
        manifest,rows=load_project(path)
        if not self.confirm_map_close(): return
        self.reset_map_project(manifest['name'])
        for row in reversed(rows): self.add_map_layer(row)
        self._map_loading=True
        scale=manifest['scale']; self.scale_on.setChecked(scale['enabled'])
        self.scale_step.setValue(scale['step']); self.scale_count.setValue(scale['count']); self._map_loading=False
        self.redraw()
        if manifest['view'] and self.figure.axes:
            ax=self.figure.axes[0]; ax.set_xlim(manifest['view'][0]); ax.set_ylim(manifest['view'][1])
        self.map_project_path=path; self.map_dirty=False; self.canvas.draw_idle()
