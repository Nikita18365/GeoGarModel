"""Independent MDI model windows with a shared, path-addressed model catalogue."""
from dataclasses import dataclass
from pathlib import Path
from PyQt5.QtCore import Qt,QSize,QEvent,pyqtSignal
from PyQt5.QtGui import QBrush,QColor,QPalette
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QSplitter,QMdiArea,QMdiSubWindow,
    QListWidget,QListWidgetItem,QComboBox,QLabel,QLineEdit,QToolBar,QMenu,QInputDialog,QMessageBox,QAction,QApplication)
from .inversion_view import ModelView
from .view_icons import view_icon


@dataclass(eq=False)
class ModelEntry:
    key: str
    name: str
    path: str
    result: object


class ModelDocument(QMdiSubWindow):
    changed=pyqtSignal()
    closed=pyqtSignal(object)

    def __init__(self,title,parent=None):
        super().__init__(parent); self.setWindowTitle(title); self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowIcon(view_icon('iso'))
        palette=self.palette(); palette.setColor(QPalette.Highlight,QColor('#165c7a')); palette.setColor(QPalette.HighlightedText,QColor('#f1f7ff')); self.setPalette(palette)
        self.view=ModelView(); self.setWidget(self.view); self.resize(900,650)
        self.entries=[]; self.states={}; self.selected=None
        self.systemMenu().addSeparator()
        self.systemMenu().addAction('Переименовать окно…',self.rename_window)

    def rename_window(self):
        title,ok=QInputDialog.getText(self,'Переименовать 3D-окно','Название окна',text=self.windowTitle())
        if ok and title.strip() and title.strip()!=self.windowTitle():
            self.setWindowTitle(title.strip())
            self.changed.emit()

    def add_entry(self,entry):
        if entry not in self.entries: self.entries.append(entry)
        self.select(entry)

    def select(self,entry):
        if self.selected is entry:
            self.changed.emit(); return
        if self.view._expanded_dialog is not None: self.view._expanded_dialog.accept()
        if self.selected is not None: self.states[self.selected.key]=self.view.capture_state()
        if entry.key in self.states: self.view.restore_state(self.states[entry.key],refresh=False)
        self.view.render(entry.result)
        if entry.key in self.states: self.view.restore_state(self.states[entry.key])
        self.selected=entry; self.changed.emit()

    def remove_entry(self,entry):
        self.entries.remove(entry); self.states.pop(entry.key,None)
        if self.selected is entry:
            self.selected=None
            if self.entries: self.select(self.entries[0])
            else:
                self.view.result=None
                if self.view.plotter is not None: self.view.plotter.clear(); self.view.plotter.render()
        self.changed.emit()

    def closeEvent(self,event):
        self.view.shutdown(); self.closed.emit(self); super().closeEvent(event)

    def remember_geometry(self):
        if not self.isMinimized() and not self.isMaximized(): self._normal_rect=self.geometry().getRect()

    def resizeEvent(self,event):
        super().resizeEvent(event); self.remember_geometry()

    def moveEvent(self,event):
        super().moveEvent(event); self.remember_geometry()


class ModelWorkspace(QWidget):
    changed=pyqtSignal()
    def __init__(self,parent=None,import_callback=None):
        super().__init__(parent); self.registry={}; self._next_window=1; self._pending_result=None
        root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0)
        bar=QToolBar(); bar.setIconSize(QSize(24,24)); bar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        bar.setStyleSheet('QToolButton { padding: 3px; min-width: 28px; min-height: 28px; }'); root.addWidget(bar)
        bar.addAction(view_icon('new'),'Новое 3D-окно').triggered.connect(lambda:self.new_window())
        self.load_action=bar.addAction(view_icon('load'),'Загрузить 3D-модель с диска…')
        if import_callback: self.load_action.triggered.connect(lambda:import_callback())
        bar.addSeparator()
        self.window_selector=QComboBox(); self.window_selector.setMinimumContentsLength(12); self.window_selector.setToolTip('Активное окно. Выберите свёрнутое окно, чтобы открыть его.')
        self.window_selector.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.rename_action=QAction('Переименовать окно…',self.window_selector)
        self.rename_action.triggered.connect(self.rename_active_window)
        self.window_selector.addAction(self.rename_action)
        bar.addWidget(self.window_selector); self.window_selector.activated.connect(self.restore_window)
        bar.addAction(view_icon('tile'),'Расположить окна рядом').triggered.connect(lambda:self.mdi.tileSubWindows())
        bar.addAction(view_icon('cascade'),'Расположить окна каскадом').triggered.connect(lambda:self.mdi.cascadeSubWindows())
        splitter=QSplitter(); root.addWidget(splitter,1)
        sidebar=QWidget(); rows=QVBoxLayout(sidebar); rows.setContentsMargins(3,3,3,3)
        self.window_title=QLabel('Модели активного окна'); self.window_title.setWordWrap(True); rows.addWidget(self.window_title)
        self.models=QListWidget(); self.models.setContextMenuPolicy(Qt.CustomContextMenu); self.models.customContextMenuRequested.connect(self.model_menu)
        self.models.setStyleSheet('QListWidget::item { color: #dbe7f6; padding: 4px; } QListWidget::item:selected { color: #f1f7ff; background: #165c7a; }')
        self.models.currentItemChanged.connect(self.model_selected); rows.addWidget(self.models,1)
        rows.addWidget(QLabel('Источник выбранной модели'))
        self.source_path=QLineEdit(); self.source_path.setReadOnly(True); rows.addWidget(self.source_path)
        rows.addWidget(QLabel('Все загруженные модели'))
        add_row=QHBoxLayout(); rows.addLayout(add_row)
        self.catalogue=QComboBox(); self.catalogue.setMinimumContentsLength(12); self.catalogue.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon); add_row.addWidget(self.catalogue,1)
        add_bar=QToolBar(); add_bar.setIconSize(QSize(20,20)); add_bar.setStyleSheet('QToolButton { padding: 3px; }'); add_row.addWidget(add_bar)
        self.add_action=add_bar.addAction(view_icon('new'),'Добавить выбранную модель в активное окно'); self.add_action.triggered.connect(self.add_from_catalogue)
        self.mdi=QMdiArea(); self.mdi.setBackground(QBrush(QColor('#182638'))); self.mdi.setActivationOrder(QMdiArea.ActivationHistoryOrder)
        self.mdi.subWindowActivated.connect(self.active_changed)
        splitter.addWidget(sidebar); splitter.addWidget(self.mdi); splitter.setSizes([230,950]); splitter.setStretchFactor(1,1)
        self._documents=[]; self.refresh()

    def active_document(self):
        active=self.mdi.currentSubWindow()
        return active if isinstance(active,ModelDocument) else None

    @property
    def active_view(self):
        document=self.active_document(); return document.view if document else None

    @property
    def result(self):
        return self._pending_result if self._pending_result is not None else (self.active_view.result if self.active_view else None)

    @result.setter
    def result(self,result): self._pending_result=result

    @property
    def plotter(self): return self.active_view.plotter if self.active_view else None

    def register(self,result,path=None,name=None,source_key=None):
        path=path or result.settings.get('source_path','')
        if path: path=str(Path(path).resolve())
        key=source_key or ((path.casefold()+'|'+result.settings['mode']+'|'+result.settings.get('source_array','')) if path else f'memory:{id(result)}')
        if key not in self.registry:
            title=name or (Path(path).name if path else ('Начальная модель' if result.status=='initial_model' else 'Результат моделирования'))
            array=result.settings.get('source_array','')
            if array: title+=' · '+Path(array).name
            title+=' · '+{'gravity':'Δρ','induced':'χ','vector':'J'}[result.settings['mode']]
            if not path: title+=f' {sum(e.path=="" for e in self.registry.values())+1}'
            self.registry[key]=ModelEntry(key,title,path,result); self.refresh_catalogue()
        return self.registry[key]

    def new_window(self,entry=None):
        document=ModelDocument(f'3D-окно {self._next_window}'); document.number=self._next_window; self._next_window+=1
        self._documents.append(document); self.mdi.addSubWindow(document)
        document.changed.connect(self.refresh)
        document.changed.connect(self.changed)
        for w in document.view.state_controls().values():
            signal=w.currentTextChanged if isinstance(w,QComboBox) else w.toggled if hasattr(w,'toggled') else w.valueChanged
            signal.connect(self.modified)
        document.view.state_changed.connect(self.modified)
        document.closed.connect(self.document_closed)
        document.show(); self.mdi.setActiveSubWindow(document)
        if entry is not None: document.add_entry(entry)
        self.refresh(); self.changed.emit(); return document

    def modified(self,*args): self.changed.emit()

    def document_closed(self,document):
        if document in self._documents: self._documents.remove(document)
        if not getattr(self,'_closing',False): self.refresh(); self.changed.emit()

    def render(self,result,path=None,name=None,source_key=None):
        entry=self.register(result,path,name,source_key)
        document=self.active_document() or self.new_window()
        self.display_entry(document,entry)
        if document.isMinimized(): document.showNormal()
        return document

    def display_entry(self,document,entry):
        self._pending_result=None
        if document.selected is entry: document.view.render(entry.result)
        else: document.add_entry(entry)
        self.refresh()

    def finish_import(self,result,target):
        entry=self.register(result)
        if target not in self._documents:
            self.changed.emit()
            return False
        self.display_entry(target,entry)
        return True

    def add_from_catalogue(self):
        key=self.catalogue.currentData()
        if key not in self.registry: return
        document=self.active_document() or self.new_window(); document.add_entry(self.registry[key]); self.refresh()

    def showEvent(self,event):
        super().showEvent(event)
        QApplication.instance().installEventFilter(self)

    def hideEvent(self,event):
        QApplication.instance().removeEventFilter(self)
        super().hideEvent(event)

    def eventFilter(self,widget,event):
        # A VTK canvas or tool button may consume the press before it reaches its MDI frame.
        if event.type()==QEvent.MouseButtonPress and isinstance(widget,QWidget) and self.mdi.isAncestorOf(widget):
            parent=widget
            while parent is not self.mdi and not isinstance(parent,ModelDocument):
                parent=parent.parentWidget()
            if isinstance(parent,ModelDocument) and parent is not self.active_document():
                self.mdi.setActiveSubWindow(parent)
        return super().eventFilter(widget,event)

    def snapshot(self):
        windows=[]; active=self.active_document()
        for document in self._documents:
            states=dict(document.states)
            if document.selected is not None: states[document.selected.key]=document.view.capture_state()
            x,y,w,h=getattr(document,'_normal_rect',document.geometry().getRect()); width=max(1,self.mdi.width()); height=max(1,self.mdi.height())
            windows.append(dict(title=document.windowTitle(),entries=[entry.key for entry in document.entries],
                selected=document.selected.key if document.selected else None,states=states,
                geometry=[x/width,y/height,w/width,h/height],
                state='minimized' if document.isMinimized() else 'maximized' if document.isMaximized() else 'normal',active=document is active))
        return windows

    def restore(self,entries,windows):
        from importlib.util import find_spec
        self.reset()
        self.registry={entry.key:entry for entry in entries}; self.refresh_catalogue()
        active=None
        for saved in windows:
            document=self.new_window(); document.setWindowTitle(saved['title'])
            document.entries=[self.registry[key] for key in saved['entries']]
            document.states=saved['states']
            key=saved['selected']
            if key is not None:
                entry=self.registry[key]
                if find_spec('pyvistaqt') is not None: document.select(entry)
                else:
                    document.selected=entry; document.view.result=entry.result
                    document.view.restore_state(document.states.get(key,{}),refresh=False)
                    document.view.info.setText('Модель восстановлена. Для интерактивного 3D установите geogarmodel[volume].')
            x,y,w,h=saved['geometry']; width=max(400,self.mdi.width()); height=max(300,self.mdi.height())
            document.setGeometry(int(min(.8,max(0,x))*width),int(min(.8,max(0,y))*height),max(350,int(w*width)),max(250,int(h*height)))
            if saved['state']=='maximized': document.showMaximized()
            elif saved['state']=='minimized': document.showMinimized()
            if saved.get('active'): active=document
        if active is not None: self.mdi.setActiveSubWindow(active)
        self.refresh()

    def active_changed(self,*args): self.refresh()

    def rename_active_window(self):
        document=self.active_document()
        if document is not None: document.rename_window()

    def refresh_catalogue(self):
        key=self.catalogue.currentData(); self.catalogue.clear()
        for entry in self.registry.values():
            text=entry.path or 'В памяти — файл ещё не сохранён'
            self.catalogue.addItem(f'{entry.name} — {text}',entry.key)
            self.catalogue.setItemData(self.catalogue.count()-1,text,Qt.ToolTipRole)
        index=self.catalogue.findData(key)
        if index>=0: self.catalogue.setCurrentIndex(index)
        self.add_action.setEnabled(bool(self.registry))

    def refresh(self):
        if getattr(self,'_closing',False): return
        active=self.active_document(); self.window_selector.blockSignals(True); self.window_selector.clear()
        self.rename_action.setEnabled(active is not None)
        for document in self._documents:
            self.window_selector.addItem(document.windowTitle(),document.number)
            if document is active: self.window_selector.setCurrentIndex(self.window_selector.count()-1)
        self.window_selector.blockSignals(False)
        self.models.blockSignals(True); self.models.clear(); self.source_path.clear()
        self.window_title.setText('Модели · '+active.windowTitle() if active else 'Создайте окно или загрузите модель')
        if active:
            for entry in active.entries:
                item=QListWidgetItem(entry.name); item.setData(Qt.UserRole,entry.key); item.setToolTip(entry.path or 'В памяти — файл ещё не сохранён'); self.models.addItem(item)
                if entry is active.selected: self.models.setCurrentItem(item); self.show_path(entry)
        self.models.blockSignals(False)

    def show_path(self,entry):
        path=entry.path or 'В памяти — файл ещё не сохранён'
        array=entry.result.settings.get('source_array','')
        if array and Path(array).is_absolute(): path+=' | '+array
        self.source_path.setText(path); self.source_path.setToolTip(path); self.source_path.setCursorPosition(0)

    def restore_window(self,index):
        number=self.window_selector.itemData(index)
        document=next((d for d in self._documents if d.number==number),None)
        if document:
            if document.isMinimized(): document.showNormal()
            self.mdi.setActiveSubWindow(document); document.raise_(); self.refresh()

    def model_selected(self,item,previous=None):
        active=self.active_document()
        if item is not None and active:
            try: active.select(self.registry[item.data(Qt.UserRole)]); self.show_path(active.selected)
            except (ValueError,ImportError) as error: QMessageBox.warning(self,'3D-модель',str(error))

    def model_menu(self,point):
        item=self.models.itemAt(point); active=self.active_document()
        if item is None or active is None: return
        menu=QMenu(self); remove=menu.addAction('Убрать модель из этого окна')
        if menu.exec_(self.models.mapToGlobal(point)) is remove: active.remove_entry(self.registry[item.data(Qt.UserRole)])

    def open_slice_maps(self):
        if self._pending_result is not None:
            # Two-dimensional maps remain available without the optional VTK package.
            view=self.active_view
            if view is None: view=self.new_window().view
            result=self._pending_result; previous=view.result
            try: view.result=result; view.open_slice_maps()
            finally: view.result=previous; self._pending_result=None
        elif self.active_view: self.active_view.open_slice_maps()

    def shutdown(self):
        self._closing=True
        self.window_selector.clear(); self.models.clear()
        for document in list(self._documents): document.close()
        self.registry.clear()

    def reset(self):
        self.shutdown(); self._closing=False; self._next_window=1; self._pending_result=None
        self.refresh_catalogue(); self.refresh()
