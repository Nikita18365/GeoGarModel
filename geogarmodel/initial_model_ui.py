"""Initial-model controls kept separate from observation and solver settings."""
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (QGroupBox,QVBoxLayout,QComboBox,QLabel,QFileDialog,QInputDialog,QDialog,QDialogButtonBox,QFormLayout,QLineEdit)
from .voxel import VoxelModel,load_voxel,combine_components,edit,resample,clip_surface,crop_xy,crop_box
from .grids import load_grid
from .ui_layout import Disclosure


class InitialModelPanel(QGroupBox):
    def __init__(self,tab):
        super().__init__('02 · Начальная модель'); self.tab=tab; self.model=None; self.undo_model=None
        layout=QVBoxLayout(self)
        self.source=QComboBox(); self.source.addItems(['Константа на новой сетке','Загруженная / отредактированная модель'])
        layout.addWidget(self.source)
        self.info=QLabel('Без файла: новая сетка заполняется полем «Начальное значение».'); self.info.setWordWrap(True); layout.addWidget(self.info)
        for label,callback in [('Загрузить воксель / UBC / VTK / центры…',self.load),('Загрузить отдельно JE, JN, JD…',self.components),('Использовать результат последнего моделирования',self.previous),('Предпросмотр модели и поля',self.preview),('Показать начальную модель в 3D',self.preview3d),('Редактор вокселя…',self.open_editor)]:
            tab.button(label,callback,layout)
        extra=QGroupBox(''); extra_rows=QVBoxLayout(extra)
        for i in (4,3):
            widget=layout.takeAt(i).widget(); extra_rows.insertWidget(0,widget)
        layout.addWidget(Disclosure('Компоненты и предыдущий результат',extra))

    def open_editor(self):
        if self.source.currentIndex()==1:
            self.current(copy=False)
        dialog=QDialog(self); dialog.setWindowTitle('Редактор начального вокселя'); dialog.resize(570,600)
        layout=QVBoxLayout(dialog)
        note=QLabel('Необязательная подготовка модели. Изменения применяются сразу. Для просмотра результата закройте редактор кнопкой «Готово».')
        note.setWordWrap(True); layout.addWidget(note)
        for title,actions in [('Свойства',[('Операции со свойствами…',self.operation),('Калькулятор вокселей…',self.formula)]),('Геометрия',[('Обрезать по поверхности…',self.clip),('Обрезка куба по XY с сохранением положения кровли и подошвы…',self.crop_xy),('Обрезать по прямоугольной области…',self.crop),('Перенести на сетку Nx / Ny / Nz…',self.transfer)]),('История и файл',[('Отменить последнюю операцию',self.undo),('Сохранить начальную модель NPZ…',self.save)])]:
            group=QGroupBox(title); rows=QVBoxLayout(group)
            for label,callback in actions: self.tab.button(label,callback,rows)
            layout.addWidget(group)
        buttons=QDialogButtonBox(); buttons.addButton('Готово',QDialogButtonBox.AcceptRole); buttons.accepted.connect(dialog.accept); layout.addWidget(buttons)
        dialog.exec_()

    def current(self,copy=True):
        if self.source.currentIndex()==1:
            if self.model is None: raise ValueError('Сначала загрузите модель или выберите константу.')
            if self.model.mode!=self.tab.mode.currentData(): raise ValueError('Тип начальной модели отличается от выбранной физики. Загрузите нужное свойство.')
            crs=self.tab.crs.currentText().split('|')[0].strip()
            if self.model.crs and self.model.crs!=crs: raise ValueError('CRS модели и наблюдений различаются; автоматического перепроецирования нет.')
            return self.tab.workbench.activity.run('Копирование модели',self.model.copy) if copy else self.model
        mesh=self.tab.new_mesh()
        mode=self.tab.mode.currentData()
        crs=self.tab.crs.currentText().split('|')[0].strip()
        initial=self.tab.initial.value()
        shape=(mesh.size,3) if mode=='vector' else (mesh.size,)
        return self.tab.workbench.activity.run('Создание начальной модели',
            lambda: VoxelModel(mesh,np.full(shape,initial),mode,crs,own_values=True))

    def assign(self,model,previous=None,check_crs=True):
        crs=self.tab.crs.currentText().split('|')[0].strip()
        if check_crs and self.tab.table is not None and model.crs and crs and model.crs!=crs:
            raise ValueError('CRS загруженной модели отличается от CRS наблюдений. Согласуйте координаты и CRS перед загрузкой.')
        old=previous if previous is not None else self.model
        self.undo_model=self.tab.workbench.activity.run('Подготовка отмены операции',old.copy) if old is not None else None
        if previous is not None: model.source_path=''; model.source_array=''
        self.model=model; self.source.setCurrentIndex(1)
        m=model.mesh
        def summary():
            mask=m.active[:,None] if model.values.ndim==2 else m.active
            return (m.active.sum(),np.min(model.values,where=mask,initial=np.inf),
                    np.max(model.values,where=mask,initial=-np.inf))
        count,minimum,maximum=self.tab.workbench.activity.run('Проверка свойств модели',summary)
        self.info.setText(f'{m.shape} · {count:,}/{m.size:,} активных · {model.mode}\nE: {m.east[0]:g}…{m.east[-1]:g}; N: {m.north[0]:g}…{m.north[-1]:g}\nDown: {m.depth[0]:g}…{m.depth[-1]:g} м; свойства {minimum:.5g}…{maximum:.5g}\nГеометрия загруженной модели используется при запуске. Nx/Ny/Nz применяются только к новой сетке или переносу.')
        if model.crs: self.tab.crs.setCurrentText(model.crs)
        self.tab.status.setText('Начальная модель готова. Проверьте единицы, диапазон свойств и границы ограничения.')

    def ask_load(self,mode,title='Начальная модель'):
        path,_=QFileDialog.getOpenFileName(self,title,'','Voxel (*.geosoft_voxel *.msh *.mesh *.npz *.ggmodel *.vo *.vtr *.vtk *.vti *.vtu *.csv *.xlsx *.txt);;Все файлы (*)')
        if not path: return None
        options={}; ext=Path(path).suffix.lower()
        if ext in ('.msh','.mesh'):
            import json
            default='-1e32' if mode in ('induced','vector','J_parallel','J_magnitude') else ''
            sidecar=Path(path).with_suffix('.json')
            if sidecar.is_file():
                try:
                    meta=json.loads(sidecar.read_text(encoding='utf-8-sig'))
                    default=str(float(meta['dummy']))
                except (OSError,ValueError,TypeError,KeyError,AttributeError):
                    pass
            if mode=='vector':
                kind,ok=QInputDialog.getItem(self,'Векторная модель UBC','Файлы свойств',
                    ['Три файла JE, JN, JD','Один файл с тремя компонентами в каждой строке'],0,False)
                if not ok: return None
            else:
                kind=''
            if kind.startswith('Три'):
                components=[]
                for name in ('JE (East)','JN (North)','JD (Down)'):
                    component,_=QFileDialog.getOpenFileName(self,'UBC · '+name,str(Path(path).parent),'Все файлы (*)')
                    if not component: return None
                    components.append(component)
                options['component_paths']=components
            else:
                model_path,_=QFileDialog.getOpenFileName(self,'UBC свойства (соответствующий MOD/SUS/DEN)',str(Path(path).parent),'Все файлы (*)')
                if not model_path: return None
                options['model_path']=model_path
            dummy,ok=QInputDialog.getText(self,'UBC dummy','Неактивное значение (пусто = все конечные активны):',text=default)
            if not ok: return None
            options['dummy']=float(dummy) if dummy.strip() else None
        elif ext in ('.vtk','.vtr','.vti','.vtu'):
            import pyvista as pv
            def arrays():
                reader=pv.get_reader(path)
                return list(reader.cell_array_names) if hasattr(reader,'cell_array_names') else list(reader.read().cell_data.keys())
            names=self.tab.workbench.activity.run('Чтение структуры VTK',arrays)
            preferred='M_END' if mode=='vector' else 'property'
            selected=names.index(preferred) if preferred in names else 0
            array,ok=QInputDialog.getItem(self,'Свойство VTK','Выберите cell-массив (Z = Up; вектор East/North/Down):',names,selected,False)
            if not ok: return None
            options['array']=array
        elif ext=='.vo':
            from .gocad_voxet import header
            try:
                _,props,meta=header(path)
            except PermissionError as exc:
                raise ValueError('Windows запретила чтение заголовка Voxet. Проверьте права файла или заново экспортируйте куб в доступную папку.') from exc
            replacements={}
            for filename in dict.fromkeys(prop['PROP_FILE'] for prop in props.values()):
                if not (Path(path).parent/filename).is_file():
                    replacement,_=QFileDialog.getOpenFileName(self,'Файл данных Voxet · '+filename,
                        str(Path(path).parent),'Все файлы (*)')
                    if not replacement: return None
                    replacements[filename]=replacement
            options['binary_paths']=replacements
            if len(props)>1 and mode!='vector' and meta.get('mode')!='vector':
                labels={f"{k} · {p.get('PROPERTY',k)}":k for k,p in props.items()}
                label,ok=QInputDialog.getItem(self,'GOCAD Voxet','Свойство',list(labels),0,False)
                if not ok: return None
                options['property_id']=labels[label]
        elif ext not in ('.npz','.ggmodel','.geosoft_voxel'):
            vertical,ok=QInputDialog.getItem(self,'Вертикальная координата','Столбец Down (глубина) или Up (высота):',['Down','Up'],0,False)
            if not ok: return None
            options['vertical']=vertical
        crs=self.tab.crs.currentText().split('|')[0].strip()
        def loader():
            try:
                model=load_voxel(path,mode,crs,**options)
            except PermissionError as exc:
                raise ValueError(f'Windows запретила чтение файла: {exc.filename or path}. Проверьте права файла или заново экспортируйте куб в доступную папку.') from exc
            model.source_path=str(Path(path).resolve())
            model.source_array=options.get('array',options.get('model_path',options.get('property_id','')))
            if 'component_paths' in options:
                model.source_array=' · '.join(options['component_paths'])
            return model
        loader.path = str(Path(path).resolve())
        if 'component_paths' in options:
            loader.path += ' · ' + ' · '.join(options['component_paths'])
        elif 'model_path' in options:
            loader.path += ' · ' + str(options['model_path'])
        elif 'array' in options or 'property_id' in options:
            loader.path += ' · ' + str(options.get('array',options.get('property_id')))
        return loader

    def load(self):
        loader=self.ask_load(self.tab.mode.currentData())
        if loader is not None: self.tab.model_task(loader,self.loaded)

    def loaded(self,model): self.assign(model); self.preview()

    def components(self):
        loaders=[]
        for name in ('JE (East)','JN (North)','JD (Down)'):
            self.tab.status.setText('Загрузка компоненты '+name+' в A/m')
            loader=self.ask_load('gravity')  # Three scalar sources; no density conversion applied.
            if loader is None: return
            loaders.append(loader)
        def done(model):
            self.tab.mode.setCurrentIndex(2); self.loaded(model)
        self.tab.model_task(lambda:combine_components([loader() for loader in loaders]),done)

    def previous(self):
        r=self.tab.require_result()
        self.assign(VoxelModel(r.mesh,r.model,r.settings['mode'],r.settings.get('crs',''),[dict(operation='previous inversion')]))
        self.preview()

    def preview(self): self.tab.preview_initial(self.tab.prepared_model(copy=False))
    def preview3d(self):
        self.tab.scene.render(self.tab.prepared_model(copy=False).result()); self.tab.tabs.setCurrentIndex(0)

    def formula(self):
        from .voxel_formula import Formula, calculate, MODEL_NAMES
        from PyQt5.QtWidgets import QScrollArea,QWidget,QPushButton,QSpinBox,QHBoxLayout
        dialog = QDialog(self)
        dialog.setWindowTitle('Калькулятор вокселей')
        form = QFormLayout(dialog)
        count = QSpinBox()
        count.setRange(1,len(MODEL_NAMES))
        count.setValue(2)
        form.addRow('Количество моделей',count)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        rows = QWidget()
        layout = QFormLayout(rows)
        scroll.setWidget(rows)
        form.addRow(scroll)
        dialog.resize(620,440)
        loaders = {}
        controls = []
        def choose(name,field):
            loader = self.ask_load(self.tab.mode.currentData(),'Выберите воксель '+name)
            if loader is not None:
                loaders[name] = loader
                field.setText(loader.path)
                field.setToolTip(loader.path)
        def reset_a(field):
            loaders.pop('A',None)
            field.setText('Текущая модель редактора')
            field.setToolTip('')
        for i,name in enumerate(MODEL_NAMES):
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0,0,0,0)
            field = QLineEdit()
            field.setReadOnly(True)
            line.addWidget(field)
            button = QPushButton('Выбрать…')
            button.setObjectName('file_'+name)
            button.clicked.connect(lambda _,n=name,f=field:choose(n,f))
            line.addWidget(button)
            if name=='A':
                reset_a(field)
                reset = QPushButton('Текущая')
                reset.clicked.connect(lambda _,f=field:reset_a(f))
                line.addWidget(reset)
            layout.addRow(name,row)
            controls.append((layout.labelForField(row),row))
        def resize_rows(n):
            for i,pair in enumerate(controls):
                for widget in pair:
                    widget.setVisible(i<n)
        count.valueChanged.connect(resize_rows)
        resize_rows(count.value())
        form.addRow(QLabel('A задаёт сетку результата. X, Y, Z — координаты центра ячейки.'))
        expression = QComboBox()
        expression.setEditable(True)
        expression.addItems(['A + B','A - B','A * B','A * 2','where(B != 0, A / B, 0)',
                             'where(Z >= 1000 and A > 0, A, 0)','clip(A, -1, 1)'])
        form.addRow('Формула',expression)
        form.addRow(QLabel('where(условие, значение если верно, значение иначе)'))
        error = QLabel()
        error.setWordWrap(True)
        form.addRow(error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        form.addRow(buttons)
        buttons.rejected.connect(dialog.reject)
        def confirm():
            try:
                dialog.formula = Formula(expression.currentText())
                used = dialog.formula.names.intersection(MODEL_NAMES)
                outside = used.difference(MODEL_NAMES[:count.value()])
                missing = used.difference(loaders,{'A'})
                if outside:
                    raise ValueError('Увеличьте количество моделей для '+', '.join(sorted(outside)))
                if missing:
                    raise ValueError('Выберите файл для '+', '.join(sorted(missing)))
            except (ValueError,OverflowError) as exc:
                error.setText(str(exc))
                return
            dialog.accept()
        buttons.accepted.connect(confirm)
        if dialog.exec_()!=QDialog.Accepted: return
        formula = dialog.formula
        previous = self.model if 'A' in loaders else self.current(copy=False)
        def run():
            base = loaders['A']() if 'A' in loaders else previous
            if base.mode != mode:
                raise ValueError('Свойство модели A отличается от выбранного режима.')
            operands = {n:loaders[n]() for n in MODEL_NAMES[1:] if n in formula.names}
            return calculate(base,formula,models=operands)
        mode = self.tab.mode.currentData()
        result = self.tab.workbench.activity.run('Калькулятор вокселей',run)
        self.assign(result,previous=previous)
        self.preview()

    def operation(self):
        model=self.current(copy=False)
        labels={'Заполнить константой (a)':'constant','Умножить на a':'multiply','Прибавить a':'add','Ограничить [a,b]':'clamp','Нормальное распределение (a=среднее,b=σ)':'normal','Равномерное распределение [a,b]':'uniform','Градиент с глубиной: a + b × Down(м)':'depth_gradient'}
        label,ok=QInputDialog.getItem(self,'Операции','Операция над активными ячейками:',list(labels),0,False)
        if not ok: return
        component=None
        if model.mode=='vector':
            c,ok=QInputDialog.getItem(self,'Компоненты','Применить к:',['Все компоненты','JE','JN','JD'],0,False)
            if not ok: return
            component=None if c=='Все компоненты' else ['JE','JN','JD'].index(c)
        text,ok=QInputDialog.getText(self,'Параметры','a, b, seed (для случайных распределений):',text='0, 1, 42')
        if not ok: return
        a,b,seed=map(float,text.split(','))
        self.assign(self.tab.workbench.activity.run('Изменение свойств',edit,model,labels[label],a,b,int(seed),component),previous=model); self.preview()

    def clip(self):
        model=self.current(copy=False)
        path,_=QFileDialog.getOpenFileName(self,'Поверхность: East North Down/Up или ZMAP','','Surface (*.csv *.xlsx *.txt *.dat *.zmap *.npz)')
        if not path: return
        vertical,ok=QInputDialog.getItem(self,'Единицы поверхности','Значения поверхности в метрах:',['Down','Up'],0,False)
        if not ok: return
        keep,ok=QInputDialog.getItem(self,'Сторона поверхности','Что оставить?',['Ниже (фундамент)','Выше (осадочный чехол)'],0,False)
        if not ok: return
        if Path(path).suffix.lower() in ('.dat','.zmap','.npz'):
            g=self.tab.workbench.activity.run('Загрузка поверхности',load_grid,path); x,y=np.meshgrid(g.x,g.y); valid=np.isfinite(g.values)
            east,north,z=x[valid],y[valid],g.values[valid]
        else:
            from .models import read_table
            t=self.tab.workbench.activity.run('Загрузка поверхности',read_table,path); east,north,z=[t[c].to_numpy(float) for c in ('East','North',vertical)]
        self.assign(self.tab.workbench.activity.run('Обрезка по поверхности',clip_surface,model,east,north,z,'below' if keep.startswith('Ниже') else 'above',vertical),previous=model); self.preview()

    def crop(self):
        self.crop_dialog(True)

    def crop_xy(self):
        self.crop_dialog(False)

    def crop_dialog(self, rectangular):
        previous = self.current(copy=False)
        mesh = previous.mesh
        dialog = QDialog(self)
        title = 'Обрезка по прямоугольной области' if rectangular else 'Обрезка куба по XY'
        dialog.setWindowTitle(title)
        form = QFormLayout(dialog)
        fields = []
        coordinates = [('X минимум (East), м',mesh.east[0]),('X максимум (East), м',mesh.east[-1]),
                            ('Y минимум (North), м',mesh.north[0]),('Y максимум (North), м',mesh.north[-1])]
        if rectangular:
            coordinates.extend([('Z минимум (Down), м',mesh.depth[0]),('Z максимум (Down), м',mesh.depth[-1])])
            note = QLabel('Будет создан сплошной блок между заданными глубинами. Пустые ячейки получат значение 0. Частично заполненные сохранят свойство на весь оставшийся объём.')
            note.setWordWrap(True)
            form.addRow(note)
        for label,value in coordinates:
            field = QLineEdit(str(value))
            fields.append(field)
            form.addRow(label,field)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() == QDialog.Accepted:
            try:
                limits = [float(field.text().strip().replace(',', '.')) for field in fields]
            except ValueError:
                raise ValueError('Введите числовые координаты в метрах.') from None
            result = self.tab.workbench.activity.run(title,crop_box if rectangular else crop_xy,previous,*limits)
            self.assign(result,previous=previous)
            self.preview()

    def transfer(self):
        method,ok=QInputDialog.getItem(self,'Перенос на новую сетку','Новая сетка — Nx/Ny/Nz, верх/низ и площадь наблюдений. Ближайшая ячейка сохраняет значения, линейная сглаживает. Масса и поле могут измениться. Вне исходной области ячейки неактивны.',['nearest','linear'],0,False)
        if not ok: return
        model=self.current(copy=False); self.assign(self.tab.workbench.activity.run('Перенос на сетку',resample,model,self.tab.new_mesh(),method),previous=model); self.preview()

    def undo(self):
        if self.undo_model is None: raise ValueError('Нет операции для отмены.')
        old=self.undo_model; self.assign(old); self.preview()

    def save(self):
        model=self.current(copy=False); path,_=QFileDialog.getSaveFileName(self,'Начальная модель','','NPZ (*.npz)')
        if path: self.tab.workbench.activity.run('Сохранение модели',model.save,path)
