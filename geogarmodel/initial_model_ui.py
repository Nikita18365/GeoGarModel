"""Initial-model controls kept separate from observation and solver settings."""
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (QGroupBox,QVBoxLayout,QComboBox,QLabel,QFileDialog,QInputDialog,QDialog,QDialogButtonBox)
from .voxel import VoxelModel,load_voxel,combine_components,edit,resample,clip_surface
from .grids import load_grid


class InitialModelPanel(QGroupBox):
    def __init__(self,tab):
        super().__init__('02 · Начальная модель'); self.tab=tab; self.model=None; self.undo_model=None
        layout=QVBoxLayout(self)
        self.source=QComboBox(); self.source.addItems(['Константа на новой сетке','Загруженная / отредактированная модель'])
        layout.addWidget(self.source)
        self.info=QLabel('Без файла: новая сетка заполняется полем «Начальное значение».'); self.info.setWordWrap(True); layout.addWidget(self.info)
        for label,callback in [('Загрузить воксель / UBC / VTK / центры…',self.load),('Загрузить отдельно JE, JN, JD…',self.components),('Использовать результат последнего моделирования',self.previous),('Предпросмотр модели и поля',self.preview),('Показать начальную модель в 3D',self.preview3d),('Редактор вокселя…',self.open_editor)]:
            tab.button(label,callback,layout)

    def open_editor(self):
        self.current()
        dialog=QDialog(self); dialog.setWindowTitle('Редактор начального вокселя'); dialog.resize(570,600)
        layout=QVBoxLayout(dialog)
        note=QLabel('Необязательная подготовка модели. Изменения применяются сразу. Для просмотра результата закройте редактор кнопкой «Готово».')
        note.setWordWrap(True); layout.addWidget(note)
        for title,actions in [('Свойства',[('Операции со свойствами…',self.operation)]),('Геометрия',[('Обрезать по поверхности…',self.clip),('Обрезать по прямоугольной области…',self.crop),('Перенести на сетку Nx / Ny / Nz…',self.transfer)]),('История и файл',[('Отменить последнюю операцию',self.undo),('Сохранить начальную модель NPZ…',self.save)])]:
            group=QGroupBox(title); rows=QVBoxLayout(group)
            for label,callback in actions: self.tab.button(label,callback,rows)
            layout.addWidget(group)
        buttons=QDialogButtonBox(); buttons.addButton('Готово',QDialogButtonBox.AcceptRole); buttons.accepted.connect(dialog.accept); layout.addWidget(buttons)
        dialog.exec_()

    def current(self):
        if self.source.currentIndex()==1:
            if self.model is None: raise ValueError('Сначала загрузите модель или выберите константу.')
            if self.model.mode!=self.tab.mode.currentData(): raise ValueError('Тип начальной модели отличается от выбранной физики. Загрузите нужное свойство.')
            crs=self.tab.crs.currentText().split('|')[0].strip()
            if self.model.crs and self.model.crs!=crs: raise ValueError('CRS модели и наблюдений различаются; автоматического перепроецирования нет.')
            return self.model.copy()
        mesh=self.tab.new_mesh()
        return VoxelModel(mesh,np.full((mesh.size,3) if self.tab.mode.currentData()=='vector' else (mesh.size,),self.tab.initial.value()),self.tab.mode.currentData(),self.tab.crs.currentText().split('|')[0].strip())

    def assign(self,model,previous=None):
        crs=self.tab.crs.currentText().split('|')[0].strip()
        if self.tab.table is not None and model.crs and crs and model.crs!=crs:
            raise ValueError('CRS загруженной модели отличается от CRS наблюдений. Согласуйте координаты и CRS перед загрузкой.')
        self.undo_model=previous.copy() if previous is not None else (self.model.copy() if self.model is not None else None)
        self.model=model; self.source.setCurrentIndex(1)
        m=model.mesh; v=model.values[m.active]
        self.info.setText(f'{m.shape} · {m.active.sum():,}/{m.size:,} активных · {model.mode}\nE: {m.east[0]:g}…{m.east[-1]:g}; N: {m.north[0]:g}…{m.north[-1]:g}\nDown: {m.depth[0]:g}…{m.depth[-1]:g} м; свойства {v.min():.5g}…{v.max():.5g}\nГеометрия загруженной модели используется при запуске. Nx/Ny/Nz применяются только к новой сетке или переносу.')
        if model.crs: self.tab.crs.setCurrentText(model.crs)
        self.tab.status.setText('Начальная модель готова. Проверьте единицы, диапазон свойств и границы ограничения.')

    def ask_load(self,mode):
        path,_=QFileDialog.getOpenFileName(self,'Начальная модель','','Voxel (*.geosoft_voxel *.msh *.mesh *.npz *.vtr *.vtk *.vti *.csv *.xlsx *.txt);;Все файлы (*)')
        if not path: return None
        options={}; ext=Path(path).suffix.lower()
        if ext in ('.msh','.mesh'):
            model_path,_=QFileDialog.getOpenFileName(self,'UBC свойства (соответствующий MOD/SUS/DEN)','','Все файлы (*)')
            if not model_path: return None
            options['model_path']=model_path
            dummy,ok=QInputDialog.getText(self,'UBC dummy','Неактивное значение (пусто = все конечные активны; Geosoft export часто −1):')
            if not ok: return None
            options['dummy']=float(dummy) if dummy.strip() else None
        elif ext in ('.vtk','.vtr','.vti'):
            import pyvista as pv
            names=list(pv.read(path).cell_data.keys())
            array,ok=QInputDialog.getItem(self,'Свойство VTK','Выберите cell-массив (Z = Up; вектор East/North/Down):',names,0,False)
            if not ok: return None
            options['array']=array
        elif ext not in ('.npz','.geosoft_voxel'):
            vertical,ok=QInputDialog.getItem(self,'Вертикальная координата','Столбец Down (глубина) или Up (высота):',['Down','Up'],0,False)
            if not ok: return None
            options['vertical']=vertical
        crs=self.tab.crs.currentText().split('|')[0].strip()
        return lambda:load_voxel(path,mode,crs,**options)

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

    def preview(self): self.tab.preview_initial(self.current())
    def preview3d(self):
        self.tab.scene.render(self.current().result()); self.tab.tabs.setCurrentIndex(0)

    def operation(self):
        model=self.current()
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
        self.assign(edit(model,labels[label],a,b,int(seed),component),previous=model); self.preview()

    def clip(self):
        model=self.current()
        path,_=QFileDialog.getOpenFileName(self,'Поверхность: East North Down/Up или ZMAP','','Surface (*.csv *.xlsx *.txt *.dat *.zmap *.npz)')
        if not path: return
        vertical,ok=QInputDialog.getItem(self,'Единицы поверхности','Значения поверхности в метрах:',['Down','Up'],0,False)
        if not ok: return
        keep,ok=QInputDialog.getItem(self,'Сторона поверхности','Что оставить?',['Ниже (фундамент)','Выше (осадочный чехол)'],0,False)
        if not ok: return
        if Path(path).suffix.lower() in ('.dat','.zmap','.npz'):
            g=load_grid(path); x,y=np.meshgrid(g.x,g.y); valid=np.isfinite(g.values)
            east,north,z=x[valid],y[valid],g.values[valid]
        else:
            from .models import read_table
            t=read_table(path); east,north,z=[t[c].to_numpy(float) for c in ('East','North',vertical)]
        self.assign(clip_surface(model,east,north,z,'below' if keep.startswith('Ниже') else 'above',vertical),previous=model); self.preview()

    def crop(self):
        model=self.current(); previous=model.copy()
        text,ok=QInputDialog.getText(self,'Область (метры)','Emin,Emax,Nmin,Nmax,Dmin,Dmax (сохраняются пересекающиеся части призм):')
        if not ok: return
        limits=np.asarray([float(v) for v in text.split(',')]).reshape(3,2)
        if np.any(limits[:,1]<=limits[:,0]): raise ValueError('Максимумы должны быть больше минимумов.')
        cells=model.mesh.cells(); cells[:,:3]=np.maximum(cells[:,:3],limits[:,0]); cells[:,3:]=np.minimum(cells[:,3:],limits[:,1])
        active=model.mesh.active&np.all(cells[:,3:]>cells[:,:3],axis=1)
        if not active.any(): raise ValueError('Область не пересекает активную модель.')
        cells[~active]=model.mesh.cells()[~active]; model.mesh.bounds=cells; model.mesh.active=active; model.values[~active]=0
        model.history.append(dict(operation='box crop',limits=limits.tolist())); self.assign(model,previous=previous); self.preview()

    def transfer(self):
        method,ok=QInputDialog.getItem(self,'Перенос на новую сетку','Новая сетка — Nx/Ny/Nz, верх/низ и площадь наблюдений. Ближайшая ячейка сохраняет значения, линейная сглаживает. Масса и поле могут измениться. Вне исходной области ячейки неактивны.',['nearest','linear'],0,False)
        if not ok: return
        model=self.current(); self.assign(resample(model,self.tab.new_mesh(),method),previous=model); self.preview()

    def undo(self):
        if self.undo_model is None: raise ValueError('Нет операции для отмены.')
        old=self.undo_model; self.assign(old); self.preview()

    def save(self):
        model=self.current(); path,_=QFileDialog.getSaveFileName(self,'Начальная модель','','NPZ (*.npz)')
        if path: model.save(path)
