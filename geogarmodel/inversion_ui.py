"""Third workbench tab: independent gravity / induced / vector magnetic inversion."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QGroupBox,
    QComboBox,QDoubleSpinBox,QSpinBox,QPushButton,QLabel,QScrollArea,QSplitter,
    QTabWidget,QFileDialog,QProgressBar,QPlainTextEdit,QCheckBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,NavigationToolbar2QT
from threading import Event
from .inversion import Mesh,FieldOperator,invert,InversionResult
from .inversion_view import ModelView,vtk_grid
from .models import read_table
from .coordinates import gsk2011_zones,validate_crs
from .compute import ComputationCancelled
from .initial_model_ui import InitialModelPanel
from .theme import style_figure


class InversionWorker(QThread):
    progress=pyqtSignal(object); result=pyqtSignal(object); failed=pyqtSignal(str)
    def __init__(self,function,parent):
        super().__init__(parent); self.function=function; self.cancel=Event()
    def run(self):
        try: self.result.emit(self.function(self.progress.emit,self.cancel))
        except ComputationCancelled: self.failed.emit('Расчёт остановлен. История итераций сохранена на графике.')
        except Exception as exc: self.failed.emit(str(exc))


class InversionTab(QWidget):
    def __init__(self,workbench):
        super().__init__(workbench)
        self.workbench=workbench; self.worker=None; self.table=None; self.result=None
        root=QHBoxLayout(self); splitter=QSplitter(); root.addWidget(splitter)
        left=QWidget(); panel=QVBoxLayout(left)
        self.parameter_panel=left
        self.mode=QComboBox()
        for label,key in [('Гравитация · Δρ, g/cm³','gravity'),('Магнитная · индуцированная χ, SI','induced'),('Магнитная · векторная J, A/m','vector')]: self.mode.addItem(label,key)
        panel.addWidget(self.mode)
        self.hint=QLabel(); self.hint.setWordWrap(True); panel.addWidget(self.hint)
        self.load_button=self.button('Наблюдения XLSX / CSV…',self.load_table,panel)
        self.button('Наблюдённое поле ZMAP…',self.load_field,panel)
        self.grid_button=self.button('Взять выбранную карту',self.load_grid,panel)
        self.demo_button=self.button('Синтетический пример для режима',self.demo,panel)
        self.model_load=self.button('Открыть сохранённый результат NPZ…',self.load_result,panel)
        self.data_info=QLabel('Загрузите аномалию Δg (mGal) или ΔT (nT), не полное поле.'); self.data_info.setWordWrap(True); panel.addWidget(self.data_info)
        data=QGroupBox('01 · Координаты и наблюдения'); form=QFormLayout(data)
        self.east,self.north,self.height,self.value,self.error=[QComboBox() for _ in range(5)]
        self.crs=QComboBox(); self.crs.setEditable(True)
        for code,name in gsk2011_zones(): self.crs.addItem(f'{code} | {name}')
        self.crs.setCurrentText('EPSG:20910 | GSK-2011 / Gauss-Kruger zone 10')
        self.height_constant=self.number(-1e6,1e6,0)
        self.sigma=self.number(1e-8,1e8,1,6)
        self.stride=self.integer(1,100000,1)
        for name,w in [('East, м',self.east),('North, м',self.north),('Высота Up, м',self.height),('Высота константа',self.height_constant),('Аномалия',self.value),('Ошибка σ: столбец',self.error),('σ константа',self.sigma),('CRS',self.crs)]: form.addRow(name,w)
        panel.addWidget(data)
        form.addRow('Использовать каждую N-ю строку',self.stride)
        self.initial_panel=InitialModelPanel(self); panel.addWidget(self.initial_panel)
        mesh_group=QGroupBox('03 · Новая сетка / сетка переноса'); f=QFormLayout(mesh_group)
        self.nx,self.ny,self.nz=[self.integer(2,200,v) for v in (12,12,8)]
        self.top=self.number(-1e6,1e6,100)
        self.bottom=self.number(-1e6,1e6,3000)
        self.padding=self.number(0,10,.2)
        self.cache=self.integer(0,2048,128)
        for name,w in [('Nx',self.nx),('Ny',self.ny),('Nz',self.nz),('Верх: глубина Down, м',self.top),('Низ: глубина Down, м',self.bottom),('Запас площади, доля',self.padding)]: f.addRow(name,w)
        panel.addWidget(mesh_group)
        forms=[form,f]
        settings=QGroupBox('04 · Свойства и ограничения'); f=QFormLayout(settings)
        self.field=self.number(1,200000,50000)
        self.inc=self.number(-90,90,60); self.dec=self.number(-360,360,0)
        self.lower=self.number(-1e5,1e5,-1,6); self.upper=self.number(-1e5,1e5,1,6)
        self.initial=self.number(-1e5,1e5,0,6)
        self.iterations=self.integer(1,10000,100)
        self.damping=self.number(0,1e8,.01,6); self.smooth=self.number(0,1e8,.1,6)
        self.depth_index=self.number(0,4,1)
        self.target=self.number(0,1000,1,4)
        self.step=self.number(.001,2,1,3)
        self.offset=QCheckBox('Подбирать постоянный сдвиг поля')
        self.device=QComboBox(); self.device.addItem('CPU · NumPy','cpu'); self.device.addItem('GPU · NVIDIA CUDA','gpu')
        self.gpu=self.integer(0,31,0)
        for name,w in [('Свойство: минимум',self.lower),('Свойство: максимум',self.upper),('Начальное значение',self.initial)]: f.addRow(name,w)
        panel.addWidget(settings)
        self.magnetic_group=QGroupBox('05 · Магнитное главное поле'); mf=QFormLayout(self.magnetic_group)
        for name,w in [('Главное поле F, nT',self.field),('Наклонение I вниз, °',self.inc),('Склонение D от севера, °',self.dec)]: mf.addRow(name,w)
        panel.addWidget(self.magnetic_group)
        solver=QGroupBox('06 · Решатель и регуляризация'); sf=QFormLayout(solver)
        for name,w in [('Итерации',self.iterations),('Глубинный индекс',self.depth_index),('L2 малость',self.damping),('Гладкость',self.smooth),('Масштаб первого шага',self.step),('Целевая RMS / σ',self.target)]: sf.addRow(name,w)
        sf.addRow(self.offset); panel.addWidget(solver)
        compute=QGroupBox('07 · Вычисления'); cf=QFormLayout(compute)
        cf.addRow('Устройство',self.device); cf.addRow('GPU ID',self.gpu); cf.addRow('Кэш, MiB',self.cache); panel.addWidget(compute)
        self.dec.setToolTip('Склонение относительно севера проекционной сетки. Для IGRF учтите сближение меридианов.')
        forms.extend([f,mf,sf,cf])
        for f in forms:
            f.setRowWrapPolicy(QFormLayout.WrapLongRows)
            f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        panel.addStretch()
        for combo in left.findChildren(QComboBox):
            combo.setMinimumContentsLength(10); combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(left)
        controls=QWidget(); cl=QVBoxLayout(controls); cl.addWidget(scroll)
        self.run_button=self.button('Запустить 3D-инверсию',self.run,cl); self.run_button.setObjectName('primary')
        self.stop=self.button('Остановить',self.cancel,cl); self.stop.setEnabled(False)
        self.progress=QProgressBar(); cl.addWidget(self.progress)
        controls.setMinimumWidth(400); controls.setMaximumWidth(540); splitter.addWidget(controls)
        right=QWidget(); rl=QVBoxLayout(right)
        self.status=QLabel('Прямоугольные призмы · точная ΔT · ограниченная градиентная инверсия'); self.status.setWordWrap(True); rl.addWidget(self.status)
        self.tabs=QTabWidget(); rl.addWidget(self.tabs,1)
        self.scene=ModelView(); self.tabs.addTab(self.scene,'3D-модель')
        self.figure=Figure(); self.canvas=FigureCanvasQTAgg(self.figure)
        plot=QWidget(); pl=QVBoxLayout(plot); pl.addWidget(NavigationToolbar2QT(self.canvas,self)); pl.addWidget(self.canvas)
        self.tabs.addTab(plot,'Срезы и поля')
        self.convergence=pg.PlotWidget(background='#101c2f'); self.convergence.setLabel('left','RMSE'); self.convergence.setLabel('bottom','Итерация'); self.curve=self.convergence.plot(pen=pg.mkPen('#2dd4bf',width=2)); self.tabs.addTab(self.convergence,'Сходимость')
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100); self.log.setMaximumBlockCount(1000); rl.addWidget(self.log)
        actions=QHBoxLayout(); rl.addLayout(actions)
        self.button('Открыть / обновить 3D',self.open_3d,actions)
        self.button('Модель NPZ…',self.save,actions)
        self.button('Поля CSV…',self.save_fields,actions)
        self.button('VTK…',self.save_vtk,actions)
        splitter.addWidget(right); splitter.setSizes([440,1000]); splitter.setStretchFactor(1,1)
        self.mode.currentIndexChanged.connect(self.mode_changed); self.mode_changed()
        self.history=[]

    def button(self,text,fn,layout):
        w=QPushButton(text); w.clicked.connect(lambda: self.workbench.guard(fn)); layout.addWidget(w); return w
    @staticmethod
    def number(lo,hi,val,decimals=2):
        w=QDoubleSpinBox(); w.setDecimals(decimals); w.setRange(lo,hi); w.setValue(val); return w
    @staticmethod
    def integer(lo,hi,val):
        w=QSpinBox(); w.setRange(lo,hi); w.setValue(val); return w

    def mode_changed(self):
        mode=self.mode.currentData()
        self.hint.setText({'gravity':'Контраст плотности Δρ в g/cm³; Δg положительно вниз. Поправки Буге/рельефа и региональное поле готовятся до импорта.',
            'induced':'Восприимчивость χ в СИ (без множителя 10⁻⁵); J=χH вдоль заданного главного поля.',
            'vector':'Суммарная J=(JE, JN, JD), A/m: допускает остаточную намагниченность. Разделить её на индуцированную и остаточную по одной ΔT нельзя. Границы задаются для каждой компоненты.'}[mode])
        self.lower.setValue(0 if mode=='induced' else -1)
        self.upper.setValue(.1 if mode=='induced' else 1)
        for w in (self.field,self.inc,self.dec): w.setEnabled(mode!='gravity')
        self.magnetic_group.setVisible(mode!='gravity')

    def assign_table(self,table):
        if len(table.columns)<3: raise ValueError('Нужны минимум координаты и аномалия.')
        self.table=table; self.result=None
        self.scene.result=None
        for combo in (self.east,self.north,self.height,self.value,self.error): combo.clear()
        for combo in (self.east,self.north,self.value): combo.addItems(list(table.columns))
        for combo in (self.height,self.error): combo.addItem('Константа'); combo.addItems(list(table.columns))
        def select(combo,names,fallback):
            match=next((n for n in names if n in table.columns),None)
            combo.setCurrentText(match) if match else combo.setCurrentIndex(fallback)
        select(self.east,['East','E','Y'],0); select(self.north,['North','N','X'],1)
        select(self.value,['anomaly','Δg','ΔT'],len(table.columns)-1)
        select(self.height,['height','Up','elevation'],0)
        select(self.error,['sigma'],0)
        self.data_info.setText(f'{len(table):,} точек. Проверьте столбцы, единицы, CRS и высоту. Старые результаты очищены.')
        self.figure.clear(); self.canvas.draw_idle()
        if self.scene.plotter is not None: self.scene.plotter.clear()

    def load_table(self):
        path,_=QFileDialog.getOpenFileName(self,'Наблюдения','', 'Таблицы (*.xlsx *.csv *.txt)')
        if path: self.assign_table(read_table(path))

    def load_grid(self):
        grid=self.workbench.selected_grid()
        validate_crs(grid.crs,grid.coordinate_unit)
        if grid.coordinate_unit!='m': raise ValueError('Задайте CRS карты и координаты в метрах.')
        xx,yy=np.meshgrid(grid.x,grid.y); mask=np.isfinite(grid.values)
        self.assign_table(pd.DataFrame(dict(East=xx[mask],North=yy[mask],anomaly=grid.values[mask])))
        self.crs.setCurrentText(grid.crs)

    def load_field(self):
        from .grids import load_grid
        path,_=QFileDialog.getOpenFileName(self,'Наблюдённая аномалия ZMAP','','ZMAP (*.dat *.zmap *.npz)')
        if not path: return
        g=load_grid(path); x,y=np.meshgrid(g.x,g.y); valid=np.isfinite(g.values)
        self.assign_table(pd.DataFrame(dict(East=x[valid],North=y[valid],anomaly=g.values[valid])))
        self.preview_field()

    def preview_field(self):
        if self.table is None: raise ValueError('Загрузите поле.')
        east,north,data=[pd.to_numeric(self.table[c.currentText()],errors='raise').to_numpy(float) for c in (self.east,self.north,self.value)]
        self.figure.clear(); ax=self.figure.add_subplot(111); stride=max(1,len(data)//20000)
        p=ax.scatter(east[::stride],north[::stride],c=data[::stride],s=5,cmap='coolwarm',rasterized=True)
        self.figure.colorbar(p,ax=ax,label='mGal' if self.mode.currentData()=='gravity' else 'nT')
        ax.set(xlabel='East (m)',ylabel='North (m)',title=f'Наблюдённое поле · {len(data):,} точек; предпросмотр каждой {stride}-й')
        ax.set_aspect('equal'); style_figure(self.figure); self.canvas.draw_idle(); self.tabs.setCurrentIndex(1)

    def preview_initial(self,model):
        self.figure.clear(); mesh=model.mesh
        values=np.linalg.norm(model.values,axis=1) if model.mode=='vector' else model.values.copy()
        values=values.copy(); values[~mesh.active]=np.nan; cube=values.reshape(mesh.shape)
        ix,iy,iz=[n//2 for n in mesh.shape]
        unit='g/cm³' if model.mode=='gravity' else ('SI' if model.mode=='induced' else 'A/m')
        origin=mesh.east[0],mesh.north[0]
        ax=self.figure.add_subplot(131); p=ax.pcolormesh(mesh.east-origin[0],mesh.north-origin[1],cube[:,:,iz].T,shading='flat'); self.figure.colorbar(p,ax=ax,label=unit); ax.set(xlabel='East − E₀ (m)',ylabel='North − N₀ (m)',title='Начальная модель · горизонталь')
        ax=self.figure.add_subplot(132); p=ax.pcolormesh(mesh.east-origin[0],mesh.depth,cube[:,iy,:].T,shading='flat'); self.figure.colorbar(p,ax=ax,label=unit); ax.invert_yaxis(); ax.set(xlabel='East − E₀ (m)',ylabel='Down (m)',title='Начальная модель · срез')
        ax=self.figure.add_subplot(133)
        if self.table is not None:
            cols=[pd.to_numeric(self.table[c.currentText()],errors='raise').to_numpy(float) for c in (self.east,self.north,self.value)]
            stride=max(1,len(cols[0])//20000); p=ax.scatter(cols[0][::stride]-origin[0],cols[1][::stride]-origin[1],c=cols[2][::stride],s=3,cmap='coolwarm'); self.figure.colorbar(p,ax=ax,label='mGal' if model.mode=='gravity' else 'nT'); ax.set(xlabel='East − E₀ (m)',ylabel='North − N₀ (m)',title='Наблюдённое поле')
        else: ax.text(.1,.5,'Поле ещё не загружено',transform=ax.transAxes)
        self.figure.suptitle(f'{mesh.active.sum():,} активных призм · E₀={origin[0]:g}, N₀={origin[1]:g} м · точная обрезка в 3D',color='#dbe7f6',fontsize=10)
        self.figure.tight_layout(); style_figure(self.figure); self.canvas.draw_idle(); self.tabs.setCurrentIndex(1)

    def new_mesh(self):
        if self.table is None: raise ValueError('Для новой сетки загрузите поле; готовый воксель можно просмотреть без поля.')
        east,north=[pd.to_numeric(self.table[c.currentText()],errors='raise').to_numpy(float) for c in (self.east,self.north)]
        if self.bottom.value()<=self.top.value(): raise ValueError('Низ должен быть глубже верха.')
        axes=[]
        for values,n in ((east,self.nx.value()),(north,self.ny.value())):
            span=np.ptp(values)
            if not np.isfinite(values).all() or span<=0: raise ValueError('Нужны конечные различные координаты по East и North.')
            margin=span*self.padding.value()+span/(2*max(n-1,1))
            axes.append(np.linspace(values.min()-margin,values.max()+margin,n+1))
        return Mesh(*axes,np.linspace(self.top.value(),self.bottom.value(),self.nz.value()+1))

    def demo(self):
        mode=self.mode.currentData()
        x,y=np.meshgrid(np.linspace(-1500,1500,15),np.linspace(-1500,1500,15))
        obs=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
        mesh=Mesh(np.linspace(-1800,1800,7),np.linspace(-1800,1800,7),np.linspace(100,1900,7))
        op=FieldOperator(mesh,obs,mode,field_strength=self.field.value(),inclination=self.inc.value(),declination=self.dec.value())
        truth=np.zeros(op.parameter_shape)
        mask=np.all(np.abs(mesh.centers()[:,:2])<650,axis=1)&(mesh.centers()[:,2]>400)&(mesh.centers()[:,2]<1000)
        truth[mask]=[.6,-.25,.4] if mode=='vector' else (.03 if mode=='induced' else .4)
        d=op.predict(truth); sigma=max(float(np.std(d)*.03),1e-5)
        d=d+np.random.default_rng(42).normal(0,sigma,len(d))
        self.assign_table(pd.DataFrame(dict(East=obs[:,0]+10500000,North=obs[:,1]+6500000,height=0.,anomaly=d,sigma=sigma)))
        self.nx.setValue(6); self.ny.setValue(6); self.nz.setValue(6)
        self.top.setValue(100); self.bottom.setValue(1900); self.padding.setValue(.1)
        self.crs.setCurrentText('EPSG:20910')
        self.initial_panel.source.setCurrentIndex(0)
        self.stride.setValue(1)
        self.data_info.setText('DEMO: 225 точек, шум σ=3% стандартного отклонения; смещённая суммарная J в векторном режиме.')

    def run(self):
        if self.worker or self.workbench.worker: raise ValueError('Дождитесь текущего расчёта.')
        if self.table is None: raise ValueError('Загрузите наблюдения.')
        crs=validate_crs(self.crs.currentText().split('|')[0].strip(),'m')
        if not crs: raise ValueError('Укажите метрическую CRS наблюдений.')
        selected=self.table.iloc[::self.stride.value()]
        def column(combo): return pd.to_numeric(selected[combo.currentText()],errors='raise').to_numpy(dtype=float)
        if self.east.currentText()==self.north.currentText(): raise ValueError('Выберите разные столбцы East и North.')
        east,north=column(self.east),column(self.north)
        height=column(self.height) if self.height.currentIndex()>0 else np.full(len(east),self.height_constant.value())
        data=column(self.value)
        sigma=column(self.error) if self.error.currentIndex()>0 else self.sigma.value()
        observations=np.column_stack((east,north,-height))
        if not np.isfinite(observations).all() or not np.isfinite(data).all(): raise ValueError('Таблица содержит пропуски/Inf; подготовьте наблюдения.')
        initial_model=self.initial_panel.current(); mesh=initial_model.mesh
        observation_stride=self.stride.value()
        mode=self.mode.currentData()
        operator_args=dict(device=self.device.currentData(),gpu_id=self.gpu.value(),field_strength=self.field.value(),inclination=self.inc.value(),declination=self.dec.value(),cache_mb=self.cache.value())
        if np.any(initial_model.values[mesh.active]<self.lower.value()) or np.any(initial_model.values[mesh.active]>self.upper.value()):
            raise ValueError('Начальные свойства выходят за минимум/максимум. Исправьте границы или явно ограничьте свойства через операции.')
        settings=dict(sigma=sigma,iterations=self.iterations.value(),lower=self.lower.value(),upper=self.upper.value(),initial=initial_model.values,damping=self.damping.value(),smoothness=self.smooth.value(),depth_index=self.depth_index.value(),step_scale=self.step.value(),target_rms=self.target.value(),fit_offset=self.offset.isChecked())
        def compute(progress,cancel):
            op=FieldOperator(mesh,observations,mode,**operator_args,cancel_event=cancel)
            result=invert(op,data,**settings,progress=progress,cancel_event=cancel)
            result.settings.update(crs=crs,gpu_id=operator_args['gpu_id'])
            result.settings.update(initial_model_history=initial_model.history,observation_stride=observation_stride)
            return result
        self.result=None; self.history=[]; self.curve.setData([],[]); self.log.clear()
        self.progress.setRange(0,settings['iterations']); self.progress.setValue(0)
        self.status.setText(f'{len(data):,} наблюдений × {mesh.active.sum():,} активных ячеек · {operator_args["device"].upper()} · кэш ≤{self.cache.value()} MiB; блоки 64×512')
        self.worker=InversionWorker(compute,self)
        self.parameter_panel.setEnabled(False)
        self.initial_panel.setEnabled(False)
        self.worker.progress.connect(self.update_progress); self.worker.result.connect(self.finished_result)
        self.worker.failed.connect(self.failure); self.worker.finished.connect(self.job_finished)
        for w in (self.run_button,self.load_button,self.grid_button,self.demo_button,self.model_load,self.mode): w.setEnabled(False)
        self.stop.setEnabled(True); self.tabs.setCurrentIndex(2); self.worker.start()

    def update_progress(self,row):
        self.history.append(row); self.curve.setData([r['iteration'] for r in self.history],[r['rmse'] for r in self.history])
        self.progress.setValue(row['iteration'])
        self.log.appendPlainText(f"{row['iteration']}: RMSE={row['rmse']:.6g}; RMS/σ={row['normalized_rms']:.4g}; J={row['objective']:.6g}")
    def failure(self,message): self.status.setText(message); self.log.appendPlainText(message)
    def job_finished(self):
        self.worker.function=None; self.worker.deleteLater(); self.worker=None
        for w in (self.run_button,self.load_button,self.grid_button,self.demo_button,self.model_load,self.mode): w.setEnabled(True)
        self.stop.setEnabled(False)
        self.initial_panel.setEnabled(True)
        self.parameter_panel.setEnabled(True)

    def model_task(self,function,callback):
        if self.worker or self.workbench.worker: raise ValueError('Дождитесь текущего расчёта.')
        self.status.setText('Чтение начальной модели…')
        self.parameter_panel.setEnabled(False); self.run_button.setEnabled(False)
        self.worker=InversionWorker(lambda progress,cancel:function(),self)
        self.worker.result.connect(lambda model:self.workbench.guard(lambda:callback(model)))
        self.worker.failed.connect(self.failure); self.worker.finished.connect(self.job_finished)
        self.worker.start()
    def cancel(self):
        if self.worker: self.worker.cancel.set(); self.stop.setEnabled(False)
    def finished_result(self,result):
        self.result=result
        self.status.setText(f"{result.status} · RMSE={result.history[-1]['rmse']:.6g} · RMS/σ={result.history[-1]['normalized_rms']:.4g}; {result.settings['device'].upper()}")
        self.workbench.guard(self.draw_results)
        self.progress.setValue(result.history[-1]['iteration'])

    def draw_results(self):
        r=self.require_result(); self.figure.clear()
        shape=r.mesh.shape
        values=np.linalg.norm(r.model,axis=1) if r.model.ndim==2 else r.model.copy()
        values=values.copy(); values[~r.mesh.active]=np.nan
        property_unit='g/cm³' if r.settings['mode']=='gravity' else ('SI' if r.settings['mode']=='induced' else 'A/m')
        field_unit='mGal' if r.settings['mode']=='gravity' else 'nT'
        cube=values.reshape(shape)
        for n,(image,title,x,y) in enumerate([
            (cube[:,:,shape[2]//2].T,'Горизонтальный срез',r.mesh.east,r.mesh.north),
            (cube[:,shape[1]//2,:].T,'Разрез East–Depth',r.mesh.east,r.mesh.depth)]):
            ax=self.figure.add_subplot(2,3,n+1); im=ax.pcolormesh(x,y,image,shading='flat',cmap='viridis' if np.nanmin(values)>=0 else 'coolwarm'); ax.set_title(title); self.figure.colorbar(im,ax=ax,label=property_unit)
            if n==1: ax.invert_yaxis()
        ax=self.figure.add_subplot(2,3,3); ax.hist(values[np.isfinite(values)],bins=30,color='#2dd4bf'); ax.set_title('Распределение свойства / |J|')
        obs=r.observations
        for n,(v,title) in enumerate([(r.observed,'Наблюдения'),(r.predicted,'Расчёт'),(r.predicted-r.observed,'Расчёт − наблюдения')]):
            ax=self.figure.add_subplot(2,3,n+4)
            lim=np.max(np.abs(v)) if n==2 else max(np.max(np.abs(r.observed)),np.max(np.abs(r.predicted)))
            im=ax.scatter(obs[:,0]-obs[:,0].min(),obs[:,1]-obs[:,1].min(),c=v,s=12,cmap='coolwarm',vmin=-lim,vmax=lim)
            ax.set_title(title); self.figure.colorbar(im,ax=ax,label=field_unit)
        style_figure(self.figure); self.figure.tight_layout(); self.canvas.draw_idle(); self.tabs.setCurrentIndex(1)
    def require_result(self):
        if self.result is None: raise ValueError('Сначала завершите инверсию.')
        return self.result
    def open_3d(self): self.scene.render(self.require_result()); self.tabs.setCurrentIndex(0)
    def load_result(self):
        path,_=QFileDialog.getOpenFileName(self,'Сохранённая 3D-инверсия','','NPZ (*.npz)')
        if path:
            result=InversionResult.load(path)
            self.history=list(result.history)
            self.curve.setData([r['iteration'] for r in self.history],[r['rmse'] for r in self.history])
            self.finished_result(result)
    def save(self):
        result=self.require_result(); path,_=QFileDialog.getSaveFileName(self,'3D-модель и отчёт','inversion.npz','NPZ (*.npz)')
        if path: result.save(path if path.lower().endswith('.npz') else path+'.npz')
    def save_fields(self):
        r=self.require_result(); path,_=QFileDialog.getSaveFileName(self,'Поля и невязки','inversion-fields.csv','CSV (*.csv)')
        if path: pd.DataFrame(dict(East=r.observations[:,0],North=r.observations[:,1],height=-r.observations[:,2],observed=r.observed,predicted=r.predicted,residual=r.predicted-r.observed)).to_csv(path,index=False,encoding='utf-8-sig')
    def save_vtk(self):
        r=self.require_result(); grid=vtk_grid(r); ext='.vtu' if grid.__class__.__name__=='UnstructuredGrid' else '.vtr'
        path,_=QFileDialog.getSaveFileName(self,'VTK-модель','inversion'+ext,'VTK (*'+ext+')')
        if path: grid.save(path if path.lower().endswith(ext) else path+ext)
