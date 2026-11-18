"""Third workbench tab: density, susceptibility and magnetization modelling."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from time import perf_counter
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QGroupBox,
    QComboBox,QDoubleSpinBox,QSpinBox,QPushButton,QLabel,QScrollArea,
    QTabWidget,QFileDialog,QProgressBar,QPlainTextEdit,QCheckBox,QAbstractSpinBox,QLineEdit,QAbstractButton,QDialog)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,NavigationToolbar2QT
from threading import Event
from .inversion import Mesh,FieldOperator,invert,InversionResult
from .inversion_view import vtk_grid
from .model_workspace import ModelWorkspace
from .layer_fft import LayerFFT, HEIGHT_ERROR, model_axis
from .models import read_table
from .coordinates import gsk2011_zones,validate_crs
from .compute import ComputationCancelled
from .initial_model_ui import InitialModelPanel
from .voxel import extend_density_model
from .theme import style_figure
from .inversion_workflow_ui import arrange_workflow
from .inversion_export import field_table, export_zmap, export_ubc
from .inversion_progress import duration
from .ui_layout import Disclosure,ParameterSplitter,SummaryLabel,details,form_details,quiet_button,prepare_forms


class InversionWorker(QThread):
    progress=pyqtSignal(object); result=pyqtSignal(object); failed=pyqtSignal(str)
    def __init__(self,function,parent):
        super().__init__(parent); self.function=function; self.cancel=Event()
    def run(self):
        try: self.result.emit(self.function(self.progress.emit,self.cancel))
        except ComputationCancelled: self.failed.emit('Расчёт остановлен до завершения оценки начальной модели. Готовой модели и поля пока нет.')
        except Exception as exc: self.failed.emit(str(exc))


class InversionTab(QWidget):
    def __init__(self,workbench):
        super().__init__(workbench)
        self.workbench=workbench; self.worker=None; self.table=None; self.result=None
        root=QHBoxLayout(self)
        left=QWidget(); panel=QVBoxLayout(left)
        self.parameter_panel=left
        left.setObjectName('inversionParameters')
        self._locked_inputs=[]
        self.mode=QComboBox()
        for label,key in [('Плотностное моделирование · Δρ, g/cm³','gravity'),('Моделирование магнитной восприимчивости · χ, SI','induced'),('Моделирование векторной намагниченности · J, A/m','vector')]: self.mode.addItem(label,key)
        panel.addWidget(self.mode)
        self.method=QComboBox()
        self.method.addItem('Прямой расчёт по призмам','prisms')
        self.method.addItem('Послойная 2D FFT','layer_fft')
        self.method.setAccessibleName('Метод расчёта плотностной модели')
        self.hint=QLabel(); self.hint.setWordWrap(True); panel.addWidget(self.hint)
        self.load_button=self.button('Наблюдения XLSX / CSV…',self.load_table,panel)
        self.button('Наблюдённое поле ZMAP…',self.load_field,panel)
        self.grid_button=self.button('Взять выбранную карту',self.load_grid,panel)
        self.button('Показать наблюдённое поле',self.preview_field,panel)
        self.demo_button=self.button('Синтетический пример для режима',self.demo,panel)
        self.model_load=self.button('Открыть сохранённый результат NPZ…',self.load_result,panel)
        self.data_info=QLabel('Загрузите аномалии силы тяжести в редукции Буге (mGal) или аномальное магнитное поле (nT).'); self.data_info.setWordWrap(True); panel.addWidget(self.data_info)
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
        self.nx,self.ny,self.nz=[self.integer(2,10000,v) for v in (12,12,8)]
        self.fft_grid=QComboBox()
        self.fft_grid.addItems(['По шагу карты','Задать Nx и Ny'])
        self.top=self.number(-1e6,1e6,100)
        self.bottom=self.number(-1e6,1e6,3000)
        self.padding=self.number(0,10,.2)
        self.cache=self.integer(0,2147483647,512)
        self.cache.setEnabled(False)
        self.memory_policy=QComboBox()
        for label,key in [('Авто · сбалансированный','balanced'),('Авто · производительность','performance'),('Вручную','manual')]:
            self.memory_policy.addItem(label,key)
        self.cache.setToolTip('Предел кэша в MiB: GPU — видеопамять, CPU — оперативная память. 0 отключает кэш. Бюджет проверяется при запуске.')
        for name,w in [('Горизонтальная сетка',self.fft_grid),('Nx',self.nx),('Ny',self.ny),('Nz',self.nz),('Верх: глубина Down, м',self.top),('Низ: глубина Down, м',self.bottom),('Запас с каждой стороны, доля',self.padding)]: f.addRow(name,w)
        self.padding.setToolTip('0,20 добавляет минимум 20% от ширины карты с каждой стороны на новой FFT-сетке.')
        self.extend_fft=QCheckBox('Расширить загруженную модель')
        self.extend_fft.setToolTip('Дополняет модель активными ячейками до указанного запаса от края карты. '
                                  'Плотность добавленных ячеек берётся из поля «Начальное значение».')
        f.addRow(self.extend_fft)
        self.fft_transfer_hint=QLabel('Чтобы изменить Nx, Ny или Nz загруженной модели, '
                                      'выберите перенос на новую сетку в редакторе вокселя.')
        self.fft_transfer_hint.setWordWrap(True)
        f.addRow(self.fft_transfer_hint)
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
        self.early_stop=QCheckBox('Ранняя остановка по слабому улучшению RMS/σ')
        self.early_stop.setToolTip(self.early_stop.text()); self.early_stop.setText('Ранняя остановка')
        self.early_tolerance=self.number(0,99,.1,4)
        self.early_patience=self.integer(1,1000,5)
        sf.addRow(self.early_stop)
        sf.addRow('Минимальное улучшение, %',self.early_tolerance)
        sf.addRow('Последовательных итераций',self.early_patience)
        self.early_tolerance.setEnabled(False); self.early_patience.setEnabled(False)
        self.early_stop.toggled.connect(self.early_tolerance.setEnabled)
        self.early_stop.toggled.connect(self.early_patience.setEnabled)
        self.early_tolerance.setToolTip('Если относительное уменьшение RMS/σ меньше порога несколько итераций подряд, возвращается лучшая завершённая модель.')
        compute=QGroupBox('07 · Вычисления'); cf=QFormLayout(compute)
        cf.addRow('Устройство',self.device); cf.addRow('GPU ID',self.gpu)
        cf.addRow('Режим памяти',self.memory_policy); cf.addRow('Лимит кэша, МиБ',self.cache)
        self._cache_label=cf.labelForField(self.cache)
        self.memory_policy.currentIndexChanged.connect(self.update_memory_controls)
        self.memory_info=QLabel('Свободная RAM/VRAM, бюджет кэша и размер пакетов определяются при запуске.'); self.memory_info.setWordWrap(True)
        cf.addRow(self.memory_info); panel.addWidget(compute)
        self.dec.setToolTip('Склонение относительно севера проекционной сетки. Для IGRF учтите сближение меридианов.')
        forms.extend([f,mf,sf,cf])
        for f in forms:
            f.setRowWrapPolicy(QFormLayout.WrapLongRows)
            f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        details(panel,self.hint,'О выбранном режиме')
        form_details(sf,[self.depth_index,self.damping,self.smooth,self.step,self.offset],'Регуляризация и подбор шага')
        form_details(sf,[self.early_tolerance,self.early_patience],'Параметры ранней остановки')
        self._solver_group=solver
        self._fft_hidden=[self.stride,form.labelForField(self.stride)]
        for index in range(sf.count()):
            widget=sf.itemAt(index).widget()
            if widget is not None and widget not in (self.iterations,sf.labelForField(self.iterations)):
                self._fft_hidden.append(widget)
        mesh_form=mesh_group.layout()
        self._fft_new_grid=mesh_group
        self._initial_label=settings.layout().labelForField(self.initial)
        self._grid_rows={widget:mesh_form.labelForField(widget)
                         for widget in (self.fft_grid,self.nx,self.ny,self.nz,self.top,self.bottom,self.padding)}
        self._height_label=form.labelForField(self.height_constant)
        self._sigma_label=form.labelForField(self.sigma)
        self.height.currentIndexChanged.connect(self.update_fft_inputs)
        self.error.currentIndexChanged.connect(self.update_fft_inputs)
        self.initial_panel.source.currentIndexChanged.connect(self.update_fft_inputs)
        self.fft_grid.currentIndexChanged.connect(self.update_fft_inputs)
        self.extend_fft.toggled.connect(self.update_fft_inputs)
        self.height.activated.connect(lambda _:self.workbench.guard(self.check_fft_height))
        self.workflow=arrange_workflow(panel,self.initial_panel,mesh_group,settings,self.magnetic_group,solver,compute)
        panel.insertWidget(1,self.method)
        prepare_forms(left)
        for button in left.findChildren(QPushButton): quiet_button(button)
        for label in left.findChildren(QLabel): label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        for combo in left.findChildren(QComboBox):
            combo.setMinimumContentsLength(10); combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(left)
        controls=QWidget(); cl=QVBoxLayout(controls); cl.addWidget(scroll)
        self.run_button=self.button('Запустить 3D-моделирование',self.run,cl); self.run_button.setObjectName('primary')
        self.stop=self.button('Остановить',self.cancel,cl); self.stop.setEnabled(False)
        self.progress=QProgressBar(); cl.addWidget(self.progress)
        self.progress.setFormat('%v / %m итераций (%p%)')
        right=QWidget(); rl=QVBoxLayout(right)
        self.status=SummaryLabel('Загрузите поле, задайте модель и запустите расчёт.'); rl.addWidget(self.status)
        self.phase_label=QLabel('Этап расчёта появится после запуска'); self.phase_label.setWordWrap(True); rl.addWidget(self.phase_label)
        self.phase_progress=QProgressBar(); self.phase_progress.setRange(0,100); rl.addWidget(self.phase_progress)
        self.timing_label=QLabel('Время расчёта: —'); self.timing_label.setWordWrap(True); rl.addWidget(self.timing_label)
        timing_details=QWidget(); timing_rows=QVBoxLayout(timing_details)
        rl.removeWidget(self.phase_label); rl.removeWidget(self.timing_label)
        timing_rows.addWidget(self.phase_label); timing_rows.addWidget(self.timing_label)
        self.runtime_details=Disclosure('Время и этапы расчёта',timing_details)
        rl.addWidget(self.runtime_details)
        self._run_started=None; self._iteration_started=None; self._iteration_number=0; self._iteration_limit=0
        self._notify_inversion=False
        self.clock_timer=QTimer(self); self.clock_timer.setInterval(250); self.clock_timer.timeout.connect(self.update_clock)
        self.tabs=QTabWidget(); rl.addWidget(self.tabs,1)
        self.scene=ModelWorkspace(self,import_callback=lambda:self.workbench.guard(self.import_display_model)); self.tabs.addTab(self.scene,'3D-модели')
        self.figure=Figure(); self.canvas=FigureCanvasQTAgg(self.figure)
        plot=QWidget(); pl=QVBoxLayout(plot); pl.addWidget(NavigationToolbar2QT(self.canvas,self)); pl.addWidget(self.canvas)
        self.tabs.addTab(plot,'Срезы и поля')
        self.convergence=pg.PlotWidget(background='#101c2f'); self.convergence.setLabel('left','RMSE'); self.convergence.setLabel('bottom','Итерация'); self.curve=self.convergence.plot(pen=pg.mkPen('#2dd4bf',width=2)); self.tabs.addTab(self.convergence,'Сходимость')
        timing_plot=pg.PlotWidget(background='#101c2f'); timing_plot.setLabel('left','Время','s'); timing_plot.setLabel('bottom','Итерация')
        self.timing_curve=timing_plot.plot(pen=pg.mkPen('#38bdf8',width=2),symbol='o',symbolSize=5)
        self.tabs.addTab(timing_plot,'Время итераций')
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100); self.log.setMaximumBlockCount(1000); rl.addWidget(self.log)
        details(rl,self.log,'Журнал расчёта')
        actions=QHBoxLayout(); rl.addLayout(actions)
        self.button('Открыть / обновить 3D',self.open_3d,actions)
        self.button('Карты срезов…',self.open_slice_maps,actions)
        from PyQt5.QtWidgets import QMenu
        export_button=QPushButton('Сохранить результат…'); export_menu=QMenu(export_button)
        for title,callback in [('Модель и отчёт · NPZ',self.save),('Модель · UBC',self.save_ubc),
                               ('Поля · CSV / ZMAP',self.save_fields),('Модель · VTK',self.save_vtk)]:
            export_menu.addAction(title,lambda checked=False,fn=callback:self.workbench.guard(fn))
        export_button.setMenu(export_menu); actions.addWidget(export_button)
        root.addWidget(ParameterSplitter(controls,right))
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.method.currentIndexChanged.connect(self.update_fft_inputs)
        self.mode_changed()
        self.history=[]

    def button(self,text,fn,layout):
        w=QPushButton(text); w.clicked.connect(lambda: self.workbench.guard(fn)); layout.addWidget(w); return w

    def lock_inputs(self):
        """Freeze settings, retaining enabled page navigation and scroll areas."""
        self._locked_inputs=[]
        for widget in self.parameter_panel.findChildren(QWidget):
            if isinstance(widget,QAbstractSpinBox):
                self._locked_inputs.append((widget,'readonly',widget.isReadOnly()))
            elif isinstance(widget,QLineEdit) and not isinstance(widget.parent(),QAbstractSpinBox):
                self._locked_inputs.append((widget,'readonly',widget.isReadOnly()))
            elif isinstance(widget,(QComboBox,QAbstractButton)) and not widget.property('workflowNavigation'):
                self._locked_inputs.append((widget,'enabled',widget.isEnabled()))
        for widget,kind,_ in self._locked_inputs:
            if kind=='readonly': widget.setReadOnly(True)
            else: widget.setEnabled(False)

    def unlock_inputs(self):
        for widget,kind,value in self._locked_inputs:
            if kind=='readonly': widget.setReadOnly(value)
            else: widget.setEnabled(value)
        self._locked_inputs=[]
    @staticmethod
    def number(lo,hi,val,decimals=2):
        w=QDoubleSpinBox(); w.setDecimals(decimals); w.setRange(lo,hi); w.setValue(val); return w
    @staticmethod
    def integer(lo,hi,val):
        w=QSpinBox(); w.setRange(lo,hi); w.setValue(val); return w

    def mode_changed(self):
        mode=self.mode.currentData()
        self.hint.setText({'gravity':'Плотностное моделирование: контраст плотности Δρ в g/cm³ по аномалиям силы тяжести. Поправки, редукция Буге и региональный фон готовятся до импорта.',
            'induced':'Моделирование магнитной восприимчивости χ в СИ (без множителя 10⁻⁵); индуцированная J=χH направлена вдоль главного поля.',
            'vector':'Моделирование суммарной векторной намагниченности J=(JE, JN, JD), A/m с возможным остаточным вкладом. Разделить индуцированную и остаточную части по одной карте аномального магнитного поля нельзя.'}[mode])
        self.lower.setValue(0 if mode=='induced' else -1)
        self.upper.setValue(.1 if mode=='induced' else 1)
        for w in (self.field,self.inc,self.dec): w.setEnabled(mode!='gravity')
        self.magnetic_group.setVisible(mode!='gravity')
        self.update_fft_inputs()

    def is_fft(self):
        return self.mode.currentData()=='gravity' and self.method.currentData()=='layer_fft'

    def update_memory_controls(self):
        manual=self.memory_policy.currentData()=='manual'
        self.cache.setEnabled(manual)
        self.cache.setVisible(manual)
        self._cache_label.setVisible(manual)

    def update_fft_inputs(self,*args):
        fft=self.is_fft()
        self.update_memory_controls()
        self.method.setVisible(self.mode.currentData()=='gravity')
        self._solver_group.setTitle('Предел итераций' if fft else '06 · Решатель и регуляризация')
        for widget in self._fft_hidden: widget.setVisible(not fft)
        new_grid=not fft or self.initial_panel.source.currentIndex()==0
        self._fft_new_grid.setTitle('03 · Сетка и запас по краям' if fft else '03 · Новая сетка / сетка переноса')
        manual=not fft or self.fft_grid.currentIndex()==1
        for widget, label in self._grid_rows.items():
            visible=True
            if widget is self.fft_grid: visible=fft
            elif widget in (self.nx,self.ny): visible=manual
            widget.setVisible(visible)
            label.setVisible(visible)
        self.extend_fft.setVisible(fft and not new_grid)
        self.fft_transfer_hint.setVisible(fft and not new_grid)
        show_initial=new_grid or self.extend_fft.isChecked()
        self.initial.setVisible(show_initial)
        self._initial_label.setVisible(show_initial)
        self._initial_label.setText('Начальная плотность запаса' if fft and not new_grid else 'Начальное значение')
        self._height_label.setText('Единая высота Up, м' if fft else 'Высота константа')
        for combo,field,label in ((self.height,self.height_constant,self._height_label),
                                  (self.error,self.sigma,self._sigma_label)):
            visible=not fft or combo.currentIndex()<=0
            field.setVisible(visible)
            label.setVisible(visible)
        if fft:
            self.hint.setText('Послойная FFT рассчитана на регулярную карту с единой высотой Up. '
                              'Для новой модели можно выбрать шаг карты или задать Nx и Ny.')
        elif self.mode.currentData()=='gravity':
            self.hint.setText('Контраст плотности Δρ в g/cm³ по аномалиям силы тяжести. '
                              'Поправки, редукция Буге и региональный фон готовятся до импорта.')

    def check_fft_height(self):
        if self.is_fft() and self.table is not None and self.height.currentIndex()>0:
            values=pd.to_numeric(self.table[self.height.currentText()],errors='raise').to_numpy(float)
            if not np.isfinite(values).all() or np.ptp(values)>1e-7:
                raise ValueError(HEIGHT_ERROR)

    def assign_table(self,table):
        if len(table.columns)<3: raise ValueError('Нужны минимум координаты и аномалия.')
        self.table=table; self.result=None
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
        self.data_info.setText(f'{len(table):,} точек. Проверьте столбцы, единицы, CRS и высоту. Результат расчёта очищен; открытые 3D-модели сохранены.')
        self.figure.clear(); self.canvas.draw_idle()

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
        nx, ny, nz = mesh.shape
        title = f'{nx} × {ny} × {nz} · {mesh.active.sum():,} активных призм · E₀={origin[0]:g}, N₀={origin[1]:g} м'
        if self.is_fft():
            title += f'\nШаг модели X {(mesh.east[-1]-mesh.east[0])/nx:,.2f} м · Y {(mesh.north[-1]-mesh.north[0])/ny:,.2f} м'
        self.figure.suptitle(title,color='#dbe7f6',fontsize=10)
        self.figure.tight_layout(); style_figure(self.figure); self.canvas.draw_idle(); self.tabs.setCurrentIndex(1)

    def new_mesh(self):
        if self.table is None: raise ValueError('Для новой сетки загрузите поле; готовый воксель можно просмотреть без поля.')
        east,north=[pd.to_numeric(self.table[c.currentText()],errors='raise').to_numpy(float) for c in (self.east,self.north)]
        if self.bottom.value()<=self.top.value(): raise ValueError('Низ должен быть глубже верха.')
        if self.is_fft():
            import psutil
            counts=(self.nx.value(),self.ny.value()) if self.fft_grid.currentIndex()==1 else (None,None)
            specs=[model_axis(values,n,self.padding.value()) for values,n in zip((east,north),counts)]
            size=specs[0][2]*specs[1][2]*self.nz.value()
            if size*40>psutil.virtual_memory().available*.8:
                raise ValueError('FFT-сетка слишком велика для доступной RAM. Проверьте шаг координат.')
            axes=[origin+np.arange(count+1)*step for origin,step,count in specs]
            return Mesh(*axes,np.linspace(self.top.value(),self.bottom.value(),self.nz.value()+1))
        axes=[]
        for values,n in ((east,self.nx.value()),(north,self.ny.value())):
            span=np.ptp(values)
            if not np.isfinite(values).all() or span<=0: raise ValueError('Нужны конечные различные координаты по East и North.')
            margin=span*self.padding.value()+span/(2*max(n-1,1))
            axes.append(np.linspace(values.min()-margin,values.max()+margin,n+1))
        return Mesh(*axes,np.linspace(self.top.value(),self.bottom.value(),self.nz.value()+1))

    def prepared_model(self):
        model = self.initial_panel.current()
        if self.is_fft() and self.initial_panel.source.currentIndex()==1 and self.extend_fft.isChecked():
            if self.table is None:
                raise ValueError('Загрузите наблюдения, чтобы определить границы расширения.')
            east, north = [pd.to_numeric(self.table[c.currentText()], errors='raise').to_numpy(float)
                           for c in (self.east, self.north)]
            model = extend_density_model(model,east,north,self.padding.value(),self.initial.value())
        return model

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
        fft=self.is_fft()
        self.check_fft_height()
        observation_stride=1 if fft else self.stride.value()
        selected=self.table.iloc[::observation_stride]
        def column(combo): return pd.to_numeric(selected[combo.currentText()],errors='raise').to_numpy(dtype=float)
        if self.east.currentText()==self.north.currentText(): raise ValueError('Выберите разные столбцы East и North.')
        east,north=column(self.east),column(self.north)
        height=column(self.height) if self.height.currentIndex()>0 else np.full(len(east),self.height_constant.value())
        data=column(self.value)
        sigma=column(self.error) if self.error.currentIndex()>0 else self.sigma.value()
        observations=np.column_stack((east,north,-height))
        if not np.isfinite(observations).all() or not np.isfinite(data).all(): raise ValueError('Таблица содержит пропуски/Inf; подготовьте наблюдения.')
        initial_model=self.prepared_model(); mesh=initial_model.mesh
        mode=self.mode.currentData()
        if np.any(initial_model.values[mesh.active]<self.lower.value()) or np.any(initial_model.values[mesh.active]>self.upper.value()):
            raise ValueError('Начальные свойства выходят за минимум/максимум. Исправьте границы или явно ограничьте свойства через операции.')
        settings=dict(sigma=sigma,iterations=self.iterations.value(),lower=self.lower.value(),upper=self.upper.value(),initial=initial_model.values)
        operator_args=dict(device=self.device.currentData(),gpu_id=self.gpu.value(),
                           cache_mb=self.cache.value(),memory_policy=self.memory_policy.currentData())
        if not fft:
            operator_args.update(field_strength=self.field.value(),inclination=self.inc.value(),declination=self.dec.value())
            settings.update(damping=self.damping.value(),smoothness=self.smooth.value(),
                            depth_index=self.depth_index.value(),step_scale=self.step.value(),target_rms=self.target.value(),
                            fit_offset=self.offset.isChecked(),early_stop=self.early_stop.isChecked(),
                            early_tolerance=self.early_tolerance.value()/100,early_patience=self.early_patience.value())
        def compute(progress,cancel):
            started=perf_counter()
            op=(LayerFFT(mesh,observations,**operator_args,cancel_event=cancel) if fft else
                FieldOperator(mesh,observations,mode,**operator_args,cancel_event=cancel))
            progress(dict(event='resources',**op.resources))
            result=invert(op,data,**settings,progress=progress,cancel_event=cancel,work_progress=progress)
            result.settings['run_elapsed_seconds']=perf_counter()-started
            result.settings.update(crs=crs,gpu_id=operator_args['gpu_id'])
            result.settings.update(initial_model_history=initial_model.history,observation_stride=observation_stride)
            return result
        self.result=None; self.history=[]; self.curve.setData([],[]); self.log.clear()
        self.timing_curve.setData([],[])
        self._run_started=perf_counter(); self._iteration_started=self._run_started
        self._iteration_number=0; self._iteration_limit=settings['iterations']
        self.phase_label.setText('Подготовка устройства и оператора…'); self.phase_progress.setRange(0,0)
        self.clock_timer.start(); self.update_clock()
        self.progress.setRange(0,settings['iterations']); self.progress.setValue(0)
        method = 'Послойная 2D FFT · ' if fft else ''
        self.status.setText(f'{method}{len(data):,} наблюдений × {mesh.active.sum():,} активных ячеек · {operator_args["device"].upper()} · определение бюджета памяти…')
        self.worker=InversionWorker(compute,self)
        self._notify_inversion=True
        self.lock_inputs()
        self.worker.progress.connect(self.update_progress); self.worker.result.connect(self.finished_result)
        self.worker.failed.connect(self.failure); self.worker.finished.connect(self.job_finished)
        self.run_button.setEnabled(False)
        self.stop.setEnabled(True); self.tabs.setCurrentIndex(2); self.worker.start()

    def update_progress(self,row):
        if row.get('event')=='work':
            self._iteration_number=row['iteration']
            self._iteration_started=perf_counter()-row['iteration_seconds']
            stage={'field':'Рассчитанное поле','gradient':'Градиент','regularization':'Регуляризация'}[row['stage']]
            iteration='Начальная оценка (0)' if row['iteration']==0 else f'Итерация {row["iteration"]} / {self._iteration_limit}'
            trial=f' · пробный шаг {row["trial"]}' if row['trial'] else ''
            blocks=f' · блоки {row["completed"]:,} / {row["total"]:,}' if row['total'] else ''
            estimate=f' · осталось на этапе ≈ {duration(row["stage_remaining_seconds"])}' if row['total'] else ''
            self.phase_label.setText(f'{iteration}{trial} · {stage}{blocks}\nЭтап: {duration(row["stage_seconds"])}{estimate}')
            if row['total']:
                self.phase_progress.setRange(0,1000)
                self.phase_progress.setValue(round(1000*row['completed']/row['total']))
                self.phase_progress.setFormat('%p% текущего этапа')
            else: self.phase_progress.setRange(0,0)
            self.update_clock()
            return
        if row.get('event')=='resources':
            if row.get('method')=='layer_fft':
                text=(f'Послойная 2D FFT · рабочая сетка {row["fft_shape"][0]:,} × {row["fft_shape"][1]:,} · '
                      f'слоёв {row["layers"]} · частичных призм {row["partial_cells"]:,}')
                memory=(f'Свободно при старте {row["free_mib"]:,} МиБ · кэш спектров ≤ {row["cache_mib"]:,.1f} МиБ '
                        f'({row["cached_layers"]} слоёв).')
                self.memory_info.setText(memory); self.log.appendPlainText(memory)
                self.status.setText(text); self.log.appendPlainText(text)
                return
            text=(f'Свободно при старте: {row["free_mib"]:,} MiB; кэш ≤{row["cache_mib"]:,} MiB; '
                  f'полный оператор ≤{row["operator_mib"]:,} MiB; пакет до '
                  f'{row["observation_block"]} наблюдений × {row["cell_block"]} вокселей.')
            self.memory_info.setText(text); self.log.appendPlainText(text)
            self.status.setText(text+' Расчёт начального поля и градиента…')
            return
        self.history.append(row); self.curve.setData([r['iteration'] for r in self.history],[r['rmse'] for r in self.history])
        self.update_timing_plot()
        self.progress.setValue(row['iteration'])
        self.status.setText(f'Итерация {row["iteration"]}: RMSE={row["rmse"]:.6g}; RMS/σ={row["normalized_rms"]:.4g}')
        self.log.appendPlainText(f"{row['iteration']}: RMSE={row['rmse']:.6g}; RMS/σ={row['normalized_rms']:.4g}; J={row['objective']:.6g}; время={row.get('iteration_seconds',0):.2f} с; всего={row.get('elapsed_seconds',0):.2f} с")
        self.update_clock()

    def update_timing_plot(self):
        rows=[r for r in self.history if 'iteration_seconds' in r]
        self.timing_curve.setData([r['iteration'] for r in rows],[r['iteration_seconds'] for r in rows])

    def update_clock(self):
        if self._run_started is None: return
        now=perf_counter(); elapsed=now-self._run_started
        current=now-self._iteration_started if self._iteration_started is not None else 0
        measured=[r['iteration_seconds'] for r in self.history if r['iteration']>0 and 'iteration_seconds' in r]
        completed=self.history[-1]['iteration'] if self.history else 0
        remaining=None
        iteration_remaining=None
        if measured:
            mean=float(np.mean(measured[-5:])); remaining=max(0,self._iteration_limit-completed)*mean
            if self._iteration_number>completed:
                remaining=max(0,remaining-current)
                iteration_remaining=max(0,mean-current) if current<=mean else None
                if current>mean: remaining=None
        last=self.history[-1].get('iteration_seconds') if self.history else None
        self.timing_label.setText(f'Прошло: {duration(elapsed)} · текущая итерация: {duration(current)} · последняя: {duration(last)}\n'
            f'Остаток итерации ≈ {duration(iteration_remaining)}\n'
            f'До лимита итераций ≈ {duration(remaining)} · полное время ≈ {duration(elapsed+remaining) if remaining is not None else "ожидаются замеры"}')

    def stop_timing(self,elapsed=None):
        if self._run_started is None: return
        elapsed=perf_counter()-self._run_started if elapsed is None else elapsed
        last=self.history[-1].get('iteration_seconds') if self.history else None
        self.timing_label.setText(f'Общее время расчёта: {duration(elapsed)} ({elapsed:.2f} с) · последняя завершённая итерация: {duration(last)}')
        self.clock_timer.stop(); self._run_started=None; self._iteration_started=None
        self.phase_progress.setRange(0,100); self.phase_progress.setValue(0)
        self.phase_progress.setFormat('Расчёт завершён')

    def failure(self,message):
        if self._notify_inversion:
            self._notify_inversion=False
            # Arbitrary exception text may contain paths/data. Keep outbound error messages generic.
            cancelled=self.worker is not None and self.worker.cancel.is_set()
            event='cancelled' if cancelled else 'error'
            text=('Расчёт остановлен пользователем до получения готового результата.' if cancelled
                  else 'Расчёт 3D-модели завершился без готового результата.')
            self.workbench.notifications.send(event,'GeoGarModel: '+text+'\nПодробности доступны в программе.')
        self.stop_timing(); self.phase_label.setText(message)
        self.status.setText(message); self.log.appendPlainText(message)
    def job_finished(self):
        self.stop_timing()
        self.worker.function=None; self.worker.deleteLater(); self.worker=None
        self.unlock_inputs()
        self.run_button.setEnabled(True)
        self.stop.setEnabled(False)

    def model_task(self,function,callback):
        if self.worker or self.workbench.worker: raise ValueError('Дождитесь текущего расчёта.')
        self.status.setText('Чтение начальной модели…')
        self.lock_inputs(); self.run_button.setEnabled(False)
        self.worker=InversionWorker(lambda progress,cancel:function(),self)
        self.worker.result.connect(lambda model:self.workbench.guard(lambda:callback(model)))
        self.worker.failed.connect(self.failure); self.worker.finished.connect(self.job_finished)
        self.worker.start()
    def cancel(self):
        if self.worker:
            self.worker.cancel.set(); self.stop.setEnabled(False)
            self.status.setText('Остановка… Возвращаем лучшую завершённую модель и соответствующее поле.')
    def finished_result(self,result,offer_export=True):
        self.result=result
        self.history=list(result.history); self.update_timing_plot()
        elapsed=result.settings.get('run_elapsed_seconds',result.settings.get('elapsed_seconds'))
        if self._run_started is not None: self.stop_timing(elapsed)
        elif elapsed is not None: self.timing_label.setText(f'Общее время сохранённого расчёта: {duration(elapsed)} ({elapsed:.2f} с)')
        else: self.timing_label.setText('В сохранённом результате нет замеров времени.')
        self.phase_label.setText('Вычисления завершены. Подготовка результатов…')
        selected=result.settings.get('selected_iteration',result.history[-1]['iteration'])
        row=next(r for r in result.history if r['iteration']==selected)
        reason={'cancelled':'Остановлено пользователем','early_stop':'Ранняя остановка'}.get(result.status,result.status)
        if self._notify_inversion:
            self._notify_inversion=False
            from .telegram_notifications import result_message
            event=result.status if result.status in ('cancelled','early_stop') else 'finished'
            self.workbench.notifications.send(event,result_message(result))
        self.phase_label.setText(f'{reason} · завершено {result.history[-1]["iteration"]} итераций (начальная оценка — 0).')
        self.status.setText(f"{reason} · модель итерации {selected} · RMSE={row['rmse']:.6g} · RMS/σ={row['normalized_rms']:.4g}; {result.settings['device'].upper()}")
        if result.status=='cancelled' and offer_export:
            self.stopped_export_dialog()
        self.workbench.guard(self.draw_results)
        if result.status in ('cancelled','early_stop'):
            from importlib.util import find_spec
            if find_spec('pyvistaqt') is not None:
                self.workbench.guard(lambda:self.scene.render(result))
            else:
                self.scene.result=result
                self.log.appendPlainText('3D-сцена требует пакет volume. Модель и двумерные срезы доступны.')
        self.progress.setValue(result.history[-1]['iteration'])

    def stopped_export_dialog(self):
        dialog=QDialog(self); dialog.setWindowTitle('Инверсия остановлена — сохранить результат')
        layout=QVBoxLayout(dialog)
        text=QLabel('Лучшая завершённая модель выбрана по минимальной RMS/σ.\n'
                    'Можно сохранить несколько форматов. После закрытия окна модель и поля\n'
                    'останутся в программе; экспорт будет доступен и позднее.\n'
                    'Автоматического сохранения на диск нет.')
        text.setWordWrap(True); layout.addWidget(text)
        for title,callback in [('Модель и полный результат NPZ…',self.save),('Модель UBC…',self.save_ubc),
                               ('Модель VTK / VTR / VTU…',self.save_vtk),('Рассчитанное поле ZMAP / поля CSV…',self.save_fields)]:
            button=QPushButton(title); layout.addWidget(button)
            button.clicked.connect(lambda checked=False,fn=callback:self.workbench.guard(fn))
        done=QPushButton('Показать модель и поля'); done.clicked.connect(dialog.accept); layout.addWidget(done)
        dialog.exec_()

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
        if self.result is None: raise ValueError('Сначала завершите 3D-моделирование.')
        return self.result
    def open_3d(self): self.scene.render(self.require_result()); self.tabs.setCurrentIndex(0)
    def import_display_model(self):
        from PyQt5.QtWidgets import QInputDialog
        if self.worker or self.workbench.worker: raise ValueError('Дождитесь текущего расчёта.')
        target=self.scene.active_document()
        labels={'Плотностная модель · г/см³':'gravity','Магнитная восприимчивость · SI':'induced','Векторная намагниченность · А/м':'vector'}
        label,ok=QInputDialog.getItem(self,'Свойство 3D-модели','Для VTK, UBC и таблиц выберите свойство. NPZ хранит тип модели внутри файла.',list(labels),0,False)
        if not ok: return
        loader=self.initial_panel.ask_load(labels[label])
        if loader is not None:
            if target is None: target=self.scene.new_window()

            def imported(model):
                displayed=self.scene.finish_import(model.result(),target)
                self.status.setText('3D-модель загружена.' if displayed else
                    'Окно загрузки закрыто. Модель доступна в списке «Все загруженные модели».')
            self.model_task(loader,imported)
    def open_slice_maps(self):
        self.scene.result=self.require_result(); self.scene.open_slice_maps()
    def load_result(self):
        path,_=QFileDialog.getOpenFileName(self,'Сохранённый результат 3D-моделирования','','NPZ (*.npz)')
        if path:
            result=InversionResult.load(path)
            result.settings['source_path']=str(Path(path).resolve())
            self.history=list(result.history)
            self.curve.setData([r['iteration'] for r in self.history],[r['rmse'] for r in self.history])
            self.finished_result(result,offer_export=False)
    def save(self):
        result=self.require_result(); path,_=QFileDialog.getSaveFileName(self,'3D-модель и отчёт','inversion.npz','NPZ (*.npz)')
        if path: result.save(path if path.lower().endswith('.npz') else path+'.npz')
    def save_fields(self):
        r=self.require_result(); path,kind=QFileDialog.getSaveFileName(self,'Поля и невязки','inversion-fields.csv','CSV (*.csv);;Рассчитанное поле ZMAP (*.zmap)')
        if path:
            if Path(path).suffix.lower()=='.zmap' or ('ZMAP' in kind and not Path(path).suffix):
                export_zmap(r,path if path.lower().endswith('.zmap') else path+'.zmap')
            else:
                field_table(r).to_csv(path if path.lower().endswith('.csv') else path+'.csv',index=False,encoding='utf-8-sig')
    def save_ubc(self):
        r=self.require_result(); path,_=QFileDialog.getSaveFileName(self,'UBC: mesh, свойства и описание','inversion.msh','UBC mesh (*.msh)')
        if path: export_ubc(r,path)
    def save_vtk(self):
        r=self.require_result(); grid=vtk_grid(r); ext='.vtu' if grid.__class__.__name__=='UnstructuredGrid' else '.vtr'
        path,_=QFileDialog.getSaveFileName(self,'VTK-модель','inversion'+ext,'VTK (*'+ext+')')
        if path: grid.save(path if path.lower().endswith(ext) else path+ext)
