"""One project lifecycle for maps, ML/Optuna and 3D modelling."""
from pathlib import Path
from PyQt5.QtCore import Qt,QSignalBlocker
from PyQt5.QtWidgets import (QApplication,QComboBox,QCheckBox,QLineEdit,QSpinBox,QDoubleSpinBox,
    QPlainTextEdit,QFileDialog,QMessageBox,QPushButton,QTabWidget)
from .maps_ui import MapWorkspace
from .project_archive import write_project,read_project
from .model_workspace import ModelEntry

ML_CONTROLS=('x_column','y_column','target','spatial','block','test_fraction','seed','parameters','search',
    'table_crs','north_first','model_choice','device_choice','gpu_id','optimize_check','trials','folds','timeout',
    'validation_method','group_column','buffer_size')
INV_CONTROLS=('mode','east','north','height','value','error','crs','height_constant','sigma','stride','nx','ny','nz',
    'top','bottom','padding','cache','memory_policy','field','inc','dec','lower','upper','initial','iterations','damping',
    'smooth','depth_index','target','step','offset','device','gpu','early_stop','early_tolerance','early_patience',
    'method','fft_grid','extend_fft','reference_source','penalty_norm','penalty_weight','weight_power',
    'vertical_factor','penalty_epsilon','structural','boundary_jump','freeze_reference')


def controls_state(owner,names):
    result={}
    for name in names:
        w=getattr(owner,name)
        result[name]=w.currentText() if isinstance(w,QComboBox) else w.isChecked() if isinstance(w,QCheckBox) else w.text() if isinstance(w,QLineEdit) else w.value()
    return result


def apply_controls(owner,values):
    blockers=[QSignalBlocker(getattr(owner,name)) for name in values]
    for name,value in values.items():
        w=getattr(owner,name)
        if isinstance(w,QComboBox):
            if w.findText(value)<0 and value and not w.isEditable(): w.addItem(value)
            w.setCurrentText(value)
        elif isinstance(w,QCheckBox): w.setChecked(value)
        elif isinstance(w,QLineEdit): w.setText(value)
        else: w.setValue(value)
    del blockers


class ProjectWorkspace:
    def setup_project(self):
        self._project_loading=False; self._project_dirty=False
        self._ml_defaults=controls_state(self,ML_CONTROLS); self._inv_defaults=controls_state(self.inversion,INV_CONTROLS)
        self.inversion.scene.changed.connect(self.mark_project_dirty)
        for w in self.centralWidget().findChildren(QWidgetTypes):
            signal=w.currentTextChanged if isinstance(w,QComboBox) else w.toggled if isinstance(w,QCheckBox) else w.textChanged if isinstance(w,(QLineEdit,QPlainTextEdit)) else w.valueChanged if isinstance(w,(QSpinBox,QDoubleSpinBox)) else w.clicked if isinstance(w,QPushButton) else w.currentChanged
            signal.connect(self.mark_project_dirty)
        self.feature_list.itemChanged.connect(self.mark_project_dirty)
        self.table_crs.valueChanged.connect(self.mark_project_dirty)
        self._project_dirty=False

    def mark_project_dirty(self,*args):
        if not getattr(self,'_project_loading',True) and (self.layers.count() or self.training is not None or self.prediction is not None or self.inversion.table is not None or self.inversion.initial_panel.model is not None or self.inversion.scene.registry or self.inversion.scene.window_selector.count() or self.map_project_name!='Без названия'): self._project_dirty=True

    def update_project_label(self):
        MapWorkspace.update_project_label(self)
        self.setWindowTitle('GeoGarModel 2.3 · '+self.map_project_name)

    def project_idle(self):
        if self.worker is not None or self.inversion.worker is not None: raise ValueError('Дождитесь окончания расчёта перед открытием, закрытием или созданием другого проекта.')

    def confirm_map_close(self):
        self.project_idle()
        if not getattr(self,'_project_dirty',False) and not self.map_dirty: return True
        answer=QMessageBox.question(self,'Проект GeoGarModel','Сохранить изменения во всех трёх разделах проекта?',QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel)
        if answer==QMessageBox.Cancel: return False
        return self.save_map_project() if answer==QMessageBox.Save else True

    def reset_map_project(self,name='Без названия'):
        # Called by project creation and explicit resets; map restore calls the base directly.
        MapWorkspace.reset_map_project(self,name)
        if not hasattr(self,'_ml_defaults'): return
        self._project_loading=True
        try:
            self.training=self.prediction=self.result=self.output=None; self.optimization_context=None
            for w in (self.x_column,self.y_column,self.target,self.group_column): w.clear()
            self.group_column.addItem("")
            self.feature_panel.search.clear(); self.feature_list.clear(); apply_controls(self,self._ml_defaults)
            self.update_device_controls(); self.update_validation_controls(); self.training_workflow.setCurrentIndex(0)
            self.dashboard.reset(); self.log.clear(); self.progress_bar.setValue(0)
            self.best_trial_label.setText('Лучший trial: ещё нет завершённых результатов')
            self.table_info.setText('Загрузите обучающую и прогнозную таблицы.')
            t=self.inversion; t.result=t.table=None; t.history=[]; t.initial_panel.model=t.initial_panel.undo_model=None
            t.set_reference(None)
            for w in (t.east,t.north,t.height,t.value,t.error): w.clear()
            apply_controls(t,self._inv_defaults); t.mode_changed(); t.initial_panel.source.setCurrentIndex(0)
            t.initial_panel.info.setText('Без файла: новая сетка заполняется полем «Начальное значение».')
            t.close_result_windows()
            t.scene.reset()
            t.figure.clear(); t.canvas.draw_idle(); t.curve.setData([],[]); t.timing_curve.setData([],[]); t.log.clear()
            t.status.setText('Загрузите поле, задайте модель и запустите расчёт.'); t.progress.setValue(0)
            t.phase_progress.setValue(0); t.phase_label.setText('Этап расчёта появится после запуска'); t.timing_label.setText('Время расчёта: —'); t.workflow.setCurrentIndex(0)
            self._project_dirty=False; self.map_dirty=False
        finally: self._project_loading=False

    def new_map_project(self):
        self.project_idle(); self.project_crs_dialog(new=True)

    def close_map_project(self):
        if self.confirm_map_close(): self.reset_map_project()

    def save_map_project(self,save_as=False):
        path=self.map_project_path
        if save_as or not path:
            path,_=QFileDialog.getSaveFileName(self,'Сохранить общий проект GeoGarModel',self.map_project_name+'.ggproject','GeoGarModel · все разделы (*.ggproject)')
        if not path: return False
        if not Path(path).suffix: path+='.ggproject'
        self.save_project_to(path); return True

    def save_project_to(self,path):
        self.map_layer_rows()
        if self._map_selected and self._map_selected.data(Qt.UserRole)['kind']=='grid': self._map_selected.data(Qt.UserRole)['style'].update(self.read_map_style())
        ax=self.figure.axes[0] if self.figure.axes else None
        maps=dict(name=self.map_project_name,layers=self.map_layer_rows(),view=[list(ax.get_xlim()),list(ax.get_ylim())] if ax else None,
            scale=dict(enabled=self.scale_on.isChecked(),step=self.scale_step.value(),count=self.scale_count.value(),unit=self.scale_unit.currentText(),show_crs=self.scale_crs.isChecked(),north='arrow'),
            crs=self.map_project_crs,presentation=self.read_map_presentation())
        t=self.inversion; models={}; entries=[]; seen={}
        for entry in t.scene.registry.values():
            key=seen.setdefault(id(entry.result),str(len(seen))); models[key]=entry.result
            entries.append(dict(key=entry.key,name=entry.name,path=entry.path,model=key))
        inversion_key=None
        if t.result is not None:
            inversion_key=seen.setdefault(id(t.result),str(len(seen))); models[inversion_key]=t.result
        voxels={key:value for key,value in [('initial',t.initial_panel.model),('undo',t.initial_panel.undo_model),
                                            ('reference',t.reference_model)] if value is not None}
        state=dict(name=self.map_project_name,entries=entries,windows=t.scene.snapshot(),inversion_result=inversion_key,
            ml_controls=controls_state(self,ML_CONTROLS),inv_controls=controls_state(t,INV_CONTROLS),
            transform_controls=controls_state(self,('operation','height','wave','degree')),
            features=[self.feature_list.item(i).text() for i in range(self.feature_list.count()) if self.feature_list.item(i).checkState()==Qt.Checked],
            optimization_context=self.optimization_context,trial_history=list(self.dashboard.history),
            fitted_crs=getattr(self,'fitted_crs',''),fitted_north_first=getattr(self,'fitted_north_first',True),
            initial_source=t.initial_panel.source.currentIndex(),initial_sources={key:dict(path=value.source_path,array=value.source_array) for key,value in voxels.items()},
            map_layout=self._map_layout,map_selected=self.layers.currentRow(),map_view=self.map_views.currentIndex(),
            active_tab=self.tabs.currentIndex(),inversion_tab=t.tabs.currentIndex(),training_tab=self.dashboard.tabs.currentIndex(),
            training_workflow=self.training_workflow.currentIndex(),inversion_workflow=t.workflow.currentIndex(),
            training_log=self.log.toPlainText(),inversion_log=t.log.toPlainText(),
            progress=[self.progress_bar.minimum(),self.progress_bar.maximum(),self.progress_bar.value()],
            inversion_progress=[t.progress.minimum(),t.progress.maximum(),t.progress.value()],
            ml_views={name:plot.getViewBox().viewRange() for name,plot in self.dashboard.plots.items()})
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try: self.activity.run('Сохранение проекта',write_project,path,state,maps,dict(training=self.training,prediction=self.prediction,output=self.output,observations=t.table),self.result,models,voxels)
        finally: QApplication.restoreOverrideCursor()
        self.map_project_path=str(path); self._project_dirty=False; self.map_dirty=False
        self.statusBar().showMessage('Сохранён общий проект: карты, ML/Optuna и 3D-моделирование.')

    def open_map_project(self):
        self.project_idle()
        path,_=QFileDialog.getOpenFileName(self,'Открыть проект GeoGarModel','','GeoGarModel (*.ggproject *.ggmaps)')
        if path: self.open_project_from(path)

    def open_project_from(self,path):
        self.project_idle(); data=self.activity.run('Открытие проекта',read_project,path)
        if 'state' in data:
            state=data['state']
            if set(state['ml_controls'])-set(ML_CONTROLS) or set(state['inv_controls'])-set(INV_CONTROLS): raise ValueError('Неизвестные настройки в проекте.')
        if not self.confirm_map_close(): return False
        self._project_loading=True; QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.reset_map_project(); self._project_loading=True
            if 'legacy_maps' in data:
                self.restore_map_project(*data['legacy_maps'],path)
            else:
                state=data['state']; self.restore_map_project(*data['maps'],path)
                self._map_layout=state['map_layout']
                apply_controls(self,state.get('transform_controls',{})); self.update_transform_parameters()
                self.training=data['tables'].get('training'); self.prediction=data['tables'].get('prediction'); self.output=data['tables'].get('output')
                if self.training is not None: self.configure_columns()
                ml_controls = dict(state['ml_controls'])
                if 'validation_method' not in ml_controls:
                    ml_controls['validation_method'] = 'BlockKFold' if ml_controls.get('spatial', True) else 'ShuffleKFold'
                apply_controls(self,ml_controls); self.update_validation_controls(); self.update_features()
                for i in range(self.feature_list.count()):
                    item=self.feature_list.item(i); item.setCheckState(Qt.Checked if item.text() in state['features'] else Qt.Unchecked)
                self.update_device_controls(); self.optimization_context=state['optimization_context']
                self.fitted_crs=state['fitted_crs']; self.fitted_north_first=state['fitted_north_first']
                self.dashboard.reset()
                for record in state['trial_history']: self.trial_progress(record)
                self.result=data['training_result']
                if self.result is not None: self.dashboard.show_result(self.result,self.fitted_north_first)
                for name,limits in state.get('ml_views',{}).items():
                    if name in self.dashboard.plots: self.dashboard.plots[name].setRange(xRange=limits[0],yRange=limits[1],padding=0)
                self.log.setPlainText(state['training_log']); self.dashboard.tabs.setCurrentIndex(state['training_tab'])
                if 'progress' in state: self.progress_bar.setRange(*state['progress'][:2]); self.progress_bar.setValue(state['progress'][2])
                self.training_workflow.setCurrentIndex(state['training_workflow'])
                if self.training is not None: self.table_info.setText(f'Из проекта: {len(self.training):,} обучающих строк; исходные данные встроены в файл.')
                t=self.inversion
                if 'observations' in data['tables']: t.assign_table(data['tables']['observations'])
                inversion_controls = dict(state['inv_controls'])
                if inversion_controls['mode']=='Послойная 2D FFT · плотностное моделирование':
                    inversion_controls.update(mode=t.mode.itemText(0), method='Послойная 2D FFT', padding=0.)
                if inversion_controls.get('method')=='Послойная 2D FFT' and 'penalty_norm' not in inversion_controls:
                    for name in ('depth_index','smooth','damping','step','target','offset','early_stop','early_patience','early_tolerance'):
                        inversion_controls[name]=self._inv_defaults[name]
                t.mode.setCurrentText(inversion_controls['mode'])
                t.mode_changed()
                apply_controls(t,inversion_controls)
                t.cache.setEnabled(t.memory_policy.currentData()=='manual'); t.early_tolerance.setEnabled(t.early_stop.isChecked()); t.early_patience.setEnabled(t.early_stop.isChecked())
                if 'initial' in data['voxels']:
                    t.initial_panel.assign(data['voxels']['initial'],check_crs=False)
                for key,model in data['voxels'].items():
                    model.source_path=state['initial_sources'][key]['path']; model.source_array=state['initial_sources'][key]['array']
                t.set_reference(data['voxels'].get('reference'))
                t.initial_panel.undo_model=data['voxels'].get('undo'); t.initial_panel.source.setCurrentIndex(state['initial_source'])
                apply_controls(t,inversion_controls)
                t.update_fft_inputs()
                if state['inversion_result'] is not None:
                    result=data['models'][state['inversion_result']]; t.finished_result(result,offer_export=False)
                    t.curve.setData([r['iteration'] for r in result.history],[r['rmse'] for r in result.history])
                entries=[ModelEntry(row['key'],row['name'],row['path'],data['models'][row['model']]) for row in state['entries']]
                t.scene.restore(entries,state['windows']); t.log.setPlainText(state['inversion_log'])
                if 'inversion_progress' in state: t.progress.setRange(*state['inversion_progress'][:2]); t.progress.setValue(state['inversion_progress'][2])
                t.workflow.setCurrentIndex(state['inversion_workflow']); t.tabs.setCurrentIndex(state['inversion_tab'])
                self.layers.setCurrentRow(state['map_selected']); self.map_views.setCurrentIndex(state['map_view']); self.tabs.setCurrentIndex(state['active_tab'])
                self.redraw()
            self.map_project_path=str(path); self.map_dirty=False; self._project_dirty=False
            self.statusBar().showMessage('Проект восстановлен: карты, ML/Optuna и 3D-моделирование.'); return True
        finally: self._project_loading=False; QApplication.restoreOverrideCursor()


QWidgetTypes=(QComboBox,QCheckBox,QLineEdit,QSpinBox,QDoubleSpinBox,QPlainTextEdit,QPushButton,QTabWidget)
