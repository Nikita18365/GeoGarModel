import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
import numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox,QPushButton,QDialog,QInputDialog
from PyQt5.QtCore import QEventLoop,QTimer
from geogarmodel.app import MainWindow


@pytest.mark.parametrize('outcome',['result','failed','cancelled'])
def test_worker_releases_device_before_publishing(monkeypatch,outcome):
    from geogarmodel import compute
    from geogarmodel.inversion_ui import InversionWorker
    events=[]
    def solve(progress,cancel):
        if outcome=='failed': raise MemoryError('GPU memory')
        if outcome=='cancelled': raise compute.ComputationCancelled()
        return 42
    worker=InversionWorker(solve,None,gpu_id=2)
    def release(device):
        assert worker.function is None
        events.append(device)
    monkeypatch.setattr(compute,'release_inversion_memory',release)
    worker.result.connect(lambda _:events.append('result'))
    worker.failed.connect(lambda _:events.append('failed'))
    worker.run()
    assert events==[2,'result' if outcome=='result' else 'failed']


def test_previous_overview_exports_its_own_model_and_field(monkeypatch,tmp_path):
    from PyQt5.QtWidgets import QFileDialog
    from geogarmodel.inversion import InversionResult
    import pandas as pd
    window=MainWindow()
    tab=window.inversion
    try:
        tab.method.setCurrentIndex(1)
        tab.demo()
        tab.target.setValue(0)
        for iterations in (1,2):
            tab.iterations.setValue(iterations)
            loop=QEventLoop()
            timer=QTimer()
            timer.setSingleShot(True); timer.timeout.connect(loop.quit)
            tab.run(); tab.worker.finished.connect(loop.quit)
            timer.start(30000); loop.exec_()
            assert tab.worker is None
        first,second=tab.result_windows.values()
        assert first.isVisible() and second.result is tab.result
        assert not np.array_equal(first.result.predicted,second.result.predicted)
        first.tabs.setCurrentIndex(1)
        actions=next(button.menu().actions() for button in first.findChildren(QPushButton)
                     if button.text()=='Сохранить…')
        for action,path in [(actions[0],tmp_path/'first.npz'),(actions[-1],tmp_path/'first.csv')]:
            monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args:(str(path),''))
            action.trigger()
        saved=InversionResult.load(tmp_path/'first.npz')
        np.testing.assert_array_equal(saved.model,first.result.model)
        np.testing.assert_array_equal(saved.predicted,first.result.predicted)
        frame=pd.read_csv(tmp_path/'first.csv')
        np.testing.assert_allclose(frame['predicted'],first.result.predicted)
        first.close()
        assert list(tab.result_windows.values())==[second]
        path=tmp_path/'second.ggmodel'
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args:(str(path),'Быстрая модель (*.ggmodel)'))
        tab.save()
        monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args:(str(path),''))
        tab.load_result()
        overview=list(tab.result_windows.values())[-1]
        from geogarmodel.model_archive import ModelArchive
        assert isinstance(overview.result,ModelArchive)
        overview.tabs.setCurrentIndex(1)
        monkeypatch.setattr(QInputDialog,'getInt',lambda *args:(1,True))
        overview.open_layer(overview.result)
        button=next(b for b in overview.findChildren(QPushButton) if b.text()=='Загрузить полностью')
        button.click()
        loop=QEventLoop()
        tab.worker.finished.connect(loop.quit)
        QTimer.singleShot(30000,loop.quit)
        loop.exec_()
        assert not button.isEnabled()
        np.testing.assert_array_equal(overview.result.model,second.result.model)
    finally: window.close()


@pytest.fixture(scope='module',autouse=True)
def app_session():
    app=QApplication.instance() or QApplication([])
    yield app


@pytest.mark.parametrize('mode',[0,1,2])
def test_inversion_tab_background_workflow(mode,monkeypatch):
    errors=[]
    monkeypatch.setattr(QMessageBox,'warning',lambda *a: errors.append(a[-1]))
    window=MainWindow(); tab=window.inversion
    assert window.tabs.count()==3
    tab.mode.setCurrentIndex(mode); tab.demo()
    tab.iterations.setValue(4); tab.target.setValue(0)
    loop=QEventLoop(); timer=QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit)
    tab.run(); tab.worker.finished.connect(loop.quit); timer.start(30000); loop.exec_()
    assert tab.worker is None
    assert not errors,errors
    assert tab.result is not None,tab.status.text()
    assert tab.result.settings['resources']['policy']=='balanced'
    assert 'Свободно при старте' in tab.memory_info.text()
    assert len(tab.result.predicted)==225
    assert not tab.clock_timer.isActive()
    assert 'Общее время расчёта' in tab.timing_label.text()
    assert tab.result.settings['run_elapsed_seconds']>=tab.result.settings['elapsed_seconds']
    assert len(tab.timing_curve.xData)==len(tab.result.history)
    assert len(tab.figure.axes)==11
    assert tab.result.history[-1]['rmse']<tab.result.history[0]['rmse']
    tab.initial_panel.previous()
    initial=tab.initial_panel.current()
    np.testing.assert_allclose(initial.values,tab.result.model)
    assert initial.mesh.shape==tab.result.mesh.shape
    window.close(); QApplication.processEvents()


def test_vtk_coordinates_and_vector_order(tmp_path):
    pytest.importorskip('pyvista')
    from geogarmodel.inversion import Mesh,InversionResult
    from geogarmodel.inversion_view import vtk_grid
    mesh=Mesh(np.arange(3.)+100,np.arange(4.)+200,np.arange(5.)+10)
    model=np.arange(mesh.size*3).reshape(-1,3).astype(float)
    result=InversionResult(mesh,model,np.zeros((1,3)),np.zeros(1),np.zeros(1),[],{'mode':'vector'},'test')
    grid=vtk_grid(result)
    expected=mesh.centers(); expected[:,2]*=-1
    centers=grid.cell_centers().points
    for point,vector in zip(expected,model):
        index=np.where(np.all(np.isclose(centers,point),axis=1))[0][0]
        np.testing.assert_allclose(grid['M_END'][index],vector)
        assert grid['property'][index]==np.linalg.norm(vector)
    grid.save(tmp_path/'model.vtr')


def test_display_scale_keeps_physical_axes_and_model():
    pytest.importorskip('pyvistaqt')
    from geogarmodel.inversion import Mesh,InversionResult
    from geogarmodel.inversion_view import ModelView,vtk_grid
    mesh=Mesh(np.linspace(18000000,18010000,4),np.linspace(6000000,6020000,4),np.linspace(100,30000,4))
    result=InversionResult(mesh,np.ones(mesh.size),np.zeros((1,3)),np.zeros(1),np.zeros(1),[],{'mode':'gravity'},'test')
    original=vtk_grid(result).points.copy()
    view=ModelView(); view.resize(1100,850)
    try:
        view.render(result)
        view.axis_scales[2].setValue(3)
        np.testing.assert_allclose(view.bounds_axes.bounds,[0,10000,0,20000,-90000,-300])
        meshes=[actor.mapper.dataset for actor in view.plotter.renderer.actors.values()
                if getattr(getattr(actor,'mapper',None),'dataset',None) is not None]
        assert any(np.allclose(g.bounds,[0,10000,0,20000,-90000,-300]) for g in meshes)
        np.testing.assert_allclose(view.bounds_axes.z_axis_range,[-30000,-100])
        view.axis_unit.setCurrentText('км')
        np.testing.assert_allclose(view.bounds_axes.z_axis_range,[-30,-.1])
        view.plotter.camera.zoom(1.2)
        zoom=view.plotter.camera.parallel_scale
        view.axis_font.setValue(22)
        assert view.plotter.camera.parallel_scale==pytest.approx(zoom)
        view.plotter.camera.azimuth+=20
        direction=np.array(view.plotter.camera.direction)
        view.axis_scales[0].setValue(2)
        np.testing.assert_allclose(view.plotter.camera.direction,direction)
        np.testing.assert_allclose(view.display_grid.bounds,[0,20000,0,20000,-90000,-300])
        view.axis_scales[0].setValue(1)
        view.camera_view.setCurrentIndex(1); view.fit_model()
        view.axis_scales[2].setValue(4)
        assert view.camera_view.currentIndex()==0  # Top XY cannot show a Z stretch.
        assert abs(view.plotter.camera.direction[2])<.99
        view.axis_scales[2].setValue(3)
        for mode in range(5):
            view.kind.setCurrentIndex(mode)
            np.testing.assert_allclose(view.bounds_axes.z_axis_range,[-30,-.1])
        view.reset_scales()
        np.testing.assert_allclose(view.bounds_axes.bounds,[0,10000,0,20000,-30000,-100])
        np.testing.assert_allclose(vtk_grid(result).points,original)
        result.model[:]=0
        view.kind.setCurrentIndex(0); view.render(result)
        assert any(getattr(getattr(actor,'mapper',None),'dataset',None) is not None
                   and actor.mapper.dataset.n_cells==mesh.size for actor in view.plotter.renderer.actors.values())
        view.expanded_view()
        assert view._expanded_dialog.isVisible()
        view._expanded_dialog.accept()
        assert view._expanded_dialog is None
    finally:
        view.shutdown(); view.close()


def test_model_colors_apply_to_all_modes_and_expanded_view():
    pytest.importorskip('pyvistaqt')  # Exercised by the volume-enabled CI job.
    from geogarmodel.inversion import Mesh,InversionResult
    from geogarmodel.inversion_view import ModelView
    mesh=Mesh(np.arange(5.)*100,np.arange(5.)*100,np.arange(5.)*100+10)
    result=InversionResult(mesh,np.linspace(-1,1,mesh.size),np.zeros((1,3)),np.zeros(1),np.zeros(1),[],{'mode':'gravity'},'test')
    view=ModelView()
    try:
        view.render(result)
        view.palette.setCurrentText('plasma'); view.reverse_palette.setChecked(True)
        view.color_range.setCurrentIndex(2)
        view.color_min.setValue(-.3); view.color_max.setValue(.7)
        for mode in range(5):
            view.kind.setCurrentIndex(mode); view.apply_colors.click()
            assert not view.color_note.text()
            bar=next(iter(view.plotter.scalar_bars.values()))
            np.testing.assert_allclose(bar.GetLookupTable().GetRange(),[-.3,.7],err_msg=f'mode={mode}')
        view.color_min.setValue(.8); view.apply_colors.click()
        assert 'меньше' in view.color_note.text()
        view.color_min.setValue(-.3); view.apply_colors.click()
        view.expanded_view()
        expanded=view._expanded_dialog.findChild(ModelView)
        assert expanded.palette.currentText()=='plasma'
        assert expanded.reverse_palette.isChecked()
        assert expanded.color_min.value()==pytest.approx(-.3)
        assert expanded.color_max.value()==pytest.approx(.7)
        view._expanded_dialog.accept()
        np.testing.assert_array_equal(result.model,np.linspace(-1,1,mesh.size))
    finally:
        view.shutdown(); view.close()


def test_workflow_and_separate_editor(monkeypatch):
    window=MainWindow(); tab=window.inversion; tab.demo()
    tab.memory_policy.setCurrentIndex(2)
    tab.cache.setValue(1048576)
    assert tab.cache.value()==1048576 and tab.cache.isEnabled()
    tab.memory_policy.setCurrentIndex(0)
    assert not tab.cache.isEnabled()
    assert [tab.mode.itemText(i) for i in range(3)]==[
        'Плотностное моделирование · Δρ, g/cm³',
        'Моделирование магнитной восприимчивости · χ, SI',
        'Моделирование векторной намагниченности · J, A/m']
    assert tab.workflow.count()==3
    for i in range(2):
        tab.workflow.setCurrentIndex(i)
        button=next(b for b in tab.workflow.widget(i).findChildren(QPushButton) if b.text().startswith('Далее'))
        button.click(); assert tab.workflow.currentIndex()==i+1
    assert not any(b.text()=='Операции со свойствами…' for b in tab.initial_panel.findChildren(QPushButton))
    monkeypatch.setattr(QInputDialog,'getItem',lambda *args,**kwargs:('Заполнить константой (a)',True))
    monkeypatch.setattr(QInputDialog,'getText',lambda *args,**kwargs:('0.25, 1, 42',True))
    observed=[]
    def edit_and_close():
        dialog=QApplication.activeModalWidget()
        try:
            assert isinstance(dialog,QDialog)
            button=next(b for b in dialog.findChildren(QPushButton) if b.text()=='Операции со свойствами…')
            button.click(); observed.append(tab.initial_panel.current().values.copy())
        finally: dialog.accept()
    QTimer.singleShot(0,edit_and_close); tab.initial_panel.open_editor()
    assert len(observed)==1
    np.testing.assert_allclose(observed[0],.25)
    window.close(); QApplication.processEvents()


def test_running_job_keeps_navigation_but_freezes_inputs(monkeypatch):
    from threading import Event
    from PyQt5.QtTest import QTest
    from PyQt5.QtCore import Qt
    import geogarmodel.inversion_ui as ui
    gate=Event(); original=ui.invert
    def delayed(*args,**kwargs):
        assert gate.wait(15)
        return original(*args,**kwargs)
    monkeypatch.setattr(ui,'invert',delayed)
    window=MainWindow(); tab=window.inversion; tab.demo(); tab.iterations.setValue(1)
    tab.workflow.setCurrentIndex(2); tab.run()
    try:
        for i,button in enumerate(tab.workflow.buttons):
            assert button.isEnabled()
            button.click(); assert tab.workflow.currentIndex()==i
        assert tab.parameter_panel.isEnabled()
        assert tab.nx.isReadOnly() and tab.initial.isReadOnly()
        value=tab.nx.value(); QTest.keyClick(tab.nx,Qt.Key_Up)
        assert tab.nx.value()==value
        assert not tab.east.isEnabled() and not tab.mode.isEnabled()
        assert not tab.initial_panel.source.isEnabled()
        assert all(not b.isEnabled() for b in tab.initial_panel.findChildren(QPushButton) if not b.property('workflowNavigation'))
        assert not tab.run_button.isEnabled() and tab.stop.isEnabled()
        assert not tab.cache.isEnabled()  # Automatic memory policy remains disabled after unlock too.
        tab.workflow.setCurrentIndex(0)
        next(b for b in tab.workflow.widget(0).findChildren(QPushButton) if b.text().startswith('Далее')).click()
        assert tab.workflow.currentIndex()==1
    finally:
        gate.set()
        loop=QEventLoop(); timer=QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit)
        tab.worker.finished.connect(loop.quit); timer.start(15000); loop.exec_()
        assert tab.worker is None
        assert not tab.nx.isReadOnly() and tab.east.isEnabled()
        assert not tab.cache.isEnabled()
        assert tab.initial_panel.source.isEnabled()
        window.close(); QApplication.processEvents()


def test_stop_dialog_exports_then_displays_model(monkeypatch,tmp_path):
    from PyQt5.QtWidgets import QFileDialog
    import geogarmodel.inversion_ui as ui
    errors=[]; dialogs=[]
    monkeypatch.setattr(QMessageBox,'warning',lambda *a:errors.append(a[-1]))
    original=ui.invert
    def cancelled(*args,**kwargs):
        notify=kwargs['progress']
        def progress(row):
            notify(row)
            if row['iteration']==2: kwargs['cancel_event'].set()
        kwargs['progress']=progress
        return original(*args,**kwargs)
    monkeypatch.setattr(ui,'invert',cancelled)
    window=MainWindow(); tab=window.inversion; tab.demo(); tab.target.setValue(0)
    selected=tmp_path/'stopped.npz'
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(selected),'NPZ (*.npz)'))
    original_dialog=tab.stopped_export_dialog
    def dialog():
        def save_and_close():
            modal=QApplication.activeModalWidget()
            dialogs.append(modal.windowTitle())
            next(b for b in modal.findChildren(QPushButton) if b.text().startswith('Модель и полный')).click()
            next(b for b in modal.findChildren(QPushButton) if b.text()=='Показать модель и поля').click()
        QTimer.singleShot(0,save_and_close)
        original_dialog()
    monkeypatch.setattr(tab,'stopped_export_dialog',dialog)
    loop=QEventLoop(); timer=QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit)
    tab.run(); tab.worker.finished.connect(loop.quit); timer.start(30000); loop.exec_()
    assert tab.worker is None
    assert not errors,errors
    assert len(dialogs)==1 and selected.exists()
    assert tab.result.status=='cancelled'
    assert not tab.clock_timer.isActive()
    assert tab.scene.result is tab.result
    assert len(tab.figure.axes)==11
    assert tab.tabs.currentIndex()==1
    assert not tab.early_tolerance.isEnabled()
    tab.early_stop.setChecked(True)
    assert tab.early_tolerance.isEnabled() and tab.early_patience.isEnabled()
    window.close(); QApplication.processEvents()
