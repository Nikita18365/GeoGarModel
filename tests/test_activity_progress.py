import threading

import numpy as np
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow

from geogarmodel.activity_ui import Activities
from geogarmodel.work_progress import report, reporting

def test_export_live_map_via_toolbar(qt_application_lifetime,tmp_path,monkeypatch):
    from PyQt5.QtWidgets import QFileDialog,QMessageBox
    from geogarmodel.app import MainWindow
    from geogarmodel.map_navigation import MapToolbar
    window=MainWindow()
    window.load_demo()
    path=tmp_path/'map.png'
    errors=[]
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args:(str(path),''))
    monkeypatch.setattr(QMessageBox,'warning',lambda *args:errors.append(args[-1]))
    try:
        window.findChild(MapToolbar).save_figure()
        assert not errors,errors
        assert path.read_bytes().startswith(b'\x89PNG')
        assert window.activity.bar.value()==100
    finally:
        window.close()


def test_gui_stays_live_and_percent_waits_for_success(qt_application_lifetime):
    window=QMainWindow()
    window.resize(850,400)
    activity=Activities(window)
    window.show()
    window.statusBar().showMessage('Прежнее сообщение')
    progress=[]
    activity.bar.valueChanged.connect(progress.append)
    gui_thread=threading.get_ident()
    pulse=threading.Event()
    def work():
        assert threading.get_ident()!=gui_thread
        report(1,2,'Слои')
        assert pulse.wait(3)
        report(2,2,'Слои')
        return 'результат'
    timer=QTimer(window)
    def inspect():
        if activity.bar.value()==50 and activity.panel.isVisible() and activity.panel.x()<30:
            pulse.set()
    timer.timeout.connect(inspect)
    timer.start(5)
    try:
        assert activity.run('Экспорт',work)=='результат'
        assert 50 in progress and progress[-1]==100
        outer=activity.start('Инверсия')
        activity.update(outer,'Итерации',20)
        with pytest.raises(ValueError):
            activity.run('Файл',lambda: int('ошибка'))
        assert activity.bar.value()==20
        activity.finish(outer,'Остановлено')
        assert activity.bar.value()==20 and 'Остановлено' in activity.label.text()
        with pytest.raises(OSError):
            activity.run('Файл',lambda: open('/missing/geogarmodel/file'))
        assert activity.bar.value()<100 and 'Ошибка' in activity.label.text()
    finally:
        window.close()


def test_voxet_and_ubc_report_real_work(tmp_path):
    from geogarmodel.inversion import Mesh
    from geogarmodel.voxel import VoxelModel,read_ubc
    from geogarmodel.inversion_export import export_ubc
    from geogarmodel.gocad_voxet import read_voxet,write_voxet
    mesh=Mesh(np.arange(11.),np.arange(11.),np.arange(11.))
    model=VoxelModel(mesh,np.arange(mesh.size,dtype=float)).result()
    records=[]
    with reporting(lambda stage,pct: records.append((stage,pct))):
        export_ubc(model,tmp_path/'a.msh')
        read_ubc(tmp_path/'a.msh',tmp_path/'a.mod')
        write_voxet(model,tmp_path/'a.vo')
        read_voxet(tmp_path/'a.vo','gravity')
    for stage in ('Запись свойств UBC','Запись слоёв Voxet','Слои Voxet','Свойства UBC'):
        values=[value for label,value in records if label==stage]
        assert values==sorted(values) and values[-1]==99
