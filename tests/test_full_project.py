import json
import numpy as np
import pandas as pd
import pytest
from PyQt5.QtCore import Qt
from geogarmodel.app import MainWindow
from geogarmodel.grids import Grid
from geogarmodel.models import train_horizon
from geogarmodel.inversion import Mesh,InversionResult
from geogarmodel.voxel import VoxelModel
from geogarmodel.map_project import load_project as load_maps


def test_complete_project_restores_data_optuna_predictions_and_model_windows(qt_application_lifetime,tmp_path):
    window=MainWindow(); window.resize(1400,900); window.show(); qt_application_lifetime.processEvents()
    restored=None
    try:
        window.map_project_name='Общий проект'; window.map_project_crs='EPSG:20918'
        window.add_grid(Grid(np.arange(4.)+18000000,np.arange(4.)+7800000,np.arange(16.).reshape(4,4),'Карта','mGal','EPSG:20918','m'))
        window.palette.setCurrentText('terrain'); window.contours.setChecked(True); window.map_title.setText('Общий заголовок'); window.redraw()
        table=pd.DataFrame(dict(X=np.arange(24.)*1000,Y=np.arange(24.)*500,feature=np.arange(24.)*.7,depth=np.arange(24.)*2+10))
        window.training=table; window.prediction=table.drop(columns='depth').iloc[:6].copy(); window.configure_columns()
        window.model_choice.setCurrentText('Ridge'); window.spatial.setChecked(False); window.parameters.setText('{"alpha": 0.2}')
        window.optimization_context=dict(model='Ridge',parameters={},seed=42,device='cpu',gpu_id=0)
        fitted=train_horizon(table,'Ridge',spatial=False,optimize=True,n_trials=2,cv_folds=2,search={'alpha':[.1,1.]})
        window.fitted_crs='EPSG:20918'; window.fitted_north_first=True
        for row in fitted.optimization_history: window.trial_progress(row)
        window.trained(fitted); window.output=fitted.predict(window.prediction); expected=window.output.copy()
        mesh=Mesh(np.linspace(18000000,18003000,4),np.linspace(7800000,7803000,4),np.array([10.,100.,300.,600.]))
        cells=mesh.cells(); cells[0,2]=25; mesh.bounds=cells; mesh.active[-1]=False
        values=np.linspace(-.2,.3,mesh.size); observations=np.array([[18000000,7800000,0],[18001000,7800000,0],[18000000,7800100,0],[18001000,7800100,0]])
        history=[dict(iteration=0,rmse=2.,normalized_rms=2.),dict(iteration=1,rmse=1.,normalized_rms=1.,seconds=.25)]
        result=InversionResult(mesh,values,observations,np.arange(4.),np.arange(4.)+.5,history,dict(mode='gravity',device='cpu',crs='EPSG:20918'),'iterations')
        t=window.inversion; t.crs.setCurrentText('EPSG:20918')
        t.assign_table(pd.DataFrame(dict(East=observations[:,0],North=observations[:,1],anomaly=result.observed)))
        initial=VoxelModel(mesh,values,'gravity','EPSG:20918'); t.initial_panel.assign(initial)
        initial.history.append(dict(operation='test preparation'))
        t.finished_result(result,offer_export=False); t.nz.setValue(35); t.early_stop.setChecked(True)
        path=tmp_path/'original.npz'; initial.save(path)
        entry=t.scene.register(result,path=str(path)); other=t.scene.register(initial.result(),name='Начальная')
        # Catalogue/window metadata is independent of the optional OpenGL renderer.
        document=t.scene.new_window(); document.entries=[entry,other]; document.selected=entry; document.view.result=result
        document.view.slice_coordinates[2].setValue(250); document.view.palette.setCurrentText('plasma')
        document.showMinimized(); window.tabs.setCurrentIndex(2)
        project=tmp_path/'portable.ggproject'; window.save_project_to(project)
        path.unlink()  # Embedded model must open without its original source file.
        assert project.exists(); manifest,rows=load_maps(project); assert rows[0]['name']=='Карта'
        restored=MainWindow(); restored.resize(1400,900); restored.show(); qt_application_lifetime.processEvents()
        assert restored.open_project_from(project)
        assert restored.map_project_name=='Общий проект' and restored.map_title.text()=='Общий заголовок'
        assert restored.selected_grid().name=='Карта' and restored.contours.isChecked()
        pd.testing.assert_frame_equal(restored.training,table)
        pd.testing.assert_frame_equal(restored.output,expected)
        pd.testing.assert_frame_equal(restored.result.predict(restored.prediction),expected)
        assert len(restored.dashboard.history)==2 and restored.parameters.text()=='{"alpha": 0.2}'
        np.testing.assert_array_equal(restored.inversion.result.model,values)
        np.testing.assert_array_equal(restored.inversion.initial_panel.model.mesh.bounds,cells)
        assert restored.inversion.nz.value()==35 and restored.inversion.early_stop.isChecked()
        assert len(restored.inversion.scene.registry)==2
        doc=restored.inversion.scene._documents[0]; assert len(doc.entries)==2 and doc.isMinimized()
        assert doc.view.palette.currentText()=='plasma'
        assert doc.selected.path==str(path.resolve())
        assert next(e for e in restored.inversion.scene.registry.values() if e.name.startswith('Начальная')).result.settings['voxel_history']==initial.history
        assert doc.states[entry.key]['coordinate_2']==250
        # Saving a restored model must preserve its fitted pipeline too.
        second=tmp_path/'again.ggproject'; restored.save_project_to(second)
        from geogarmodel.project_archive import read_project
        pd.testing.assert_frame_equal(read_project(second)['training_result'].predict(restored.prediction),expected)
    finally:
        window.close()
        if restored is not None: restored.close()
        qt_application_lifetime.processEvents()


def test_failed_save_is_atomic_and_bad_open_preserves_current_project(qt_application_lifetime,tmp_path):
    window=MainWindow()
    try:
        path=tmp_path/'safe.ggproject'; window.save_project_to(path); original=path.read_bytes()
        window.optimization_context={'bad':object()}
        with pytest.raises(TypeError): window.save_project_to(path)
        assert path.read_bytes()==original
        window.training=pd.DataFrame({'original':[1,2,3]})
        bad=tmp_path/'bad.ggproject'; bad.write_bytes(b'not a project')
        with pytest.raises(Exception): window.open_project_from(bad)
        assert list(window.training)==['original']
    finally: window.close(); qt_application_lifetime.processEvents()
