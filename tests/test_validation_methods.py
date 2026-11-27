import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from geogarmodel.validation import make_splits, METHODS
from geogarmodel.models import train_horizon, model_defaults, make_model
from geogarmodel.compute import device_parameters
from geogarmodel.optimization import default_space

@pytest.fixture
def survey():
    a, b = np.meshgrid(np.arange(12)*1000., np.arange(12)*1000.)
    f = np.sin(a.ravel()/3000) + np.cos(b.ravel()/3000)
    return pd.DataFrame(dict(X=a.ravel(), Y=b.ravel(), profile=a.ravel(), feature=f, depth=100+20*f))

@pytest.mark.parametrize("method", METHODS)
def test_folds_and_oof(survey, method):
    xy=survey[["X","Y"]].to_numpy()
    labels=survey.profile.to_numpy()
    args=dict(labels=labels, block_size=3000., buffer_size=600., seed=17)
    splits=make_splits(method,xy,3,**args)
    assert np.array_equal(np.sort(np.concatenate([v for _,v in splits])),np.arange(len(xy)))
    for train,test in splits:
        assert not set(train)&set(test)
        if method in ("GroupKFold","BufferedGroupKFold"):
            assert not set(labels[train])&set(labels[test])
        if method=="BlockKFold":
            blocks=np.floor(xy/3000).astype(int)
            assert not set(map(tuple,blocks[train]))&set(map(tuple,blocks[test]))
        if method=="BufferedGroupKFold":
            assert cdist(xy[train],xy[test]).min()>600.
    result=train_horizon(survey,"Linear",features=["feature"],validation_method=method,
        group_column="profile",block_size=3000,buffer_size=600,cv_folds=3)
    assert len(result.validation)==len(survey)
    assert result.metrics["RMSE"]<1e-8
    assert set(result.validation.fold)=={1,2,3}
    assert len(result.settings["fold_metrics"])==3

def test_buffered_group_kfold_matches_group_control_and_filters_training(survey):
    # Two neighbouring profiles at exactly the buffer radius; other profiles
    # are well separated. Interleave rows so plain KFold would split profiles.
    survey=survey.copy()
    profile=survey.profile.to_numpy()
    survey["X"]=np.where(profile<=1000.,profile,profile*5.)
    survey=survey.sample(frac=1,random_state=7).reset_index(drop=True)
    xy=survey[["X","Y"]].to_numpy()
    labels=survey.profile.to_numpy()
    plain=make_splits("GroupKFold",xy,3,labels=labels)
    first=make_splits("BufferedGroupKFold",xy,3,labels=labels,block_size=0,buffer_size=1000.)
    second=make_splits("BufferedGroupKFold",xy,3,labels=labels,block_size=1e9,buffer_size=1000.)
    removed=0
    for (train,test),(other_train,other_test),(raw_train,raw_test) in zip(first,second,plain):
        np.testing.assert_array_equal(test,raw_test)
        np.testing.assert_array_equal(train,other_train)
        np.testing.assert_array_equal(test,other_test)
        expected=raw_train[cdist(xy[raw_train],xy[raw_test]).min(axis=1)>1000.]
        np.testing.assert_array_equal(train,expected)
        assert not set(labels[train])&set(labels[test])
        removed+=len(raw_train)-len(train)
    # Explicit boundary case independent of the group assignment/tie-breaking.
    boundary_xy=np.array([[0.,0.],[0.,10.],[1.,0.],[1.,10.],
                          [100.,0.],[100.,10.],[200.,0.],[200.,10.]])
    boundary_labels=np.array([0,0,1,1,2,2,3,3])
    base=make_splits("GroupKFold",boundary_xy,2,labels=boundary_labels)
    buffered=make_splits("BufferedGroupKFold",boundary_xy,2,labels=boundary_labels,buffer_size=1.)
    for (raw_train,raw_test),(train,test) in zip(base,buffered):
        np.testing.assert_array_equal(raw_test,test)
        np.testing.assert_array_equal(train,raw_train[cdist(boundary_xy[raw_train],boundary_xy[test]).min(axis=1)>1.])
    result=train_horizon(survey,"Linear",features=["feature"],validation_method="BufferedGroupKFold",
        group_column="profile",block_size=0,buffer_size=1000.,cv_folds=3)
    assert result.settings["block_size"] is None
    assert result.settings["block_origin"] is None
    assert result.settings["buffer_size"]==1000.
    assert result.settings["group_column"]=="profile"
    assert sum(f["excluded_rows"] for f in result.settings["fold_metrics"])==removed
    assert result.metrics["RMSE"]<1e-8


@pytest.mark.parametrize("method", ["BlockKFold", "BufferedGroupKFold"])
def test_nested_validation_outer_labels_do_not_leak(survey, method):
    args=dict(model_name="Ridge",features=["feature"],validation_method=method,
        group_column="profile",buffer_size=600.,block_size=3000,cv_folds=3,optimize=True,n_trials=2,search={"alpha":[.01,1.]})
    first=train_horizon(survey,**args)
    altered=survey.copy()
    held=first.settings["fold_splits"][0]["test"]
    altered.iloc[held,altered.columns.get_loc("depth")]+=10000
    second=train_horizon(altered,**args)
    np.testing.assert_allclose(first.validation.iloc[held].predicted,second.validation.iloc[held].predicted)
    assert first.settings["fold_metrics"][0]["parameters"]==second.settings["fold_metrics"][0]["parameters"]
    assert len(first.optimization_history)==2

def test_validation_rejects_impossible_splits(survey):
    xy=survey[["X","Y"]].to_numpy()
    with pytest.raises(ValueError):
        make_splits("BlockKFold",xy,3,block_size=1e9)
    with pytest.raises(ValueError):
        make_splits("BufferedGroupKFold",xy,3,labels=survey.profile.to_numpy(),buffer_size=1e9,block_size=3000)
    with pytest.raises(ValueError):
        make_splits("GroupKFold",xy,3,labels=None)
    with pytest.raises(ValueError, match="столбец групп"):
        make_splits("BufferedGroupKFold",xy,3,labels=None)
    with pytest.raises(ValueError, match="столбец групп"):
        train_horizon(survey,"Linear",features=["feature"],validation_method="BufferedGroupKFold",cv_folds=3)

def test_defaults_and_mlp_device():
    assert model_defaults("Ridge")["alpha"]==1.
    assert [s["units"] for s in model_defaults("MLP")["layer_specs"] if s["type"]=="Dense"]==[128,64,32]
    assert model_defaults("MLP")==model_defaults("MLP",device="gpu")
    assert device_parameters("MLP","gpu",1)["device"]=="cuda:1"
    assert "learning_rate" in default_space("MLP","gpu")
    assert default_space("MLP","cpu")==default_space("MLP","gpu")
    assert "device" not in model_defaults("MLP")
    make_model("MLP (sklearn)",model_defaults("MLP (sklearn)"))
    if "MLP" in __import__("geogarmodel.models",fromlist=["available_models"]).available_models():
        make_model("MLP",model_defaults("MLP"))

def test_torch_mlp_architecture_cpu():
    pytest.importorskip("torch")
    from geogarmodel.neural import TorchRegressor
    x=np.arange(40.).reshape(20,2)/40
    model=TorchRegressor(architecture="mlp",width=8,layers=2,epochs=1).fit(x,x[:,0])
    assert model.predict(x).shape==(20,)

def test_validation_ui_and_parameter_editor(qt_application_lifetime):
    from geogarmodel.app import MainWindow
    from geogarmodel.ml_settings_ui import ParameterDialog
    from geogarmodel.models import MODEL_NAMES
    window=MainWindow()
    try:
        assert [window.model_choice.itemText(i) for i in range(window.model_choice.count())]==MODEL_NAMES
        window.training=pd.DataFrame(dict(X=np.arange(12),Y=np.arange(12),line=np.arange(12)//3,feature=np.arange(12),depth=np.arange(12)))
        window.configure_columns()
        window.validation_method.setCurrentText("GroupKFold")
        window.group_column.setCurrentText("line")
        assert "line" not in [window.feature_list.item(i).text() for i in range(window.feature_list.count())]
        window.validation_method.setCurrentText("BufferedGroupKFold")
        assert not window.buffer_size.isHidden()
        assert not window.group_column.isHidden()
        assert "line" not in [window.feature_list.item(i).text() for i in range(window.feature_list.count())]
        assert window.validation_method.findText("BufferedKFold")==-1
        assert window.block.isHidden()
        assert window.parameters.isHidden()
        window.validation_method.setCurrentText("BlockKFold")
        assert not window.block.isHidden()
        assert window.buffer_size.isHidden()
        window.model_choice.setCurrentText("MLP")
        assert window.device_choice.model().item(1).isEnabled()
        dialog=ParameterDialog("Ridge",42,"cpu",0,{},window)
        assert dialog.checked()["alpha"]==1.
        dialog.show()
        qt_application_lifetime.processEvents()
        dialog.help.resizeRowsToContents()
        assert dialog.help.verticalHeader().isHidden()
        assert all(dialog.help.rowHeight(i)>=dialog.help.fontMetrics().height()+14
                   for i in range(dialog.help.rowCount()))
        assert all(dialog.help.item(i,2).toolTip()==dialog.help.item(i,2).text()
                   for i in range(dialog.help.rowCount()))
        dialog.editor.setPlainText('{"alpha":0.25}')
        dialog.apply()
        assert dialog.parameters=={"alpha":0.25}
        dialog.close()
    finally:
        window.close()


def test_old_buffered_project_requires_explicit_group_selection():
    from geogarmodel.project_ui import migrate_ml_controls
    old={"validation_method":"BufferedKFold","group_column":"stale","buffer_size":2000.}
    migrated=migrate_ml_controls(old)
    assert migrated=={"validation_method":"BufferedGroupKFold","group_column":"","buffer_size":2000.}
    assert old["validation_method"]=="BufferedKFold"
    assert migrate_ml_controls(migrated)==migrated
