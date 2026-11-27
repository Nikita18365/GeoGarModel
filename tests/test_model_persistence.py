import numpy as np
import pytest
from geogarmodel.models import make_model
from geogarmodel.model_persistence import dump_pipeline,StoredPipeline,load_skops


@pytest.mark.parametrize('name,params,package',[
    ('Linear',{},None),('Ridge',{},None),('Lasso',{},None),('Random Forest',{'n_estimators':2},None),
    ('Extra Trees',{'n_estimators':2},None),('Gradient Boosting',{'n_estimators':2},None),
    ('Histogram Boosting',{'max_iter':2},None),('MLP (sklearn)',{'max_iter':2},None),('MLP (pytorch)',{'epochs':2,'layer_specs':[{'type':'Dense','units':8},{'type':'Dropout','rate':.2}]},'torch'),
    ('XGBoost',{'n_estimators':2},'xgboost'),('LightGBM',{'n_estimators':2},'lightgbm'),
    ('CatBoost',{'iterations':2},'catboost'),('ResNet',{'epochs':1,'width':8,'layers':1},'torch'),
    ('Transformer',{'epochs':1,'width':8,'layers':1},'torch')])
def test_fitted_pipeline_roundtrip_without_training(name,params,package):
    if package: pytest.importorskip(package)
    X=np.arange(80.).reshape(20,4); X[0,0]=np.nan; y=np.arange(20.)+np.sin(np.arange(20.))
    original=make_model(name,params); original.fit(X,y); expected=original.predict(X)
    meta,blobs=dump_pipeline(original); stored=StoredPipeline(meta,blobs)
    np.testing.assert_allclose(stored.predict(X),expected,rtol=1e-6,atol=1e-6)
    again=StoredPipeline(*dump_pipeline(stored)); np.testing.assert_allclose(again.predict(X),expected,rtol=1e-6,atol=1e-6)


def test_unexpected_skops_types_are_rejected():
    import skops.io as sio
    from pathlib import Path
    with pytest.raises(ValueError,match='Неизвестные типы'): load_skops(sio.dumps(Path('unexpected')))
