import numpy as np
import pytest
from geogarmodel.mlp_config import defaults,validate_config,OPTIMIZERS,LOSSES
from geogarmodel.models import model_defaults,make_model,evaluation_scores

def test_mlp_config_validation():
    cfg=validate_config({"optimizer_name":"SGD","loss_name":"Huber"})
    assert cfg["optimizer_params"]["momentum"]==0.
    assert cfg["loss_params"]["delta"]==1.
    assert model_defaults("MLP (pytorch)")==model_defaults("MLP (pytorch)",device="gpu")
    with pytest.raises(ValueError):
        validate_config({"layer_specs":[{"type":"Dropout","rate":1.}]})
    with pytest.raises(ValueError):
        validate_config({"layer_specs":[{"type":"Dense","units":0}]})
    with pytest.raises(ValueError):
        validate_config({"optimizer_name":"SGD","optimizer_params":{"nesterov":True}})
    scores=evaluation_scores([1.,2.,3.],[1.,3.,5.],"MLP (pytorch)",{"evaluation_metrics":["MedianAE","ExplainedVariance"]})
    assert scores["MedianAE"]==1.
    assert "ExplainedVariance" in scores

@pytest.mark.parametrize("optimizer",list(OPTIMIZERS))
def test_mlp_order_penalty_training_and_clone(optimizer):
    torch=pytest.importorskip("torch")
    from sklearn.base import clone
    from geogarmodel.mlp import ConfigurableMLP
    specs=[{"type":"Dense","units":8,"activation":"gelu","l1":.001,"l2":.002},
           {"type":"Dropout","rate":.2},{"type":"LayerNorm"},
           {"type":"Dense","units":4,"activation":"tanh"}]
    model=ConfigurableMLP(layer_specs=specs,optimizer_name=optimizer,loss_name="Huber",
                          epochs=2,batch_size=8,output_l2=.003)
    x=np.arange(40,dtype=np.float32).reshape(20,2)/40
    y=x[:,0]
    fitted=clone(model).fit(x,y)
    assert [type(l).__name__ for l in fitted.network_]==["Linear","GELU","Dropout","LayerNorm","Linear","Tanh","Linear","Flatten"]
    assert fitted.network_[2].p==.2
    expected=.001*fitted.network_[0].weight.abs().sum()+.002*fitted.network_[0].weight.square().sum()+.003*fitted.network_[6].weight.square().sum()
    torch.testing.assert_close(fitted.penalty(),expected)
    assert len(fitted.loss_curve_)==2
    assert len(fitted.metric_curves_["RMSE"])==2
    assert np.isfinite(fitted.predict(x)).all()

@pytest.mark.parametrize("loss",list(LOSSES))
def test_mlp_losses(loss):
    pytest.importorskip("torch")
    x=np.arange(24,dtype=float).reshape(12,2)/24
    model=make_model("MLP (pytorch)",{"epochs":1,"loss_name":loss,"layer_specs":[{"type":"Dense","units":4}]})
    model.fit(x,x[:,0])
    assert np.isfinite(model.predict(x)).all()

def test_mlp_editor(qt_application_lifetime):
    from geogarmodel.mlp_ui import MLPDialog
    dialog=MLPDialog(42,{})
    try:
        dialog.add_layer("Dropout")
        dialog.move(-1)
        assert dialog.collect()["layer_specs"][-2]["type"]=="Dropout"
        dialog.optimizer.setCurrentText("SGD")
        dialog.loss.setCurrentText("MAE")
        dialog.tabs.setCurrentIndex(3)
        assert dialog.checked()["optimizer_name"]=="SGD"
        assert dialog.checked()["loss_name"]=="MAE"
        dialog.apply()
        assert dialog.parameters["optimizer_name"]=="SGD"
    finally:
        dialog.close()


def test_batchnorm_and_separate_activation_roundtrip():
    torch=pytest.importorskip("torch")
    from geogarmodel.model_persistence import dump_pipeline, StoredPipeline
    specs=[{"type":"Dense","units":8,"activation":"linear"},
           {"type":"BatchNorm","momentum":.2,"eps":1e-4},
           {"type":"Activation","activation":"gelu"},
           {"type":"LayerNorm","eps":1e-4,"bias":False},
           {"type":"Dense","units":4,"activation":"linear"},
           {"type":"Activation","activation":"tanh"}]
    x=np.arange(34,dtype=np.float32).reshape(17,2)/34
    model=make_model("MLP (pytorch)",{"layer_specs":specs,"epochs":2,"batch_size":8})
    model.fit(x,x[:,0])
    net=model.regressor_.steps[-1][1].network_
    bn=net[2]
    assert isinstance(bn,torch.nn.BatchNorm1d)
    assert isinstance(net[3],torch.nn.GELU)
    assert bn.num_batches_tracked.item()==4
    before=bn.running_mean.clone()
    expected=model.predict(x)
    np.testing.assert_allclose(model.predict(x[:1]),expected[:1],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(bn.running_mean,before)
    restored=StoredPipeline(*dump_pipeline(model))
    np.testing.assert_allclose(restored.predict(x),expected,rtol=1e-5,atol=1e-6)


def test_batchnorm_batch_sampler_preserves_all_rows():
    torch=pytest.importorskip("torch")
    from geogarmodel.mlp import NonSingletonBatches
    for n in (2,3,8,9,17,25):
        for size in (2,4,8,128):
            batches=NonSingletonBatches(torch.utils.data.BatchSampler(range(n),size,False))
            actual=list(batches)
            assert len(actual)==len(batches)
            assert min(map(len,actual))>=2
            assert [i for batch in actual for i in batch]==list(range(n))


def test_normalization_config_rejects_invalid_options():
    specs=[{"type":"Dense","units":8},{"type":"BatchNorm"}]
    with pytest.raises(ValueError,match="batch_size"):
        validate_config({"layer_specs":specs,"batch_size":1})
    with pytest.raises(ValueError,match="track_running_stats"):
        validate_config({"layer_specs":[specs[0],{"type":"BatchNorm","track_running_stats":False}]})
    with pytest.raises(ValueError,match="eps"):
        validate_config({"layer_specs":[specs[0],{"type":"LayerNorm","eps":0}]})


def test_editor_explicit_activation_and_normalization(qt_application_lifetime):
    from geogarmodel.mlp_ui import MLPDialog
    dialog=MLPDialog(42,{})
    try:
        dialog.add_layer("Dense")
        dialog.add_layer("BatchNorm")
        dialog.add_layer("Activation")
        dialog.layers.item(dialog.layers.rowCount()-1,2).setText("gelu")
        specs=dialog.collect()["layer_specs"]
        assert [s["type"] for s in specs[-3:]]==["Dense","BatchNorm","Activation"]
        assert specs[-3]["activation"]=="linear"
        assert specs[-2]["track_running_stats"] is True
        assert specs[-1]["activation"]=="gelu"
        dialog.tabs.setCurrentIndex(3)
        assert dialog.checked()["layer_specs"]==specs
    finally:
        dialog.close()
