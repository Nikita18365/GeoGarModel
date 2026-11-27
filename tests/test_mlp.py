import numpy as np
import pytest
from geogarmodel.mlp_config import defaults,validate_config,OPTIMIZERS,LOSSES
from geogarmodel.models import model_defaults,make_model,evaluation_scores

def test_mlp_config_validation():
    cfg=validate_config({"optimizer_name":"SGD","loss_name":"Huber"})
    assert cfg["optimizer_params"]["momentum"]==0.
    assert cfg["loss_params"]["delta"]==1.
    assert model_defaults("MLP")==model_defaults("MLP",device="gpu")
    with pytest.raises(ValueError):
        validate_config({"layer_specs":[{"type":"Dropout","rate":1.}]})
    with pytest.raises(ValueError):
        validate_config({"layer_specs":[{"type":"Dense","units":0}]})
    with pytest.raises(ValueError):
        validate_config({"optimizer_name":"SGD","optimizer_params":{"nesterov":True}})
    scores=evaluation_scores([1.,2.,3.],[1.,3.,5.],"MLP",{"evaluation_metrics":["MedianAE","ExplainedVariance"]})
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
    model=make_model("MLP",{"epochs":1,"loss_name":loss,"layer_specs":[{"type":"Dense","units":4}]})
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
