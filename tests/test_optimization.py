from threading import Event
import os
import numpy as np
import pytest
from geogarmodel.demo import demo_survey
from geogarmodel.models import train_horizon, make_model
from geogarmodel.compute import device_parameters, verify_device, ComputationCancelled
from geogarmodel.optimization import default_space, suggest_parameters
from optuna.trial import FixedTrial


def test_optuna_holdout_isolation_and_progress():
    _, table, _ = demo_survey()
    progress = []
    args = dict(model_name="Ridge", block_size=3000, optimize=True, n_trials=4, cv_folds=3,
                search={"alpha": {"type": "float", "low": .001, "high": 10., "log": True}})
    first = train_horizon(table, progress=progress.append, **args)
    changed = table.copy()
    changed.loc[first.settings["test_indices"], "Depth"] += 1e5
    second = train_horizon(changed, **args)
    assert len(progress) == 4
    assert all(r["state"] == "COMPLETE" for r in progress)
    assert all(len(r["fold_rmse"]) == 3 for r in progress)
    assert np.all(np.diff([r["best_rmse"] for r in progress]) <= 0)
    np.testing.assert_allclose([r["rmse"] for r in first.optimization_history],
                               [r["rmse"] for r in second.optimization_history])
    assert first.settings["best_parameters"] == second.settings["best_parameters"]
    assert second.metrics["RMSE"] > 90000


def test_cancellation_does_not_return_partial_model():
    _, table, _ = demo_survey()
    event = Event()
    with pytest.raises(ComputationCancelled):
        train_horizon(table, "Ridge", block_size=3000, optimize=True, n_trials=5,
                      progress=lambda _: event.set(), cancel_event=event)


def test_search_schema_and_device_parameters():
    spec = {"n_estimators": {"type": "int", "low": 5, "high": 10}, "max_depth": [None, 3]}
    assert suggest_parameters(FixedTrial({"n_estimators": 6, "max_depth": None}), spec) == {
        "n_estimators": 6, "max_depth": None}
    with pytest.raises(ValueError):
        suggest_parameters(FixedTrial({"device": "cuda"}), {"device": ["cuda"]})
    with pytest.raises(ValueError):
        make_model("Ridge", device="gpu")
    with pytest.raises(ValueError):
        make_model("Ridge", {"device": "cpu"})
    assert device_parameters("XGBoost", "gpu", 2)["device"] == "cuda:2"
    assert device_parameters("CatBoost", "gpu", 1)["devices"] == "1"
    assert device_parameters("LightGBM", "gpu")["device_type"] == "gpu"
    assert device_parameters("Transformer", "gpu")["device"] == "cuda:0"
    for name in ("Ridge", "MLP", "Transformer", "LightGBM"):
        assert default_space(name)


def test_xgboost_silent_fallback_is_rejected():
    class Fake:
        def save_config(self):
            return '{"learner":{"generic_param":{"device":"cpu"}}}'
        def get_booster(self):
            return self
        @property
        def regressor_(self):
            return self
        @property
        def steps(self):
            return [("model", self)]
    with pytest.raises(RuntimeError, match="CPU"):
        verify_device(Fake(), "XGBoost", "gpu")


@pytest.mark.skipif(os.environ.get("GEOGARMODEL_GPU_TESTS") != "1", reason="Explicit GPU smoke test")
def test_real_xgboost_gpu_optuna():
    import json
    pytest.importorskip("xgboost")
    _, table, prediction = demo_survey()
    result = train_horizon(table, "XGBoost", block_size=3000, device="gpu", optimize=True,
                           n_trials=2, cv_folds=2, parameters={"n_estimators": 10}, search={"max_depth": [2, 3]})
    estimator = result.model.regressor_.steps[-1][1]
    assert json.loads(estimator.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda")
    assert np.isfinite(result.predict(prediction).to_numpy()).all()
