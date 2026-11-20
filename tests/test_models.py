import numpy as np
import pandas as pd
import pytest
from geogarmodel.models import train_horizon, make_model


@pytest.fixture
def table():
    rng = np.random.default_rng(7)
    x, y = np.meshgrid(np.arange(15) * 1000., np.arange(12) * 1000.)
    f = rng.normal(size=(x.size, 3))
    data = pd.DataFrame({"X": x.ravel(), "Y": y.ravel(), "gravity": f[:, 0],
                         "magnetic": f[:, 1], "seismic": f[:, 2],
                         "horizon": 1200 + 100 * f[:, 0] - 40 * f[:, 1] + 10 * f[:, 2]})
    return data


def test_spatial_holdout_and_schema(table):
    result = train_horizon(table, "Linear", block_size=3000)
    assert result.metrics["RMSE"] < 1e-9
    train = result.settings["train_indices"]
    test = result.settings["test_indices"]
    blocks = np.floor(table[["X", "Y"]].to_numpy() / 3000).astype(int)
    assert not (set(map(tuple, blocks[train])) & set(map(tuple, blocks[test])))
    prediction = table.drop(columns="horizon")
    # Different column order must not change the model input schema.
    a = result.predict(prediction)
    b = result.predict(prediction[prediction.columns[::-1]])
    np.testing.assert_allclose(a, b)
    assert list(a.columns) == ["X", "Y", "horizon_pred"]
    with pytest.raises(ValueError, match="Отсутствуют"):
        result.predict(prediction.drop(columns="gravity"))


def test_inner_search_and_scaling_no_holdout_leakage(table):
    result = train_horizon(table, "Ridge", block_size=3000, search={"alpha": [.01, 1.]})
    assert "alpha" in result.settings["best_parameters"]
    assert result.metrics["RMSE"] < 2
    assert len(result.validation) == result.metrics["test_rows"]
    # Final deployment scaler must reflect all labelled training data.
    scaler = result.model.regressor_.steps[1][1]
    np.testing.assert_allclose(scaler.mean_, table[result.features].mean())


@pytest.mark.parametrize("name,params", [("Random Forest", {"n_estimators": 5}),
    ("Extra Trees", {"n_estimators": 5}), ("Gradient Boosting", {"n_estimators": 5}),
    ("Histogram Boosting", {"max_iter": 5}), ("Lasso", {}), ("MLP", {"max_iter": 5})])
def test_base_estimators(table, name, params):
    result = train_horizon(table, name, parameters=params, spatial=False)
    assert np.isfinite(result.predict(table)["horizon_pred"]).all()


def test_imputation_and_bad_inputs(table):
    table.loc[0, "gravity"] = np.nan
    result = train_horizon(table, "Ridge", spatial=False)
    assert np.isfinite(result.predict(table).to_numpy()).all()
    with pytest.raises(ValueError):
        train_horizon(table, features=["horizon"])
    with pytest.raises(ValueError):
        train_horizon(table, block_size=1e9)
    table.loc[0, "horizon"] = np.nan
    with pytest.raises(ValueError):
        train_horizon(table)


@pytest.mark.parametrize("name", ["Transformer", "ResNet"])
def test_optional_neural(table, name):
    pytest.importorskip("torch")
    result = train_horizon(table, name, spatial=False, parameters={"epochs": 2, "width": 8, "layers": 1})
    assert np.isfinite(result.predict(table).to_numpy()).all()


@pytest.mark.parametrize("name,module,params", [
    ("XGBoost", "xgboost", {"n_estimators": 5}),
    ("LightGBM", "lightgbm", {"n_estimators": 5}),
    ("CatBoost", "catboost", {"iterations": 5}),
])
def test_optional_boosting(table, name, module, params):
    pytest.importorskip(module)
    result = train_horizon(table, name, spatial=False, parameters=params)
    assert np.isfinite(result.predict(table).to_numpy()).all()
