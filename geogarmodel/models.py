"""Horizon regression with a single preprocessing pipeline and isolated holdout."""
from dataclasses import dataclass, field
from importlib.util import find_spec
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .compute import device_parameters, RESERVED, check_cancel, verify_device

MODEL_NAMES = ["Ridge", "Linear", "Lasso", "Random Forest", "Extra Trees", "Gradient Boosting",
               "Histogram Boosting", "MLP", "XGBoost", "LightGBM", "CatBoost", "ResNet", "Transformer"]
OPTIONAL = {"XGBoost": "xgboost", "LightGBM": "lightgbm", "CatBoost": "catboost",
            "ResNet": "torch", "Transformer": "torch"}


def available_models():
    return [n for n in MODEL_NAMES if n not in OPTIONAL or find_spec(OPTIONAL[n]) is not None]


def make_model(name, parameters=None, seed=42, device="cpu", gpu_id=0):
    params = dict(parameters or {})
    if RESERVED & params.keys():
        raise ValueError("Устройство задаётся через CPU/GPU, а не параметры модели.")
    compute = device_parameters(name, device, gpu_id)
    if name in OPTIONAL and find_spec(OPTIONAL[name]) is None:
        raise ValueError(f"Для {name} установите пакет {OPTIONAL[name]} (см. docs/MODERNIZATION.md).")
    factories = {
        "Ridge": lambda: Ridge(), "Linear": lambda: LinearRegression(),
        "Lasso": lambda: Lasso(max_iter=10000),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "Extra Trees": lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=seed),
        "Histogram Boosting": lambda: HistGradientBoostingRegressor(random_state=seed, early_stopping=False),
        "MLP": lambda: MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=seed),
    }
    if name in factories:
        estimator = factories[name]()
    elif name == "XGBoost":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=.05, random_state=seed, n_jobs=-1)
    elif name == "LightGBM":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(n_estimators=400, verbosity=-1, random_state=seed,
                                  n_jobs=1, force_col_wise=True)
    elif name == "CatBoost":
        from catboost import CatBoostRegressor
        estimator = CatBoostRegressor(iterations=400, verbose=False, random_seed=seed, allow_writing_files=False)
    elif name in {"Transformer", "ResNet"}:
        from .neural import TorchRegressor
        estimator = TorchRegressor(architecture=name.lower(), random_state=seed)
    else:
        raise ValueError(f"Неизвестная модель: {name}")
    estimator.set_params(**params)
    estimator.set_params(**compute)
    # Imputation/scaling is fitted within each fold, never to the prediction area.
    return TransformedTargetRegressor(
        regressor=make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), estimator),
        transformer=StandardScaler())


def read_table(path):
    if str(path).lower().endswith(".xlsx"):
        table = pd.read_excel(path)
    else:
        table = pd.read_csv(path, sep=None, engine="python")
    table.columns = table.columns.map(str)
    if table.columns.duplicated().any() or not table.index.is_unique:
        raise ValueError("Имена столбцов и индексы должны быть уникальны.")
    return table


def numeric_features(table, features):
    missing = set(features) - set(table.columns)
    if missing:
        raise ValueError(f"Отсутствуют признаки: {sorted(missing)}")
    result = table.loc[:, features].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Признаки содержат Inf.")
    return result


@dataclass
class TrainingResult:
    model: object
    features: list
    target: str
    coordinates: list
    model_name: str
    metrics: dict
    validation: pd.DataFrame
    settings: dict
    optimization_history: list = field(default_factory=list)

    def predict(self, table):
        values = numeric_features(table, self.features)
        coords = table.loc[:, self.coordinates].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(coords.to_numpy()).all():
            raise ValueError("Координаты прогнозной выборки должны быть конечными.")
        output = coords.copy()
        output[self.target + "_pred"] = self.model.predict(values)
        return output


def train_horizon(table, model_name="Histogram Boosting", *, target=None, coordinates=None,
                  features=None, test_size=.2, spatial=True, block_size=10000., seed=42,
                  parameters=None, search=None, optimize=False, n_trials=30, timeout=None,
                  cv_folds=3, device="cpu", gpu_id=0, progress=None, cancel_event=None):
    check_cancel(cancel_event)
    if table.columns.duplicated().any() or not table.index.is_unique:
        raise ValueError("Имена столбцов и индексы должны быть уникальны.")
    coordinates = list(coordinates or table.columns[:2])
    target = target or table.columns[-1]
    features = list(features if features is not None else
                    [c for c in table.columns if c not in coordinates + [target]])
    if len(coordinates) != 2 or len(set(coordinates)) != 2 or target in coordinates:
        raise ValueError("Выберите два различных столбца координат и отдельную цель.")
    if not features or target in features or set(features) & set(coordinates):
        raise ValueError("Признаки должны исключать координаты и целевую переменную.")
    x = numeric_features(table, features)
    y = pd.to_numeric(table[target], errors="raise").astype(float)
    xy = table[coordinates].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if len(table) < 10 or not np.isfinite(y).all() or not np.isfinite(xy).all():
        raise ValueError("Нужно ≥ 10 строк с конечными координатами и глубинами.")
    if not 0 < test_size < 1:
        raise ValueError("Доля контроля должна быть между 0 и 1.")
    groups = None
    if spatial:
        if not np.isfinite(block_size) or block_size <= 0:
            raise ValueError("Размер пространственного блока должен быть > 0.")
        blocks = np.floor((xy - xy.min(axis=0)) / block_size).astype(np.int64)
        _, groups = np.unique(blocks, axis=0, return_inverse=True)
        if len(np.unique(groups)) < 4:
            raise ValueError("Нужно ≥ 4 пространственных блока. Уменьшите размер блока.")
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(x, y, groups))
    else:
        train, test = train_test_split(np.arange(len(x)), test_size=test_size, random_state=seed)
    if min(len(train), len(test)) < 2:
        raise ValueError("Недостаточно точек в обучении или контроле.")
    if x.iloc[train].isna().all().any():
        raise ValueError("В обучающей части есть полностью пустой признак.")
    model = make_model(model_name, parameters, seed, device, gpu_id)
    best = {}
    history, best_cv = [], None
    if optimize or search:
        from .optimization import default_space, optimize_model
        if not isinstance(cv_folds, int) or cv_folds < 2:
            raise ValueError("Для CV нужно минимум 2 фолда.")
        if spatial:
            folds = min(cv_folds, len(np.unique(groups[train])))
            if folds < 2:
                raise ValueError("Недостаточно обучающих блоков для подбора.")
            cv = GroupKFold(folds)
        else:
            cv = KFold(min(cv_folds, len(train)), shuffle=True, random_state=seed)
        inner_x, inner_y = x.iloc[train], y.iloc[train]
        splits = list(cv.split(inner_x, inner_y, groups=None if groups is None else groups[train]))
        best, history, best_cv = optimize_model(
            lambda p: make_model(model_name, p, seed, device, gpu_id), inner_x, inner_y, splits,
            model_name=model_name, device=device, space=search or default_space(model_name), fixed=parameters,
            n_trials=n_trials, timeout=timeout, seed=seed, progress=progress, cancel_event=cancel_event)
        model = make_model(model_name, {**(parameters or {}), **best}, seed, device, gpu_id)
    check_cancel(cancel_event)
    model.fit(x.iloc[train], y.iloc[train])
    verify_device(model, model_name, device)
    predicted = model.predict(x.iloc[test])
    mse = mean_squared_error(y.iloc[test], predicted)
    metrics = {"MAE": float(mean_absolute_error(y.iloc[test], predicted)), "RMSE": float(np.sqrt(mse)),
               "MSE": float(mse), "R2": float(r2_score(y.iloc[test], predicted)),
               "train_rows": len(train), "test_rows": len(test)}
    validation = table.iloc[test][coordinates].copy()
    validation["observed"] = y.iloc[test].to_numpy()
    validation["predicted"] = predicted
    validation["residual_pred_minus_observed"] = predicted - y.iloc[test].to_numpy()
    settings = {"spatial": spatial, "block_size": block_size, "seed": seed,
                "test_size": test_size, "parameters": parameters or {}, "best_parameters": best,
                "search": search or {}, "train_indices": train.tolist(), "test_indices": test.tolist(),
                "optimizer": "Optuna TPE" if history else None, "best_cv_rmse": best_cv,
                "device": device, "gpu_id": gpu_id, "n_trials": n_trials, "cv_folds": cv_folds,
                "timeout_seconds": timeout}
    # Final deployment model uses all labelled observations AFTER holdout scoring.
    check_cancel(cancel_event)
    final_model = clone(model).fit(x, y)
    verify_device(final_model, model_name, device)
    check_cancel(cancel_event)
    return TrainingResult(final_model, features, target, coordinates, model_name, metrics, validation, settings, history)
