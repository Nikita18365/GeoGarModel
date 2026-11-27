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
from .validation import make_splits, SPATIAL_METHODS
from .work_progress import report

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
    if name == "MLP" and device == "gpu":
        if find_spec("torch") is None:
            raise ValueError("GPU MLP требует CUDA-сборку PyTorch (пакет torch).")
        from .neural import TorchRegressor
        estimator = TorchRegressor(architecture="mlp", width=128, layers=3, epochs=100, random_state=seed)
    elif name in factories:
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
    if name in {"MLP", "ResNet", "Transformer"} and "architecture" in params:
        raise ValueError("Архитектура определяется выбранной моделью.")
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
                  cv_folds=3, device="cpu", gpu_id=0, progress=None, cancel_event=None,
                  validation_method=None, group_column=None, buffer_size=1000.):
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
    if validation_method is not None:
        return cross_validate_horizon(table, x, y, xy, model_name, features, target, coordinates,
            validation_method, group_column, block_size, buffer_size, cv_folds, seed,
            parameters, search, optimize, n_trials, timeout, device, gpu_id, progress, cancel_event)
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
            model_name=model_name, device=device, space=search or default_space(model_name, device), fixed=parameters,
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


def cross_validate_horizon(table, x, y, xy, name, features, target, coordinates,
                           method, group_column, block_size, buffer_size, folds, seed,
                           parameters, search, optimize, trials, timeout, device, gpu_id,
                           progress, cancel_event):
    """Nested CV: external labels never enter the corresponding inner search.
    Final full-data search is distinct from outer evaluation and used for deployment.
    """
    from .optimization import default_space, optimize_model
    if method == "GroupKFold":
        if group_column not in table.columns or group_column in coordinates + [target]:
            raise ValueError("Выберите отдельный столбец идентификаторов групп.")
        if group_column in features:
            raise ValueError("Исключите столбец групп из предикторов.")
        labels = table[group_column].to_numpy()
    else:
        labels = None
    origin = xy.min(axis=0)
    options = dict(block_size=block_size, buffer_size=buffer_size, seed=seed, origin=origin)
    splits = make_splits(method, xy, folds, labels=labels, **options)
    do_search = bool(optimize or search)
    factory = lambda p: make_model(name, p, seed, device, gpu_id)
    fixed = dict(parameters or {})
    space = search or default_space(name, device) if do_search else {}
    oof = np.full(len(x), np.nan)
    fold_ids = np.zeros(len(x), dtype=int)
    scores = []
    # Preflight all inner splits before any expensive fits.
    inner_splits = [make_splits(method, xy[train], folds,
                       labels=None if labels is None else labels[train],
                       inner=True, **options) for train, _ in splits] if do_search else []
    for i, (train, valid) in enumerate(splits):
        if do_search and any(x.iloc[train[inner_train]].isna().all().any() for inner_train, _ in inner_splits[i]):
            raise ValueError(f"Внутренний фолд {i+1}: полностью пустой обучающий признак.")
    for i, (train, valid) in enumerate(splits):
        check_cancel(cancel_event)
        report(i, folds, f"Внешняя валидация · фолд {i+1}/{folds}")
        if x.iloc[train].isna().all().any():
            raise ValueError(f"Фолд {i+1}: полностью пустой обучающий признак.")
        best = {}
        if do_search:
            best, _, _ = optimize_model(factory, x.iloc[train], y.iloc[train], inner_splits[i],
                model_name=name, device=device, space=space, fixed=fixed, n_trials=trials,
                timeout=timeout, seed=seed, cancel_event=cancel_event)
        estimator = factory({**fixed, **best})
        estimator.fit(x.iloc[train], y.iloc[train])
        verify_device(estimator, name, device)
        pred = estimator.predict(x.iloc[valid])
        oof[valid] = pred
        fold_ids[valid] = i + 1
        scores.append(dict(fold=i+1, train_rows=len(train), test_rows=len(valid),
                           excluded_rows=len(x)-len(train)-len(valid),
                           rmse=float(np.sqrt(mean_squared_error(y.iloc[valid], pred))),
                           mae=float(mean_absolute_error(y.iloc[valid], pred)),
                           r2=float(r2_score(y.iloc[valid], pred)), parameters=best,
                           inner_folds=len(inner_splits[i]) if do_search else 0))
    check_cancel(cancel_event)
    mse = mean_squared_error(y, oof)
    metrics = dict(MAE=float(mean_absolute_error(y, oof)), RMSE=float(np.sqrt(mse)),
                   MSE=float(mse), R2=float(r2_score(y, oof)),
                   train_rows=len(x), test_rows=len(x))
    validation = table[coordinates].copy()
    validation["observed"] = y.to_numpy()
    validation["predicted"] = oof
    validation["residual_pred_minus_observed"] = oof-y.to_numpy()
    validation["fold"] = fold_ids
    best, history, best_cv = {}, [], None
    if do_search:
        report(0, trials, "Optuna · итоговый подбор на всех данных")
        best, history, best_cv = optimize_model(factory, x, y, splits,
            model_name=name, device=device, space=space, fixed=fixed, n_trials=trials,
            timeout=timeout, seed=seed, progress=progress, cancel_event=cancel_event)
    check_cancel(cancel_event)
    report(0, 1, "Обучение итоговой модели")
    final = factory({**fixed, **best}).fit(x, y)
    verify_device(final, name, device)
    check_cancel(cancel_event)
    settings = dict(validation_method=method, group_column=group_column, buffer_size=buffer_size,
        spatial=method in SPATIAL_METHODS or method=="GroupKFold", block_size=block_size,
        seed=seed, parameters=fixed, best_parameters=best, search=space,
        cv_folds=folds, fold_metrics=scores, optimizer="Optuna TPE" if do_search else None,
        best_cv_rmse=best_cv, device=device, gpu_id=gpu_id, n_trials=trials,
        timeout_seconds=timeout, evaluation="nested_oof" if do_search else "oof",
        block_origin=origin.tolist(), final_train_rows=len(x),
        fold_splits=[dict(train=train.tolist(), test=valid.tolist()) for train, valid in splits])
    return TrainingResult(final, features, target, coordinates, name, metrics, validation, settings, history)


def model_defaults(name, seed=42, device="cpu", gpu_id=0):
    estimator = make_model(name, seed=seed, device=device, gpu_id=gpu_id).regressor.steps[-1][1]
    result = {}
    for key, value in estimator.get_params(deep=False).items():
        if key in RESERVED or key == "architecture" or "__" in key:
            continue
        if isinstance(value, tuple):
            value = list(value)
        if value is None or isinstance(value, (str, int, bool, list)) or (
                isinstance(value, float) and np.isfinite(value)):
            result[key] = value
    if name == "CatBoost":
        for key in ("depth", "learning_rate", "l2_leaf_reg", "random_strength", "border_count"):
            result.setdefault(key, None)
    return result
