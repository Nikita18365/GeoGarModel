"""Horizon regression with a single preprocessing pipeline and isolated holdout."""
from dataclasses import dataclass
from importlib.util import find_spec
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold, train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["Ridge", "Linear", "Lasso", "Random Forest", "Extra Trees", "Gradient Boosting",
               "Histogram Boosting", "MLP", "XGBoost", "LightGBM", "CatBoost", "ResNet", "Transformer"]
OPTIONAL = {"XGBoost": "xgboost", "LightGBM": "lightgbm", "CatBoost": "catboost",
            "ResNet": "torch", "Transformer": "torch"}


def available_models():
    return [n for n in MODEL_NAMES if n not in OPTIONAL or find_spec(OPTIONAL[n]) is not None]


def make_model(name, parameters=None, seed=42):
    params = dict(parameters or {})
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
        estimator = LGBMRegressor(n_estimators=400, verbosity=-1, random_state=seed)
    elif name == "CatBoost":
        from catboost import CatBoostRegressor
        estimator = CatBoostRegressor(iterations=400, verbose=False, random_seed=seed, allow_writing_files=False)
    elif name in {"Transformer", "ResNet"}:
        from .neural import TorchRegressor
        estimator = TorchRegressor(architecture=name.lower(), random_state=seed)
    else:
        raise ValueError(f"Неизвестная модель: {name}")
    estimator.set_params(**params)
    # Imputation/scaling is fitted within each fold, never to the prediction area.
    return TransformedTargetRegressor(
        regressor=make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), estimator),
        transformer=StandardScaler())


def read_table(path):
    if str(path).lower().endswith(".xlsx"):
        table = pd.read_excel(path)
    else:
        table = pd.read_csv(path, sep=None, engine="python")
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
                  parameters=None, search=None):
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
    model = make_model(model_name, parameters, seed)
    best = {}
    if search:
        prefix = "regressor__" + model.regressor.steps[-1][0] + "__"
        if spatial:
            folds = min(5, len(np.unique(groups[train])))
            if folds < 2:
                raise ValueError("Недостаточно обучающих блоков для подбора.")
            cv = GroupKFold(folds)
        else:
            cv = KFold(min(5, len(train)), shuffle=True, random_state=seed)
        tuner = GridSearchCV(model, {prefix + k: v for k, v in search.items()}, cv=cv,
                             scoring="neg_mean_squared_error", n_jobs=1, error_score="raise")
        tuner.fit(x.iloc[train], y.iloc[train], groups=None if groups is None else groups[train])
        model = tuner.best_estimator_
        best = {k.removeprefix(prefix): v for k, v in tuner.best_params_.items()}
    else:
        model.fit(x.iloc[train], y.iloc[train])
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
                "search": search or {}, "train_indices": train.tolist(), "test_indices": test.tolist()}
    # Final deployment model uses all labelled observations AFTER holdout scoring.
    final_model = clone(model).fit(x, y)
    return TrainingResult(final_model, features, target, coordinates, model_name, metrics, validation, settings)
