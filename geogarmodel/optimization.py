"""Optuna TPE search using fixed INNER folds and fold-local sklearn pipelines."""
import math
import time
import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.metrics import mean_squared_error
from .compute import RESERVED, check_cancel, verify_device


def default_space(name):
    def f(low, high, log=False):
        return {"type": "float", "low": low, "high": high, "log": log}
    def i(low, high):
        return {"type": "int", "low": low, "high": high}
    common = {"learning_rate": f(.01, .2, True)}
    spaces = {
        "Linear": {"fit_intercept": [True, False]},
        "Ridge": {"alpha": f(1e-4, 1e3, True)},
        "Lasso": {"alpha": f(1e-5, 1., True)},
        "Random Forest": {"n_estimators": i(100, 500), "max_depth": [None, 5, 10, 20], "min_samples_leaf": i(1, 10)},
        "Extra Trees": {"n_estimators": i(100, 500), "max_depth": [None, 5, 10, 20], "min_samples_leaf": i(1, 10)},
        "Gradient Boosting": {**common, "n_estimators": i(100, 500), "max_depth": i(2, 6)},
        "Histogram Boosting": {**common, "max_iter": i(100, 500), "max_leaf_nodes": i(8, 63), "l2_regularization": f(1e-5, 10, True)},
        "XGBoost": {**common, "n_estimators": i(100, 500), "max_depth": i(2, 8), "reg_lambda": f(1e-3, 10, True)},
        "LightGBM": {**common, "n_estimators": i(100, 500), "num_leaves": i(8, 63)},
        "CatBoost": {**common, "iterations": i(100, 500), "depth": i(3, 8), "l2_leaf_reg": f(.1, 10, True)},
        "MLP": {"alpha": f(1e-6, .1, True), "learning_rate_init": f(1e-4, .01, True)},
    }
    neural = {"width": [32, 64, 128], "layers": i(1, 4), "learning_rate": f(1e-4, .01, True), "dropout": f(0, .3)}
    spaces["ResNet"], spaces["Transformer"] = neural.copy(), neural.copy()
    return spaces[name]


def suggest_parameters(trial, space):
    if not isinstance(space, dict) or not space:
        raise ValueError("Нужно непустое пространство параметров Optuna.")
    result = {}
    for name, spec in space.items():
        if name in RESERVED or "__" in name:
            raise ValueError(f"Параметр {name} управляется выбором устройства / конвейером.")
        if isinstance(spec, list):
            if not spec or any(not isinstance(v, (str, int, float, bool, type(None))) for v in spec):
                raise ValueError(f"{name}: список должен содержать скалярные варианты.")
            result[name] = trial.suggest_categorical(name, spec)
        elif isinstance(spec, dict) and spec.get("type") in {"int", "float"}:
            kind = spec["type"]
            low, high = spec["low"], spec["high"]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (low, high)) or low > high:
                raise ValueError(f"{name}: некорректные границы.")
            log = bool(spec.get("log", False))
            if log and low <= 0:
                raise ValueError(f"{name}: логарифмический диапазон требует low > 0.")
            if kind == "int" and (int(low) != low or int(high) != high):
                raise ValueError(f"{name}: целочисленные границы обязательны.")
            kwargs = {"log": log}
            if "step" in spec:
                kwargs["step"] = spec["step"]
            suggest = trial.suggest_int if kind == "int" else trial.suggest_float
            result[name] = suggest(name, int(low) if kind == "int" else low,
                                   int(high) if kind == "int" else high, **kwargs)
        else:
            raise ValueError(f"{name}: задайте список или объект type/low/high.")
    return result


def optimize_model(factory, x, y, splits, *, model_name, device, space, fixed=None,
                   n_trials=30, timeout=None, seed=42, progress=None, cancel_event=None):
    if not isinstance(n_trials, int) or n_trials < 1 or (timeout is not None and (not np.isfinite(timeout) or timeout <= 0)):
        raise ValueError("Число trials и лимит времени должны быть > 0.")
    # Validate the distribution schema before starting expensive trials.
    suggest_parameters(optuna.trial.FixedTrial({key: (value[0] if isinstance(value, list) else value["low"])
                                               for key, value in space.items()}), space)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    history = []
    started = time.monotonic()

    def objective(trial):
        check_cancel(cancel_event)
        params = {**(fixed or {}), **suggest_parameters(trial, space)}
        squared, count, fold_scores = 0., 0, []
        for step, (train, valid) in enumerate(splits):
            check_cancel(cancel_event)
            estimator = factory(params)
            estimator.fit(x.iloc[train], y.iloc[train])
            verify_device(estimator, model_name, device)
            prediction = estimator.predict(x.iloc[valid])
            mse = mean_squared_error(y.iloc[valid], prediction)
            if not np.isfinite(mse):
                raise ValueError("Получена неконечная ошибка модели.")
            squared += mse * len(valid)
            count += len(valid)
            fold_scores.append(float(np.sqrt(mse)))
            trial.set_user_attr("fold_rmse", fold_scores)
            trial.report(float(np.sqrt(squared / count)), step)
            check_cancel(cancel_event)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.sqrt(squared / count))

    def callback(study, trial):
        complete = [t.value for t in study.trials if t.state == TrialState.COMPLETE]
        record = {"trial": trial.number, "state": trial.state.name, "rmse": trial.value,
                  "best_rmse": min(complete) if complete else None,
                  "seconds": trial.duration.total_seconds() if trial.duration else 0.,
                  "elapsed_seconds": time.monotonic() - started, "parameters": dict(trial.params),
                  "fold_rmse": trial.user_attrs.get("fold_rmse", [])}
        history.append(record)
        if progress:
            progress(dict(record))

    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1,
                   callbacks=[callback], gc_after_trial=True)
    check_cancel(cancel_event)
    return study.best_params, history, float(study.best_value)
