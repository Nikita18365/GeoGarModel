"""Optuna TPE search using fixed INNER folds and fold-local sklearn pipelines."""
import math
import time
import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.metrics import mean_squared_error
from .compute import RESERVED, check_cancel, verify_device
from .search_space import merge_suggestions, changed_parameters
from .conditional_search import sample_configuration


def default_space(name, device="cpu", fixed=None):
    """Conservative search domains, not the unrestricted library API."""
    from .search_space import recommended_space
    return recommended_space(name, device, fixed or {})


def suggest_parameters(trial, space):
    if not isinstance(space, dict) or not space:
        raise ValueError("Нужно непустое пространство параметров Optuna.")
    result = {}
    for name, spec in space.items():
        if name.split(".")[0] in RESERVED or "__" in name:
            raise ValueError(f"Параметр {name} управляется выбором устройства / конвейером.")
        if isinstance(spec, list):
            if not spec or any(not isinstance(v, (str, int, float, bool, type(None))) for v in spec):
                raise ValueError(f"{name}: список должен содержать скалярные варианты.")
            result[name] = trial.suggest_categorical(name, spec)
        elif isinstance(spec, dict) and spec.get("type") in {"int", "float"}:
            kind = spec["type"]
            low, high = spec["low"], spec["high"]
            if not all(isinstance(v, (int, float)) and not isinstance(v,bool) and math.isfinite(v) for v in (low, high)) or low > high:
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
                   n_trials=30, timeout=None, seed=42, progress=None, cancel_event=None, best_oof=None):
    if not isinstance(n_trials, int) or n_trials < 1 or (timeout is not None and (not np.isfinite(timeout) or timeout <= 0)):
        raise ValueError("Число trials и лимит времени должны быть > 0.")
    # Validate the distribution schema before starting expensive trials.
    try:
        actual={k:v for k,v in space.items() if not (isinstance(v,dict) and v.get("type")=="fixed")}
        initial = {key: (value[0] if isinstance(value, list) else value["low"]) for key, value in actual.items()}
        suggest_parameters(optuna.trial.FixedTrial(initial), actual)
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("Некорректное пространство Optuna: используйте список или объект type/low/high.") from exc
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    history = []
    if best_oof is not None: best_oof.clear()
    started = time.monotonic()

    def objective(trial):
        check_cancel(cancel_event)
        params = sample_configuration(trial,model_name,device,fixed or {},space,suggest_parameters)
        trial.set_user_attr("effective_parameters",params)
        squared, count, fold_scores = 0., 0, []
        prediction_oof = np.full(len(x), np.nan) if best_oof is not None else None
        for step, (train, valid) in enumerate(splits):
            check_cancel(cancel_event)
            estimator = factory(params)
            estimator.fit(x.iloc[train], y.iloc[train])
            verify_device(estimator, model_name, device)
            prediction = estimator.predict(x.iloc[valid])
            if prediction_oof is not None: prediction_oof[valid] = prediction
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
        value = float(np.sqrt(squared / count))
        if best_oof is not None and value < best_oof.get("rmse", float("inf")):
            best_oof.update(rmse=value, trial=trial.number, predicted=prediction_oof)
        return value

    def callback(study, trial):
        complete = [t.value for t in study.trials if t.state == TrialState.COMPLETE]
        record = {"trial": trial.number, "state": trial.state.name, "rmse": trial.value,
                  "best_rmse": min(complete) if complete else None,
                  "seconds": trial.duration.total_seconds() if trial.duration else 0.,
                  "elapsed_seconds": time.monotonic() - started, "parameters": dict(trial.params),
                  "fold_rmse": trial.user_attrs.get("fold_rmse", []),
                  "effective_parameters": trial.user_attrs.get("effective_parameters",{})}
        history.append(record)
        if progress:
            progress(dict(record))

    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1,
                   callbacks=[callback], gc_after_trial=True)
    check_cancel(cancel_event)
    if best_oof is not None and (best_oof.get("trial") != study.best_trial.number or not np.isfinite(best_oof["predicted"]).all()):
        raise RuntimeError("Не удалось сохранить OOF-прогнозы лучшего trial.")
    return {k:v for k,v in study.best_trial.user_attrs["effective_parameters"].items() if k not in (fixed or {}) or v!=(fixed or {})[k]}, history, float(study.best_value)
