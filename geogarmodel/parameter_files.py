"""Portable JSON hyperparameters; never deserialize executable model objects."""
import math
from .compute import RESERVED


def validate_parameters(parameters):
    if not isinstance(parameters, dict):
        raise ValueError("Гиперпараметры должны быть объектом JSON.")
    for key in parameters:
        if not isinstance(key, str) or key in RESERVED or "__" in key:
            raise ValueError(f"Параметр {key} управляется устройством / конвейером.")
    return dict(parameters)


def read_parameters(payload):
    """Accept our snapshot, training report, or a plain parameter dictionary."""
    if not isinstance(payload, dict):
        raise ValueError("Файл должен содержать объект JSON.")
    if payload.get("format") == "geogarmodel.optuna-best.v1":
        return payload["model"], validate_parameters(payload["parameters"])
    if "settings" in payload and "model" in payload:
        settings = payload["settings"]
        return payload["model"], validate_parameters({**settings.get("parameters", {}),
                                                      **settings.get("best_parameters", {})})
    return None, validate_parameters(payload)


def best_snapshot(context, history):
    completed = [r for r in history if r["state"] == "COMPLETE" and
                 r.get("rmse") is not None and math.isfinite(r["rmse"])]
    if not completed:
        raise ValueError("Пока нет завершённых trials. Экспорт доступен после первого COMPLETE trial.")
    best = min(completed, key=lambda r: (r["rmse"], r["trial"]))
    return {"format": "geogarmodel.optuna-best.v1", "model": context["model"],
            "parameters": {**context["parameters"], **best["parameters"]},
            "suggested_parameters": dict(best["parameters"]), "best_trial": best["trial"],
            "best_cv_rmse": best["rmse"], "completed_trials": len(completed),
            "recorded_trials": len(history), "settings": context,
            "note": "Параметры лучшего завершённого trial; веса модели и Optuna study не включены."}
