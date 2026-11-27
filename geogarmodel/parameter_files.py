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


def migrate_model_selection(name, parameters, search=None):
    """Only migrate editable settings. Fitted legacy pipelines keep their own engine."""
    parameters = dict(parameters)
    search = dict(search or {})
    keys = set(parameters) | set(search)
    sklearn_keys = {"hidden_layer_sizes", "max_iter", "alpha", "learning_rate_init", "solver"}
    legacy_keys = {"width", "layers", "dropout", "weight_decay", "heads"}
    if name == "MLP" and keys & sklearn_keys:
        return "MLP (sklearn)", parameters, search
    if name == "MLP (legacy CUDA)" or (name == "MLP" and keys & legacy_keys):
        width = parameters.pop("width", 128)
        count = parameters.pop("layers", 3)
        dropout = parameters.pop("dropout", .1)
        decay = parameters.pop("weight_decay", .0001)
        parameters.pop("heads", None)
        if not isinstance(count, int) or not 1 <= count <= 33:
            raise ValueError("Старая архитектура MLP: некорректное число слоёв.")
        layers = []
        for _ in range(count):
            layers.extend([dict(type="Dense", units=width, activation="linear"),
                           dict(type="Activation", activation="relu"),
                           dict(type="Dropout", rate=dropout)])
        parameters.update(layer_specs=layers, optimizer_name="AdamW",
                          optimizer_params={"weight_decay": decay}, loss_name="MSE")
        search = {k:v for k,v in search.items() if k not in legacy_keys}
        return "MLP (pytorch)", parameters, search
    if name == "MLP":
        name = "MLP (pytorch)"
    return name, parameters, search


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
