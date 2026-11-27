"""JSON schema for configurable MLP; no PyTorch import required."""
import math
from copy import deepcopy

ACTIVATIONS = ("relu", "leaky_relu", "gelu", "elu", "selu", "tanh", "sigmoid", "softplus", "linear")
NORMALIZATION = {
    "BatchNorm": {"eps": 1e-5, "momentum": .1, "affine": True, "track_running_stats": True},
    "LayerNorm": {"eps": 1e-5, "elementwise_affine": True, "bias": True},
}
OPTIMIZERS = {
    "Adam": {"betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.},
    "AdamW": {"betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.},
    "SGD": {"momentum": 0., "dampening": 0., "nesterov": False, "weight_decay": 0.},
    "RMSprop": {"alpha": .99, "eps": 1e-8, "momentum": 0., "centered": False, "weight_decay": 0.},
    "Adagrad": {"lr_decay": 0., "eps": 1e-10, "initial_accumulator_value": 0., "weight_decay": 0.},
}
LOSSES = {"MSE": {}, "MAE": {}, "Huber": {"delta": 1.}, "SmoothL1": {"beta": 1.},
          "RMSE": {"epsilon": 1e-8}}
TRAIN_METRICS = ("MSE", "RMSE", "MAE", "R2")
EVAL_METRICS = ("MSE", "RMSE", "MAE", "R2", "MedianAE", "ExplainedVariance")


def defaults(seed=42):
    return dict(layer_specs=[
        dict(type="Dense", units=128, activation="linear", l1=0., l2=0.),
        dict(type="Activation", activation="relu"),
        dict(type="Dropout", rate=.1),
        dict(type="Dense", units=64, activation="linear", l1=0., l2=0.),
        dict(type="Activation", activation="relu"),
        dict(type="Dense", units=32, activation="linear", l1=0., l2=0.),
        dict(type="Activation", activation="relu")],
        output_l1=0., output_l2=0., optimizer_name="AdamW",
        optimizer_params=deepcopy(OPTIMIZERS["AdamW"]), loss_name="MSE",
        loss_params={}, epochs=100, batch_size=128, learning_rate=.001,
        gradient_clip=1., shuffle=True, random_state=seed,
        training_metrics=["RMSE", "MAE"], evaluation_metrics=["RMSE", "MAE", "R2"])


def number(value, key, low=0., high=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
        raise ValueError(f"{key}: требуется конечное число.")
    if value < low or (strict and value == low) or (high is not None and value > high):
        raise ValueError(f"{key}: недопустимый диапазон.")
    return float(value)


def integer(value, key, low=1, high=1000000):
    if isinstance(value,bool) or not isinstance(value,int) or not low <= value <= high:
        raise ValueError(f"{key}: целое число от {low} до {high}.")
    return value


def option_dict(name, values, catalog):
    if name not in catalog or not isinstance(values,dict):
        raise ValueError(f"Неизвестный метод или некорректные параметры: {name}.")
    unknown=set(values)-set(catalog[name])
    if unknown:
        raise ValueError(f"{name}: неизвестные параметры {sorted(unknown)}.")
    return {**deepcopy(catalog[name]), **values}


def validate_config(parameters, seed=42):
    config=defaults(seed)
    unknown=set(parameters)-set(config)
    if unknown:
        raise ValueError(f"MLP: неизвестные параметры {sorted(unknown)}. Для старых настроек выберите прежнюю реализацию.")
    config.update(deepcopy(parameters))
    if "optimizer_params" not in parameters:
        config["optimizer_params"] = {}
    if "loss_params" not in parameters:
        config["loss_params"] = {}
    layers=config["layer_specs"]
    if not isinstance(layers,list) or not 1 <= len(layers) <= 100:
        raise ValueError("layer_specs: список из 1–100 слоёв.")
    dense=0
    for i, layer in enumerate(layers,1):
        if not isinstance(layer,dict):
            raise ValueError(f"Слой {i}: требуется объект JSON.")
        kind=layer.get("type")
        if kind=="Dense":
            if set(layer)-{"type","units","activation","l1","l2"}:
                raise ValueError(f"Dense {i}: неизвестные параметры.")
            integer(layer.get("units"), f"Dense {i}: units",1,16384)
            if layer.get("activation","relu") not in ACTIVATIONS:
                raise ValueError(f"Dense {i}: неизвестная активация.")
            layer.setdefault("activation","relu")
            for key in ("l1","l2"):
                layer[key]=number(layer.get(key,0.),f"Dense {i}: {key}")
            dense+=1
        elif kind=="Dropout":
            if set(layer)-{"type","rate"}:
                raise ValueError(f"Dropout {i}: неизвестные параметры.")
            layer["rate"]=number(layer.get("rate"),f"Dropout {i}: rate",high=1.)
            if layer["rate"]>=1.:
                raise ValueError("Dropout rate должен быть меньше 1.")
        elif kind=="Activation":
            if set(layer)-{"type","activation"} or layer.get("activation") not in ACTIVATIONS:
                raise ValueError(f"Activation {i}: выберите допустимую активацию.")
        elif kind in NORMALIZATION:
            if set(layer)-{"type",*NORMALIZATION[kind]}:
                raise ValueError(f"{kind} {i}: неизвестные параметры.")
            for key,value in NORMALIZATION[kind].items():
                layer.setdefault(key,value)
            layer["eps"]=number(layer["eps"],f"{kind} {i}: eps",strict=True)
            flags=("affine","track_running_stats") if kind=="BatchNorm" else ("elementwise_affine","bias")
            for key in flags:
                if not isinstance(layer[key],bool):
                    raise ValueError(f"{kind} {i}: {key} — true или false.")
            if kind=="BatchNorm":
                if layer["momentum"] is not None:
                    layer["momentum"]=number(layer["momentum"],f"BatchNorm {i}: momentum",high=1.)
                if not layer["track_running_stats"]:
                    raise ValueError("BatchNorm: track_running_stats должен быть true для независимого прогноза точек.")
        else:
            raise ValueError(f"Слой {i}: неизвестный тип {kind}.")
    if not dense:
        raise ValueError("Нужен хотя бы один скрытый Dense-слой.")
    for key in ("output_l1","output_l2","gradient_clip"):
        config[key]=number(config[key],key)
    config["learning_rate"]=number(config["learning_rate"],"learning_rate",strict=True)
    for key in ("epochs","batch_size"):
        integer(config[key],key)
    if any(layer["type"]=="BatchNorm" for layer in layers) and config["batch_size"]<2:
        raise ValueError("BatchNorm требует batch_size >= 2.")
    integer(config["random_state"],"random_state",0,2147483647)
    if not isinstance(config["shuffle"],bool):
        raise ValueError("shuffle: true или false.")
    opt=config["optimizer_name"]
    opts=option_dict(opt,config["optimizer_params"],OPTIMIZERS)
    for key,value in opts.items():
        if key=="betas":
            if not isinstance(value,(list,tuple)) or len(value)!=2:
                raise ValueError("betas: два коэффициента.")
            opts[key]=[number(v,key,high=1.) for v in value]
            if any(v>=1. for v in opts[key]):
                raise ValueError("betas должны быть меньше 1.")
        elif key in ("nesterov","centered"):
            if not isinstance(value,bool):
                raise ValueError(f"{key}: true или false.")
        else:
            number(value,key,high=1. if key=="alpha" else None,strict=key=="eps")
    if opt=="SGD" and opts["nesterov"] and (opts["momentum"]<=0 or opts["dampening"]!=0):
        raise ValueError("Nesterov требует momentum > 0 и dampening = 0.")
    config["optimizer_params"]=opts
    loss=option_dict(config["loss_name"],config["loss_params"],LOSSES)
    for key,value in loss.items():
        number(value,key,strict=key in ("delta","epsilon"))
    config["loss_params"]=loss
    for key,allowed in (("training_metrics",TRAIN_METRICS),("evaluation_metrics",EVAL_METRICS)):
        values=config[key]
        if not isinstance(values,list) or any(v not in allowed for v in values) or len(set(values))!=len(values):
            raise ValueError(f"{key}: список уникальных допустимых метрик.")
        if key=="evaluation_metrics" and not values:
            raise ValueError("Выберите хотя бы одну метрику оценки.")
    return config
