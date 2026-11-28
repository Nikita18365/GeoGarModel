"""Model-aware conservative Optuna domains and JSON-compatible nested parameters."""
from copy import deepcopy
import math
from .compute import RESERVED,device_parameters
from .mlp_config import validate_config, ACTIVATIONS

def integer(low,high,step=1):
    return dict(type="int",low=low,high=high,step=step)

def floating(low,high,log=False):
    return dict(type="float",low=low,high=high,log=log)

def recommended_space(name,device="cpu",fixed=None):
    p=fixed or {}
    f,i=floating,integer
    lr=f(.005,.2,True)
    tree=dict(n_estimators=i(100,500,50),max_depth=[None,5,10,20],
              min_samples_split=i(2,20),min_samples_leaf=i(1,10),
              max_features=[.5,.75,1.],min_weight_fraction_leaf=f(0,.05),
              min_impurity_decrease=f(0,.01),ccp_alpha=f(0,.01))
    forest=dict(tree,bootstrap=[True,False])
    # max_samples and oob_score are conditional on bootstrap.
    if p.get("max_samples") is not None or p.get("oob_score",False):
        forest.pop("bootstrap")
    if p.get("criterion")=="poisson":
        # The target is standardized; Poisson cannot be used.
        forest.pop("criterion",None)
    spaces={
        "Linear":dict(fit_intercept=[True,False],positive=[True,False]),
        "Ridge":dict(alpha=f(1e-5,1e3,True),fit_intercept=[True,False],
                     tol=f(1e-7,1e-2,True),max_iter=i(500,5000,500)),
        "Lasso":dict(alpha=f(1e-6,1.,True),fit_intercept=[True,False],positive=[True,False],
                     selection=["cyclic","random"],tol=f(1e-7,1e-2,True),max_iter=i(1000,10000,1000)),
        "Random Forest":dict(forest),"Extra Trees":dict(forest),
        "Gradient Boosting":dict(tree,learning_rate=lr,subsample=f(.5,1.),
                                 loss=["squared_error","absolute_error","huber","quantile"],
                                 alpha=f(.1,.9)),
        "Histogram Boosting":dict(learning_rate=lr,max_iter=i(100,500,50),
                                 max_leaf_nodes=i(8,63),max_depth=[None,3,5,8,12],
                                 min_samples_leaf=i(2,30),l2_regularization=f(1e-6,10,True),
                                 max_bins=i(32,255),max_features=f(.5,1.)),
        "XGBoost":dict(learning_rate=lr,n_estimators=i(100,600,50),max_depth=i(2,8),
                       min_child_weight=f(.1,10,True),gamma=f(0,5),subsample=f(.5,1.),
                       colsample_bytree=f(.5,1.),colsample_bylevel=f(.5,1.),
                       colsample_bynode=f(.5,1.),reg_alpha=f(1e-6,10,True),
                       reg_lambda=f(1e-4,10,True),max_bin=i(64,256,32)),
        "LightGBM":dict(learning_rate=lr,n_estimators=i(100,600,50),num_leaves=i(8,63),
                        min_child_samples=i(2,30),min_child_weight=f(1e-4,1,True),
                        min_split_gain=f(0,1),subsample=f(.5,1.),subsample_freq=i(1,5),
                        colsample_bytree=f(.5,1.),reg_alpha=f(1e-6,10,True),
                        reg_lambda=f(1e-6,10,True)),
        "CatBoost":dict(learning_rate=lr,iterations=i(100,600,50),depth=i(3,8),
                        l2_leaf_reg=f(.1,20,True),random_strength=f(0,5),border_count=i(32,128,16)),
        "MLP (sklearn)":dict(alpha=f(1e-6,.1,True),learning_rate_init=f(1e-4,.01,True),
                             activation=["identity","logistic","tanh","relu"],
                             max_iter=i(100,500,50),batch_size=[32,64,128,256],
                             tol=f(1e-6,1e-3,True),shuffle=[True,False]),
        "MLP (pytorch)":dict(learning_rate=f(1e-4,.01,True),batch_size=[32,64,128,256],
                             epochs=i(30,150,10),gradient_clip=[0.,.5,1.,2.,5.],shuffle=[True,False]),
    }
    neural=dict(width=[32,64,128],layers=i(1,4),learning_rate=f(1e-4,.01,True),
                dropout=f(0,.3),epochs=i(30,150,10),batch_size=[32,64,128,256],
                weight_decay=f(1e-6,.01,True))
    spaces["ResNet"]=dict(neural)
    heads=p.get("heads",4)
    widths=[v for v in (32,64,128) if isinstance(heads,int) and heads>0 and v%heads==0]
    spaces["Transformer"]=dict(neural)
    if widths:
        spaces["Transformer"]["width"]=widths
    else:
        spaces["Transformer"].pop("width")
    if name=="CatBoost" and device=="gpu":
        spaces[name].pop("random_strength",None)
    if name=="MLP (pytorch)":
        cfg=validate_config(p)
        for index,layer in enumerate(cfg["layer_specs"]):
            prefix=f"layer_specs.{index}."
            if layer["type"]=="Dense":
                spaces[name][prefix+"units"]=[32,64,128,256]
                spaces[name][prefix+"l1"]=f(1e-8,.001,True)
                spaces[name][prefix+"l2"]=f(1e-8,.001,True)
                if layer["activation"]!="linear":
                    spaces[name][prefix+"activation"]=["relu","gelu","tanh","elu"]
            elif layer["type"]=="Activation":
                spaces[name][prefix+"activation"]=["relu","gelu","tanh","elu"]
            elif layer["type"]=="Dropout":
                spaces[name][prefix+"rate"]=f(0,.4)
        for key in ("output_l1","output_l2"):
            spaces[name][key]=f(1e-8,.001,True)
        if "weight_decay" in cfg["optimizer_params"]:
            spaces[name]["optimizer_params.weight_decay"]=f(1e-8,.001,True)
    if name in ("Random Forest","Extra Trees") and p.get("monotonic_cst") is not None:
        spaces[name].pop("criterion",None)
    if name=="MLP (sklearn)" and p.get("solver")=="lbfgs":
        for key in ("learning_rate_init","batch_size","shuffle"):
            spaces[name].pop(key,None)
    if name=="XGBoost" and p.get("booster") not in (None,"gbtree","dart"):
        for key in ("max_depth","min_child_weight","gamma","subsample","colsample_bytree","colsample_bylevel","colsample_bynode","max_bin"):
            spaces[name].pop(key,None)
    return {k:v for k,v in spaces[name].items() if k not in device_parameters(name,device)}

def get_path(values,path):
    for part in path.split("."):
        values=values[int(part)] if isinstance(values,list) else values[part]
    return values

def set_path(values,path,value):
    parts=path.split(".")
    for part in parts[:-1]:
        values=values[int(part)] if isinstance(values,list) else values[part]
    if isinstance(values,list):
        values[int(parts[-1])]=deepcopy(value)
    else:
        values[parts[-1]]=deepcopy(value)

def merge_suggestions(fixed,suggestions):
    result=deepcopy(fixed)
    for key,value in suggestions.items():
        set_path(result,key,value)
    return result

def changed_parameters(fixed,suggestions):
    merged=merge_suggestions(fixed,suggestions)
    return {key:merged[key] for key in dict.fromkeys(path.split(".")[0] for path in suggestions)}

def resolved_fixed(name,parameters,seed=42,device="cpu",gpu_id=0):
    from .models import model_defaults
    if name=="MLP (pytorch)":
        return validate_config(parameters,seed)
    combined={**model_defaults(name,seed,device,gpu_id),**deepcopy(parameters)}
    return {k:v for k,v in combined.items() if k not in RESERVED and k not in device_parameters(name,device,gpu_id) and k!="architecture"}

def search_rows(name,fixed,device="cpu"):
    """Show all exposed constructor parameters, plus supported nested scalar paths."""
    domains=recommended_space(name,device,fixed)
    blocked=set(RESERVED)|set(device_parameters(name,device))|{"architecture"}
    rows={k:deepcopy(v) for k,v in fixed.items() if k not in blocked}
    for key in domains:
        if "." in key:
            rows[key]=get_path(fixed,key)
        else:
            rows.setdefault(key,None)
    return rows,domains

def unpack_space(payload):
    if not isinstance(payload,dict):
        raise ValueError("Диапазоны должны быть объектом JSON.")
    if payload.get("format")=="geogarmodel.search-space.v1":
        if not isinstance(payload.get("space"),dict):
            raise ValueError("В файле отсутствует объект space.")
        return payload["space"]
    return payload

def validate_space(space,name,fixed,device="cpu"):
    """Validate domains before starting any folds; no uncontrolled model/device keys."""
    from .optimization import suggest_parameters
    from optuna.trial import FixedTrial
    if not isinstance(space,dict):
        raise ValueError("Диапазоны должны быть объектом JSON.")
    rows,domains=search_rows(name,fixed,device)
    # Only curated parameters are independently searchable. Other library settings remain fixed.
    for key,spec in space.items():
        if key not in domains:
            raise ValueError(f"{key}: параметр фиксированный или не поддерживается для подбора этой модели.")
        if isinstance(spec,list):
            if not spec:
                raise ValueError(f"{key}: пустой список.")
            if any(isinstance(v,float) and not math.isfinite(v) for v in spec):
                raise ValueError(f"{key}: неконечное значение.")
        elif not isinstance(spec,dict):
            raise ValueError(f"{key}: ожидается диапазон или список.")
        # Enforce type/domain compatibility, allowing user-defined numeric ranges.
        expected=domains[key]
        if isinstance(expected,list):
            if not isinstance(spec,list):
                # Numeric categorical defaults may be replaced by numeric ranges.
                if not all(isinstance(v,int) and not isinstance(v,bool) for v in expected) or spec.get("type")!="int":
                    raise ValueError(f"{key}: используйте список вариантов.")
            elif any(type(v) not in {type(x) for x in expected} for v in spec):
                raise ValueError(f"{key}: несовместимый тип вариантов.")
            elif any(isinstance(v,str) and v not in expected for v in spec):
                raise ValueError(f"{key}: неподдерживаемый категориальный вариант.")
        else:
            vals=spec if isinstance(spec,list) else [spec.get("low"),spec.get("high")]
            if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in vals):
                raise ValueError(f"{key}: нужны конечные числовые значения.")
            if expected["type"]=="int" and any(int(v)!=v for v in vals):
                raise ValueError(f"{key}: нужны целые значения.")
        values=spec if isinstance(spec,list) else [spec.get("low"),spec.get("high")]
        numeric=[v for v in values if isinstance(v,(int,float)) and not isinstance(v,bool)]
        leaf=key.split(".")[-1]
        positive={"learning_rate","learning_rate_init","tol","n_estimators","iterations","max_iter","epochs",
                  "batch_size","width","units","layers","depth","max_depth","max_leaf_nodes","num_leaves",
                  "min_child_samples","min_samples_leaf","min_samples_split","max_bins","max_bin","border_count"}
        if leaf in positive and any(v<=0 for v in numeric):
            raise ValueError(f"{key}: значения должны быть положительными.")
        if leaf in ("subsample","colsample_bytree","colsample_bylevel","colsample_bynode","max_features") and any(v<=0 or v>1 for v in numeric):
            raise ValueError(f"{key}: допустимая доля (0, 1].")
        if leaf in ("rate","dropout") and any(v<0 or v>=1 for v in numeric):
            raise ValueError(f"{key}: допустимая доля [0, 1).")
        if leaf in ("alpha","l1","l2","output_l1","output_l2","weight_decay","reg_alpha","reg_lambda",
                    "l2_leaf_reg","l2_regularization","gamma","gradient_clip","min_child_weight",
                    "min_impurity_decrease","ccp_alpha","min_split_gain","random_strength") and any(v<0 for v in numeric):
            raise ValueError(f"{key}: значения не должны быть отрицательными.")
        if leaf=="min_weight_fraction_leaf" and any(v<0 or v>.5 for v in numeric):
            raise ValueError(f"{key}: диапазон от 0 до 0.5.")
        if leaf=="min_samples_split" and any(v<2 for v in numeric):
            raise ValueError(f"{key}: минимум 2.")
        if leaf=="max_bins" and any(v<2 or v>255 for v in numeric):
            raise ValueError(f"{key}: от 2 до 255.")
        if leaf=="units" and any(v>16384 for v in numeric):
            raise ValueError(f"{key}: максимум 16384.")
        if name=="Gradient Boosting" and leaf=="alpha" and any(v<=0 or v>=1 for v in numeric):
            raise ValueError("Gradient Boosting: alpha должна быть между 0 и 1.")
        if name=="MLP (pytorch)" and leaf=="batch_size" and any(layer["type"]=="BatchNorm" for layer in fixed["layer_specs"]) and any(v<2 for v in numeric):
            raise ValueError("BatchNorm требует batch_size >= 2.")
        if name=="Transformer" and key=="width":
            if not isinstance(spec,list) or any(not isinstance(v,int) or v<=0 or v%fixed.get("heads",4) for v in spec):
                raise ValueError("Transformer: ширина должна делиться на фиксированное число heads.")
    if space:
        initial={k:v[0] if isinstance(v,list) else v.get("low") for k,v in space.items()}
        try:
            suggest_parameters(FixedTrial(initial),space)
            # Validate actual distribution objects, including step/log constraints.
            from optuna.distributions import IntDistribution, FloatDistribution
            for spec in space.values():
                if isinstance(spec,dict):
                    dist=IntDistribution if spec["type"]=="int" else FloatDistribution
                    dist(spec["low"],spec["high"],log=spec.get("log",False),**({"step":spec["step"]} if "step" in spec else {}))
        except (KeyError,TypeError,ValueError) as error:
            raise ValueError(f"Некорректный диапазон Optuna: {error}") from error
    return deepcopy(space)
