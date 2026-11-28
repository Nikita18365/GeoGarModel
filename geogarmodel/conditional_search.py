"""Conditional search policy. Trial names are stable; inactive branches are not sampled."""
from copy import deepcopy
from .mlp_config import OPTIMIZERS, LOSSES, ACTIVATIONS, validate_config

def f(low,high,log=False):
    return dict(type="float",low=low,high=high,log=log)
def i(low,high):
    return dict(type="int",low=low,high=high)

OPT_DOMAINS={
    "betas.0":f(.8,.99),"betas.1":f(.9,.9999),"eps":f(1e-10,1e-5,True),
    "weight_decay":f(0,.01),"momentum":f(.01,.99),"dampening":f(0,.5),
    "nesterov":[True,False],"alpha":f(.9,.9999),"centered":[True,False],
    "lr_decay":f(0,.1),"initial_accumulator_value":f(0,.1)}
LOSS_DOMAINS={"delta":f(.1,5),"beta":f(0,5),"epsilon":f(1e-10,1e-5,True)}

def conditional_domains(name,device,p):
    d={}
    if name=="MLP (pytorch)":
        cfg=validate_config(p)
        d.update(optimizer_name=list(OPTIMIZERS),loss_name=list(LOSSES),learning_rate=f(1e-5,.05,True),
                 batch_size=[16,32,64,128,256,512],gradient_clip=f(0,10))
        for opt,defaults in OPTIMIZERS.items():
            for key in defaults:
                if key=="betas":
                    for j in (0,1): d[f"optimizer_branches.{opt}.betas.{j}"]=OPT_DOMAINS[f"betas.{j}"]
                elif key!="weight_decay": d[f"optimizer_branches.{opt}.{key}"]=OPT_DOMAINS[key]
        for loss,defaults in LOSSES.items():
            for key in defaults: d[f"loss_branches.{loss}.{key}"]=LOSS_DOMAINS[key]
        for j,layer in enumerate(cfg["layer_specs"]):
            prefix=f"layer_specs.{j}."
            if layer["type"]=="Activation" or layer["type"]=="Dense" and layer["activation"]!="linear":
                d[prefix+"activation"]=list(ACTIVATIONS)
            if layer["type"] in ("BatchNorm","LayerNorm"):
                d[prefix+"eps"]=f(1e-7,1e-3,True)
            if layer["type"]=="BatchNorm":
                d[prefix+"momentum"]=f(.01,.9)
                d[prefix+"affine"]=[True,False]
            elif layer["type"]=="LayerNorm":
                d[prefix+"elementwise_affine"]=[True,False]
                d[prefix+"bias"]=[True,False]
    elif name=="Ridge":
        d.update(solver=["auto","svd","cholesky","lsqr","sparse_cg","sag","saga","lbfgs"],positive=[True,False])
    elif name in ("Random Forest","Extra Trees"):
        d.update(criterion=["squared_error","absolute_error","friedman_mse"],
                 bootstrap=[True,False],max_samples=f(.5,1.),oob_score=[True,False],
                 max_leaf_nodes=[None,16,32,64,128])
        if p.get("monotonic_cst") is not None: d.pop("criterion")
    elif name=="Gradient Boosting":
        d.update(criterion=["friedman_mse","squared_error"],max_leaf_nodes=[None,8,16,32],
                 validation_fraction=f(.1,.3),n_iter_no_change=[None,5,10,20],tol=f(1e-6,.01,True))
    elif name=="Histogram Boosting":
        d.update(loss=["squared_error","absolute_error","quantile"],quantile=f(.1,.9),
                 early_stopping=[True,False],validation_fraction=f(.1,.3),
                 n_iter_no_change=i(5,30),tol=f(1e-8,.001,True))
    elif name=="XGBoost":
        d.update(booster=["gbtree","dart"],grow_policy=["depthwise","lossguide"],
                 max_leaves=[0,16,32,64],rate_drop=f(0,.3),skip_drop=f(0,.3),
                 sample_type=["uniform","weighted"],normalize_type=["tree","forest"],
                 objective=["reg:squarederror","reg:absoluteerror","reg:pseudohubererror","reg:quantileerror"],
                 huber_slope=f(.1,5,True),quantile_alpha=f(.1,.9),max_delta_step=f(0,5),
                 sampling_method=["uniform","gradient_based"] if device=="gpu" else ["uniform"])
    elif name=="LightGBM":
        d.update(boosting_type=["gbdt","dart"],data_sample_strategy=["bagging","goss"],max_depth=[-1,3,5,8,12],
                 drop_rate=f(.05,.3),skip_drop=f(0,.5),max_drop=i(10,50),uniform_drop=[True,False],
                 top_rate=f(.1,.3),other_rate=f(.05,.2),objective=["regression","regression_l1","huber","quantile"],
                 alpha=f(.1,.9),max_bin=i(32,255),extra_trees=[True,False])
    elif name=="CatBoost":
        d.update(bootstrap_type=["Bayesian","Bernoulli","MVS"] if device=="cpu" else ["Bayesian","Bernoulli","Poisson"],bagging_temperature=f(0,5),
                 subsample=f(.5,1.),grow_policy=["SymmetricTree","Depthwise","Lossguide"],
                 min_data_in_leaf=i(1,30),max_leaves=i(8,32),
                 loss_function=["RMSE","MAE","Huber:delta=1.0"] if device=="cpu" else ["RMSE","MAE"],
                 boosting_type=["Plain","Ordered"] if device=="cpu" else ["Plain"])
        if device=="cpu": d["mvs_reg"]=f(1e-5,10,True)
        if device=="cpu": d["random_strength"]=f(0,5)
    elif name=="MLP (sklearn)":
        d.update(solver=["adam","sgd","lbfgs"],learning_rate=["constant","invscaling","adaptive"],
                 momentum=f(.01,.99),nesterovs_momentum=[True,False],power_t=f(.1,.9),
                 beta_1=f(.8,.99),beta_2=f(.9,.9999),epsilon=f(1e-10,1e-5,True),
                 early_stopping=[True,False],validation_fraction=f(.1,.3),n_iter_no_change=i(5,30),
                 learning_rate_init=f(1e-4,.01,True),batch_size=[32,64,128,256],shuffle=[True,False])
    elif name=="Transformer":
        d.update(width=[32,64,128],heads=[1,2,4,8])
        if p.get("heads") and not any(w%p["heads"]==0 for w in d["width"]):
            d["width"].append(p.get("width",p["heads"]))
    return d

def conditional_value(name,fixed,key):
    parts=key.split(".")
    if parts[0] in ("optimizer_branches","loss_branches"):
        catalog=OPTIMIZERS if parts[0]=="optimizer_branches" else LOSSES
        prefix="optimizer" if parts[0]=="optimizer_branches" else "loss"
        values=deepcopy(catalog[parts[1]])
        if fixed.get(prefix+"_name")==parts[1]: values.update(fixed.get(prefix+"_params",{}))
        parts=parts[2:]
    else:
        values=fixed
    for part in parts:
        values=values[int(part)] if isinstance(values,list) else values[part]
    return values

def is_fixed(spec):
    return isinstance(spec,dict) and spec.get("type")=="fixed"

def has_search(space):
    return any(not is_fixed(spec) for spec in space.values())

def is_branch(key):
    return key.startswith(("optimizer_branches.","loss_branches."))

def active_condition(name,key):
    p=key.split(".")
    if p[0]=="optimizer_branches": return "Optimizer = "+p[1]+("; Nesterov = false" if key.endswith(".dampening") else "")
    if p[0]=="loss_branches": return "Loss = "+p[1]
    if name in ("Random Forest","Extra Trees") and key in ("max_samples","oob_score"): return "bootstrap = true"
    if name=="Ridge" and key=="positive": return "solver = auto; lbfgs требует true"
    if name=="Gradient Boosting" and key in ("validation_fraction","tol"): return "n_iter_no_change задан"
    if name=="Histogram Boosting" and key in ("validation_fraction","n_iter_no_change","tol"): return "early_stopping = true"
    if name=="Gradient Boosting" and key=="alpha": return "loss = huber / quantile"
    if name=="Histogram Boosting" and key=="quantile": return "loss = quantile"
    if name=="XGBoost" and key in ("rate_drop","skip_drop","sample_type","normalize_type"): return "booster = dart"
    if name=="XGBoost" and key=="max_leaves": return "grow_policy = lossguide"
    if name=="XGBoost" and key=="huber_slope": return "objective = reg:pseudohubererror"
    if name=="XGBoost" and key=="quantile_alpha": return "objective = reg:quantileerror"
    if name=="LightGBM":
        if key in ("drop_rate","skip_drop","max_drop","uniform_drop"): return "boosting_type = dart"
        if key in ("top_rate","other_rate"): return "data_sample_strategy = goss"
        if key in ("subsample","subsample_freq"): return "data_sample_strategy = bagging"
        if key=="alpha": return "objective = huber / quantile"
    if name=="Transformer" and key in ("width","heads"): return "width % heads = 0"
    if name=="MLP (pytorch)" and key.endswith(".bias"): return "elementwise_affine = true"
    if name=="CatBoost":
        if key=="bagging_temperature": return "bootstrap_type = Bayesian"
        if key=="subsample": return "bootstrap_type = Bernoulli / MVS / Poisson"
        if key=="mvs_reg": return "bootstrap_type = MVS"
        if key=="boosting_type": return "Ordered: только SymmetricTree"
        if key=="max_leaves": return "grow_policy = Lossguide"
        if key=="min_data_in_leaf": return "grow_policy != SymmetricTree"
    if name=="MLP (sklearn)":
        if key in ("beta_1","beta_2","epsilon"): return "solver = adam"
        if key in ("momentum","nesterovs_momentum","learning_rate","power_t"): return "solver = sgd"
        if key in ("learning_rate_init","batch_size","shuffle","early_stopping","validation_fraction","n_iter_no_change"): return "solver != lbfgs"
    return ""

def sample_configuration(trial,name,device,fixed,space,suggest):
    from .search_space import set_path
    p=deepcopy(fixed)
    remaining=dict(space)
    def draw(key,spec):
        return spec["value"] if isinstance(spec,dict) and spec.get("type")=="fixed" else suggest(trial,{key:spec})[key]
    def take(key,default=None):
        if key in remaining:
            value=draw(key,remaining.pop(key))
            set_path(p,key,value)
            return value
        return p.get(key,default)
    def skip(*keys):
        for key in keys: remaining.pop(key,None)
    if name=="MLP (pytorch)":
        p=validate_config(p)
        old_opt,old_loss=p["optimizer_name"],p["loss_name"]
        opt=take("optimizer_name",old_opt); loss=take("loss_name",old_loss)
        if opt!=old_opt:
            decay=p["optimizer_params"]["weight_decay"]
            p["optimizer_params"]=deepcopy(OPTIMIZERS[opt])
            p["optimizer_params"]["weight_decay"]=decay
        if loss!=old_loss: p["loss_params"]=deepcopy(LOSSES[loss])
        # Legacy flat optimizer paths remain supported when the optimizer is fixed.
        for key in list(remaining):
            if key.startswith(("optimizer_branches.","loss_branches.")):
                parts=key.split(".")
                selected=opt if parts[0]=="optimizer_branches" else loss
                if parts[1]!=selected: skip(key); continue
                target=("optimizer_params." if parts[0]=="optimizer_branches" else "loss_params.")+".".join(parts[2:])
                if opt=="SGD" and target in ("optimizer_params.momentum","optimizer_params.dampening","optimizer_params.nesterov"):
                    continue
                value=draw(key,remaining.pop(key))
                set_path(p,target,value)
        if opt=="SGD":
            for key in ("optimizer_branches.SGD.momentum","optimizer_branches.SGD.nesterov","optimizer_branches.SGD.dampening"):
                if key in remaining:
                    opts=p["optimizer_params"]
                    if (key.endswith("nesterov") and opts["momentum"]<=0) or (key.endswith("dampening") and opts["nesterov"]):
                        skip(key); continue
                    opts[key.split(".")[-1]]=draw(key,remaining.pop(key))
            if p["optimizer_params"]["momentum"]<=0:
                p["optimizer_params"]["nesterov"]=False
            if p["optimizer_params"]["nesterov"]:
                p["optimizer_params"]["dampening"]=0.
        # Shared weight_decay applies to every optimizer; old flat paths remain valid.
        for key in list(remaining):
            if key.startswith("layer_specs.") and key.endswith(".bias"):
                parent=key.rsplit(".",1)[0]+".elementwise_affine"
                if parent in remaining:
                    value=draw(parent,remaining.pop(parent))
                    set_path(p,parent,value)
                idx=int(key.split(".")[1])
                if not p["layer_specs"][idx]["elementwise_affine"]:
                    skip(key); p["layer_specs"][idx]["bias"]=False
    elif name=="Ridge" and any(k in space for k in ("solver","positive")):
        solver=take("solver","auto")
        if solver=="lbfgs":
            skip("positive"); p["positive"]=True
        elif solver!="auto":
            skip("positive"); p["positive"]=False
    elif name in ("Random Forest","Extra Trees") and any(k in space for k in ("bootstrap","max_samples","oob_score")):
        bootstrap=take("bootstrap",True if name=="Random Forest" else False)
        if not bootstrap:
            skip("max_samples","oob_score"); p.update(max_samples=None,oob_score=False)
    elif name=="Gradient Boosting":
        if take("loss","squared_error") not in ("huber","quantile"): skip("alpha")
        if take("n_iter_no_change",None) is None: skip("validation_fraction","tol")
    elif name=="Histogram Boosting":
        if take("loss","squared_error")!="quantile": skip("quantile")
        if not take("early_stopping",False): skip("validation_fraction","n_iter_no_change","tol")
    elif name=="XGBoost":
        if take("booster","gbtree")!="dart": skip("rate_drop","skip_drop","sample_type","normalize_type")
        if take("grow_policy","depthwise")!="lossguide": skip("max_leaves")
        objective=take("objective","reg:squarederror")
        if objective!="reg:pseudohubererror": skip("huber_slope")
        if objective!="reg:quantileerror": skip("quantile_alpha")
    elif name=="LightGBM":
        boosting=take("boosting_type","gbdt")
        if boosting!="dart": skip("drop_rate","skip_drop","max_drop","uniform_drop")
        if boosting=="dart":
            skip("data_sample_strategy"); strategy="bagging"
            if "boosting_type" in space or "data_sample_strategy" in space: p["data_sample_strategy"]="bagging"
        else: strategy=take("data_sample_strategy","bagging")
        if strategy=="goss" or boosting=="goss":
            skip("subsample","subsample_freq"); p.update(subsample=1.,subsample_freq=0)
        else: skip("top_rate","other_rate")
        if take("objective","regression") not in ("huber","quantile"): skip("alpha")
    elif name=="CatBoost" and any(k in space for k in ("bootstrap_type","bagging_temperature","subsample","mvs_reg","grow_policy","max_leaves","min_data_in_leaf","boosting_type","random_strength")):
        bootstrap=take("bootstrap_type",p.get("bootstrap_type") or "Bayesian")
        if bootstrap=="Bayesian":
            skip("subsample"); p["subsample"]=None
        else:
            skip("bagging_temperature"); p["bagging_temperature"]=None
        if bootstrap!="MVS": skip("mvs_reg"); p["mvs_reg"]=None
        grow=take("grow_policy",p.get("grow_policy") or "SymmetricTree")
        if grow!="Lossguide": skip("max_leaves"); p["max_leaves"]=None
        if grow=="SymmetricTree": skip("min_data_in_leaf"); p["min_data_in_leaf"]=None
        if grow!="SymmetricTree":
            p["random_strength"]=None; skip("random_strength","boosting_type"); p["boosting_type"]="Plain"
    elif name=="MLP (sklearn)":
        solver=take("solver","adam")
        if solver!="adam": skip("beta_1","beta_2","epsilon")
        if solver!="sgd": skip("momentum","nesterovs_momentum","learning_rate","power_t")
        elif take("learning_rate","constant")!="invscaling": skip("power_t")
        if solver=="lbfgs":
            skip("learning_rate_init","batch_size","shuffle","early_stopping","validation_fraction","n_iter_no_change")
            if "solver" in space: p["early_stopping"]=False
        elif not take("early_stopping",False): skip("validation_fraction")
    elif name=="Transformer":
        width=take("width",p.get("width",32))
        if "heads" in remaining:
            choices=remaining.pop("heads")
            compatible=[h for h in choices if width%h==0]
            if not compatible: raise ValueError("Transformer: нет совместимого числа heads.")
            # Branch-specific distribution names avoid Optuna dynamic categorical spaces.
            p["heads"]=draw(f"heads@width={width}",compatible)
        elif width%p.get("heads",4): raise ValueError("Transformer: width должно делиться на heads.")
    if remaining:
        for key,spec in remaining.items(): set_path(p,key,draw(key,spec))
    return validate_config(p) if name=="MLP (pytorch)" else p
