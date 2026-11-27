"""Configurable dense neural network on a single CPU/CUDA PyTorch backend."""
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from .mlp_config import defaults, validate_config
from .work_progress import report


class ConfigurableMLP(RegressorMixin, BaseEstimator):
    # Store constructor parameters unchanged for sklearn.clone.
    def __init__(self, layer_specs=None, output_l1=0., output_l2=0.,
                 optimizer_name="AdamW", optimizer_params=None, loss_name="MSE",
                 loss_params=None, epochs=100, batch_size=128, learning_rate=.001,
                 gradient_clip=1., shuffle=True, random_state=42,
                 training_metrics=None, evaluation_metrics=None, device="cpu"):
        self.layer_specs=layer_specs
        self.output_l1=output_l1
        self.output_l2=output_l2
        self.optimizer_name=optimizer_name
        self.optimizer_params=optimizer_params
        self.loss_name=loss_name
        self.loss_params=loss_params
        self.epochs=epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.gradient_clip=gradient_clip
        self.shuffle=shuffle
        self.random_state=random_state
        self.training_metrics=training_metrics
        self.evaluation_metrics=evaluation_metrics
        self.device=device

    def configuration(self):
        params=self.get_params(deep=False)
        params.pop("device")
        return validate_config({k:v for k,v in params.items() if v is not None},self.random_state)

    def build_network(self, features):
        config=self.configuration()
        activation={"relu":nn.ReLU,"leaky_relu":nn.LeakyReLU,"gelu":nn.GELU,
                    "elu":nn.ELU,"selu":nn.SELU,"tanh":nn.Tanh,"sigmoid":nn.Sigmoid,
                    "softplus":nn.Softplus,"linear":nn.Identity}
        layers=[]
        self.regularized_=[]
        width=features
        for spec in config["layer_specs"]:
            if spec["type"]=="Dense":
                linear=nn.Linear(width,spec["units"])
                layers.extend([linear,activation[spec["activation"]]()])
                self.regularized_.append((linear,spec["l1"],spec["l2"]))
                width=spec["units"]
            elif spec["type"]=="Dropout":
                layers.append(nn.Dropout(spec["rate"]))
            elif spec["type"]=="LayerNorm":
                layers.append(nn.LayerNorm(width))
        output=nn.Linear(width,1)
        self.regularized_.append((output,config["output_l1"],config["output_l2"]))
        layers.extend([output,nn.Flatten(0)])
        return nn.Sequential(*layers)

    def penalty(self):
        result=next(self.network_.parameters()).new_zeros(())
        for layer,l1,l2 in self.regularized_:
            if l1:
                result=result+l1*layer.weight.abs().sum()
            if l2:
                result=result+l2*layer.weight.square().sum()
        return result

    def data_loss(self, prediction, target):
        name=self.loss_name
        params=self.config_["loss_params"]
        if name=="MSE":
            return nn.functional.mse_loss(prediction,target)
        if name=="MAE":
            return nn.functional.l1_loss(prediction,target)
        if name=="Huber":
            return nn.functional.huber_loss(prediction,target,delta=params["delta"])
        if name=="SmoothL1":
            return nn.functional.smooth_l1_loss(prediction,target,beta=params["beta"])
        if name=="RMSE":
            return torch.sqrt(nn.functional.mse_loss(prediction,target)+params["epsilon"])
        raise ValueError("Неизвестная функция потерь.")

    def fit(self,X,y):
        X,y=check_X_y(X,y,dtype=np.float32,y_numeric=True)
        y=np.asarray(y,dtype=np.float32)
        self.config_=self.configuration()
        if self.device!="cpu" and not self.device.startswith("cuda:"):
            raise ValueError("PyTorch device: cpu или cuda:N.")
        if self.device.startswith("cuda:") and (not torch.cuda.is_available()
                or int(self.device.split(":")[1])>=torch.cuda.device_count()):
            raise ValueError("Выбранный CUDA GPU недоступен. Проверьте драйвер и PyTorch.")
        torch.manual_seed(self.random_state)
        if self.device.startswith("cuda:"):
            torch.cuda.manual_seed_all(self.random_state)
        self.device_=torch.device(self.device)
        self.n_features_in_=X.shape[1]
        self.network_=self.build_network(X.shape[1]).to(self.device_)
        kwargs=dict(self.config_["optimizer_params"])
        if "betas" in kwargs:
            kwargs["betas"]=tuple(kwargs["betas"])
        optimizers={"Adam":torch.optim.Adam,"AdamW":torch.optim.AdamW,"SGD":torch.optim.SGD,
                    "RMSprop":torch.optim.RMSprop,"Adagrad":torch.optim.Adagrad}
        optimizer=optimizers[self.optimizer_name](self.network_.parameters(),lr=self.learning_rate,**kwargs)
        generator=torch.Generator().manual_seed(self.random_state)
        dataset=torch.utils.data.TensorDataset(torch.from_numpy(X.copy()),torch.from_numpy(y.copy()))
        loader=torch.utils.data.DataLoader(dataset,batch_size=self.batch_size,shuffle=self.shuffle,generator=generator)
        self.loss_curve_=[]
        self.data_loss_curve_=[]
        self.metric_curves_={name:[] for name in self.config_["training_metrics"]}
        self.loss_label_=self.loss_name+" + L1/L2 · нормированная цель"
        for epoch in range(self.epochs):
            self.network_.train()
            total=0.
            data_total=0.
            for xb,yb in loader:
                xb,yb=xb.to(self.device_),yb.to(self.device_)
                optimizer.zero_grad()
                data_loss=self.data_loss(self.network_(xb),yb)
                loss=data_loss+self.penalty()
                if not torch.isfinite(loss):
                    raise ValueError("Нейросеть разошлась; уменьшите learning_rate или регуляризацию.")
                loss.backward()
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.network_.parameters(),self.gradient_clip)
                optimizer.step()
                total+=float(loss.detach())*len(xb)
                data_total+=float(data_loss.detach())*len(xb)
            self.loss_curve_.append(total/len(X))
            self.data_loss_curve_.append(data_total/len(X))
            self.network_.eval()
            if self.metric_curves_:
                pred=self.predict(X).astype(np.float64)
                actual=y.astype(np.float64)
                residual=pred-actual
                mse=float(np.mean(residual**2))
                ss=float(np.sum((actual-actual.mean())**2))
                scores={"MSE":mse,"RMSE":float(np.sqrt(mse)),"MAE":float(np.mean(np.abs(residual))),
                        "R2":1.-float(np.sum(residual**2))/ss if ss>0 else (1. if mse==0 else 0.)}
                for name,curve in self.metric_curves_.items():
                    curve.append(scores[name])
            report(epoch+1,self.epochs,f"MLP · эпоха {epoch+1}/{self.epochs}")
        return self

    def predict(self,X):
        check_is_fitted(self,"network_")
        X=check_array(X,dtype=np.float32)
        if X.shape[1]!=self.n_features_in_:
            raise ValueError("Число признаков отличается от обучения.")
        self.network_.eval()
        with torch.inference_mode():
            return np.concatenate([self.network_(torch.from_numpy(X[i:i+self.batch_size].copy()).to(self.device_)).cpu().numpy()
                                   for i in range(0,len(X),self.batch_size)])
