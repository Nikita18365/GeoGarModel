"""Portable fitted pipelines: skops for sklearn, native numeric formats otherwise."""
from copy import copy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin

# Reviewed numerical sklearn implementation types used by our supported estimators.
# Never trust arbitrary classes declared by an input archive.
TRUSTED_TYPES={'numpy.dtype','sklearn.tree._tree.Tree',
    'sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor',
    'sklearn.neural_network._stochastic_optimizers.AdamOptimizer',
    'sklearn.neural_network._stochastic_optimizers.SGDOptimizer',
    'sklearn._loss.loss.HalfSquaredError','sklearn._loss.link.IdentityLink',
    '_loss.CyHalfSquaredError','sklearn._loss._loss.CyHalfSquaredError'}


def load_skops(blob):
    import skops.io as sio
    unknown=set(sio.get_untrusted_types(data=blob))
    if unknown-TRUSTED_TYPES: raise ValueError('Неизвестные типы сохранённой ML-модели: '+', '.join(sorted(unknown-TRUSTED_TYPES)))
    return sio.loads(blob,trusted=sorted(unknown))


def dump_pipeline(model):
    import skops.io as sio
    if isinstance(model,StoredPipeline): return model.metadata,model.blobs
    estimator=model.regressor_.steps[-1][1]; module=type(estimator).__module__
    losses=list(getattr(estimator,'loss_curve_',[]))
    metadata={'engine':'skops','loss_curve':losses}
    if module.startswith('sklearn.'):
        return metadata,{'pipeline':sio.dumps(model)}
    shell=copy(model); shell.regressor=None; shell.regressor_=copy(model.regressor_)
    shell.regressor_.steps=[*model.regressor_.steps[:-1],(model.regressor_.steps[-1][0],None)]
    blobs={'pipeline':sio.dumps(shell)}
    if module.startswith('xgboost.') or module.startswith('catboost.'):
        engine='xgboost' if module.startswith('xgboost.') else 'catboost'
        with TemporaryDirectory() as folder:
            path=Path(folder)/('model.json' if engine=='xgboost' else 'model.cbm')
            estimator.save_model(str(path)); blobs['native']=path.read_bytes()
        metadata['engine']=engine
    elif module.startswith('lightgbm.'):
        metadata['engine']='lightgbm'; blobs['native']=estimator.booster_.model_to_string().encode('utf-8')
    elif module in ('geogarmodel.neural','geogarmodel.mlp'):
        metadata.update(engine='configurable_mlp' if module=='geogarmodel.mlp' else 'torch',
                        parameters=estimator.get_params(),features=int(estimator.n_features_in_),
                        metric_curves=getattr(estimator,'metric_curves_',{}),
                        data_loss_curve=getattr(estimator,'data_loss_curve_',[]),
                        loss_label=getattr(estimator,'loss_label_','MSE · нормированная цель'))
        stream=BytesIO(); np.savez_compressed(stream,**{key:value.detach().cpu().numpy() for key,value in estimator.network_.state_dict().items()}); blobs['native']=stream.getvalue()
    else: raise ValueError('Этот тип обученной модели не поддерживает сохранение в проект.')
    return metadata,blobs


class NativeBooster(RegressorMixin,BaseEstimator):
    def __init__(self,booster): self.booster=booster; self.n_features_in_=booster.num_feature()
    def fit(self,X,y): raise ValueError('Сохранённый Booster предназначен для прогноза. Для обучения запустите новый расчёт.')
    def __sklearn_is_fitted__(self): return True
    def predict(self,X): return self.booster.predict(X)


class StoredPipeline:
    """Delay optional imports until prediction; preserve original bytes on resave."""
    def __init__(self,metadata,blobs):
        self.metadata=metadata; self.blobs=blobs; self._pipeline=None
        self.regressor_=SimpleNamespace(steps=[('estimator',SimpleNamespace(loss_curve_=metadata.get('loss_curve',[]),metric_curves_=metadata.get('metric_curves',{}),
            data_loss_curve_=metadata.get('data_loss_curve',[]),loss_label_=metadata.get('loss_label','MSE · нормированная цель')))])

    def predict(self,X):
        if self._pipeline is None:
            try: self._pipeline=self.load()
            except ImportError as error: raise ValueError('Для прогноза сохранённой модели установите её библиотеку: '+str(error)) from error
        return self._pipeline.predict(X)

    def load(self):
        pipeline=load_skops(self.blobs['pipeline']); engine=self.metadata['engine']
        if engine=='skops': return pipeline
        if engine=='lightgbm':
            from lightgbm import Booster
            estimator=NativeBooster(Booster(model_str=self.blobs['native'].decode('utf-8')))
        elif engine in ('xgboost','catboost'):
            with TemporaryDirectory() as folder:
                path=Path(folder)/('model.json' if engine=='xgboost' else 'model.cbm'); path.write_bytes(self.blobs['native'])
                if engine=='xgboost':
                    from xgboost import XGBRegressor
                    estimator=XGBRegressor(); estimator.load_model(str(path)); estimator.set_params(device='cpu')
                else:
                    from catboost import CatBoostRegressor
                    estimator=CatBoostRegressor(); estimator.load_model(str(path))
        elif engine in ('torch','configurable_mlp'):
            import torch
            from .neural import TorchRegressor
            from .mlp import ConfigurableMLP
            cls=ConfigurableMLP if engine=='configurable_mlp' else TorchRegressor
            estimator=cls(**{**self.metadata['parameters'],'device':'cpu'})
            estimator.n_features_in_=self.metadata['features']; estimator.device_=torch.device('cpu')
            estimator.network_=estimator.build_network(estimator.n_features_in_)
            with np.load(BytesIO(self.blobs['native']),allow_pickle=False) as data:
                estimator.network_.load_state_dict({key:torch.from_numpy(data[key].copy()) for key in data.files})
            estimator.network_.eval(); estimator.loss_curve_=self.metadata.get('loss_curve',[])
            estimator.metric_curves_=self.metadata.get('metric_curves',{})
            estimator.data_loss_curve_=self.metadata.get('data_loss_curve',[])
            estimator.loss_label_=self.metadata.get('loss_label','MSE · нормированная цель')
        else: raise ValueError('Неизвестный формат обученной модели.')
        pipeline.regressor_.steps[-1]=(pipeline.regressor_.steps[-1][0],estimator)
        return pipeline
