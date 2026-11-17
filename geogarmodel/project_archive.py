"""Self-contained project archives with atomic writes and numeric-only datasets."""
from io import BytesIO,StringIO
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import pandas as pd
from .map_project import save_project as save_maps,load_project as load_maps
from .inversion import InversionResult
from .voxel import VoxelModel
from .models import TrainingResult
from .model_persistence import dump_pipeline,StoredPipeline


def json_bytes(value):
    def convert(value):
        if isinstance(value,np.ndarray): return value.tolist()
        if isinstance(value,np.generic): return value.item()
        raise TypeError(f'Неподдерживаемое значение проекта: {type(value).__name__}')
    return json.dumps(value,ensure_ascii=False,default=convert).encode('utf-8')


def write_project(path,state,maps,tables,training_result,models,voxels):
    """Write only completed snapshots. Worker state and device caches are excluded."""
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as stream: temporary=stream.name
    try:
        with ZipFile(temporary,'w',compression=ZIP_DEFLATED,compresslevel=3) as archive:
            stream=BytesIO(); save_maps(stream,**maps); archive.writestr('maps.npz',stream.getvalue())
            from importlib.metadata import version,PackageNotFoundError
            versions={}
            for name in ('numpy','pandas','scipy','scikit-learn','skops','optuna','xgboost','lightgbm','catboost','torch','pyvista','pyvistaqt'):
                try: versions[name]=version(name)
                except PackageNotFoundError: pass
            state={**state,'format':'GeoGarModel','version':1,'tables':[],'models':list(models),'voxels':list(voxels),'versions':versions}
            for name,table in tables.items():
                if table is not None:
                    archive.writestr(f'tables/{name}.json',table.to_json(orient='table',double_precision=15,date_unit='ns'))
                    state['tables'].append(name)
            for key,result in models.items():
                stream=BytesIO(); result.save(stream); archive.writestr(f'models/{key}.npz',stream.getvalue())
            for key,model in voxels.items():
                stream=BytesIO(); model.save(stream); archive.writestr(f'voxels/{key}.npz',stream.getvalue())
            if training_result is not None:
                r=training_result; metadata,blobs=dump_pipeline(r.model)
                state['training_result']={key:getattr(r,key) for key in ('features','target','coordinates','model_name','metrics','settings','optimization_history')}
                state['training_result'].update(model_metadata=metadata,blobs=list(blobs))
                archive.writestr('training/validation.json',r.validation.to_json(orient='table',double_precision=15,date_unit='ns'))
                for key,blob in blobs.items(): archive.writestr(f'training/{key}.bin',blob)
            archive.writestr('manifest.json',json_bytes(state))
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def read_project(path):
    """Read all numeric data before replacing the current GUI project."""
    with ZipFile(path) as archive:
        if 'manifest.json' not in archive.namelist(): return {'legacy_maps':load_maps(path)}
        state=json.loads(archive.read('manifest.json'))
        if state.get('format')!='GeoGarModel' or state.get('version')!=1: raise ValueError('Неподдерживаемый формат общего проекта.')
        maps=load_maps(BytesIO(archive.read('maps.npz')))
        tables={name:pd.read_json(StringIO(archive.read(f'tables/{name}.json').decode()),orient='table') for name in state['tables']}
        models={key:InversionResult.load(BytesIO(archive.read(f'models/{key}.npz'))) for key in state['models']}
        voxels={}
        from .inversion import Mesh
        for key in state['voxels']:
            with np.load(BytesIO(archive.read(f'voxels/{key}.npz')),allow_pickle=False) as data:
                meta=json.loads(str(data['voxel_metadata'])); mesh=Mesh(data['east'],data['north'],data['depth'],data['active'],data['bounds'] if data['bounds'].size else None)
                voxels[key]=VoxelModel(mesh,data['model'],**meta)
        trained=None
        if 'training_result' in state:
            meta=dict(state['training_result']); model_meta=meta.pop('model_metadata'); blob_keys=meta.pop('blobs')
            trained=TrainingResult(StoredPipeline(model_meta,{key:archive.read(f'training/{key}.bin') for key in blob_keys}),
                validation=pd.read_json(StringIO(archive.read('training/validation.json').decode()),orient='table'),**meta)
        # Validate cross references before the GUI is changed.
        keys={entry['key'] for entry in state.get('entries',[])}
        for entry in state.get('entries',[]):
            if entry['model'] not in models: raise ValueError('В проекте отсутствует 3D-модель.')
        for window in state.get('windows',[]):
            if set(window['entries'])-keys or (window['selected'] is not None and window['selected'] not in window['entries']): raise ValueError('Некорректный список моделей 3D-окна.')
        return dict(state=state,maps=maps,tables=tables,models=models,voxels=voxels,training_result=trained)
