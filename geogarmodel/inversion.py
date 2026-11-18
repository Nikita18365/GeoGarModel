"""3D rectangular-prism inversion. Internal coordinates: East, North, Down (m).

Independent implementation of prism integrals and projected gradient inversion.
CPU NumPy / GPU CuPy share the same double precision kernels and objective.
"""
from dataclasses import dataclass
from itertools import product
import json
import numpy as np
from time import perf_counter
from .compute import check_cancel, ComputationCancelled
from .inversion_progress import WorkProgress
from .inversion_resources import available_memory, resource_plan


@dataclass
class Mesh:
    east: np.ndarray
    north: np.ndarray
    depth: np.ndarray
    active: np.ndarray | None = None
    bounds: np.ndarray | None = None

    def __post_init__(self):
        for key in ('east', 'north', 'depth'):
            a = np.asarray(getattr(self, key), dtype=float)
            if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or not (np.diff(a) > 0).all():
                raise ValueError('Границы ячеек должны быть конечными и возрастающими.')
            setattr(self, key, a)
        self.active = np.ones(self.size, dtype=bool) if self.active is None else np.asarray(self.active, dtype=bool).reshape(self.size).copy()
        if self.bounds is not None:
            self.bounds = np.asarray(self.bounds, dtype=float).reshape(self.size,6).copy()
            if not np.isfinite(self.bounds).all() or np.any(self.bounds[self.active,3:] <= self.bounds[self.active,:3]):
                raise ValueError('Некорректные границы активных призм.')
            bounds=self.bounds; self.bounds=None
            base=self.cells()
            if np.any(bounds[self.active,:3]<base[self.active,:3]) or np.any(bounds[self.active,3:]>base[self.active,3:]):
                raise ValueError('Обрезанные призмы должны лежать внутри исходных ячеек.')
            self.bounds=None if np.array_equal(bounds,base) else bounds

    @property
    def shape(self):
        return tuple(len(a)-1 for a in (self.east, self.north, self.depth))

    @property
    def size(self):
        return int(np.prod(self.shape))

    def cells(self):
        if self.bounds is not None:
            return self.bounds.copy()
        indices = np.indices(self.shape).reshape(3, -1)
        return np.column_stack([a[i] for a, i in zip((self.east, self.north, self.depth), indices)] +
                               [a[i+1] for a, i in zip((self.east, self.north, self.depth), indices)])

    def centers(self):
        c = self.cells()
        return (c[:, :3] + c[:, 3:]) / 2


def direction(inclination, declination):
    """Declination clockwise from North, inclination positive down."""
    if not np.isfinite([inclination, declination]).all() or not -90 <= inclination <= 90:
        raise ValueError('Наклонение должно быть в [-90, 90], углы — конечными.')
    i, d = np.deg2rad([inclination, declination])
    return np.array([np.cos(i)*np.sin(d), np.cos(i)*np.cos(d), np.sin(i)])


def backend(device='cpu', gpu_id=0):
    if device == 'cpu':
        return np
    if device != 'gpu' or not isinstance(gpu_id, int) or gpu_id < 0:
        raise ValueError('Выберите CPU или GPU и неотрицательный ID.')
    try:
        import cupy as cp
        cp.cuda.Device(gpu_id).use()
        cp.zeros(1).sum().item()
        return cp
    except Exception as exc:
        raise RuntimeError('CUDA недоступна. Установите geogarmodel[gpu] и совместимый драйвер; CPU выберите явно.') from exc


def host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


def prism_kernel(observations, cells, magnetic=False, xp=np):
    """Eight-corner analytical integral; observations must be above cell tops.

    Gravity returned in mGal per g/cm³; magnetic Hessian in nT per A/m.
    Magnetic entries are xx, yy, zz, xy, xz, yz in END coordinates.
    """
    obs, cells = xp.asarray(observations), xp.asarray(cells)
    out = xp.zeros((len(obs), len(cells), 6) if magnetic else (len(obs), len(cells)))
    tiny = np.finfo(float).tiny
    def logplus(a, b, c, r):
        # Stable r+a for a<0; all observations are strictly above the mesh.
        return xp.log(xp.maximum(xp.where(a < 0, (b*b+c*c)/xp.maximum(r-a, tiny), r+a), tiny))
    def atan(a, b):
        angle = xp.arctan2(a, b)
        return xp.where(angle > np.pi/2, angle-np.pi, xp.where(angle < -np.pi/2, angle+np.pi, angle))
    for i, j, k in product(range(2), repeat=3):
        x = cells[None, :, 3*i] - obs[:, None, 0]
        y = cells[None, :, 1+3*j] - obs[:, None, 1]
        z = cells[None, :, 2+3*k] - obs[:, None, 2]
        r = xp.sqrt(x*x+y*y+z*z)
        sign = (-1.)**(3-i-j-k)
        lx, ly = logplus(x,y,z,r), logplus(y,x,z,r)
        if magnetic:
            out[..., 0] -= sign * atan(y*z, x*r)
            out[..., 1] -= sign * atan(x*z, y*r)
            out[..., 2] -= sign * atan(x*y, z*r)
            out[..., 3] += sign * logplus(z,x,y,r)
            out[..., 4] += sign * ly
            out[..., 5] += sign * lx
        else:
            out -= sign * (x*ly+y*lx-z*atan(x*y,z*r))
    return out * (100. if magnetic else 6.67430e-3)


class MisfitOperator:
    """Keep the field and adjoint input for an accepted line-search evaluation."""

    def misfit(self, model, data, sigma, fit_offset=False):
        xp = self.xp
        predicted, unit = self.field_and_direction(model)
        self.last_offset = float(xp.sum((data-predicted)/sigma**2) /
                                 xp.sum(xp.broadcast_to(1/sigma**2,data.shape))) if fit_offset else 0.
        predicted = predicted+self.last_offset
        residual = predicted-data
        weighted = residual/(sigma*sigma*len(data))
        return .5*xp.mean((residual/sigma)**2), predicted, weighted, unit

    def value_gradient(self, model, data, sigma, fit_offset=False):
        value, predicted, weighted, unit = self.misfit(model,data,sigma,fit_offset)
        return value, self.data_gradient(weighted,unit), predicted


class FieldOperator(MisfitOperator):
    def __init__(self, mesh, observations, mode='gravity', *, device='cpu', gpu_id=0,
                 field_strength=50000., inclination=60., declination=0., cache_mb=128,
                 cancel_event=None, memory_policy='manual'):
        if mode not in ('gravity', 'induced', 'vector'):
            raise ValueError('Неизвестный режим 3D-моделирования.')
        self.mesh, self.mode, self.cancel_event = mesh, mode, cancel_event
        self.observations = np.asarray(observations, dtype=float)
        if self.observations.ndim != 2 or self.observations.shape[1] != 3 or not len(self.observations) or not np.isfinite(self.observations).all():
            raise ValueError('Наблюдения: конечный массив N×3 East, North, Down.')
        if not mesh.active.any():
            raise ValueError('В модели нет активных ячеек.')
        self.cells = mesh.cells()
        if np.max(self.observations[:,2]) > np.min(self.cells[mesh.active,2]):
            raise ValueError('Все точки наблюдения должны находиться выше или на уровне самой высокой активной грани модели.')
        if not np.isfinite(field_strength) or field_strength <= 0 or not np.isfinite(cache_mb) or cache_mb < 0:
            raise ValueError('Модуль главного поля должен быть >0; кэш — ≥0.')
        self.xp = backend(device, gpu_id)
        self.resources=resource_plan(available_memory(self.xp),mesh.size,len(self.observations),
                                     mode,device,memory_policy,cache_mb)
        self.field_strength = float(field_strength)
        self.f = self.xp.asarray(direction(inclination, declination))
        self.h = field_strength / (400*np.pi)  # nT to A/m, μ0=4π·10^-7
        self.cache = {}
        self.cache_limit = self.resources['cache_mib'] * 1024**2
        self.cache_bytes = 0
        self.monitor = None

    @property
    def parameter_shape(self):
        return (self.mesh.size, 3) if self.mode == 'vector' else (self.mesh.size,)

    def blocks(self, stage='field'):
        ob,cb=self.resources['observation_block'],self.resources['cell_block']
        monitor=self.monitor
        if monitor:
            total=((len(self.observations)+ob-1)//ob)*sum(self.mesh.active[c:c+cb].any() for c in range(0,self.mesh.size,cb))
            monitor.begin_stage(stage,int(total))
        completed=0
        for o in range(0, len(self.observations), ob):
            for c in range(0, self.mesh.size, cb):
                check_cancel(self.cancel_event)
                os, cs = slice(o,o+ob), np.flatnonzero(self.mesh.active[c:c+cb])+c
                if not len(cs):
                    continue
                key = o,c
                a = self.cache.get(key)
                if a is None:
                    if self.xp is np:
                        a = prism_kernel(self.observations[os],self.cells[cs],self.mode!='gravity')
                    else:
                        from .prism_cuda import prism_block
                        a = prism_block(self.observations[os],self.cells[cs],self.mode!='gravity')
                    if self.cache_bytes+a.nbytes <= self.cache_limit:
                        self.cache[key] = a
                        self.cache_bytes += a.nbytes
                yield os,cs,a
                completed+=1
                if monitor: monitor.update(completed)

    def field(self, model):
        xp = self.xp
        m = xp.asarray(model).reshape(self.parameter_shape)
        out = xp.zeros((len(self.observations),3)) if self.mode != 'gravity' else xp.zeros(len(self.observations))
        for os,cs,a in self.blocks('field'):
            if self.mode == 'gravity':
                out[os] += a @ m[cs]
            else:
                v = m[cs] if self.mode == 'vector' else m[cs,None]*self.h*self.f
                out[os,0] += a[...,0]@v[:,0]+a[...,3]@v[:,1]+a[...,4]@v[:,2]
                out[os,1] += a[...,3]@v[:,0]+a[...,1]@v[:,1]+a[...,5]@v[:,2]
                out[os,2] += a[...,4]@v[:,0]+a[...,5]@v[:,1]+a[...,2]@v[:,2]
        return out

    def predict(self, model):
        field = self.field(model)
        if self.mode == 'gravity':
            return field
        return self.xp.linalg.norm(field+self.field_strength*self.f, axis=1)-self.field_strength

    def field_and_direction(self, model):
        xp = self.xp
        field = self.field(model)
        if self.mode == 'gravity':
            return field, None
        total = field+self.field_strength*self.f
        norm = xp.linalg.norm(total, axis=1)
        if bool(xp.any(norm < 1e-8)):
            raise ValueError('Суммарное магнитное поле близко к нулю; измените ограничения.')
        return norm-self.field_strength, total/norm[:,None]

    def data_gradient(self, weighted, unit):
        xp = self.xp
        gradient = xp.zeros(self.parameter_shape)
        for os,cs,a in self.blocks('gradient'):
            if self.mode == 'gravity':
                gradient[cs] += a.T @ weighted[os]
            else:
                v = weighted[os,None]*unit[os]
                g = xp.stack((a[...,0].T@v[:,0]+a[...,3].T@v[:,1]+a[...,4].T@v[:,2],
                              a[...,3].T@v[:,0]+a[...,1].T@v[:,1]+a[...,5].T@v[:,2],
                              a[...,4].T@v[:,0]+a[...,5].T@v[:,1]+a[...,2].T@v[:,2]),axis=1)
                gradient[cs] += g if self.mode == 'vector' else self.h*(g@self.f)
        return gradient


def regularization(model, mesh, damping, smoothness, xp=np, *, need_value=True, need_gradient=True):
    """Mean square smallness and first differences scaled by cell aspect ratio."""
    active = xp.asarray(mesh.active).reshape((mesh.size,)+(() if model.ndim==1 else (1,)))
    count = int(mesh.active.sum())*(1 if model.ndim==1 else 3)
    grad = damping*model*active/max(count,1) if need_gradient else None
    value = .5*damping*xp.sum(model*model*active)/max(count,1) if need_value else None
    shape = mesh.shape+(() if model.ndim == 1 else (3,))
    cube = model.reshape(shape)
    g = grad.reshape(shape) if need_gradient else None
    axes = (mesh.east, mesh.north, mesh.depth)
    widths = [np.min(np.diff(a)) for a in axes]
    mask = xp.asarray(mesh.active.reshape(mesh.shape))
    for axis in range(3):
        if shape[axis] < 2:
            continue
        d = xp.diff(cube, axis=axis)
        lo,hi = [slice(None)]*len(shape),[slice(None)]*len(shape)
        lo[axis],hi[axis] = slice(None,-1),slice(1,None)
        valid = mask[tuple(lo[:3])]&mask[tuple(hi[:3])]
        distances = np.diff((axes[axis][1:]+axes[axis][:-1])/2)
        dims = [1]*len(shape); dims[axis] = len(distances)
        valid = valid if model.ndim==1 else valid[...,None]
        coefficient = smoothness*(min(widths)/xp.asarray(distances).reshape(dims))**2*valid/max(int(xp.sum(valid))*(1 if model.ndim==1 else 3),1)
        if need_value:
            value += .5*xp.sum(coefficient*d*d)
        if need_gradient:
            g[tuple(lo)] -= coefficient*d
            g[tuple(hi)] += coefficient*d
    return value, grad


@dataclass
class InversionResult:
    mesh: Mesh
    model: np.ndarray
    observations: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    history: list
    settings: dict
    status: str

    def save(self, path):
        np.savez_compressed(path, east=self.mesh.east, north=self.mesh.north, depth=self.mesh.depth,
                            active=self.mesh.active, bounds=np.empty(0) if self.mesh.bounds is None else self.mesh.bounds,
                            model=self.model, observations=self.observations, observed=self.observed,
                            predicted=self.predicted, metadata=json.dumps(dict(settings=self.settings,
                            status=self.status, history=self.history),ensure_ascii=False))

    @classmethod
    def load(cls, path):
        with np.load(path,allow_pickle=False) as saved:
            meta=json.loads(str(saved['metadata']))
            mesh=Mesh(saved['east'],saved['north'],saved['depth'],saved['active'] if 'active' in saved else None,saved['bounds'] if 'bounds' in saved and saved['bounds'].size else None)
            model=saved['model'].copy()
            expected=(mesh.size,3) if meta['settings']['mode']=='vector' else (mesh.size,)
            if model.shape!=expected or not np.isfinite(model).all():
                raise ValueError('Некорректная размерность / значения 3D-модели.')
            return cls(mesh,model,saved['observations'].copy(),saved['observed'].copy(),
                       saved['predicted'].copy(),meta['history'],meta['settings'],meta['status'])


def invert(operator, data, *, sigma=1., iterations=100, lower=-1., upper=1.,
           initial=None, damping=.01, smoothness=.1, depth_index=1., step_scale=1.,
           target_rms=1., fit_offset=False, progress=None, cancel_event=None,
           early_stop=False, early_tolerance=.001, early_patience=5, work_progress=None):
    """Projected preconditioned gradient with Armijo backtracking, exact ΔT."""
    xp,mesh = operator.xp,operator.mesh
    timer=WorkProgress(work_progress,xp)
    data = np.asarray(data,dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma,dtype=float),data.shape)
    if data.shape != (len(operator.observations),) or not np.isfinite(data).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):
        raise ValueError('Нужны конечные данные и положительные ошибки для всех точек.')
    if not np.isfinite([lower,upper,damping,smoothness,depth_index,step_scale,target_rms]).all() or lower>=upper or min(damping,smoothness,target_rms)<0 or not 0<=depth_index<=4 or not 0<step_scale<=2 or iterations<1:
        raise ValueError('Проверьте границы, регуляризацию, индекс глубины [0,4], шаг (0,2] и итерации.')
    if operator.mode=='induced' and lower<0:
        raise ValueError('Для индуцированной восприимчивости задайте нижнюю границу ≥0.')
    if not np.isfinite(early_tolerance) or not 0 <= early_tolerance < 1 or not isinstance(early_patience, int) or early_patience < 1:
        raise ValueError('Ранняя остановка: относительный порог [0,1), терпение ≥1 итерации.')
    data,sigma = xp.asarray(data),xp.asarray(sigma)
    m = xp.zeros(operator.parameter_shape) if initial is None else xp.asarray(initial,dtype=float).reshape(operator.parameter_shape).copy()
    if not bool(xp.isfinite(m).all()):
        raise ValueError('Начальная модель содержит пропуски.')
    m = xp.clip(m,lower,upper)
    active = xp.asarray(mesh.active)
    m[~active] = 0
    if mesh.bounds is None:
        z = np.tile((mesh.depth[1:]+mesh.depth[:-1])/2, mesh.size//mesh.shape[2])
    else:
        z = (mesh.bounds[:,2]+mesh.bounds[:,5])/2
    z -= np.max(operator.observations[:,2])
    zmin=z[mesh.active].min()
    weights = xp.asarray((np.maximum(z,zmin)/zmin)**depth_index)
    weights[~active]=0
    weights /= weights.max()
    if m.ndim==2:
        weights = weights[:,None]
    history,status = [],'iteration_limit'
    step = 1.
    def objective(v):
        val,pred,weighted,unit = operator.misfit(v,data,sigma,fit_offset)
        timer.begin_stage('regularization',0)
        rv,_ = regularization(v,mesh,damping,smoothness,xp,need_gradient=False)
        return val+rv,pred,val,rv,weighted,unit
    check_cancel(cancel_event)
    previous_monitor=operator.monitor
    operator.monitor=timer if work_progress else None
    best = None
    stale = 0
    try:
        value,pred,dv,rv,weighted,unit = objective(m)
        offset = operator.last_offset
        for iteration in range(iterations+1):
            row = dict(iteration=iteration,objective=float(value),data_objective=float(dv),
                       regularization=float(rv),rmse=float(xp.sqrt(xp.mean((pred-data)**2))),
                       normalized_rms=float(xp.sqrt(2*dv)))
            timer.sync()
            row.update(iteration_seconds=perf_counter()-timer.iteration_started,
                       elapsed_seconds=perf_counter()-timer.started,
                       trial_evaluations=timer.trial)
            history.append(row)
            if best is None or row['normalized_rms'] < best[2]['normalized_rms']:
                best = (m.copy(), pred.copy(), dict(row), offset)
            if progress:
                progress(dict(row))
            check_cancel(cancel_event)
            if row['normalized_rms']<=target_rms:
                status='target_rms'; break
            if early_stop and len(history)>1:
                previous = history[-2]['normalized_rms']
                improvement = (previous-row['normalized_rms'])/max(abs(previous),1e-30)
                stale = stale+1 if improvement < early_tolerance else 0
                if stale >= early_patience:
                    status='early_stop'; break
            if iteration==iterations:
                break
            timer.begin_iteration(iteration+1)
            gradient = operator.data_gradient(weighted,unit)
            timer.begin_stage('regularization',0)
            _,rg = regularization(m,mesh,damping,smoothness,xp,need_value=False)
            gradient += rg
            del rg
            direction_ = -weights*gradient
            if iteration==0:
                step = step_scale*(upper-lower)*.05/max(float(xp.max(xp.abs(direction_))),1e-30)
            accepted=False
            for trial_index in range(40):
                timer.trial=trial_index+1
                check_cancel(cancel_event)
                candidate=xp.clip(m+step*direction_,lower,upper)
                candidate[~active] = 0
                delta=candidate-m
                if float(xp.max(xp.abs(delta))) < 1e-12*max(1.,upper-lower):
                    status='stationary'; break
                trial=objective(candidate)
                if float(trial[0]) <= float(value+1e-4*xp.sum(gradient*delta)):
                    m=candidate
                    value,pred,dv,rv,weighted,unit=trial
                    offset=operator.last_offset
                    accepted=True; step*=1.5
                    break
                step*=.5
            if not accepted:
                status='stationary' if status=='stationary' else 'line_search_limit'
                break
    except ComputationCancelled:
        if best is None:
            raise
        status='cancelled'
    finally:
        operator.monitor=previous_monitor
    timer.sync()
    elapsed_seconds=perf_counter()-timer.started
    selected_iteration=history[-1]['iteration']
    if status in ('cancelled','early_stop'):
        m,pred,selected,offset=best
        selected_iteration=selected['iteration']
    settings=dict(mode=operator.mode,device='cpu' if xp is np else 'gpu',
                  field_strength_nT=operator.field_strength,field_direction_END=host(operator.f).tolist(),
                  lower=lower,upper=upper,damping=damping,smoothness=smoothness,depth_index=depth_index,
                  step_scale=step_scale,target_rms=target_rms,iterations=iterations,
                  coordinate_order='East North Down, m',density_unit='g/cm3',magnetization_unit='A/m',
                  susceptibility_unit='SI',magnetic_anomaly='exact total field magnitude',
                  fit_offset=bool(fit_offset),constant_offset=offset,
                  early_stop=bool(early_stop),early_tolerance=early_tolerance,early_patience=early_patience,
                  selected_iteration=selected_iteration,best_criterion='minimum normalized RMS among completed iterates',
                  resources=operator.resources,
                  elapsed_seconds=elapsed_seconds,initial_evaluation_seconds=history[0]['iteration_seconds'],
                  sigma=host(sigma).tolist())
    return InversionResult(mesh,host(m),operator.observations.copy(),host(data),host(pred),history,settings,status)
