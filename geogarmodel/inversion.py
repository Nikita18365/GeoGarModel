"""3D rectangular-prism inversion. Internal coordinates: East, North, Down (m).

Independent implementation of prism integrals and projected gradient inversion.
CPU NumPy / GPU CuPy share the same double precision kernels and objective.
"""
from dataclasses import dataclass
from itertools import product
import json
import numpy as np
from .compute import check_cancel


@dataclass
class Mesh:
    east: np.ndarray
    north: np.ndarray
    depth: np.ndarray

    def __post_init__(self):
        for key in ('east', 'north', 'depth'):
            a = np.asarray(getattr(self, key), dtype=float)
            if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or not (np.diff(a) > 0).all():
                raise ValueError('Границы ячеек должны быть конечными и возрастающими.')
            if not np.allclose(np.diff(a), np.diff(a)[0]):
                raise ValueError('Требуется равномерная сетка по каждой оси.')
            setattr(self, key, a)

    @property
    def shape(self):
        return tuple(len(a)-1 for a in (self.east, self.north, self.depth))

    @property
    def size(self):
        return int(np.prod(self.shape))

    def cells(self):
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


class FieldOperator:
    def __init__(self, mesh, observations, mode='gravity', *, device='cpu', gpu_id=0,
                 field_strength=50000., inclination=60., declination=0., cache_mb=128,
                 cancel_event=None):
        if mode not in ('gravity', 'induced', 'vector'):
            raise ValueError('Неизвестный тип инверсии.')
        self.mesh, self.mode, self.cancel_event = mesh, mode, cancel_event
        self.observations = np.asarray(observations, dtype=float)
        if self.observations.ndim != 2 or self.observations.shape[1] != 3 or not len(self.observations) or not np.isfinite(self.observations).all():
            raise ValueError('Наблюдения: конечный массив N×3 East, North, Down.')
        if np.max(self.observations[:,2]) >= mesh.depth[0]:
            raise ValueError('Все точки наблюдения должны находиться строго выше верхней грани сетки.')
        if not np.isfinite(field_strength) or field_strength <= 0 or not np.isfinite(cache_mb) or cache_mb < 0:
            raise ValueError('Модуль главного поля должен быть >0; кэш — ≥0.')
        self.xp = backend(device, gpu_id)
        self.field_strength = float(field_strength)
        self.f = self.xp.asarray(direction(inclination, declination))
        self.h = field_strength / (400*np.pi)  # nT to A/m, μ0=4π·10^-7
        self.cells = mesh.cells()
        self.cache = {}
        self.cache_limit = cache_mb * 1024**2
        self.cache_bytes = 0

    @property
    def parameter_shape(self):
        return (self.mesh.size, 3) if self.mode == 'vector' else (self.mesh.size,)

    def blocks(self):
        for o in range(0, len(self.observations), 64):
            for c in range(0, self.mesh.size, 512):
                check_cancel(self.cancel_event)
                os, cs = slice(o,o+64), slice(c,c+512)
                key = o,c
                a = self.cache.get(key)
                if a is None:
                    a = prism_kernel(self.observations[os], self.cells[cs], self.mode != 'gravity', self.xp)
                    if self.cache_bytes+a.nbytes <= self.cache_limit:
                        self.cache[key] = a
                        self.cache_bytes += a.nbytes
                yield os,cs,a

    def field(self, model):
        xp = self.xp
        m = xp.asarray(model).reshape(self.parameter_shape)
        out = xp.zeros((len(self.observations),3)) if self.mode != 'gravity' else xp.zeros(len(self.observations))
        for os,cs,a in self.blocks():
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

    def value_gradient(self, model, data, sigma, fit_offset=False):
        xp = self.xp
        field = self.field(model)
        if self.mode == 'gravity':
            predicted, unit = field, None
        else:
            total = field+self.field_strength*self.f
            norm = xp.linalg.norm(total, axis=1)
            if bool(xp.any(norm < 1e-8)):
                raise ValueError('Суммарное магнитное поле близко к нулю; измените ограничения.')
            predicted, unit = norm-self.field_strength, total/norm[:,None]
        if fit_offset:
            predicted = predicted+xp.sum((data-predicted)/sigma**2)/xp.sum(1/sigma**2)
        residual = predicted-data
        weighted = residual/(sigma*sigma*len(data))
        gradient = xp.zeros(self.parameter_shape)
        for os,cs,a in self.blocks():
            if self.mode == 'gravity':
                gradient[cs] += a.T @ weighted[os]
            else:
                v = weighted[os,None]*unit[os]
                g = xp.stack((a[...,0].T@v[:,0]+a[...,3].T@v[:,1]+a[...,4].T@v[:,2],
                              a[...,3].T@v[:,0]+a[...,1].T@v[:,1]+a[...,5].T@v[:,2],
                              a[...,4].T@v[:,0]+a[...,5].T@v[:,1]+a[...,2].T@v[:,2]),axis=1)
                gradient[cs] += g if self.mode == 'vector' else self.h*(g@self.f)
        return .5*xp.mean((residual/sigma)**2), gradient, predicted


def regularization(model, mesh, damping, smoothness, xp=np):
    """Mean square smallness and first differences scaled by cell aspect ratio."""
    grad = damping*model/model.size
    value = .5*damping*xp.mean(model*model)
    shape = mesh.shape+(() if model.ndim == 1 else (3,))
    cube, g = model.reshape(shape), grad.reshape(shape)
    widths = [a[1]-a[0] for a in (mesh.east, mesh.north, mesh.depth)]
    for axis in range(3):
        if shape[axis] < 2:
            continue
        d = xp.diff(cube, axis=axis)
        coefficient = smoothness*(min(widths)/widths[axis])**2/d.size
        value += .5*coefficient*xp.sum(d*d)
        lo,hi = [slice(None)]*len(shape),[slice(None)]*len(shape)
        lo[axis],hi[axis] = slice(None,-1),slice(1,None)
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
                            model=self.model, observations=self.observations, observed=self.observed,
                            predicted=self.predicted, metadata=json.dumps(dict(settings=self.settings,
                            status=self.status, history=self.history),ensure_ascii=False))

    @classmethod
    def load(cls, path):
        with np.load(path,allow_pickle=False) as saved:
            meta=json.loads(str(saved['metadata']))
            mesh=Mesh(saved['east'],saved['north'],saved['depth'])
            model=saved['model'].copy()
            expected=(mesh.size,3) if meta['settings']['mode']=='vector' else (mesh.size,)
            if model.shape!=expected or not np.isfinite(model).all():
                raise ValueError('Некорректная размерность / значения 3D-модели.')
            return cls(mesh,model,saved['observations'].copy(),saved['observed'].copy(),
                       saved['predicted'].copy(),meta['history'],meta['settings'],meta['status'])


def invert(operator, data, *, sigma=1., iterations=100, lower=-1., upper=1.,
           initial=None, damping=.01, smoothness=.1, depth_index=1., step_scale=1.,
           target_rms=1., fit_offset=False, progress=None, cancel_event=None):
    """Projected preconditioned gradient with Armijo backtracking, exact ΔT."""
    xp,mesh = operator.xp,operator.mesh
    data = np.asarray(data,dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma,dtype=float),data.shape)
    if data.shape != (len(operator.observations),) or not np.isfinite(data).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):
        raise ValueError('Нужны конечные данные и положительные ошибки для всех точек.')
    if not np.isfinite([lower,upper,damping,smoothness,depth_index,step_scale,target_rms]).all() or lower>=upper or min(damping,smoothness,target_rms)<0 or not 0<=depth_index<=4 or not 0<step_scale<=2 or iterations<1:
        raise ValueError('Проверьте границы, регуляризацию, индекс глубины [0,4], шаг (0,2] и итерации.')
    if operator.mode=='induced' and lower<0:
        raise ValueError('Для индуцированной восприимчивости задайте нижнюю границу ≥0.')
    data,sigma = xp.asarray(data),xp.asarray(sigma)
    m = xp.zeros(operator.parameter_shape) if initial is None else xp.asarray(initial,dtype=float).reshape(operator.parameter_shape).copy()
    if not bool(xp.isfinite(m).all()):
        raise ValueError('Начальная модель содержит пропуски.')
    m = xp.clip(m,lower,upper)
    z = mesh.centers()[:,2]-np.max(operator.observations[:,2])
    weights = xp.asarray((z/z.min())**depth_index)
    weights /= weights.max()
    if m.ndim==2:
        weights = weights[:,None]
    history,status = [],'iteration_limit'
    step = 1.
    def objective(v):
        val,grad,pred = operator.value_gradient(v,data,sigma,fit_offset)
        rv,rg = regularization(v,mesh,damping,smoothness,xp)
        return val+rv,grad+rg,pred,val,rv
    check_cancel(cancel_event)
    value,gradient,pred,dv,rv = objective(m)
    for iteration in range(iterations+1):
        check_cancel(cancel_event)
        row = dict(iteration=iteration,objective=float(value),data_objective=float(dv),
                   regularization=float(rv),rmse=float(xp.sqrt(xp.mean((pred-data)**2))),
                   normalized_rms=float(xp.sqrt(2*dv)))
        history.append(row)
        if progress:
            progress(dict(row))
        if row['normalized_rms']<=target_rms:
            status='target_rms'; break
        if iteration==iterations:
            break
        direction_ = -weights*gradient
        # Estimate first step in model units; backtracking always checks the objective.
        if iteration==0:
            step = step_scale*(upper-lower)*.05/max(float(xp.max(xp.abs(direction_))),1e-30)
        accepted=False
        for _ in range(40):
            check_cancel(cancel_event)
            candidate=xp.clip(m+step*direction_,lower,upper)
            delta=candidate-m
            if float(xp.max(xp.abs(delta))) < 1e-12*max(1.,upper-lower):
                status='stationary'; break
            trial=objective(candidate)
            if float(trial[0]) <= float(value+1e-4*xp.sum(gradient*delta)):
                m=candidate
                value,gradient,pred,dv,rv=trial
                accepted=True; step*=1.5
                break
            step*=.5
        if not accepted:
            status='stationary' if status=='stationary' else 'line_search_limit'
            break
    settings=dict(mode=operator.mode,device='cpu' if xp is np else 'gpu',
                  field_strength_nT=operator.field_strength,field_direction_END=host(operator.f).tolist(),
                  lower=lower,upper=upper,damping=damping,smoothness=smoothness,depth_index=depth_index,
                  step_scale=step_scale,target_rms=target_rms,iterations=iterations,
                  coordinate_order='East North Down, m',density_unit='g/cm3',magnetization_unit='A/m',
                  susceptibility_unit='SI',magnetic_anomaly='exact total field magnitude',
                  fit_offset=bool(fit_offset),constant_offset=float(xp.mean(pred-operator.predict(m))),
                  sigma=host(sigma).tolist())
    return InversionResult(mesh,host(m),operator.observations.copy(),host(data),host(pred),history,settings,status)
