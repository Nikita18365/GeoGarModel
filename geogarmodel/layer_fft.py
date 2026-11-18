"""Layerwise linear convolution of analytical density-prism responses (East, North, Down)."""
from fractions import Fraction
import math

import numpy as np
from scipy.fft import next_fast_len

from .compute import check_cancel
from .inversion import MisfitOperator, backend, prism_kernel
from .inversion_resources import MIB, available_memory


HEIGHT_ERROR = 'Для послойной 2D FFT высота наблюдений должна быть постоянной для всех точек.'


def lattice_indices(values, step):
    positions = (values - values.min()) / step
    rounded = np.rint(positions)
    if not np.isfinite(positions).all() or rounded.max() > 2**31 or np.max(np.abs(positions-rounded)) > 1e-6:
        raise ValueError('Для FFT точки должны лежать на регулярной сетке с шагом модели по East и North.')
    return rounded.astype(np.int64)


def observation_edges(values):
    axis = np.unique(values)
    if len(axis) < 2 or not np.isfinite(axis).all():
        raise ValueError('Для новой FFT-сетки нужны минимум два различных конечных значения по каждой оси.')
    step = np.diff(axis).min()
    indices = lattice_indices(axis, step)
    return axis[0], step, int(indices[-1])+1


def model_axis(values, count, padding):
    """Return the first edge, pitch and cell count for a padded axis."""
    origin, step, nodes = observation_edges(values)
    span = (nodes - 1) * step
    if count is None:
        extra = math.ceil(padding * (nodes - 1))
        return origin - (extra + .5) * step, step, nodes + 2 * extra
    width = span * (1 + 2 * padding) + step
    ratio = width / (count * step)
    # Choose a nearby compatible width with a small common FFT lattice.
    ratios = [Fraction(max(1, math.ceil(ratio * n - 1e-10)), n) for n in range(1, 65)]
    best = min(ratios)
    selected = min((r for r in ratios if r <= best * 1.05), key=lambda r: (r.denominator, r))
    pitch = float(selected) * step
    return origin + (span - count * pitch) / 2, pitch, count


def uniform_step(axis):
    step = (axis[-1] - axis[0]) / (len(axis) - 1)
    if np.max(np.abs((axis - axis[0]) / step - np.arange(len(axis)))) > 1e-6:
        raise ValueError('Размер ячеек FFT должен быть постоянным по каждой горизонтальной оси.')
    return step


def common_axis(coords, step):
    try:
        return lattice_indices(coords, step), step, 1
    except ValueError:
        _, obs_step, _ = observation_edges(coords)
        ratio = Fraction(step / obs_step).limit_denominator(64)
        if ratio.numerator == 0 or not math.isclose(float(ratio), step / obs_step, rel_tol=1e-9):
            raise ValueError('Шаги поля и модели несовместимы для FFT. Выберите новую сетку или перенесите модель.') from None
        pitch = step / ratio.numerator
        return lattice_indices(coords, pitch), pitch, ratio.numerator


def spectrum_budget(free, workspace, spectrum_bytes, layers, policy, cache_mib):
    fractions = {'balanced': .5, 'performance': .7}
    if policy not in (*fractions, 'manual'):
        raise ValueError('Неизвестный режим памяти.')
    if not math.isfinite(cache_mib) or cache_mib < 0:
        raise ValueError('Кэш должен быть конечным и ≥0 MiB.')
    safe = int(free * .85) - workspace
    required = layers * spectrum_bytes
    if policy == 'manual':
        limit = min(required, int(cache_mib * MIB))
        if (limit // spectrum_bytes) * spectrum_bytes > safe:
            raise ValueError(f'После выделения рабочих массивов для кэша FFT доступно {safe // MIB:,} МиБ. Уменьшите лимит.')
    else:
        limit = min(safe, required, int(free * fractions[policy]))
    capacity = limit // spectrum_bytes
    return capacity, capacity * spectrum_bytes


class LayerFFT(MisfitOperator):
    mode = 'gravity'
    field_strength = 0.
    f = np.zeros(3)

    def __init__(self, mesh, observations, *, device='cpu', gpu_id=0, cancel_event=None,
                 memory_policy='balanced', cache_mb=512):
        self.mesh = mesh
        self.parameter_shape = (mesh.size,)
        self.observations = np.asarray(observations, dtype=float)
        obs = self.observations
        if obs.ndim != 2 or obs.shape[1] != 3 or not len(obs) or not np.isfinite(obs).all():
            raise ValueError('Для FFT нужны конечные координаты East, North, Down.')
        if np.ptp(obs[:, 2]) > 1e-7:
            raise ValueError(HEIGHT_ERROR)
        if not mesh.active.any():
            raise ValueError('Нет активных ячеек.')
        self.cancel_event = cancel_event
        self.monitor = None
        self.xp = xp = backend(device, gpu_id)
        free = available_memory(xp)
        self.xy_indices = []
        self.lags = []
        self.fft_shape = []
        self.widths = []
        self.model_strides = []
        lattice_steps = []
        for axis, coords in zip((mesh.east, mesh.north), obs.T[:2]):
            step = uniform_step(axis)
            indices, pitch, stride = common_axis(coords, step)
            count = int(indices.max())+1
            self.xy_indices.append(xp.asarray(indices))
            self.fft_shape.append(next_fast_len(count+(len(axis)-2)*stride, real=True))
            self.widths.append(step)
            self.model_strides.append(stride)
            lattice_steps.append(pitch)
        # Positive observation-source lags come first. Negative lags wrap at
        # the far edge, so the retained output is the linear convolution.
        self.fft_shape = tuple(self.fft_shape)
        plane = math.prod(self.fft_shape)
        # Solver vectors, complex transforms and a strip of prism integrals.
        workspace = 160*mesh.size + 96*plane + len(obs)*96 + 32*MIB
        if workspace > free*.85:
            raise ValueError('Для модели и рабочей FFT-сетки недостаточно свободной памяти. Проверьте шаг и охват координат.')
        spectrum_bytes = 16*self.fft_shape[0]*(self.fft_shape[1]//2+1)
        for i, axis in enumerate((mesh.east, mesh.north)):
            negative = (len(axis)-2)*self.model_strides[i]
            count = self.fft_shape[i]-negative
            shifts = np.r_[np.arange(count), np.arange(-negative, 0)]
            self.lags.append(xp.asarray(shifts*lattice_steps[i]+(obs[:, i].min()-axis[0])))
        self.spectra = {}
        full = mesh.active.reshape(mesh.shape).copy()
        partial = []
        top = np.inf
        for k in range(mesh.shape[2]):
            if not full[:,:,k].any(): continue
            if mesh.bounds is not None:
                bounds = mesh.bounds.reshape(mesh.shape+(6,))[:,:,k]
                changed = np.zeros(mesh.shape[:2], dtype=bool)
                for j, edge in enumerate((mesh.east[:-1,None],mesh.north[None,:-1],mesh.depth[k],
                                          mesh.east[1:,None],mesh.north[None,1:],mesh.depth[k+1])):
                    changed |= bounds[:,:,j] != edge
                ix, iy = np.nonzero(changed & full[:,:,k])
                if len(ix):
                    partial.append(np.ravel_multi_index((ix,iy,np.full_like(ix,k)),mesh.shape))
                    top = min(top,bounds[ix,iy,2].min())
                    full[ix,iy,k] = False
            if full[:,:,k].any(): top = min(top,mesh.depth[k])
        if obs[0,2] > top: raise ValueError('Плоскость наблюдений должна быть выше активных ячеек.')
        self.layers = np.flatnonzero(full.any(axis=(0,1)))
        self.full = xp.asarray(full)
        self.partial = np.concatenate(partial) if partial else np.empty(0,dtype=int)
        self.partial_bounds = None if not len(self.partial) else xp.asarray(mesh.bounds[self.partial])
        self.device_obs = xp.asarray(obs)
        if device == 'gpu':
            from .prism_cuda import prism_block
            self.prism = prism_block
        else:
            self.prism = prism_kernel
        self.cache_limit, budget = spectrum_budget(
            free, workspace, spectrum_bytes, len(self.layers), memory_policy, cache_mb)
        self.resources = dict(method='layer_fft', layers=len(self.layers), partial_cells=len(self.partial),
                              fft_shape=self.fft_shape, policy=memory_policy, free_mib=free//MIB,
                              cache_mib=budget/MIB, cached_layers=self.cache_limit)

    @property
    def model_slice(self):
        sx, sy, _ = self.mesh.shape
        dx, dy = self.model_strides
        return np.s_[:sx*dx:dx, :sy*dy:dy]

    def spectrum(self,k):
        if k in self.spectra: return self.spectra[k]
        xp = self.xp
        z = self.mesh.depth
        cell = xp.asarray([[0.,0.,z[k],*self.widths,z[k+1]]])
        response = xp.empty(self.fft_shape)
        flat = response.ravel()
        for start in range(0,flat.size,65536):
            check_cancel(self.cancel_event)
            idx = xp.arange(start,min(start+65536,flat.size))
            obs = xp.column_stack((self.lags[0][idx//self.fft_shape[1]],self.lags[1][idx%self.fft_shape[1]],
                                   xp.full(len(idx),self.observations[0,2])))
            flat[start:start+len(idx)] = self.prism(obs,cell,False)[:,0]
        spectrum = xp.fft.rfft2(response)
        if len(self.spectra) < self.cache_limit: self.spectra[k] = spectrum
        return spectrum

    def apply(self, values, transpose):
        xp = self.xp
        ix, iy = self.xy_indices
        model_slice = self.model_slice
        obs = self.observations
        if transpose:
            plane = xp.zeros(self.fft_shape)
            xp.add.at(plane,(ix,iy),values)
            spectrum = xp.fft.rfft2(plane)
            out = xp.zeros(self.mesh.shape)
        else:
            cube = xp.asarray(values).reshape(self.mesh.shape)
            plane = xp.zeros(self.fft_shape)
            spectrum = xp.zeros((self.fft_shape[0],self.fft_shape[1]//2+1),dtype=complex)
        pairs = ((len(obs)+127)//128)*((len(self.partial)+511)//512)
        if self.monitor: self.monitor.begin_stage('gradient' if transpose else 'field',len(self.layers)+pairs)
        for n,k in enumerate(self.layers,1):
            check_cancel(self.cancel_event)
            kernel = self.spectrum(k)
            if transpose:
                out[:,:,k] = xp.fft.irfft2(spectrum*kernel.conj(),s=self.fft_shape,axes=(0,1))[model_slice]*self.full[:,:,k]
            else:
                xp.multiply(cube[:,:,k],self.full[:,:,k],out=plane[model_slice])
                layer = xp.fft.rfft2(plane)
                layer *= kernel
                spectrum += layer
            if self.monitor: self.monitor.update(n)
        out = out.ravel() if transpose else xp.fft.irfft2(spectrum,s=self.fft_shape,axes=(0,1))[ix,iy]
        completed = len(self.layers)
        for start in range(0,len(self.partial),512):
            indices = self.partial[start:start+512]
            cells = self.partial_bounds[start:start+512]
            for row in range(0,len(obs),128):
                check_cancel(self.cancel_event)
                rows = slice(row,row+128)
                a = self.prism(self.device_obs[rows],cells,False)
                if transpose: out[indices] += a.T @ values[rows]
                else: out[rows] += a @ values[indices]
                completed += 1
                if self.monitor: self.monitor.update(completed)
        return out

    def field(self, model):
        return self.apply(self.xp.asarray(model).reshape(-1),False)

    predict = field

    def field_and_direction(self, model):
        return self.field(model), None

    def data_gradient(self, weighted, unit):
        return self.apply(weighted,True)
