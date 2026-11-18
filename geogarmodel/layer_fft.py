"""Layerwise linear convolution of analytical density-prism responses (East, North, Down)."""
import numpy as np
from scipy.fft import next_fast_len

from .compute import check_cancel
from .inversion import backend, prism_kernel
from .inversion_resources import available_memory


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


class LayerFFT:
    mode = 'gravity'
    field_strength = 0.
    f = np.zeros(3)

    def __init__(self, mesh, observations, *, device='cpu', gpu_id=0, cancel_event=None):
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
        for axis, coords in zip((mesh.east, mesh.north), obs.T[:2]):
            step = (axis[-1]-axis[0])/(len(axis)-1)
            if np.max(np.abs((axis-axis[0])/step-np.arange(len(axis)))) > 1e-6:
                raise ValueError('Размер ячеек FFT должен быть постоянным по каждой горизонтальной оси.')
            indices = lattice_indices(coords, step)
            count = int(indices.max())+1
            self.xy_indices.append(xp.asarray(indices))
            self.fft_shape.append(next_fast_len(count+len(axis)-2, real=True))
            self.widths.append(step)
        # Positive observation-source lags come first. Negative lags wrap at
        # the far edge, so the retained output is the linear convolution.
        self.fft_shape = tuple(self.fft_shape)
        plane = int(np.prod(self.fft_shape))
        # Solver vectors, complex transforms and a strip of prism integrals.
        workspace = 160*mesh.size + 96*plane + 32*1024**2
        if workspace > free*.85:
            raise ValueError('Для модели и рабочей FFT-сетки недостаточно свободной памяти. Проверьте шаг и охват координат.')
        self.cache_limit = int((free*.85-workspace)//(16*self.fft_shape[0]*(self.fft_shape[1]//2+1)))
        for i, axis in enumerate((mesh.east, mesh.north)):
            count = self.fft_shape[i]-(len(axis)-2)
            shifts = np.r_[np.arange(count), np.arange(2-len(axis), 0)]
            self.lags.append(xp.asarray(shifts*self.widths[i]+(obs[:, i].min()-axis[0])))
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
        self.resources = dict(method='layer_fft', layers=len(self.layers), partial_cells=len(self.partial))

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
            flat[start:start+len(idx)] = prism_kernel(obs,cell,xp=xp)[:,0]
        spectrum = xp.fft.rfft2(response)
        if len(self.spectra) < self.cache_limit: self.spectra[k] = spectrum
        return spectrum

    def apply(self, values, transpose):
        xp = self.xp
        ix, iy = self.xy_indices
        sx, sy, _ = self.mesh.shape
        obs = self.observations
        if transpose:
            plane = xp.zeros(self.fft_shape)
            xp.add.at(plane,(ix,iy),values)
            spectrum = xp.fft.rfft2(plane)
            out = xp.zeros(self.mesh.shape)
        else:
            cube = xp.asarray(values).reshape(self.mesh.shape)
            spectrum = xp.zeros((self.fft_shape[0],self.fft_shape[1]//2+1),dtype=complex)
        pairs = ((len(obs)+127)//128)*((len(self.partial)+511)//512)
        if self.monitor: self.monitor.begin_stage('gradient' if transpose else 'field',len(self.layers)+pairs)
        for n,k in enumerate(self.layers,1):
            check_cancel(self.cancel_event)
            kernel = self.spectrum(k)
            if transpose:
                out[:,:,k] = xp.fft.irfft2(spectrum*kernel.conj(),s=self.fft_shape,axes=(0,1))[:sx,:sy]*self.full[:,:,k]
            else:
                spectrum += xp.fft.rfft2(cube[:,:,k]*self.full[:,:,k],s=self.fft_shape,axes=(0,1))*kernel
            if self.monitor: self.monitor.update(n)
        out = out.ravel() if transpose else xp.fft.irfft2(spectrum,s=self.fft_shape,axes=(0,1))[ix,iy]
        completed = len(self.layers)
        for start in range(0,len(self.partial),512):
            indices = self.partial[start:start+512]
            cells = xp.asarray(self.mesh.bounds[indices])
            for row in range(0,len(obs),128):
                check_cancel(self.cancel_event)
                rows = slice(row,row+128)
                a = prism_kernel(obs[rows],cells,xp=xp)
                if transpose: out[indices] += a.T @ values[rows]
                else: out[rows] += a @ values[indices]
                completed += 1
                if self.monitor: self.monitor.update(completed)
        return out

    def field(self, model):
        return self.apply(self.xp.asarray(model).reshape(-1),False)

    predict = field

    def value_gradient(self, model, data, sigma, fit_offset=False):
        xp = self.xp
        predicted = self.field(model)
        self.last_offset = float(xp.sum((data-predicted)/sigma**2)/xp.sum(xp.broadcast_to(1/sigma**2,data.shape))) if fit_offset else 0.
        predicted += self.last_offset
        residual = (predicted-data)/sigma
        gradient = self.apply(residual/(len(data)*sigma),True)
        return .5*xp.mean(residual**2),gradient,predicted
