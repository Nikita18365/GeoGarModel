"""Double-precision CUDA evaluation of one rectangular-prism block."""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawKernel(r'''
    __device__ double log_plus(double a, double b, double c, double r) {
        const double tiny = 2.2250738585072014e-308;
        double arg = a < 0 ? (b*b + c*c) / fmax(r-a, tiny) : r+a;
        return log(fmax(arg, tiny));
    }

    __device__ double prism_angle(double a, double b) {
        const double pi = 3.14159265358979323846;
        double angle = atan2(a, b);
        if (angle > pi/2) return angle-pi;
        if (angle < -pi/2) return angle+pi;
        return angle;
    }

    extern "C" __global__ void prism_block(const double* obs, const double* cells,
                                            int nobs, int ncells, int magnetic,
                                            double* result) {
        int pair = blockIdx.x * blockDim.x + threadIdx.x;
        if (pair >= nobs*ncells) return;
        const double* point = obs + 3*(pair/ncells);
        const double* cell = cells + 6*(pair%ncells);
        double gravity = 0;
        double tensor[6] = {};
        for (int i=0; i<2; ++i) {
            double x = cell[3*i] - point[0];
            for (int j=0; j<2; ++j) {
                double y = cell[3*j+1] - point[1];
                for (int k=0; k<2; ++k) {
                    double z = cell[3*k+2] - point[2];
                    double r = sqrt(x*x+y*y+z*z);
                    double sign = ((i+j+k)&1) ? 1. : -1.;
                    double lx = log_plus(x,y,z,r);
                    double ly = log_plus(y,x,z,r);
                    double zz = prism_angle(x*y,z*r);
                    if (magnetic) {
                        tensor[0] -= sign*prism_angle(y*z,x*r);
                        tensor[1] -= sign*prism_angle(x*z,y*r);
                        tensor[2] -= sign*zz;
                        tensor[3] += sign*log_plus(z,x,y,r);
                        tensor[4] += sign*ly;
                        tensor[5] += sign*lx;
                    } else {
                        gravity -= sign*(x*ly+y*lx-z*zz);
                    }
                }
            }
        }
        if (magnetic) {
            for (int c=0; c<6; ++c) result[6*pair+c] = 100.*tensor[c];
        } else {
            result[pair] = 6.67430e-3*gravity;
        }
    }
    ''', 'prism_block', options=('--fmad=false',))


def prism_block(observations, cells, magnetic):
    import cupy as cp
    obs = cp.asarray(observations, dtype=cp.float64, order='C')
    cells = cp.asarray(cells, dtype=cp.float64, order='C')
    shape = (len(obs), len(cells))
    result = cp.empty(shape + ((6,) if magnetic else ()), dtype=cp.float64)
    threads = 128
    blocks = (len(obs) * len(cells) + threads - 1) // threads
    if blocks:
        args = (obs, cells, np.int32(len(obs)), np.int32(len(cells)), np.int32(magnetic), result)
        _kernel()((blocks,), (threads,), args)
    return result
