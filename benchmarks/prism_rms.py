"""Measure time to the same normalized RMS on a reproducible density model."""
import argparse
from statistics import median
from time import perf_counter

import numpy as np

from geogarmodel.inversion import FieldOperator, Mesh, invert, prism_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--cache-mb", type=int, default=0)
    args = parser.parse_args()
    if args.cache_mb < 0:
        parser.error("--cache-mb must be nonnegative")

    mesh = Mesh(np.arange(17) * 250., np.arange(17) * 250., np.arange(9) * 200. + 80.)
    rng = np.random.default_rng(852)
    points = rng.uniform([0, 0, -70], [4000, 4000, -10], (256, 3))
    body = np.all(abs(mesh.centers() - [1500, 2100, 630]) < [560, 510, 300], axis=1)
    data = prism_kernel(points, mesh.cells()[body]).sum(axis=1) * 0.12
    sigma = np.std(data) * 0.15
    times = []
    for run in range(4):
        operator = FieldOperator(mesh, points, device=args.device, cache_mb=args.cache_mb)
        if args.device == "gpu":
            operator.xp.cuda.Stream.null.synchronize()
        start = perf_counter()
        result = invert(operator, data, sigma=sigma, iterations=80, target_rms=1., step_scale=2.)
        if args.device == "gpu":
            operator.xp.cuda.Stream.null.synchronize()
        elapsed = perf_counter() - start
        if result.status != "target_rms":
            raise RuntimeError(f"Target RMS not reached ({result.status})")
        row = result.history[-1]
        print(f"Run {run}: {elapsed:.3f} s, {row['iteration']} iterations, RMS/sigma {row['normalized_rms']:.6f}")
        if run > 0:
            times.append(elapsed)
    print(f"Median after warm-up: {median(times):.3f} s ({args.device}, cache {args.cache_mb} MiB)")


if __name__ == "__main__":
    main()
