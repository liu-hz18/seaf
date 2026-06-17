"""TSPCT factor benchmark — measure performance of rolling rank optimization."""

import sys
import time

sys.path.insert(0, '.')

from seafquant.data_generator import generate_synthetic_data
from qpipe.frame3d import Frame3D
import pandas as pd


def main():
    # Build test data: 200 days x 500 stocks
    gen = generate_synthetic_data(80, 100, 0.3, 42)
    frames = []
    for _, f3d in gen:
        frames.append(f3d.df)
    big = pd.concat(frames).sort_index(level=0)
    f3d = Frame3D(big)

    from seafquant.factor.tspct import compute_tspct_factors

    # Warmup
    compute_tspct_factors('warmup', 0, f3d, None)

    # Benchmark: 10 iterations
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        result = compute_tspct_factors('bench', 0, f3d, None)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    avg = sum(times) / len(times)
    best = min(times)
    n_factors = len(result.df.columns)
    n_rows = f3d.df.shape[0]

    print(f'Data: {n_rows} rows x {n_factors} factors')
    print(f'Times: {[f"{t*1000:.0f}ms" for t in times]}')
    print(f'Best:  {best*1000:.0f}ms')
    print(f'Avg:   {avg*1000:.0f}ms')
    print(f'Per factor: {avg/n_factors*1000:.2f}ms')


if __name__ == '__main__':
    main()
