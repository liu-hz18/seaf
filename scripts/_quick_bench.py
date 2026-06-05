"""Quick bench: quality_autocorr + counting_streak only"""
import time, sys
sys.path.insert(0, '.')
import pandas as pd
from qpipe.frame3d import Frame3D
from seafquant.data_generator import generate_synthetic_data

gen = generate_synthetic_data(n_times=130, n_stocks=500, noise_ratio=0.3, seed=42)
frames = [f.df for f in gen]
big_df = pd.concat(frames, axis=0).sort_index(level=0)
f3d = Frame3D(big_df)

from seafquant.factor.factors_quality_autocorr import compute_quality_autocorr_factors
from seafquant.factor.factors_counting_streak import compute_counting_streak_factors

for name, func in [('quality_autocorr', compute_quality_autocorr_factors),
                    ('counting_streak', compute_counting_streak_factors)]:
    _ = func(name, f3d, None)  # warmup
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = func(name, f3d, None)
        ts.append(time.perf_counter() - t0)
    print(f'{name:25s}  avg={sum(ts)/len(ts):.4f}s  ncols={len(r.df.columns)}')
