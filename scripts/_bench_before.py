"""Quick bench: compare before/after optimization per module."""
import time, sys
sys.path.insert(0, '.')
import pandas as pd
from qpipe.frame3d import Frame3D
from seafquant.data_generator import generate_synthetic_data

gen = generate_synthetic_data(n_times=130, n_stocks=500, noise_ratio=0.3, seed=42)
frames = [f.df for f in gen]
big_df = pd.concat(frames, axis=0).sort_index(level=0)
f3d = Frame3D(big_df)

from seafquant.factors import FACTOR_REGISTRY

modules = [
    'momentum', 'reversal', 'volatility', 'liquidity', 'value',
    'quality_basic', 'quality_advanced', 'quality_autocorr',
    'quality_pattern', 'quality_sign',
    'trend', 'trend_macd', 'size',
    'counting', 'counting_streak', 'counting_nh',
    'intraday', 'interaction', 'cross_section', 'cross_section_neut',
]

for mod in modules:
    func = FACTOR_REGISTRY[mod]
    _ = func(mod, f3d, None)  # warmup
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = func(mod, f3d, None)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f'{mod:25s}  avg={avg:.4f}s  ncols={len(r.df.columns)}')
