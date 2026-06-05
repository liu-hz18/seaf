"""快速性能对比：quality_advanced 向量化前后"""
import time, sys, os
sys.path.insert(0, '.')
import pandas as pd
from qpipe.frame3d import Frame3D
from seafquant.data_generator import generate_synthetic_data

gen = generate_synthetic_data(n_times=80, n_stocks=100, noise_ratio=0.3, seed=42)
frames = [f.df for f in gen]
big_df = pd.concat(frames, axis=0).sort_index(level=0)
f3d = Frame3D(big_df)
print(f'Data: {f3d}\n')

from seafquant.factor.factors_quality_advanced import compute_quality_advanced_factors
from seafquant.factor.factors_quality_pattern import compute_quality_pattern_factors

t0 = time.perf_counter()
r1 = compute_quality_advanced_factors('qa', f3d, None)
t1 = time.perf_counter()
t_adv = t1 - t0
print(f'quality_advanced (8 cols): {t_adv:.3f}s  cols={list(r1.df.columns)}')

t0 = time.perf_counter()
r2 = compute_quality_pattern_factors('qp', f3d, None)
t1 = time.perf_counter()
t_pat = t1 - t0
print(f'quality_pattern  (8 cols): {t_pat:.3f}s  cols={list(r2.df.columns)}')

print(f'\nparallel max (bottleneck): {max(t_adv, t_pat):.3f}s')
print(f'total 16 factor cols: {len(r1.df.columns) + len(r2.df.columns)}')
