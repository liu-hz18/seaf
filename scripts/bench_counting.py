"""快速性能对比：counting 向量化后耗时"""
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

from seafquant.factor.factors_counting import compute_counting_factors

t0 = time.perf_counter()
r = compute_counting_factors('counting', f3d, None)
t1 = time.perf_counter()
print(f'counting (16 cols): {t1-t0:.3f}s  cols={list(r.df.columns)}')
