"""Momentum 数值对比验证：完整流程。"""
import sys, numpy as np
sys.path.insert(0, '.')
import pandas as pd
from qpipe.frame3d import Frame3D
from seafquant.data_generator import generate_synthetic_data

gen = generate_synthetic_data(n_times=130, n_stocks=100, noise_ratio=0.3, seed=42)
frames = [f.df for f in gen]
big_df = pd.concat(frames, axis=0).sort_index(level=0)
f3d = Frame3D(big_df)

from seafquant.factor.factors_momentum import compute_momentum_factors
EPS = 1e-8

# 参考实现（原始逐次 groupby pct_change）
ref = f3d.copy()
ref_df = ref.df
periods = [1, 3, 5, 10, 20, 40, 60, 120]
vol_windows = sorted(set(max(p, 5) for p in periods))
ref_df['_daily_ret'] = ref_df.groupby('name')['close'].pct_change(1)
for w in vol_windows:
    ref_df[f'_vol_{w}'] = ref_df.groupby('name')['_daily_ret'].rolling(
        w, min_periods=max(1, w // 2)).std().values
for p in periods:
    w = max(p, 5)
    ref_df[f'factor_mom_ret_{p}d'] = ref_df.groupby('name')['close'].pct_change(p)
    ref_df[f'factor_mom_voladj_{p}d'] = (
        ref_df[f'factor_mom_ret_{p}d'] / (ref_df[f'_vol_{w}'] + EPS))
factor_cols = [c for c in ref_df.columns if c.startswith('factor_mom_')]
ref = ref.cs_zscore_batch(factor_cols)  # 参考也做 cs_zscore

# 优化实现
r = compute_momentum_factors('test', f3d, None)

# 对比
ok = True
for c in factor_cols:
    same = np.allclose(r.df[c].values, ref.df[c].values, equal_nan=True, rtol=1e-12)
    if not same:
        diff = (r.df[c] - ref.df[c]).abs().max()
        print(f'FAIL {c}: max_diff={diff:.2e}', flush=True)
        ok = False
    else:
        print(f'OK   {c}', flush=True)
print(f'\nAll match: {ok}')
