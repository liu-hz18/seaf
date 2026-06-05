"""精确数值验证：对比优化前后因子计算结果。"""
import sys, numpy as np
sys.path.insert(0, '.')
import pandas as pd
from qpipe.frame3d import Frame3D
from seafquant.data_generator import generate_synthetic_data

# 小数据方便对比
gen = generate_synthetic_data(n_times=50, n_stocks=20, noise_ratio=0.3, seed=42)
frames = [f.df for f in gen]
big_df = pd.concat(frames, axis=0).sort_index(level=0)
f3d = Frame3D(big_df)

# ===== 验证 momentum：ts_pct_change_multi vs 逐次 groupby pct_change =====
print("=== momentum: ts_pct_change_multi vs per-period ===")
from seafquant.factor.factors_momentum import compute_momentum_factors

result = f3d.copy()
df = result.df
periods = [1, 3, 5, 10, 20, 40, 60, 120]

# 方法A: ts_pct_change_multi (当前实现)
rA = result.ts_pct_change_multi('close', periods, prefix='test', cp=False)
colsA = [f'test_{p}d' for p in periods]

# 方法B: 逐次 groupby pct_change
rB = f3d.copy()
dfB = rB.df
for p in periods:
    dfB[f'test_{p}d'] = dfB.groupby('name')['close'].pct_change(p)

for c in colsA:
    diff = (rA.df[c] - dfB[c]).abs().max()
    same = np.allclose(rA.df[c].values, dfB[c].values, equal_nan=True)
    print(f"  {c}: match={same}, max_diff={diff:.2e}")

# ===== 验证 momentum: factor_mom_ret 列 =====
print("\n=== momentum: factor_mom_ret values ===")
ref = f3d.copy()
ref_df = ref.df
for p in periods:
    ref_df[f'factor_mom_ret_{p}d'] = ref_df.groupby('name')['close'].pct_change(p)
    w = max(p, 5)

r = compute_momentum_factors('test', f3d, None)
for p in periods:
    c = f'factor_mom_ret_{p}d'
    diff = (r.df[c] - ref_df[c]).abs().max()
    same = np.allclose(r.df[c].values, ref_df[c].values, equal_nan=True)
    print(f"  {c}: match={same}, max_diff={diff:.2e}")

# ===== 验证 counting_streak =====
print("\n=== counting_streak: pivot vs groupby-transform ===")
from seafquant.factor.factors_counting_streak import (_streaks_2d, _run_pct_2d,
                                                       compute_counting_streak_factors)

# 先验证 _streaks_2d
df_test = f3d.df.copy()
df_test['_ret'] = df_test.groupby('name')['close'].pct_change(1)
ret_pivot = df_test['_ret'].unstack(level='name')
arr = ret_pivot.values.astype(np.float64)
up2d, down2d = _streaks_2d(arr)

# groupby-transform 参考
grp = df_test.index.get_level_values('name')
def _streaks_ref(series):
    arr_s = series.values
    n = len(arr_s)
    up = np.zeros(n); down = np.zeros(n)
    uc = dc = 0
    for i in range(n):
        r = arr_s[i]
        if np.isnan(r): uc = dc = 0
        elif r > 0: uc = min(uc+1, 10); dc = 0
        elif r < 0: dc = min(dc+1, 10); uc = 0
        else: uc = dc = 0
        up[i] = uc; down[i] = dc
    return up, down

up_ref = df_test.groupby('name')['_ret'].transform(lambda x: _streaks_ref(x)[0])
down_ref = df_test.groupby('name')['_ret'].transform(lambda x: _streaks_ref(x)[1])

# Flatten up2d
ret_pivot.iloc[:,:] = up2d
up_flat = ret_pivot.stack()
same_up = np.allclose(up_flat.values, up_ref.values, equal_nan=True)
print(f"  consec_up match: {same_up}")

ret_pivot.iloc[:,:] = down2d
down_flat = ret_pivot.stack()
same_down = np.allclose(down_flat.values, down_ref.values, equal_nan=True)
print(f"  consec_down match: {same_down}")

# 验证 run_pct
def _run_pct_ref(series, window):
    arr = series.values
    n = len(arr)
    if n < max(2, window//2):
        return np.full(n, np.nan)
    same_dir = np.zeros(n)
    valid = ~np.isnan(arr)
    both_pos = (arr[1:] > 0) & (arr[:-1] > 0)
    both_neg = (arr[1:] < 0) & (arr[:-1] < 0)
    valid_pair = valid[1:] & valid[:-1]
    same_dir[1:] = (both_pos | both_neg) & valid_pair
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(same_dir, window)
    out = np.full(n, np.nan)
    out[window-1:] = win.mean(axis=1)
    return out

rp20_2d = _run_pct_2d(arr, 20)
rp20_ref = df_test.groupby('name')['_ret'].transform(lambda x: _run_pct_ref(x, 20))
ret_pivot.iloc[:,:] = rp20_2d
rp20_flat = ret_pivot.stack()
same_rp20 = np.allclose(rp20_flat.values, rp20_ref.values, equal_nan=True)
print(f"  run_pct_20d match: {same_rp20}")

rp60_2d = _run_pct_2d(arr, 60)
rp60_ref = df_test.groupby('name')['_ret'].transform(lambda x: _run_pct_ref(x, 60))
ret_pivot.iloc[:,:] = rp60_2d
rp60_flat = ret_pivot.stack()
same_rp60 = np.allclose(rp60_flat.values, rp60_ref.values, equal_nan=True)
print(f"  run_pct_60d match: {same_rp60}")

# ===== 验证 cs_zscore_batch(cp=False) 结果一致性 =====
print("\n=== cs_zscore_batch: cp=True vs cp=False ===")
test_f3d = f3d.copy()
cols = ['open', 'high', 'low']
# cp=True (默认)
r_true = test_f3d.cs_zscore_batch(cols, cp=True)
# cp=False
r_false = test_f3d.copy().cs_zscore_batch(cols, cp=False)
for c in cols:
    same = np.allclose(r_true.df[c].values, r_false.df[c].values, equal_nan=True)
    diff = (r_true.df[c] - r_false.df[c]).abs().max()
    print(f"  {c}: match={same}, max_diff={diff:.2e}")

print("\n=== All validations complete ===")
