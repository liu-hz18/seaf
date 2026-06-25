"""
动量/反转因子 — 32 个因子。优化 v3：groupby.shift/rolling → 2D-array。
"""

from __future__ import annotations

import numpy as np

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import rolling_mean_2d, rolling_std_2d

EPS = 1e-8


def compute_momentum_factors(name: str, idx: int, f3d: Frame3D, context) -> Frame3D:
    """计算 32 个动量+反转因子 — 向量化 v3。"""
    result = f3d.copy()
    close, open_p, high, low = (f3d.df['close'], f3d.df['open'],
                                 f3d.df['high'], f3d.df['low'])
    df = result.df

    # ── 提取 2D ──
    close_2d = close.unstack(level='code').values
    open_2d = open_p.unstack(level='code').values
    # high_2d = high.unstack(level='code').values
    # low_2d = low.unstack(level='code').values

    # 日收益 r[t] = C[t]/C[t-1] - 1
    ret1_2d = np.empty_like(close_2d); ret1_2d[0] = np.nan
    ret1_2d[1:] = close_2d[1:] / np.where(close_2d[:-1] != 0,
                                           close_2d[:-1], np.nan) - 1
    df['_daily_ret'] = ret1_2d.ravel()

    # ===== 动量：16 cols =====
    # f = (C[t] - C[t-p]) / C[t-p]  (多周期收益率)
    # f = f / σ_max(p,5)(r)        (波动率调整)
    periods = [1, 3, 5, 10, 20, 40, 60, 120]
    vol_windows = sorted({max(p, 5) for p in periods})

    vol_stds = rolling_std_2d(ret1_2d, list(vol_windows))
    for w in vol_windows:
        df[f'_vol_{w}'] = vol_stds[w].ravel()

    result = result.ts_pct_change_multi('close', periods, prefix='factor_mom_ret', cp=False)
    for p in periods:
        df[f'factor_mom_voladj_{p}d'] = (
            df[f'factor_mom_ret_{p}d'] / (df[f'_vol_{max(p, 5)}'] + EPS)
        )

    # ===== 反转：16 cols =====

    # ret3, ret5 (2D pct_change)
    shifted3 = np.roll(close_2d, 3, axis=0); shifted3[:3] = np.nan
    ret3_2d = (close_2d - shifted3) / np.where(shifted3 != 0, shifted3, np.nan)
    df['_ret3'] = ret3_2d.ravel()

    shifted5 = np.roll(close_2d, 5, axis=0); shifted5[:5] = np.nan
    ret5_2d = (close_2d - shifted5) / np.where(shifted5 != 0, shifted5, np.nan)
    df['_ret5'] = ret5_2d.ravel()

    # f = -r_t  (短期反转)
    df['factor_rev_ret_1d'] = -df['_daily_ret']
    df['factor_rev_ret_3d'] = -df['_ret3']
    df['factor_rev_ret_5d'] = -df['_ret5']

    # 隔夜反转: overnight = O/C[-1]-1, f = -overnight
    prev_close_2d = np.roll(close_2d, 1, axis=0); prev_close_2d[0] = np.nan
    overnight_2d = open_2d / np.where(prev_close_2d != 0, prev_close_2d, np.nan) - 1
    df['factor_rev_overnight_1d'] = -overnight_2d.ravel()

    # overnight_5d = -RollingMean(overnight, 5)
    ovn_means = rolling_mean_2d(overnight_2d, [5])
    df['factor_rev_overnight_5d'] = -ovn_means[5].ravel()

    # 日内反转: intraday = C/O-1, f = -intraday
    intraday_2d = close_2d / np.where(open_2d != 0, open_2d, np.nan) - 1
    df['factor_rev_intraday_1d'] = -intraday_2d.ravel()

    # intraday_5d = -RollingMean(intraday, 5)
    intra_means = rolling_mean_2d(intraday_2d, [5])
    df['factor_rev_intraday_5d'] = -intra_means[5].ravel()

    # 量价反转: f = -r_t · rank(V)
    vol_rank = f3d.cs_rank('volume').df['volume']
    df['factor_rev_volrev_1d'] = -df['_daily_ret'] * vol_rank
    df['factor_rev_volrev_3d'] = -df['_ret3'] * vol_rank

    # Z-score 反转: z20 = (C - μ_20(C)) / σ_20(C), f = -z20
    z_means = rolling_mean_2d(close_2d, [20])
    z_stds = rolling_std_2d(close_2d, [20])
    z20_2d = (close_2d - z_means[20]) / np.where(z_stds[20] != 0, z_stds[20], np.nan)
    df['_z20'] = z20_2d.ravel()
    df['factor_rev_zscore_1d'] = -df['_z20']
    df['factor_rev_zscore_3d'] = -df['_z20']
    df['factor_rev_zscore_5d'] = -df['_z20']

    # 缺口反转: f = -overnight · z20
    df['factor_rev_gap_1d'] = -overnight_2d.ravel() * df['_z20'].fillna(0)

    # 振幅反转: f = -(H-L)/C
    df['factor_rev_range_rev'] = -(high - low) / (close + EPS)

    # 换手率反转: f = -r1 · rank(TO)
    to_rank = f3d.cs_rank('turnover').df['turnover']
    df['factor_rev_turnover_1d'] = -df['_daily_ret'] * to_rank

    # 复合反转: f = -0.25(r1 + overnight + intraday + r1·rank(V))
    df['factor_rev_composite'] = (
        -df['_daily_ret'] * 0.25 - overnight_2d.ravel() * 0.25
        - intraday_2d.ravel() * 0.25 - df['_daily_ret'] * vol_rank * 0.25
    )

    factor_cols = [c for c in df.columns if c.startswith(('factor_mom_', 'factor_rev_'))]
    return Frame3D(result.df[factor_cols].copy())
