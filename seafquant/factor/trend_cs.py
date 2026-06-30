"""
趋势+截面合并 — 26 因子。v2: 共享 close_2d，一次 unstack。
"""

from __future__ import annotations

import numpy as np

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import (
    ewm_2d,
    rolling_max_2d,
    rolling_mean_2d,
    rolling_min_2d,
    rolling_std_2d,
)


def compute_trend_cs_factors(name: str, idx: int, f3d: Frame3D, context: dict) -> Frame3D:
    result = f3d.copy()
    close = f3d.df['close']
    df = result.df
    close_2d = close.unstack(level='code').values

    # ═══════════════════ Part A: 趋势 16 ═══════════════════
    ma_w = [5, 10, 20, 60, 120]
    mas = rolling_mean_2d(close_2d, ma_w)
    for w in ma_w:
        df[f'_ma{w}'] = mas[w].ravel()
        df[f'factor_trend_ma_{w}d'] = close / df[f'_ma{w}'].replace(0, np.nan) - 1
    df['factor_trend_ma_cross_5_20'] = df['_ma5'] / df['_ma20'].replace(0, np.nan) - 1
    df['factor_trend_ma_cross_10_60'] = df['_ma10'] / df['_ma60'].replace(0, np.nan) - 1
    df['factor_trend_ma_cross_20_120'] = df['_ma20'] / df['_ma120'].replace(0, np.nan) - 1

    ema12 = ewm_2d(close_2d, 12); ema26 = ewm_2d(close_2d, 26)
    macd = ema12 - ema26; macd_sig = ewm_2d(macd, 9)
    df['factor_trend_macd'] = macd.ravel()
    df['factor_trend_macd_signal'] = (macd - macd_sig).ravel()

    mins = rolling_min_2d(close_2d, [20, 60]); maxs = rolling_max_2d(close_2d, [20, 60])
    for w in [20, 60]:
        df[f'_min{w}'] = mins[w].ravel(); df[f'_max{w}'] = maxs[w].ravel()
        df[f'factor_trend_channel_{w}d'] = (close - df[f'_min{w}']) / (df[f'_max{w}'] - df[f'_min{w}']).replace(0, np.nan)

    zm = rolling_mean_2d(close_2d, [20, 60]); zs = rolling_std_2d(close_2d, [20, 60])
    df['factor_trend_mom_strength_20d'] = ((close_2d - zm[20]) / np.where(zs[20] != 0, zs[20], np.nan)).ravel()
    df['factor_trend_mom_strength_60d'] = ((close_2d - zm[60]) / np.where(zs[60] != 0, zs[60], np.nan)).ravel()

    df['factor_trend_vol_confirm'] = (close / df['_ma20'].replace(0, np.nan) - 1) * f3d.cs_zscore('volume').df['volume']
    df['factor_trend_composite'] = (df['factor_trend_macd_signal'] + df['factor_trend_channel_20d']
                                     + df['factor_trend_mom_strength_20d'] + df['factor_trend_vol_confirm']) / 4

    # ═══════════════════ Part B: 截面 10 ═══════════════════
    s1 = np.roll(close_2d, 1, axis=0); s1[0] = np.nan
    df['_ret1'] = ((close_2d - s1) / np.where(s1 != 0, s1, np.nan)).ravel()
    s20 = np.roll(close_2d, 20, axis=0); s20[:20] = np.nan
    df['_ret20'] = ((close_2d - s20) / np.where(s20 != 0, s20, np.nan)).ravel()

    df['factor_cs_rank_close'] = f3d.cs_rank('close').df['close']
    df['factor_cs_rank_volume'] = f3d.cs_rank('volume').df['volume']

    rk_2d = df['factor_cs_rank_close'].unstack(level='code').values
    for p in [5, 20, 60]:
        s = np.roll(rk_2d, p, axis=0); s[:p] = np.nan
        df[f'factor_cs_rank_delta_{p}d'] = (rk_2d - s).ravel()

    zm2 = rolling_mean_2d(close_2d, [5, 20, 60]); zs2 = rolling_std_2d(close_2d, [5, 20, 60])
    for w in [5, 20, 60]:
        df[f'factor_cs_rank_zscore_{w}d'] = ((close_2d - zm2[w]) / np.where(zs2[w] != 0, zs2[w], np.nan)).ravel()

    result = Frame3D(df.copy())
    df['factor_cs_ret_rank_1d'] = result.cs_rank('_ret1').df['_ret1']
    df['factor_cs_ret_rank_20d'] = result.cs_rank('_ret20').df['_ret20']

    result = Frame3D(df.copy())
    fc = [c for c in df.columns if c.startswith(('factor_trend_','factor_cs_'))]
    result = result.cs_zscore_batch(fc, cp=False)
    return Frame3D(result.df[fc].copy())
