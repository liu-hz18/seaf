"""
波动/日内因子 — 33 个因子。优化 v2：_roll 替换为 2D-array + ravel()。
"""

from __future__ import annotations

import logging

import numpy as np

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import (
    rolling_max_2d, rolling_mean_2d, rolling_min_2d, rolling_std_2d,
)


def compute_volatility_factors(name: str, idx: int, f3d: Frame3D, context) -> Frame3D:
    """计算 33 个波动率+日内因子 — 向量化 v2。"""
    result = f3d.copy()
    close, high, low, open_p = (
        f3d.df['close'], f3d.df['high'], f3d.df['low'], f3d.df['open'],
    )
    ret = f3d.ts_pct_change('close', 1).df['close']
    df = result.df
    df['_ret'] = ret

    # ── 提取 2D-array ──
    ret_2d = ret.unstack(level='code').values
    close_2d = close.unstack(level='code').values
    high_2d = high.unstack(level='code').values
    low_2d = low.unstack(level='code').values

    # ===== 波动率：16 cols =====
    ret_stds = rolling_std_2d(ret_2d, [5, 10, 20, 60])
    for p in [5, 10, 20, 60]:
        df[f'factor_vol_realized_{p}d'] = ret_stds[p].ravel()

    neg_ret_2d = np.where(ret_2d < 0, ret_2d, 0.0)
    neg_means = rolling_mean_2d(neg_ret_2d, [5, 10, 20, 60])
    for p in [5, 10, 20, 60]:
        df[f'_ds_mean{p}'] = neg_means[p].ravel()
        df[f'factor_vol_downside_{p}d'] = np.sqrt(np.abs(df[f'_ds_mean{p}']))

    # vol-of-vol
    vol5_2d = ret_stds[5]
    vol20_2d = ret_stds[20]
    vol_of_vol5 = rolling_std_2d(vol5_2d, [20])[20]
    vol_of_vol20 = rolling_std_2d(vol20_2d, [60])[60]
    df['factor_vol_of_vol_20d'] = vol_of_vol5.ravel()
    df['factor_vol_of_vol_60d'] = vol_of_vol20.ravel()

    # Parkinson
    hl_ratio = high_2d / np.where(low_2d > 0, low_2d, np.nan)
    loghl_2d = np.where(hl_ratio > 0, np.log(hl_ratio), np.nan)
    park_factor = 1.0 / (4 * np.log(2))
    park_sq_2d = park_factor * loghl_2d ** 2
    park_means = rolling_mean_2d(park_sq_2d, [5, 20])
    df['factor_vol_parkinson_5d'] = np.sqrt(np.abs(park_means[5].ravel()))
    df['factor_vol_parkinson_20d'] = np.sqrt(np.abs(park_means[20].ravel()))

    # GK
    co_ratio = close_2d / np.where(open_p.unstack(level='code').values > 0,
                                   open_p.unstack(level='code').values, np.nan)
    log_co_2d = np.where(co_ratio > 0, np.log(co_ratio), np.nan)
    gk_2d = 0.5 * loghl_2d ** 2 - (2 * np.log(2) - 1) * log_co_2d ** 2
    gk_mean = rolling_mean_2d(gk_2d, [5])[5]
    df['factor_vol_gk_5d'] = np.sqrt(np.abs(gk_mean.ravel()))

    # vol trend
    df['factor_vol_trend_20d'] = (
        df['factor_vol_realized_20d'] / df['factor_vol_realized_60d'].replace(0, np.nan) - 1
    )
    rv120_2d = rolling_std_2d(ret_2d, [120])[120]
    df['_rv120'] = rv120_2d.ravel()
    df['factor_vol_trend_60d'] = df['factor_vol_realized_60d'] / df['_rv120'].replace(0, np.nan) - 1

    range_2d = high_2d / np.where(low_2d > 0, low_2d, np.nan) - 1
    range_mean = rolling_mean_2d(range_2d, [20])[20]
    df['factor_vol_range_20d'] = range_mean.ravel()

    # ===== 日内：17 cols =====
    itra_2d = close_2d / np.where(open_p.unstack(level='code').values > 0,
                                   open_p.unstack(level='code').values, np.nan) - 1
    itra_means = rolling_mean_2d(itra_2d, [5, 20])
    df['factor_intra_ret_mean_5d'] = itra_means[5].ravel()
    df['factor_intra_ret_mean_20d'] = itra_means[20].ravel()

    # overnight gap
    prev_close_2d = np.roll(close_2d, 1, axis=0)
    prev_close_2d[0] = np.nan
    gap_2d = (open_p.unstack(level='code').values
              / np.where(prev_close_2d != 0, prev_close_2d, np.nan) - 1)
    gap_means = rolling_mean_2d(gap_2d, [5, 20])
    df['factor_intra_overnight_gap_5d'] = gap_means[5].ravel()
    df['factor_intra_overnight_gap_20d'] = gap_means[20].ravel()

    # HL range
    hl_2d = (high_2d - low_2d) / np.where(close_2d != 0, close_2d, np.nan)
    hl_means = rolling_mean_2d(hl_2d, [5, 20, 60])
    for p in [5, 20, 60]:
        df[f'factor_intra_hl_range_{p}d'] = hl_means[p].ravel()

    # close position
    rpos_2d = ((close_2d - low_2d)
               / np.where((high_2d - low_2d) != 0, (high_2d - low_2d), np.nan))
    rpos_means = rolling_mean_2d(rpos_2d, [5, 20])
    df['factor_intra_close_position_5d'] = rpos_means[5].ravel()
    df['factor_intra_close_position_20d'] = rpos_means[20].ravel()

    # open position
    opos_2d = ((open_p.unstack(level='code').values - low_2d)
               / np.where((high_2d - low_2d) != 0, (high_2d - low_2d), np.nan))
    opos_means = rolling_mean_2d(opos_2d, [5, 20])
    df['factor_intra_open_position_5d'] = opos_means[5].ravel()
    df['factor_intra_open_position_20d'] = opos_means[20].ravel()

    # HL volatility
    hl_vol_stds = rolling_std_2d(loghl_2d, [5, 20])
    df['factor_intra_hl_vol_5d'] = hl_vol_stds[5].ravel()
    df['factor_intra_hl_vol_20d'] = hl_vol_stds[20].ravel()

    # directionality
    dir_2d = ((close_2d - open_p.unstack(level='code').values)
              / np.where((high_2d - low_2d) != 0, (high_2d - low_2d), np.nan))
    dir_means = rolling_mean_2d(dir_2d, [5])
    dir_stds = rolling_std_2d(dir_2d, [5])
    df['factor_intra_directionality_5d'] = dir_means[5].ravel()
    df['factor_intra_directionality_std_5d'] = dir_stds[5].ravel()

    # gap efficiency
    gr_2d = gap_2d / (hl_2d + 1e-6)
    gr_means = rolling_mean_2d(gr_2d, [5])
    df['factor_intra_gap_efficiency_5d'] = gr_means[5].ravel()

    # composite
    df['factor_intra_composite'] = (
        df['factor_intra_close_position_5d']
        + df['factor_intra_directionality_5d']
        - df['factor_intra_hl_range_5d']
        + df['factor_intra_gap_efficiency_5d']
    ) / 4

    factor_cols = [c for c in df.columns if c.startswith(('factor_vol_', 'factor_intra_'))]
    return Frame3D(result.df[factor_cols].copy())
