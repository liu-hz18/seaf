"""
质量合并因子 — 25 因子 (basic 19 + neut 6 回并)。共享数据提取。
"""

from __future__ import annotations

import numpy as np

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import (
    _dd_duration_2d,
    _sign_change_2d,
    rolling_max_2d,
    rolling_mean_2d,
    rolling_std_2d,
)


def compute_quality_merged_factors(name: str, idx: int, f3d: Frame3D, context) -> Frame3D:
    close, high, low = f3d.df['close'], f3d.df['high'], f3d.df['low']
    ret = f3d.ts_pct_change('close', 1).df['close']
    df = f3d.df; df['_ret'] = ret

    ret_2d = df['_ret'].unstack(level='code').values
    close_2d = close.unstack(level='code').values
    amp_2d = ((high - low) / close).unstack(level='code').values

    # ── Part A: 质量基础 19 ──
    rstd = rolling_std_2d(ret_2d, [20, 60, 120])
    rmean = rolling_mean_2d(ret_2d, [20, 60, 120])
    for p in [20, 60, 120]:
        df[f'_ret_std{p}'] = rstd[p].ravel(); df[f'_ret_mean{p}'] = rmean[p].ravel()
        df[f'factor_qb_ret_stability_{p}d'] = 1.0 / df[f'_ret_std{p}'].replace(0, np.nan)
        df[f'factor_qb_sharpe_{p}d'] = df[f'_ret_mean{p}'] / df[f'_ret_std{p}'].replace(0, np.nan)

    pos_2d = (ret_2d > 0).astype(float)
    pos_m = rolling_mean_2d(pos_2d, [20, 60, 120])
    for p in [20, 60, 120]: df[f'factor_qb_pos_days_{p}d'] = pos_m[p].ravel()

    df['factor_qb_stability_decay'] = df['factor_qb_ret_stability_20d'] / df['factor_qb_ret_stability_120d'].replace(0, np.nan)
    df['factor_qb_sharpe_decay'] = df['factor_qb_sharpe_20d'] / df['factor_qb_sharpe_120d'].replace(0, np.nan)

    amp_std = rolling_std_2d(amp_2d, [20, 60])
    for p in [20, 60]:
        df[f'_amp_std{p}'] = amp_std[p].ravel()
        df[f'factor_qb_amp_stability_{p}d'] = 1.0 / df[f'_amp_std{p}'].replace(0, np.nan)

    cmax = rolling_max_2d(close_2d, [60, 120])
    for p in [60, 120]:
        df[f'_max{p}'] = cmax[p].ravel()
        df[f'factor_qb_maxdd_{p}d'] = close / df[f'_max{p}'].replace(0, np.nan) - 1

    df['factor_qb_composite'] = (df['factor_qb_sharpe_60d'] + df['factor_qb_pos_days_60d']
                                  + df['factor_qb_maxdd_60d'] + df['factor_qb_ret_stability_60d']) / 4
    df['factor_qa_ddd_60d'] = _dd_duration_2d(close_2d, 60).ravel()
    df['factor_qa_ddd_120d'] = _dd_duration_2d(close_2d, 120).ravel()
    df['factor_qa_consec_sign_change_60d'] = _sign_change_2d((ret_2d > 0).astype(np.int8), 60).ravel()

    # ── Part B: 截面中性化 6 ──
    ret5 = f3d.ts_pct_change('close', 5).df['close']
    ret20 = f3d.ts_pct_change('close', 20).df['close']
    df['_ret5'] = ret5; df['_ret20'] = ret20

    dummy = Frame3D(df)
    df['factor_cs_momentum_5d'] = dummy.cs_zscore('_ret5', cp=False).df['_ret5']
    df['factor_cs_momentum_20d'] = dummy.cs_zscore('_ret20', cp=False).df['_ret20']
    df['factor_cs_close_zscore'] = f3d.cs_zscore('close').df['close']

    r2 = Frame3D(df.copy())
    neut = r2.cs_neutralize('close', by=['market_cap'])
    df['factor_cs_close_neut_mcap'] = neut.cs_zscore('close').df['close']
    neut2 = r2.cs_neutralize('_ret20', by=['volume'])
    r3 = Frame3D(neut2.df.copy())
    df['factor_cs_ret_neut_volume'] = r3.cs_zscore('_ret20').df['_ret20']
    df['factor_cs_composite'] = (df['factor_cs_momentum_20d'] + df['factor_cs_close_zscore']
                                  + df['factor_cs_close_neut_mcap'] + df['factor_cs_ret_neut_volume']) / 4

    result = Frame3D(df)
    fc = [c for c in df.columns if c.startswith('factor_')]
    result = result.cs_zscore_batch(fc, cp=False)
    return Frame3D(result.df[fc].copy())
