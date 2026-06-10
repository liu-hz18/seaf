"""
质量合并因子 — 25 个因子。
由 quality_basic (19cols, 0.69s) 和 cross_section_neut (6cols, 0.33s) 合并，
消除进程碎片，提高并行节点效率均衡度。

质量基础 + 符号 (19): 收益稳定性×3, Sharpe×3, 正收益占比×3, 衰减率×2,
    振幅稳定性×2, 最大回撤×2, 复合×1, 回撤持续时间×2, 符号变化频次×1.
截面中性化 (6): 截面动量×2, Z-score, 市值中性化, 量价残差, 复合×1.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from qpipe.frame3d import Frame3D


def compute_quality_merged_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 25 个质量合并因子 (19 质量基础/符号 + 6 截面中性化)。"""
    # 注意：本函数直接修改传入的 f3d.df（添加 _ 中间列），
    # 调用方（node.py）保证每次调用传入的是临时 concat DataFrame，无需拷贝。
    close, high, low = f3d.df['close'], f3d.df['high'], f3d.df['low']
    ret = f3d.ts_pct_change('close', 1).df['close']
    df = f3d.df
    df['_ret'] = ret
    grp = df.index.get_level_values('name')

    def _roll(src, dst, window, agg):
        df[dst] = (
            df.groupby('name')[src].rolling(window, min_periods=max(1, window // 2)).agg(agg).reset_index(level=0, drop=True)
        )

    # ====================================================================
    # Part A: 质量基础 (quality_basic) — 19 cols (prefix factor_qb_ / factor_qa_)
    # ====================================================================

    # ----- 收益稳定性 (3 cols) -----
    for p in [20, 60, 120]:
        _roll('_ret', f'_ret_std{p}', p, 'std')
        df[f'factor_qb_ret_stability_{p}d'] = 1.0 / df[f'_ret_std{p}'].replace(0, np.nan)

    # ----- Sharpe-like (3 cols) -----
    for p in [20, 60, 120]:
        _roll('_ret', f'_ret_mean{p}', p, 'mean')
        df[f'factor_qb_sharpe_{p}d'] = df[f'_ret_mean{p}'] / df[f'_ret_std{p}'].replace(0, np.nan)

    # ----- 正收益占比 (3 cols) -----
    df['_pos'] = (ret > 0).astype(float)
    for p in [20, 60, 120]:
        _roll('_pos', f'factor_qb_pos_days_{p}d', p, 'mean')

    # ----- 衰减比 (2 cols) -----
    df['factor_qb_stability_decay'] = df['factor_qb_ret_stability_20d'] / df[
        'factor_qb_ret_stability_120d'
    ].replace(0, np.nan)
    df['factor_qb_sharpe_decay'] = df['factor_qb_sharpe_20d'] / df['factor_qb_sharpe_120d'].replace(
        0, np.nan
    )

    # ----- 振幅稳定性 (2 cols) -----
    df['_amp'] = (high - low) / close
    for p in [20, 60]:
        _roll('_amp', f'_amp_std{p}', p, 'std')
        df[f'factor_qb_amp_stability_{p}d'] = 1.0 / df[f'_amp_std{p}'].replace(0, np.nan)

    # ----- 最大回撤代理 (2 cols) -----
    for p in [60, 120]:
        _roll('close', f'_max{p}', p, 'max')
        df[f'factor_qb_maxdd_{p}d'] = close / df[f'_max{p}'].replace(0, np.nan) - 1

    # ----- 基础复合 (1 col) -----
    df['factor_qb_composite'] = (
        df['factor_qb_sharpe_60d']
        + df['factor_qb_pos_days_60d']
        + df['factor_qb_maxdd_60d']
        + df['factor_qb_ret_stability_60d']
    ) / 4

    # ----- 回撤持续时间 (2 cols, prefix factor_qa_) -----
    def _dd_duration_vec(series, window):
        arr = series.values
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        win = sliding_window_view(arr, window)
        peak_idx = np.argmax(win, axis=1)
        last = win[np.arange(len(win)), -1]
        peak_val = win[np.arange(len(win)), peak_idx]
        ddd = np.where(last < peak_val, window - 1 - peak_idx, 0.0)
        out = np.full(n, np.nan)
        out[window - 1 :] = ddd
        return out

    df['factor_qa_ddd_60d'] = close.groupby(grp).transform(lambda x: _dd_duration_vec(x, 60))
    df['factor_qa_ddd_120d'] = close.groupby(grp).transform(lambda x: _dd_duration_vec(x, 120))

    # ----- 符号变化频次 (1 col, prefix factor_qa_) -----
    def _sign_change_vec(series, window):
        arr = (series.values > 0).astype(np.int8)
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        win = sliding_window_view(arr, window)
        changes = np.count_nonzero(np.diff(win, axis=1), axis=1)
        out = np.full(n, np.nan)
        out[window - 1 :] = changes.astype(float) / window
        return out

    df['factor_qa_consec_sign_change_60d'] = ret.groupby(grp).transform(
        lambda x: _sign_change_vec(x, 60)
    )

    # ====================================================================
    # Part B: 截面中性化 (cross_section_neut) — 6 cols (prefix factor_cs_)
    # ====================================================================

    ret5 = f3d.ts_pct_change('close', 5).df['close']
    ret20 = f3d.ts_pct_change('close', 20).df['close']
    df['_ret5'] = ret5
    df['_ret20'] = ret20

    # ---- 1-2: 截面动量 ----
    # cs_zscore(cp=False) 直接修改 _ret5/_ret20 列（中间列，值可覆盖）
    dummy = Frame3D(df)
    df['factor_cs_momentum_5d'] = dummy.cs_zscore('_ret5', cp=False).df['_ret5']
    df['factor_cs_momentum_20d'] = dummy.cs_zscore('_ret20', cp=False).df['_ret20']

    # ---- 3: 截面 Z-score ----
    df['factor_cs_close_zscore'] = f3d.cs_zscore('close').df['close']

    # ---- 4: 市值中性化 ----
    result2 = Frame3D(df.copy())
    neut = result2.cs_neutralize('close', by=['market_cap'])
    df['factor_cs_close_neut_mcap'] = neut.cs_zscore('close').df['close']

    # ---- 5: 量价回归残差 ----
    neut2 = result2.cs_neutralize('_ret20', by=['volume'])
    result3 = Frame3D(neut2.df.copy())
    df['factor_cs_ret_neut_volume'] = result3.cs_zscore('_ret20').df['_ret20']

    # ---- 6: 复合 ----
    df['factor_cs_composite'] = (
        df['factor_cs_momentum_20d']
        + df['factor_cs_close_zscore']
        + df['factor_cs_close_neut_mcap']
        + df['factor_cs_ret_neut_volume']
    ) / 4

    # ====================================================================
    # 联合截面标准化（三种 prefix 统一批处理）
    # ====================================================================
    result = Frame3D(df)
    factor_cols = [c for c in df.columns if c.startswith('factor_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(
        f'[{name}] quality_merged NaN: { {c: result.df[c].isna().sum() for c in factor_cols} }'
    )
    return Frame3D(result.df[factor_cols].copy())
