"""
TSPCT 因子 — 时序百分位排名因子（40 个）。优化 v2：2D-array + sliding_window。

对基础 OHLCV 和隔夜/日内涨跌幅，在历史滑动窗口内计算百分位排名（0~1），
反映当前值在历史分布中的相对位置。

优化：24 次 groupby+rolling 替换为单次 2D-array sliding_window 批量计算。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from qpipe.frame3d import Frame3D

_WINDOWS = [5, 20, 60]

_BASE_COLS = ['open', 'high', 'low', 'close', 'volume', 'turnover']


def _ts_rank_pct_batch(
    arr: np.ndarray, windows: list[int], min_periods_frac: float = 0.5,
) -> dict[int, np.ndarray]:
    """批量时序百分位排名 — 一次 sliding_window_view 服务所有窗口。

    arr: (T, S) 2D-array, 每列是一只股票的时序。
    返回: {window: np.ndarray of shape (T, S)}
    """
    T, S = arr.shape
    max_w = max(windows)
    if max_w > T:
        return {w: np.full((T, S), np.nan) for w in windows}

    # 单次提取最大窗口的 sliding_window_view
    swv = sliding_window_view(arr, max_w, axis=0)  # (T-max_w+1, S, max_w)

    results: dict[int, np.ndarray] = {}
    for w in windows:
        result = np.full((T, S), np.nan)
        min_p = max(2, int(w * min_periods_frac))

        # 取最后 w 个元素作为窗口
        win = swv[:, :, -w:]  # (T-max_w+1, S, w)
        last_vals = win[:, :, -1]  # (T-max_w+1, S) — 被排名的值

        valid = ~np.isnan(win)  # (T-max_w+1, S, w)
        valid_count = valid.sum(axis=2)  # (T-max_w+1, S)
        last_nan = np.isnan(last_vals)

        # 计数 ≤ last_val 的有效元素
        le = (win <= last_vals[:, :, np.newaxis]) & valid
        le_count = le.sum(axis=2)

        # rank_pct = (count - 1) / (valid_count - 1)
        mask = (valid_count >= min_p) & (~last_nan)
        rank_pct = np.full((T - max_w + 1, S), np.nan)
        rank_pct[mask] = (le_count[mask] - 1.0) / np.maximum(
            valid_count[mask] - 1.0, 1.0,
        )

        result[max_w - 1:] = rank_pct
        results[w] = result

    return results


def compute_tspct_factors(name: str, idx: int, f3d: Frame3D, ctx: Any = None) -> Frame3D:
    """计算 40 个时序百分位排名因子 — 向量化 v2。

    对每个源列 × 每个窗口，批量计算 rolling rank percentile。
    """
    result = f3d.copy()
    df = result.df

    # ── 1. 隔夜涨跌幅 ──
    grp = df.groupby('code')
    prev_close = grp['close'].shift(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['_overnight_pct'] = (
            (df['open'] - prev_close) / prev_close.replace(0, np.nan)
        )

    # ── 2. 日内涨跌幅 ──
    with np.errstate(divide='ignore', invalid='ignore'):
        df['_intraday_pct'] = (
            (df['close'] - df['open']) / df['open'].replace(0, np.nan)
        )

    # ── 3. 所有源列 → 批量 2D rank ──
    src_cols = [*_BASE_COLS, '_overnight_pct', '_intraday_pct']
    factor_cols: list[str] = []

    for col in src_cols:
        if col == '_overnight_pct': alias = 'on'
        elif col == '_intraday_pct': alias = 'id'
        else: alias = col

        # 提取为 2D-array
        col_2d = df[col].unstack(level='code').values
        ranks = _ts_rank_pct_batch(col_2d, _WINDOWS)

        for w in _WINDOWS:
            fcol = f'factor_tspct_{alias}_{w}d'
            factor_cols.append(fcol)
            df[fcol] = ranks[w].ravel()

    return Frame3D(result.df[factor_cols].copy())
