"""
计数因子（纯计数算子，无离散化阈值） — 17 个因子。

Streak（连续涨跌）：consec_up/down ×2, run_pct ×2 = 4
CountPos/CountNeg（正负收益天数）：countpos ×2, countneg ×2 = 4
TillNow（累计收益/回撤）：tillnow_ret ×2, tillnow_dd ×2 = 4
新高新低：new_high ×2, new_low ×1 = 3
Turnover Rank Change：rank_chg ×2 = 2
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from qpipe.frame3d import Frame3D


def _streaks_2d(arr: np.ndarray, max_len: int = 10):
    """批量化连续涨跌计数：(T, S) 矩阵同时处理 S 个 stock。"""
    T, S = arr.shape
    up = np.zeros((T, S), dtype=np.float64)
    down = np.zeros((T, S), dtype=np.float64)
    for s in range(S):
        uc = dc = 0
        for i in range(T):
            r = arr[i, s]
            if np.isnan(r):
                uc = dc = 0
            elif r > 0:
                uc = min(uc + 1, max_len)
                dc = 0
            elif r < 0:
                dc = min(dc + 1, max_len)
                uc = 0
            else:
                uc = dc = 0
            up[i, s] = uc
            down[i, s] = dc
    return up, down


def _run_pct_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """向量化同向天数占比。"""
    T, S = arr.shape
    if window > T:
        return np.full((T, S), np.nan)
    valid = np.isfinite(arr)
    same_dir = np.zeros((T, S), dtype=np.float64)
    both_pos = (arr[1:] > 0) & (arr[:-1] > 0)
    both_neg = (arr[1:] < 0) & (arr[:-1] < 0)
    valid_pair = valid[1:] & valid[:-1]
    same_dir[1:] = (both_pos | both_neg) & valid_pair
    win = sliding_window_view(same_dir, window, axis=0)
    out = np.full((T, S), np.nan)
    out[window - 1 :] = win.mean(axis=-1)
    return out


def _tillnow_ret_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """向量化滚动累计收益：cumprod(1+ret) - 1。"""
    T, S = arr.shape
    if window > T:
        return np.full((T, S), np.nan)
    one_plus = 1.0 + arr
    clean = np.where(np.isfinite(one_plus), one_plus, 1.0)
    win = sliding_window_view(clean, window, axis=0)
    out = np.full((T, S), np.nan)
    out[window - 1 :] = np.prod(win, axis=-1) - 1.0
    return out


def _tillnow_dd_2d(price: np.ndarray, window: int) -> np.ndarray:
    """向量化滚动最大回撤：min(price / cummax(price)) - 1。"""
    T, S = price.shape
    if window > T:
        return np.full((T, S), np.nan)
    win = sliding_window_view(price, window, axis=0)
    cummax = np.maximum.accumulate(win, axis=-1)
    dd = win / np.where(cummax == 0, 1.0, cummax) - 1.0
    out = np.full((T, S), np.nan)
    out[window - 1 :] = np.min(dd, axis=-1)
    return out


def _new_high_count(series, window):
    """逐股票创新高天数计数。"""
    arr = series.values.astype(float)
    n = len(arr)
    if n < window:
        return np.full(n, np.nan)
    clean = np.where(np.isnan(arr), -np.inf, arr)
    cmax = np.maximum.accumulate(clean)
    is_nh = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if arr[i] > cmax[i - 1] and cmax[i - 1] != -np.inf:
            is_nh[i] = 1.0
    win = sliding_window_view(is_nh, window)
    out = np.full(n, np.nan)
    out[window - 1 :] = win.sum(axis=1)
    return out


def _new_low_count(series, window):
    """逐股票创新低天数计数。"""
    arr = series.values.astype(float)
    n = len(arr)
    if n < window:
        return np.full(n, np.nan)
    clean = np.where(np.isnan(arr), np.inf, arr)
    cmin = np.minimum.accumulate(clean)
    is_nl = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if arr[i] < cmin[i - 1] and cmin[i - 1] != np.inf:
            is_nl[i] = 1.0
    win = sliding_window_view(is_nl, window)
    out = np.full(n, np.nan)
    out[window - 1 :] = win.sum(axis=1)
    return out


def compute_counting_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 17 个纯计数因子（无离散化阈值）。"""
    result = f3d.copy()
    close = f3d.df['close']
    ret = f3d.ts_pct_change('close', 1).df['close']
    df = result.df
    df['_ret'] = ret
    grp = df.index.get_level_values('code')

    # ===== CountPos / CountNeg：4 cols =====
    pos_mask = (ret > 0).astype(float)
    neg_mask = (ret < 0).astype(float)
    df['_pos'] = pos_mask
    df['_neg'] = neg_mask
    for w, label_pos, label_neg in [(10, '10d', '10d'), (60, '60d', '60d')]:
        for src, dst in [('_pos', f'factor_cnt_countpos_{label_pos}'),
                         ('_neg', f'factor_cnt_countneg_{label_neg}')]:
            df[dst] = (
                df.groupby('code')[src]
                .rolling(w, min_periods=max(1, w // 2))
                .sum()
                .reset_index(level=0, drop=True)
            )

    # ===== Streak + RunPct：4 cols (pivot→numpy→flatten) =====
    ret_pivot = df['_ret'].unstack(level='code').astype(np.float64)
    arr = ret_pivot.values.copy()
    up, down = _streaks_2d(arr)
    rp20 = _run_pct_2d(arr, 20)
    rp60 = _run_pct_2d(arr, 60)
    ret_pivot.iloc[:, :] = up
    df['factor_cnt_consec_up_10d'] = ret_pivot.stack()
    ret_pivot.iloc[:, :] = down
    df['factor_cnt_consec_down_10d'] = ret_pivot.stack()
    ret_pivot.iloc[:, :] = rp20
    df['factor_cnt_run_pct_20d'] = ret_pivot.stack()
    ret_pivot.iloc[:, :] = rp60
    df['factor_cnt_run_pct_60d'] = ret_pivot.stack()

    # ===== TillNow：4 cols (pivot→numpy→flatten) =====
    close_pivot = close.unstack(level='code').astype(np.float64)
    tr20 = _tillnow_ret_2d(arr, 20)
    tr60 = _tillnow_ret_2d(arr, 60)
    dd20 = _tillnow_dd_2d(close_pivot.values, 20)
    dd60 = _tillnow_dd_2d(close_pivot.values, 60)
    ret_pivot.iloc[:, :] = tr20
    df['factor_cnt_tillnow_ret_20d'] = ret_pivot.stack()
    ret_pivot.iloc[:, :] = tr60
    df['factor_cnt_tillnow_ret_60d'] = ret_pivot.stack()
    ret_pivot.iloc[:, :] = dd20
    df['factor_cnt_tillnow_dd_20d'] = ret_pivot.stack()
    ret_pivot.iloc[:, :] = dd60
    df['factor_cnt_tillnow_dd_60d'] = ret_pivot.stack()

    # ===== Turnover Rank Change：2 cols =====
    to_rank = result.cs_rank('turnover').df['turnover']
    df['_rk'] = to_rank
    df['_rk_d1'] = df.groupby('code')['_rk'].shift(1)
    df['_rc'] = np.abs(df['_rk'] - df['_rk_d1'])
    for w, label in [(20, '20d'), (60, '60d')]:
        col = f'factor_cnt_turnover_rank_chg_{label}'
        df[col] = (
            df.groupby('code')['_rc']
            .rolling(w, min_periods=max(1, w // 2))
            .mean()
            .reset_index(level=0, drop=True)
        )

    # ===== 新高新低：3 cols =====
    result = result.add_column(
        'factor_cnt_new_high_20d',
        close.groupby(grp).transform(lambda x: _new_high_count(x, 20)),
        cp=False,
    )
    result = result.add_column(
        'factor_cnt_new_high_60d',
        close.groupby(grp).transform(lambda x: _new_high_count(x, 60)),
        cp=False,
    )
    result = result.add_column(
        'factor_cnt_new_low_60d',
        close.groupby(grp).transform(lambda x: _new_low_count(x, 60)),
        cp=False,
    )

    # ===== 截面标准化 =====
    factor_cols = [c for c in result.df.columns if c.startswith('factor_cnt_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f'Factor NaN: { {c: result.df[c].isna().sum() for c in factor_cols} }')
    return Frame3D(result.df[factor_cols].copy())
