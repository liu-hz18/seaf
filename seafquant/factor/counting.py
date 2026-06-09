"""
计数因子（合并：Volume/Rank + Streak + 新高新低） — 16 个因子。
成交量：成交量放量 ×2, 缩量 ×1, 换手率排名变化 ×2, 大涨大跌 ×2, 振幅突破 ×1, 复合 ×1。
Streak：连续涨跌 ×2, 同向占比 ×2。
新高新低：创新高 ×2, 创新低 ×1。
"""

from __future__ import annotations

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


def compute_counting_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个计数因子。"""
    result = f3d.copy()
    close, high, low = f3d.df['close'], f3d.df['high'], f3d.df['low']
    ret = f3d.ts_pct_change('close', 1).df['close']
    df = result.df
    df['_ret'] = ret
    grp = df.index.get_level_values('name')

    def _roll(src, dst, window, agg):
        df[dst] = (
            df.groupby('name')[src]
            .rolling(window, min_periods=max(1, window // 2))
            .agg(agg)
            .reset_index(level=0, drop=True)
        )

    # ===== Vol/Rank：9 cols =====
    volume = f3d.df['volume']
    df['_vol'] = volume
    _roll('_vol', '_vol_ma20', 20, 'mean')
    vol_ma20 = df['_vol_ma20']

    df['factor_cnt_vol_spike_20d'] = (volume > 1.5 * vol_ma20).astype(float)
    _roll('factor_cnt_vol_spike_20d', 'factor_cnt_vol_spike_20d', 20, 'sum')
    df['factor_cnt_vol_spike_60d'] = (volume > 1.5 * vol_ma20).astype(float)
    _roll('factor_cnt_vol_spike_60d', 'factor_cnt_vol_spike_60d', 60, 'sum')
    df['factor_cnt_vol_shrink_20d'] = (volume < 0.5 * vol_ma20).astype(float)
    _roll('factor_cnt_vol_shrink_20d', 'factor_cnt_vol_shrink_20d', 20, 'sum')

    to_rank = result.cs_rank('turnover').df['turnover']
    df['_rk'] = to_rank
    df['_rk_d1'] = df.groupby('name')['_rk'].shift(1)
    df['_rc'] = np.abs(df['_rk'] - df['_rk_d1'])
    _roll('_rc', 'factor_cnt_turnover_rank_chg_20d', 20, 'mean')
    _roll('_rc', 'factor_cnt_turnover_rank_chg_60d', 60, 'mean')

    df['factor_cnt_big_move_20d'] = (np.abs(ret) > 0.02).astype(float)
    _roll('factor_cnt_big_move_20d', 'factor_cnt_big_move_20d', 20, 'sum')
    df['factor_cnt_big_move_60d'] = (np.abs(ret) > 0.02).astype(float)
    _roll('factor_cnt_big_move_60d', 'factor_cnt_big_move_60d', 60, 'sum')

    amp = (high - low) / close
    df['_amp'] = amp
    _roll('_amp', '_amp_ma20', 20, 'mean')
    df['factor_cnt_amp_break_20d'] = (amp > 1.5 * df['_amp_ma20']).astype(float)
    _roll('factor_cnt_amp_break_20d', 'factor_cnt_amp_break_20d', 20, 'sum')

    # ===== Streak：4 cols (pivot→numpy→flatten) =====
    ret_pivot = df['_ret'].unstack(level='name')
    arr = ret_pivot.values.astype(np.float64)
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

    # ===== 新高新低：3 cols =====
    # 复合必须先于 add_column（否则 result 引用不同步）
    df['factor_cnt_composite'] = (
        df['factor_cnt_vol_spike_20d'] / 20
        + df['factor_cnt_big_move_20d'] / 20
        - df['factor_cnt_vol_shrink_20d'] / 20
    ) / 3

    def _new_high_count(series, window):
        arr = series.values.astype(float)
        n = len(arr)
        if n < max(2, window // 2):
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

    factor_cols = [c for c in result.df.columns if c.startswith('factor_cnt_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)
    return Frame3D(result.df[factor_cols].copy())
