"""
计数因子 — 17 个因子。v4: _streaks/_nh/_nl 用 numba JIT 2D 消除 groupby.transform。
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import njit, rolling_mean_2d, rolling_sum_2d


@njit
def _streaks_numba(arr: np.ndarray, max_len: int = 10):
    """Numba JIT 连续涨跌计数 2D。"""
    T, S = arr.shape
    up = np.zeros((T, S), dtype=np.float64)
    down = np.zeros((T, S), dtype=np.float64)
    for s in range(S):
        uc = 0; dc = 0
        for i in range(T):
            r = arr[i, s]
            if np.isnan(r): uc = dc = 0
            elif r > 0: uc = min(uc + 1, max_len); dc = 0
            elif r < 0: dc = min(dc + 1, max_len); uc = 0
            else: uc = dc = 0
            up[i, s] = uc; down[i, s] = dc
    return up, down


@njit
def _nh_numba(price: np.ndarray, window: int) -> np.ndarray:
    """Numba JIT 2D 创新高计数。"""
    T, S = price.shape
    out = np.full((T, S), np.nan)
    if window > T: return out
    for s in range(S):
        cmax = -np.inf
        count = 0
        for i in range(T):
            p = price[i, s]
            if np.isnan(p):
                cmax = -np.inf; count = 0
                continue
            if i >= window - 1:
                # roll off oldest
                pass  # simplified: just recount in sliding window
            if p > cmax and cmax != -np.inf:
                count += 1
            cmax = max(cmax, p)
            if i >= window - 1:
                out[i, s] = count
    return out


def _new_high_nb(price: np.ndarray, window: int) -> np.ndarray:
    """Numba JIT 创新高（精确实现）。"""
    T, S = price.shape
    out = np.full((T, S), np.nan)
    if window > T: return out

    swv = sliding_window_view(price, window, axis=0)  # (T-w+1, S, w)
    # 用 numba 加速循环
    @njit
    def _count(arr3d):
        R, S2, W = arr3d.shape
        res = np.zeros((R, S2))
        for i in range(R):
            for s2 in range(S2):
                cnt = 0; best = -np.inf
                first = True
                for j in range(W):
                    v = arr3d[i, s2, j]
                    if np.isnan(v): continue
                    if first: best = v; first = False
                    elif v > best: cnt += 1; best = v
                res[i, s2] = cnt
        return res

    out[window - 1:] = _count(swv)
    return out


def _run_pct_2d(arr: np.ndarray, window: int) -> np.ndarray:
    T, S = arr.shape
    if window > T: return np.full((T, S), np.nan)
    same_dir = np.zeros((T, S), dtype=np.float64)
    bp = (arr[1:] > 0) & (arr[:-1] > 0)
    bn = (arr[1:] < 0) & (arr[:-1] < 0)
    same_dir[1:] = (bp | bn).astype(float)
    win = sliding_window_view(same_dir, window, axis=0)
    out = np.full((T, S), np.nan)
    out[window - 1:] = win.mean(axis=-1)
    return out


def _tillnow_ret_2d(arr: np.ndarray, window: int) -> np.ndarray:
    T, S = arr.shape
    if window > T: return np.full((T, S), np.nan)
    clean = np.where(np.isfinite(1.0 + arr), 1.0 + arr, 1.0)
    win = sliding_window_view(clean, window, axis=0)
    out = np.full((T, S), np.nan)
    out[window - 1:] = np.prod(win, axis=-1) - 1.0
    return out


def _tillnow_dd_2d(price: np.ndarray, window: int) -> np.ndarray:
    T, S = price.shape
    if window > T: return np.full((T, S), np.nan)
    win = sliding_window_view(price, window, axis=0)
    cummax = np.maximum.accumulate(win, axis=-1)
    dd = win / np.where(cummax == 0, 1.0, cummax) - 1.0
    out = np.full((T, S), np.nan)
    out[window - 1:] = np.min(dd, axis=-1)
    return out


def compute_counting_factors(name: str, idx: int, f3d: Frame3D, context: dict) -> Frame3D:
    """计算 17 个计数因子 — v4 numba。"""
    result = f3d.copy()
    close = f3d.df['close']
    ret = f3d.ts_pct_change('close', 1).df['close']
    df = result.df
    df['_ret'] = ret

    ret_2d = ret.unstack(level='code').values
    close_2d = close.unstack(level='code').values

    # CountPos/CountNeg (4)
    pos_2d = (ret_2d > 0).astype(float)
    neg_2d = (ret_2d < 0).astype(float)
    pos_sums = rolling_sum_2d(pos_2d, [10, 60])
    neg_sums = rolling_sum_2d(neg_2d, [10, 60])
    df['factor_cnt_countpos_10d'] = pos_sums[10].ravel()
    df['factor_cnt_countpos_60d'] = pos_sums[60].ravel()
    df['factor_cnt_countneg_10d'] = neg_sums[10].ravel()
    df['factor_cnt_countneg_60d'] = neg_sums[60].ravel()

    # Streak + RunPct (4)
    arr = ret_2d.copy()
    up, down = _streaks_numba(arr)
    rp20 = _run_pct_2d(arr, 20); rp60 = _run_pct_2d(arr, 60)
    df['factor_cnt_consec_up_10d'] = up.ravel()
    df['factor_cnt_consec_down_10d'] = down.ravel()
    df['factor_cnt_run_pct_20d'] = rp20.ravel()
    df['factor_cnt_run_pct_60d'] = rp60.ravel()

    # TillNow (4)
    tr20 = _tillnow_ret_2d(arr, 20); tr60 = _tillnow_ret_2d(arr, 60)
    dd20 = _tillnow_dd_2d(close_2d, 20)
    dd60 = _tillnow_dd_2d(close_2d, 60)
    df['factor_cnt_tillnow_ret_20d'] = tr20.ravel()
    df['factor_cnt_tillnow_ret_60d'] = tr60.ravel()
    df['factor_cnt_tillnow_dd_20d'] = dd20.ravel()
    df['factor_cnt_tillnow_dd_60d'] = dd60.ravel()

    # Turnover Rank Change (2)
    to_rank = result.cs_rank('turnover').df['turnover']
    df['_rk'] = to_rank
    rk_2d = df['_rk'].unstack(level='code').values
    rk_shifted = np.roll(rk_2d, 1, axis=0); rk_shifted[0] = np.nan
    rc_2d = np.abs(rk_2d - rk_shifted)
    rc_means = rolling_mean_2d(rc_2d, [20, 60])
    df['factor_cnt_turnover_rank_chg_20d'] = rc_means[20].ravel()
    df['factor_cnt_turnover_rank_chg_60d'] = rc_means[60].ravel()

    # 新高新低 (3) — numba JIT 2D
    df['factor_cnt_new_high_20d'] = _new_high_nb(close_2d, 20).ravel()
    df['factor_cnt_new_high_60d'] = _new_high_nb(close_2d, 60).ravel()
    df['factor_cnt_new_low_60d'] = _new_high_nb(-close_2d, 60).ravel()  # low via negated price

    factor_cols = [c for c in result.df.columns if c.startswith('factor_cnt_')]
    return Frame3D(result.df[factor_cols].copy())
