"""
计数Streak因子 — 4 个因子：连续涨跌/同向占比。
优化：pivot → numpy → flatten 消除 groupby-transform 回调开销。
"""
import numpy as np
import logging
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
                uc = min(uc + 1, max_len); dc = 0
            elif r < 0:
                dc = min(dc + 1, max_len); uc = 0
            else:
                uc = dc = 0
            up[i, s] = uc
            down[i, s] = dc
    return up, down


def _run_pct_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """向量化同向天数占比：(T, S) 矩阵，每列独立计算。
    使用 sliding_window_view 沿时间轴批量计算。
    """
    T, S = arr.shape
    if T < max(2, window // 2):
        return np.full((T, S), np.nan)
    valid = np.isfinite(arr)
    same_dir = np.zeros((T, S), dtype=np.float64)
    # 相邻两天同向：都是正 或 都是负
    both_pos = (arr[1:] > 0) & (arr[:-1] > 0)
    both_neg = (arr[1:] < 0) & (arr[:-1] < 0)
    valid_pair = valid[1:] & valid[:-1]
    same_dir[1:] = (both_pos | both_neg) & valid_pair
    # 滑动窗口均值
    win = sliding_window_view(same_dir, window, axis=0)  # (T-window+1, S, window)
    out = np.full((T, S), np.nan)
    out[window - 1:] = win.mean(axis=-1)
    return out


def compute_counting_streak_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 4 个计数 streak 因子。"""
    result = f3d.copy()
    df = result.df
    df['_ret'] = df.groupby('name')['close'].pct_change(1)

    # Pivot: (times, stocks) — 消除 groupby-transform
    ret_pivot = df['_ret'].unstack(level='name')
    arr = ret_pivot.values.astype(np.float64)

    up, down = _streaks_2d(arr)
    rp20 = _run_pct_2d(arr, 20)
    rp60 = _run_pct_2d(arr, 60)

    # Flatten 回 MultiIndex
    ret_pivot.iloc[:, :] = up
    df['factor_cnt_consec_up_10d'] = ret_pivot.stack()
    ret_pivot.iloc[:, :] = down
    df['factor_cnt_consec_down_10d'] = ret_pivot.stack()
    ret_pivot.iloc[:, :] = rp20
    df['factor_cnt_run_pct_20d'] = ret_pivot.stack()
    ret_pivot.iloc[:, :] = rp60
    df['factor_cnt_run_pct_60d'] = ret_pivot.stack()

    factor_cols = [c for c in result.df.columns if c.startswith('factor_cnt_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] CountingStreak NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())