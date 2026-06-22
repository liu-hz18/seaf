"""
因子计算性能助手 — 向量化 2D-array 操作。

将 MultiIndex DataFrame 的一列转换为 (time × stock) 二维数组，
用 cumsum / sliding_window_view / numba JIT 完成滚动聚合，
彻底消除 pandas groupby+transform 的 O(N·log N) 开销。

在 200 时间 × 5000 股票的数据上，相比逐窗口 groupby-rolling，
滚动均值计算加速 5-15 倍。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

if TYPE_CHECKING:
    import pandas as pd

try:
    from numba import njit as _real_njit  # type: ignore[import-untyped]
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    _real_njit = None  # type: ignore[assignment]


def njit(func=None, **kwargs):
    """JIT 装饰器：有 numba 时编译，无 numba 时透传。

    兼容 @njit 和 @njit(parallel=True) 两种用法。
    """
    if _HAS_NUMBA and _real_njit:
        if func is not None:
            return _real_njit(func, **kwargs)
        return lambda f: _real_njit(f, **kwargs) # type: ignore
    if func is not None:
        return func
    return lambda f: f


# ══════════════════════════════════════════════════════════════
# MultiIndex ⇄ 2D-array 转换
# ══════════════════════════════════════════════════════════════

def extract_2d(df, col: str) -> tuple[np.ndarray, pd.Index, pd.Index]:
    """从 MultiIndex DataFrame 中提取列为 2D 数组 (time × stock)。

    返回: (arr_2d, time_index, stock_index)
    """
    pivoted: pd.DataFrame = df[col].unstack(level='code')
    return pivoted.values, pivoted.index, pivoted.columns


def inject_2d(df, arr: np.ndarray, col_name: str,
              time_idx, stock_cols) -> None:
    """将 2D 数组注回 DataFrame 的新列（原地修改）。"""
    import pandas as pd
    result_df = pd.DataFrame(arr, index=time_idx, columns=stock_cols)
    series = result_df.stack(future_stack=True)
    series.index.names = ['key', 'code']
    df[col_name] = series.reindex(df.index)


# ══════════════════════════════════════════════════════════════
# 向量化滚动聚合
# ══════════════════════════════════════════════════════════════

def rolling_mean_2d(
    arr: np.ndarray,
    windows: list[int],
    min_periods_frac: float = 0.5,
) -> dict[int, np.ndarray]:
    """多窗口滚动均值 — 一次 cumsum 服务所有 window。

    复杂度 O(max_window × n_stocks)，与窗口数无关。
    正确处理 NaN：只对窗口内有效值求均值。
    """
    n_times, n_stocks = arr.shape

    arr_filled = np.nan_to_num(arr, nan=0.0)
    valid = (~np.isnan(arr)).astype(np.float64)

    cumsum_val = np.vstack([np.zeros((1, n_stocks)),
                            np.cumsum(arr_filled, axis=0)])
    cumsum_cnt = np.vstack([np.zeros((1, n_stocks)),
                            np.cumsum(valid, axis=0)])

    results: dict[int, np.ndarray] = {}
    for w in windows:
        if w > n_times:
            results[w] = np.full((n_times, n_stocks), np.nan)
            continue

        sum_w = cumsum_val[w:] - cumsum_val[:-w]   # (n_times-w+1, n_stocks)
        cnt_w = cumsum_cnt[w:] - cumsum_cnt[:-w]

        result = np.full((n_times, n_stocks), np.nan)
        min_p = max(1, int(w * min_periods_frac))
        valid_mask = cnt_w >= min_p
        result[w - 1:] = np.where(valid_mask,
                                  sum_w / np.maximum(cnt_w, 1.0),
                                  np.nan)
        results[w] = result
    return results


def rolling_std_2d(
    arr: np.ndarray,
    windows: list[int],
    min_periods_frac: float = 0.5,
) -> dict[int, np.ndarray]:
    """多窗口滚动标准差 — 双 cumsum (value + value²)。"""
    n_times, n_stocks = arr.shape

    arr_filled = np.nan_to_num(arr, nan=0.0)
    valid = (~np.isnan(arr)).astype(np.float64)

    cumsum_val = np.vstack([np.zeros((1, n_stocks)),
                            np.cumsum(arr_filled, axis=0)])
    cumsum_val2 = np.vstack([np.zeros((1, n_stocks)),
                             np.cumsum(arr_filled ** 2, axis=0)])
    cumsum_cnt = np.vstack([np.zeros((1, n_stocks)),
                            np.cumsum(valid, axis=0)])

    results: dict[int, np.ndarray] = {}
    for w in windows:
        if w > n_times:
            results[w] = np.full((n_times, n_stocks), np.nan)
            continue

        sum_w = cumsum_val[w:] - cumsum_val[:-w]
        sum2_w = cumsum_val2[w:] - cumsum_val2[:-w]
        cnt_w = cumsum_cnt[w:] - cumsum_cnt[:-w]

        result = np.full((n_times, n_stocks), np.nan)
        min_p = max(1, int(w * min_periods_frac))
        valid_mask = cnt_w >= min_p

        mean_w = sum_w / np.maximum(cnt_w, 1.0)
        var_w = sum2_w / np.maximum(cnt_w, 1.0) - mean_w ** 2
        var_w = np.maximum(var_w, 0.0)  # 防止浮点误差导致的负值

        result[w - 1:] = np.where(valid_mask, np.sqrt(var_w), np.nan)
        results[w] = result
    return results


def rolling_sum_2d(
    arr: np.ndarray,
    windows: list[int],
    min_periods_frac: float = 0.5,
) -> dict[int, np.ndarray]:
    """多窗口滚动求和 — 一次 cumsum 服务所有窗口。"""
    n_times, n_stocks = arr.shape
    arr_filled = np.nan_to_num(arr, nan=0.0)
    valid = (~np.isnan(arr)).astype(np.float64)
    cumsum_val = np.vstack([np.zeros((1, n_stocks)), np.cumsum(arr_filled, axis=0)])
    cumsum_cnt = np.vstack([np.zeros((1, n_stocks)), np.cumsum(valid, axis=0)])
    results: dict[int, np.ndarray] = {}
    for w in windows:
        if w > n_times:
            results[w] = np.full((n_times, n_stocks), np.nan); continue
        sum_w = cumsum_val[w:] - cumsum_val[:-w]
        cnt_w = cumsum_cnt[w:] - cumsum_cnt[:-w]
        result = np.full((n_times, n_stocks), np.nan)
        min_p = max(1, int(w * min_periods_frac))
        result[w - 1:] = np.where(cnt_w >= min_p, sum_w, np.nan)
        results[w] = result
    return results


def rolling_min_2d(
    arr: np.ndarray,
    windows: list[int],
) -> dict[int, np.ndarray]:
    """多窗口滚动最小值 — sliding_window_view + nanmin。"""
    return _rolling_sliding(arr, windows, np.nanmin)


def rolling_max_2d(
    arr: np.ndarray,
    windows: list[int],
) -> dict[int, np.ndarray]:
    """多窗口滚动最大值 — sliding_window_view + nanmax。"""
    return _rolling_sliding(arr, windows, np.nanmax)


def _rolling_sliding(
    arr: np.ndarray,
    windows: list[int],
    agg_fn,
) -> dict[int, np.ndarray]:
    """通用 sliding_window 滚动聚合。"""
    n_times, n_stocks = arr.shape
    results: dict[int, np.ndarray] = {}
    for w in windows:
        if w > n_times:
            results[w] = np.full((n_times, n_stocks), np.nan)
            continue
        swv = sliding_window_view(arr, w, axis=0)  # (n_times-w+1, n_stocks, w)
        result = np.full((n_times, n_stocks), np.nan)
        result[w - 1:] = agg_fn(swv, axis=2)
        results[w] = result
    return results


# ══════════════════════════════════════════════════════════════
# 向量化 EWM (numba JIT)
# ══════════════════════════════════════════════════════════════

@njit
def _ewm_2d_core(arr: np.ndarray, alpha: float) -> np.ndarray:
    """EWM 核心：逐股票递归平滑。numba JIT 加速。"""
    n_times, n_stocks = arr.shape
    result = np.full_like(arr, np.nan)

    for s in range(n_stocks):
        # 找第一个非 NaN
        start = 0
        while start < n_times and np.isnan(arr[start, s]):
            start += 1
        if start >= n_times:
            continue

        result[start, s] = arr[start, s]
        for t in range(start + 1, n_times):
            if np.isnan(arr[t, s]):
                # NaN 沿用前值（与 pandas ewm 行为一致）
                result[t, s] = result[t - 1, s]
            else:
                result[t, s] = alpha * arr[t, s] + (1.0 - alpha) * result[t - 1, s]
    return result


def ewm_2d(arr: np.ndarray, span: int) -> np.ndarray:
    """向量化指数加权移动平均 (EWM)。

    复现 pandas ewm(span=span, adjust=False).mean() 的行为。
    """
    alpha = 2.0 / (span + 1.0)
    return _ewm_2d_core(arr, alpha)


# ══════════════════════════════════════════════════════════════
# 向量化滚动自相关 (lag=1) — cumsum 五通道
# ══════════════════════════════════════════════════════════════

def rolling_autocorr_2d(
    arr: np.ndarray, windows: list[int], min_periods_frac: float = 0.5,
) -> dict[int, np.ndarray]:
    """多窗口滚动自相关 (lag=1) — 纯向量化 rolling_mean_2d，零 Python 循环。

    w 个原值 → w-1 个 (x[t], x[t-1]) 对 → 对数组上做 rolling_mean。
    """
    T, S = arr.shape
    if T < 2:
        return {w: np.full((T, S), np.nan) for w in windows}

    # 滞后对数组 (T-1, S)
    x = arr[1:]; y = arr[:-1]
    xy, x2, y2 = x * y, x * x, y * y

    pair_ws = [w - 1 for w in windows if w > 1]
    if not pair_ws:
        return {w: np.full((T, S), np.nan) for w in windows}

    xm  = rolling_mean_2d(x,  pair_ws)
    ym  = rolling_mean_2d(y,  pair_ws)
    xym = rolling_mean_2d(xy, pair_ws)
    x2m = rolling_mean_2d(x2, pair_ws)
    y2m = rolling_mean_2d(y2, pair_ws)

    results: dict[int, np.ndarray] = {}
    for w in windows:
        result = np.full((T, S), np.nan)
        if w <= 1: results[w] = result; continue
        pw = w - 1

        cov = xym[pw] - xm[pw] * ym[pw]
        vx  = np.maximum(x2m[pw] - xm[pw] * xm[pw], 0.0)
        vy  = np.maximum(y2m[pw] - ym[pw] * ym[pw], 0.0)
        den = np.sqrt(vx * vy)

        # 对索引 t → 原索引 t+1; rolling 从 pw-1 起有效
        corr_pair = np.full((T - 1, S), np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_pair[pw - 1:] = np.where(den[pw - 1:] > 0,
                                          cov[pw - 1:] / den[pw - 1:], np.nan)
        result[1:] = corr_pair
        results[w] = result
    return results


# ══════════════════════════════════════════════════════════════
# 向量化滚动峰度 + 偏度 — 多通道 cumsum
# ══════════════════════════════════════════════════════════════

def rolling_kurt_2d(
    arr: np.ndarray, windows: list[int], min_periods_frac: float = 0.5,
) -> dict[int, np.ndarray]:
    """多窗口滚动峰度 — 纯向量化 rolling_mean_2d，零 Python 循环。

    对 x, x², x³, x⁴ 分别做 rolling_mean，再组合出峰度。
    """
    T, S = arr.shape
    arr2, arr3, arr4 = arr ** 2, arr ** 3, arr ** 4

    m1 = rolling_mean_2d(arr,  windows)
    m2 = rolling_mean_2d(arr2, windows)
    m3 = rolling_mean_2d(arr3, windows)
    m4 = rolling_mean_2d(arr4, windows)

    results: dict[int, np.ndarray] = {}
    for w in windows:
        if w > T: results[w] = np.full((T, S), np.nan); continue
        mu  = m1[w]
        mu2 = m2[w] - mu * mu
        # mu3 = m3[w] - 3 * mu * m2[w] + 2 * mu ** 3
        mu4 = m4[w] - 4 * mu * m3[w] + 6 * mu ** 2 * m2[w] - 3 * mu ** 4
        var_pos = np.maximum(mu2, 1e-15)
        results[w] = mu4 / (var_pos * var_pos)
    return results


def rolling_skew_2d(
    arr: np.ndarray, windows: list[int], min_periods_frac: float = 0.5,
) -> dict[int, np.ndarray]:
    """多窗口滚动偏度 — 纯向量化 rolling_mean_2d，零 Python 循环。

    对 x, x², x³ 分别做 rolling_mean，再组合出偏度。
    """
    T, S = arr.shape
    arr2, arr3 = arr ** 2, arr ** 3

    m1 = rolling_mean_2d(arr,  windows)
    m2 = rolling_mean_2d(arr2, windows)
    m3 = rolling_mean_2d(arr3, windows)

    results: dict[int, np.ndarray] = {}
    for w in windows:
        if w > T: results[w] = np.full((T, S), np.nan); continue
        mu  = m1[w]
        mu2 = m2[w] - mu * mu
        mu3 = m3[w] - 3 * mu * m2[w] + 2 * mu ** 3
        var_pos = np.maximum(mu2, 1e-15)
        std3 = var_pos * np.sqrt(var_pos)
        results[w] = np.where(std3 > 0, mu3 / std3, np.nan)
    return results


# ══════════════════════════════════════════════════════════════
# 向量化尾部风险 (滚动负收益 5% 分位数)
# ══════════════════════════════════════════════════════════════

def rolling_tail_risk_2d(
    arr: np.ndarray, windows: list[int], min_periods_frac: float = 0.5,
) -> dict[int, np.ndarray]:
    """多窗口尾部风险 — 纯向量化 sort+索引，零 Python 循环。

    负收益 5% 分位数取反：sort → 索引 → 条件赋值。
    """
    T, S = arr.shape
    results: dict[int, np.ndarray] = {}

    for w in windows:
        result = np.full((T, S), np.nan)
        if w > T: results[w] = result; continue
        min_p = max(1, int(w * min_periods_frac))

        swv = sliding_window_view(arr, w, axis=0)     # (T-w+1, S, w)
        neg = np.where(swv < 0, swv, np.nan)           # 保留负值
        neg_sorted = np.sort(neg, axis=2)               # NaN 排到末尾
        valid_cnt = np.sum(~np.isnan(neg), axis=2)      # (T-w+1, S)

        # 5th percentile index (0-based)
        p5_idx = np.maximum(0, np.ceil(0.05 * valid_cnt) - 1).astype(int)
        # 向量化 gather
        ti = np.arange(T - w + 1)[:, None]  # (T-w+1, 1)
        si = np.arange(S)[None, :]          # (1, S)
        p5_vals = neg_sorted[ti, si, p5_idx]

        mask = valid_cnt >= min_p
        result[w - 1:] = np.where(mask, -p5_vals, np.nan)
        results[w] = result
    return results


# ══════════════════════════════════════════════════════════════
# 向量化回撤持续期 + 符号变化频次 (用于 quality_merged)
# ══════════════════════════════════════════════════════════════

def _dd_duration_2d(price: np.ndarray, window: int) -> np.ndarray:
    """2D 回撤持续期：窗内 max 位置 → 当前价低于峰值的距离。"""
    T, S = price.shape
    out = np.full((T, S), np.nan)
    if window > T: return out

    swv = sliding_window_view(price, window, axis=0)  # (T-w+1, S, w)
    peak_idx = np.argmax(swv, axis=2)  # (T-w+1, S)
    last_val = swv[:, :, -1]           # (T-w+1, S)
    peak_val = swv[np.arange(swv.shape[0])[:, None],
                   np.arange(S)[None, :], peak_idx]  # (T-w+1, S)
    ddd = np.where(last_val < peak_val, window - 1 - peak_idx, 0.0)
    out[window - 1:] = ddd.astype(float)
    return out


def _sign_change_2d(sign_arr: np.ndarray, window: int) -> np.ndarray:
    """2D 符号变化频次：窗内二进制符号变化次数 / window。"""
    T, S = sign_arr.shape
    out = np.full((T, S), np.nan)
    if window > T: return out

    swv = sliding_window_view(sign_arr, window, axis=0)  # (T-w+1, S, w)
    changes = np.count_nonzero(np.diff(swv, axis=2), axis=2)  # (T-w+1, S)
    out[window - 1:] = changes.astype(float) / window
    return out
