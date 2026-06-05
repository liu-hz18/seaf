"""
计数新高新低因子 — 3 个因子：20/60日创新高次数 + 60日创新低次数。
从 counting_streak 拆分而来，使用 sliding_window_view 向量化。
"""
import numpy as np
import logging
from numpy.lib.stride_tricks import sliding_window_view
from qpipe.frame3d import Frame3D


def compute_counting_nh_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 3 个新高新低因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    grp = f3d.df.index.get_level_values('name')

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
        out[window - 1:] = win.sum(axis=1)
        return out

    def _new_low_count(series, window):
        arr = series.values.astype(float)
        n = len(arr)
        if n < max(2, window // 2):
            return np.full(n, np.nan)
        clean = np.where(np.isnan(arr), np.inf, arr)
        cmin = np.minimum.accumulate(clean)
        is_nl = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if arr[i] < cmin[i - 1] and cmin[i - 1] != np.inf:
                is_nl[i] = 1.0
        win = sliding_window_view(is_nl, window)
        out = np.full(n, np.nan)
        out[window - 1:] = win.sum(axis=1)
        return out

    result = result.add_column('factor_cnt_new_high_20d',
                               close.groupby(grp).transform(lambda x: _new_high_count(x, 20)))
    result = result.add_column('factor_cnt_new_high_60d',
                               close.groupby(grp).transform(lambda x: _new_high_count(x, 60)))
    result = result.add_column('factor_cnt_new_low_60d',
                               close.groupby(grp).transform(lambda x: _new_low_count(x, 60)))

    factor_cols = [c for c in result.df.columns if c.startswith('factor_cnt_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] CountingNH NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
