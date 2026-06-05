"""
质量符号/形态因子 — 3 个因子：符号变化频次/最大连续正向占比/回撤持续时间。
从 quality_pattern 拆分而来，使用 sliding_window_view 向量化。
"""
import numpy as np
import logging
from numpy.lib.stride_tricks import sliding_window_view
from qpipe.frame3d import Frame3D


def compute_quality_sign_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 3 个质量符号/形态因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    ret = f3d.ts_pct_change('close', 1).df['close']
    grp = f3d.df.index.get_level_values('name')

    # ---- 1-2: 回撤持续时间 ----
    def _dd_duration_vec(series, window):
        arr = series.values
        n = len(arr)
        if n < window: return np.full(n, np.nan)
        win = sliding_window_view(arr, window)
        peak_idx = np.argmax(win, axis=1)
        last = win[np.arange(len(win)), -1]
        peak_val = win[np.arange(len(win)), peak_idx]
        ddd = np.where(last < peak_val, window - 1 - peak_idx, 0.0)
        out = np.full(n, np.nan)
        out[window - 1:] = ddd
        return out

    result = result.add_column('factor_qa_ddd_60d',
                               close.groupby(grp).transform(lambda x: _dd_duration_vec(x, 60)))
    result = result.add_column('factor_qa_ddd_120d',
                               close.groupby(grp).transform(lambda x: _dd_duration_vec(x, 120)))

    # ---- 3: 符号变化频次 ----
    def _sign_change_vec(series, window):
        arr = (series.values > 0).astype(np.int8)
        n = len(arr)
        if n < window: return np.full(n, np.nan)
        win = sliding_window_view(arr, window)
        changes = np.count_nonzero(np.diff(win, axis=1), axis=1)
        out = np.full(n, np.nan)
        out[window - 1:] = changes.astype(float) / window
        return out

    result = result.add_column('factor_qa_consec_sign_change_60d',
                               ret.groupby(grp).transform(lambda x: _sign_change_vec(x, 60)))

    factor_cols = [c for c in result.df.columns if c.startswith('factor_qa_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] QA-Sign NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
