"""
质量形态因子 — 5 个因子：高低价差稳定性×2 / 收益峰度×2 / 最大连续正向占比。
优化：使用直接 pandas rolling 替代 f3d.copy 链。
"""
import numpy as np
import logging
from numpy.lib.stride_tricks import sliding_window_view
from qpipe.frame3d import Frame3D


def compute_quality_pattern_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 5 个质量形态因子。"""
    result = f3d.copy()
    high = f3d.df['high']
    low = f3d.df['low']
    ret = f3d.ts_pct_change('close', 1).df['close']
    df = result.df
    df['_ret'] = ret
    grp = f3d.df.index.get_level_values('name')

    def _roll(src, dst, window, agg):
        df[dst] = df.groupby('name')[src].rolling(window, min_periods=max(1, window // 2)).agg(agg).values

    # ---- 1-2: 高低价差稳定性 ----
    df['_hl_range'] = high / low - 1
    _roll('_hl_range', '_hl_std20', 20, 'std')
    _roll('_hl_range', '_hl_std60', 60, 'std')
    df['factor_qa_hl_stability_20d'] = 1.0 / df['_hl_std20'].replace(0, np.nan)
    df['factor_qa_hl_stability_60d'] = 1.0 / df['_hl_std60'].replace(0, np.nan)

    # ---- 3-4: 收益峰度 ----
    _roll('_ret', 'factor_qa_kurt_60d', 60, 'kurt')
    _roll('_ret', 'factor_qa_kurt_120d', 120, 'kurt')

    # ---- 5: 最大连续正向天数占比 ----
    def _max_consec_pos_vec(series, window):
        arr = (series.values > 0).astype(np.int8)
        n = len(arr)
        if n < window: return np.full(n, np.nan)
        win = sliding_window_view(arr, window)
        max_runs = np.zeros(len(win), dtype=float)
        for i in range(len(win)):
            cur = 0; best = 0
            for v in win[i]:
                if v: cur += 1
                if cur > best: best = cur
                else: cur = 0
            max_runs[i] = best
        out = np.full(n, np.nan)
        out[window - 1:] = max_runs / window
        return out

    df['factor_qa_max_consec_pos_60d'] = ret.groupby(grp).transform(
        lambda x: _max_consec_pos_vec(x, 60))

    factor_cols = [c for c in df.columns if c.startswith('factor_qa_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] QA-Pattern NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
