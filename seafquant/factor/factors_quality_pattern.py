"""
质量形态因子（合并：形态 + 高级） — 9 个因子。
价格形态：HL稳定性 ×2, 收益峰度 ×2, 最大连续正 ×1,
收益分布：偏度 ×2, Up/Down ×1, 复合 ×1。
"""
import numpy as np
import logging
from numpy.lib.stride_tricks import sliding_window_view
from qpipe.frame3d import Frame3D


def compute_quality_pattern_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 9 个质量形态/高级因子。"""
    result = f3d.copy()
    high, low, close = f3d.df['high'], f3d.df['low'], f3d.df['close']
    ret = f3d.ts_pct_change('close', 1).df['close']
    df = result.df
    df['_ret'] = ret
    grp = df.index.get_level_values('name')

    def _roll(src, dst, window, agg):
        df[dst] = df.groupby('name')[src].rolling(
            window, min_periods=max(1, window // 2)).agg(agg).values

    # ===== 高低价差稳定性 (2 cols) =====
    df['_hl_range'] = high / low - 1
    _roll('_hl_range', '_hl_std20', 20, 'std')
    _roll('_hl_range', '_hl_std60', 60, 'std')
    df['factor_qa_hl_stability_20d'] = 1.0 / df['_hl_std20'].replace(0, np.nan)
    df['factor_qa_hl_stability_60d'] = 1.0 / df['_hl_std60'].replace(0, np.nan)

    # ===== 收益峰度 (2 cols) =====
    _roll('_ret', 'factor_qa_kurt_60d', 60, 'kurt')
    _roll('_ret', 'factor_qa_kurt_120d', 120, 'kurt')

    # ===== 最大连续正向占比 (1 col) =====
    def _max_consec_pos_vec(series, window):
        arr = (series.values > 0).astype(np.int8)
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
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

    # ===== 收益率偏度 (2 cols) =====
    result = result.add_column('_ret2', ret, cp=False)
    df = result.df  # 同步 df 引用
    skew60 = f3d.copy().add_column('_ret2', ret).ts_rolling('_ret2', 60, 'skew').df['_ret2']
    df['factor_qa_skew_60d'] = skew60
    skew120 = f3d.copy().add_column('_ret2', ret).ts_rolling('_ret2', 120, 'skew').df['_ret2']
    df['factor_qa_skew_120d'] = skew120

    # ===== Up/Down capture ratio (1 col) =====
    def _up_down_ratio_vec(series, window):
        arr = series.values
        n = len(arr)
        if n < max(2, window // 2):
            return np.full(n, np.nan)
        win = sliding_window_view(arr, window)
        pos_mask = win > 0; neg_mask = win < 0
        pos_cnt = pos_mask.sum(axis=1)
        neg_cnt = neg_mask.sum(axis=1)
        pos_mean = np.where(pos_cnt > 0,
                            (win * pos_mask).sum(axis=1) / np.maximum(pos_cnt, 1), 0.0)
        neg_mean = np.where(neg_cnt > 0,
                            np.abs((win * neg_mask).sum(axis=1)) / np.maximum(neg_cnt, 1), 1e-6)
        neg_mean[neg_mean == 0] = 1e-6
        ratio = pos_mean / neg_mean
        out = np.full(n, np.nan)
        out[window - 1:] = ratio
        return out

    up_down = ret.groupby(grp).transform(lambda x: _up_down_ratio_vec(x, 60))
    df['factor_qa_up_down_60d'] = up_down

    # ===== 复合 (1 col) =====
    df['factor_qa_composite'] = (
        df['factor_qa_skew_60d'] + df['factor_qa_up_down_60d']) / 2

    factor_cols = [c for c in df.columns if c.startswith('factor_qa_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)
    return Frame3D(result.df[factor_cols].copy())