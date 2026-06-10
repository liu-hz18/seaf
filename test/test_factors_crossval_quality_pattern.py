"""
quality_pattern 因子对拍验证 — HL稳定性/峰度/偏度/MaxConsecPos。
每测试以 scope='class' 的 fixture params 重复 10 组随机数据。
"""

import numpy as np
import pytest

from test.crossval_helpers import (
    _compare_factor_output,
    _make_data,
    _roll_manual,
    _ts_pct_manual,
)


class TestQualityPatternCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(80, 8)

    def test_hl_stability(self, f3d):
        """factor_qa_hl_stability_{p}d = 1 / rolling_std(high/low-1, p)。"""
        from seafquant.factor.quality_pattern import compute_quality_pattern_factors
        actual = compute_quality_pattern_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_hl'] = df['high'] / df['low'] - 1
        expected = {}
        for p in [20, 60]:
            std = _roll_manual(df, '_hl', p, 'std')
            expected[f'factor_qa_hl_stability_{p}d'] = (1.0 / std.replace(0, np.nan)).values
        _compare_factor_output(actual, expected)

    def test_kurt(self, f3d):
        """factor_qa_kurt_{p}d = rolling_kurt(daily_ret, p)。「逐股票沿时序」验证。"""
        from seafquant.factor.quality_pattern import compute_quality_pattern_factors
        actual = compute_quality_pattern_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_ret'] = _ts_pct_manual(df, 'close', 1)
        expected = {}
        for p in [60, 120]:
            expected[f'factor_qa_kurt_{p}d'] = _roll_manual(df, '_ret', p, 'kurt').values
        _compare_factor_output(actual, expected)

    def test_skew(self, f3d):
        """factor_qa_skew_{p}d = rolling_skew(daily_ret, p)。「逐股票沿时序」验证。"""
        from seafquant.factor.quality_pattern import compute_quality_pattern_factors
        actual = compute_quality_pattern_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_ret'] = _ts_pct_manual(df, 'close', 1)
        expected = {}
        for p in [60, 120]:
            expected[f'factor_qa_skew_{p}d'] = _roll_manual(df, '_ret', p, 'skew').values
        _compare_factor_output(actual, expected)

    def test_max_consec_pos(self, f3d):
        """factor_qa_max_consec_pos_60d：滑动窗口内最长连续正收益占比。"""
        from numpy.lib.stride_tricks import sliding_window_view

        from seafquant.factor.quality_pattern import compute_quality_pattern_factors
        actual = compute_quality_pattern_factors('test', f3d, None)

        def _mcp_vec(series, window):
            arr = (series.values > 0).astype(np.int8)
            n = len(arr)
            if n < window:
                return np.full(n, np.nan)
            win = sliding_window_view(arr, window)
            max_runs = np.zeros(len(win), dtype=float)
            for i in range(len(win)):
                cur = best = 0
                for v in win[i]:
                    if v:
                        cur += 1
                    if cur > best:
                        best = cur
                    else:
                        cur = 0
                max_runs[i] = best
            out = np.full(n, np.nan)
            out[window - 1:] = max_runs / window
            return out

        df = f3d.df.copy()
        df['_ret'] = _ts_pct_manual(df, 'close', 1)
        mcp = df.groupby('name')['_ret'].transform(lambda x: _mcp_vec(x, 60))
        _compare_factor_output(actual, {'factor_qa_max_consec_pos_60d': mcp.values})
