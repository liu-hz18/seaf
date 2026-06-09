"""
counting 因子对拍验证 — 放量/缩量 + 大涨大跌 + 新高新低 + 振幅突破。
每测试以 scope='class' 的 fixture params 重复 10 组随机数据。
"""

import numpy as np
import pytest
from test.crossval_helpers import (
    _compare_factor_output, _make_data, _roll_manual, _ts_pct_manual,
)


class TestCountingCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(80, 8)

    def test_vol_spike(self, f3d):
        """factor_cnt_vol_spike_{p}d = rolling_sum(vol > 1.5*vol_ma20, p)。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)
        df = f3d.df.copy()
        vol_ma20 = _roll_manual(df, 'volume', 20, 'mean')
        spike = (df['volume'] > 1.5 * vol_ma20).astype(float)
        df['_s20'] = spike
        df['_s60'] = spike
        expected = {
            'factor_cnt_vol_spike_20d': _roll_manual(df, '_s20', 20, 'sum').values,
            'factor_cnt_vol_spike_60d': _roll_manual(df, '_s60', 60, 'sum').values,
        }
        _compare_factor_output(actual, expected)

    def test_vol_shrink(self, f3d):
        """factor_cnt_vol_shrink_20d = rolling_sum(vol < 0.5*vol_ma20, 20)。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)
        df = f3d.df.copy()
        vol_ma20 = _roll_manual(df, 'volume', 20, 'mean')
        shrink = (df['volume'] < 0.5 * vol_ma20).astype(float)
        df['_sh'] = shrink
        expected = {'factor_cnt_vol_shrink_20d': _roll_manual(df, '_sh', 20, 'sum').values}
        _compare_factor_output(actual, expected)

    def test_big_move(self, f3d):
        """factor_cnt_big_move_{p}d = rolling_sum(|ret|>0.02, p)。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)
        df = f3d.df.copy()
        ret = _ts_pct_manual(df, 'close', 1)
        big = (np.abs(ret) > 0.02).astype(float)
        df['_big20'] = big
        df['_big60'] = big
        expected = {
            'factor_cnt_big_move_20d': _roll_manual(df, '_big20', 20, 'sum').values,
            'factor_cnt_big_move_60d': _roll_manual(df, '_big60', 60, 'sum').values,
        }
        _compare_factor_output(actual, expected)

    def test_amp_break(self, f3d):
        """factor_cnt_amp_break_20d = rolling_sum(amp > 1.5*amp_ma20, 20)。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)
        df = f3d.df.copy()
        amp = (df['high'] - df['low']) / df['close']
        amp_ma20 = _roll_manual(df.assign(_amp=amp), '_amp', 20, 'mean')
        brk = (amp > 1.5 * amp_ma20).astype(float)
        df['_brk'] = brk
        expected = {'factor_cnt_amp_break_20d': _roll_manual(df, '_brk', 20, 'sum').values}
        _compare_factor_output(actual, expected)

    def test_new_high(self, f3d):
        """factor_cnt_new_high_20d：逐股票创新高 days 计数。「逐股票沿时序」验证。"""
        from seafquant.factor.counting import compute_counting_factors
        from numpy.lib.stride_tricks import sliding_window_view
        actual = compute_counting_factors('test', f3d, None)

        def _new_high_manual(series, window):
            arr = series.values.astype(float)
            n = len(arr)
            if n < window:
                return np.full(n, np.nan)
            clean = np.where(np.isnan(arr), -np.inf, arr)
            cmax = np.maximum.accumulate(clean)
            is_nh = np.zeros(n, dtype=float)
            for i in range(1, n):
                if arr[i] > cmax[i - 1] and cmax[i - 1] != -np.inf:
                    is_nh[i] = 1.0
            win = sliding_window_view(is_nh, window)
            out = np.full(n, np.nan)
            out[window - 1:] = win.sum(axis=1)
            return out

        nh = f3d.df.groupby('name')['close'].transform(lambda x: _new_high_manual(x, 20))
        _compare_factor_output(actual, {'factor_cnt_new_high_20d': nh.values})
