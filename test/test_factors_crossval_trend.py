"""
trend 因子对拍验证 — MA偏离 + MA交叉 + MACD + 通道 + 时序动量。
每测试以 scope='class' 的 fixture params 重复 10 组随机数据。
"""

import numpy as np
import pytest
from test.crossval_helpers import _compare_factor_output, _make_data, _roll_manual


class TestTrendCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(60, 8)

    def test_ma_deviation(self, f3d):
        """factor_trend_ma_{p}d = close / rolling_mean(close, p) - 1。"""
        from seafquant.factor.trend import compute_trend_factors
        actual = compute_trend_factors('test', f3d, None)
        df = f3d.df.copy()
        expected = {}
        for p in [5, 10, 20, 60, 120]:
            ma = _roll_manual(df, 'close', p, 'mean')
            expected[f'factor_trend_ma_{p}d'] = (df['close'] / ma.replace(0, np.nan) - 1).values
        _compare_factor_output(actual, expected)

    def test_ma_cross(self, f3d):
        """factor_trend_ma_cross_xx = MA_short / MA_long - 1。"""
        from seafquant.factor.trend import compute_trend_factors
        actual = compute_trend_factors('test', f3d, None)
        df = f3d.df.copy()
        ma5, ma10, ma20, ma60, ma120 = (
            _roll_manual(df, 'close', p, 'mean') for p in [5, 10, 20, 60, 120]
        )
        expected = {
            'factor_trend_ma_cross_5_20': (ma5 / ma20.replace(0, np.nan) - 1).values,
            'factor_trend_ma_cross_10_60': (ma10 / ma60.replace(0, np.nan) - 1).values,
            'factor_trend_ma_cross_20_120': (ma20 / ma120.replace(0, np.nan) - 1).values,
        }
        _compare_factor_output(actual, expected)

    def test_macd(self, f3d):
        """factor_trend_macd = EMA12-EMA26, signal = macd - EMA9(macd)。"""
        from seafquant.factor.trend import compute_trend_factors
        actual = compute_trend_factors('test', f3d, None)
        df = f3d.df.copy()
        grp = df.index.get_level_values('name')

        def _ema(series, span):
            return series.ewm(span=span, adjust=False).mean()

        ema12 = df['close'].groupby(grp).transform(lambda x: _ema(x, 12))
        ema26 = df['close'].groupby(grp).transform(lambda x: _ema(x, 26))
        macd = ema12 - ema26
        macd_signal = macd.groupby(grp).transform(lambda x: _ema(x, 9))
        expected = {
            'factor_trend_macd': macd.values,
            'factor_trend_macd_signal': (macd - macd_signal).values,
        }
        _compare_factor_output(actual, expected)

    def test_channel(self, f3d):
        """factor_trend_channel_{p}d = (close-min_p) / (max_p-min_p)。"""
        from seafquant.factor.trend import compute_trend_factors
        actual = compute_trend_factors('test', f3d, None)
        df = f3d.df.copy()
        for p in [20, 60]:
            rmin = _roll_manual(df, 'close', p, 'min')
            rmax = _roll_manual(df, 'close', p, 'max')
            denom = (rmax - rmin).replace(0, np.nan)
            _compare_factor_output(
                actual, {f'factor_trend_channel_{p}d': ((df['close'] - rmin) / denom).values},
            )

    def test_mom_strength(self, f3d):
        """factor_trend_mom_strength_{p}d = ts_zscore(close, p)。"""
        from seafquant.factor.trend import compute_trend_factors
        from qpipe.frame3d import Frame3D
        actual = compute_trend_factors('test', f3d, None)
        expected = {}
        for p in [20, 60]:
            z = Frame3D(f3d.df.copy()).ts_zscore('close', p).df['close']
            expected[f'factor_trend_mom_strength_{p}d'] = z.values
        _compare_factor_output(actual, expected)
