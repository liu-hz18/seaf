"""
volatility 因子对拍验证 — 已实现波动 + Parkinson + GK + 日内。
每测试以 scope='class' 的 fixture params 重复 10 组随机数据。
"""

import numpy as np
import pytest
from test.crossval_helpers import (
    _compare_factor_output, _make_data, _roll_manual, _ts_pct_manual,
)


class TestVolatilityCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(60, 8)

    def test_realized_vol(self, f3d):
        """factor_vol_realized_{p}d = rolling_std(daily_ret, p)。"""
        from seafquant.factor.volatility import compute_volatility_factors
        actual = compute_volatility_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_ret'] = _ts_pct_manual(df, 'close', 1)
        expected = {}
        for p in [5, 10, 20, 60]:
            expected[f'factor_vol_realized_{p}d'] = _roll_manual(df, '_ret', p, 'std').values
        _compare_factor_output(actual, expected)

    def test_parkinson(self, f3d):
        """factor_vol_parkinson_{p}d = sqrt(mean(park * log(high/low)^2, p))。"""
        from seafquant.factor.volatility import compute_volatility_factors
        actual = compute_volatility_factors('test', f3d, None)
        df = f3d.df.copy()
        park = 1.0 / (4 * np.log(2))
        df['_park'] = park * (np.log(df['high'] / df['low'])) ** 2
        expected = {}
        for p in [5, 20]:
            expected[f'factor_vol_parkinson_{p}d'] = (
                np.sqrt(np.abs(_roll_manual(df, '_park', p, 'mean'))).values
            )
        _compare_factor_output(actual, expected)

    def test_gk(self, f3d):
        """factor_vol_gk_5d = sqrt(mean(Garman-Klass, 5))。"""
        from seafquant.factor.volatility import compute_volatility_factors
        actual = compute_volatility_factors('test', f3d, None)
        df = f3d.df.copy()
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        df['_gk'] = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        expected = {
            'factor_vol_gk_5d': np.sqrt(np.abs(_roll_manual(df, '_gk', 5, 'mean'))).values,
        }
        _compare_factor_output(actual, expected)

    def test_downside_vol(self, f3d):
        """factor_vol_downside_{p}d = sqrt(abs(mean(clip(ret, upper=0), p)))。"""
        from seafquant.factor.volatility import compute_volatility_factors
        actual = compute_volatility_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_ret'] = _ts_pct_manual(df, 'close', 1)
        df['_neg'] = df['_ret'].clip(upper=0)
        expected = {}
        for p in [5, 10, 20, 60]:
            expected[f'factor_vol_downside_{p}d'] = (
                np.sqrt(np.abs(_roll_manual(df, '_neg', p, 'mean'))).values
            )
        _compare_factor_output(actual, expected)

    def test_of_vol(self, f3d):
        """factor_vol_of_vol_{p}d = rolling_std(realized_vol_smaller, p)。"""
        from seafquant.factor.volatility import compute_volatility_factors
        actual = compute_volatility_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_ret'] = _ts_pct_manual(df, 'close', 1)
        df['_rv5'] = _roll_manual(df, '_ret', 5, 'std')
        df['_rv20'] = _roll_manual(df, '_ret', 20, 'std')
        expected = {
            'factor_vol_of_vol_20d': _roll_manual(df, '_rv5', 20, 'std').values,
            'factor_vol_of_vol_60d': _roll_manual(df, '_rv20', 60, 'std').values,
        }
        _compare_factor_output(actual, expected)

    def test_intraday_hl_range(self, f3d):
        """factor_intra_hl_range_{p}d = rolling_mean((high-low)/close, p)。"""
        from seafquant.factor.volatility import compute_volatility_factors
        actual = compute_volatility_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_hl'] = (df['high'] - df['low']) / df['close']
        expected = {}
        for p in [5, 20, 60]:
            expected[f'factor_intra_hl_range_{p}d'] = _roll_manual(df, '_hl', p, 'mean').values
        _compare_factor_output(actual, expected)
