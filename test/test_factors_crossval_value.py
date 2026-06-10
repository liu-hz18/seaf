"""
value 因子对拍验证 — 直接公式 + MA偏离 + 回撤 + 复合。
每测试以 scope='class' 的 fixture params 重复 10 组随机数据。
"""

import numpy as np
import pytest

from test.crossval_helpers import _compare_factor_output, _make_data, _roll_manual


class TestValueCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(60, 8)

    def test_direct_formulas(self, f3d):
        """factor_val_inv_price / log_mcap / mcap_to_price / turnover_yield。"""
        from seafquant.factor.value import compute_value_factors
        actual = compute_value_factors('test', f3d, None)
        df = f3d.df
        expected = {
            'factor_val_inv_price': (1.0 / df['close']).values,
            'factor_val_log_mcap': (-np.log(df['market_cap'])).values,
            'factor_val_mcap_to_price': (df['market_cap'] / df['close']).values,
            'factor_val_turnover_yield': (1.0 / df['turnover']).values,
        }
        _compare_factor_output(actual, expected)

    def test_dist_ma(self, f3d):
        """factor_val_dist_ma_{p}d = -(close/MA - 1)。"""
        from seafquant.factor.value import compute_value_factors
        actual = compute_value_factors('test', f3d, None)
        df = f3d.df.copy()
        for p in [20, 60, 120]:
            ma = _roll_manual(df, 'close', p, 'mean')
            val = -(df['close'] / ma.replace(0, np.nan) - 1)
            _compare_factor_output(actual, {f'factor_val_dist_ma_{p}d': val.values})

    def test_dd(self, f3d):
        """factor_val_dd_{p}d = close / rolling_max(close, p) - 1。"""
        from seafquant.factor.value import compute_value_factors
        actual = compute_value_factors('test', f3d, None)
        df = f3d.df.copy()
        for p in [60, 120]:
            rmax = _roll_manual(df, 'close', p, 'max')
            val = df['close'] / rmax.replace(0, np.nan) - 1
            _compare_factor_output(actual, {f'factor_val_dd_{p}d': val.values})

    def test_sharpe_inv(self, f3d):
        """factor_val_sharpe_inv = -ts_zscore(close, 60)。"""
        from qpipe.frame3d import Frame3D
        from seafquant.factor.value import compute_value_factors
        actual = compute_value_factors('test', f3d, None)
        z = Frame3D(f3d.df.copy()).ts_zscore('close', 60).df['close']
        _compare_factor_output(actual, {'factor_val_sharpe_inv': (-z).values})

    def test_price_to_range(self, f3d):
        """factor_val_price_to_range = -(close-min120) / (max120-min120)。"""
        from seafquant.factor.value import compute_value_factors
        actual = compute_value_factors('test', f3d, None)
        df = f3d.df.copy()
        rmin = _roll_manual(df, 'close', 120, 'min')
        rmax = _roll_manual(df, 'close', 120, 'max')
        val = -(df['close'] - rmin) / (rmax - rmin).replace(0, np.nan)
        _compare_factor_output(actual, {'factor_val_price_to_range': val.values})
