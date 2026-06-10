"""
momentum 因子对拍验证 — 动量收益率 + 波动率调整 + 反转。
每测试以 scope='class' 的 fixture params 重复 10 组随机数据。
"""

import pytest

from test.crossval_helpers import (
    EPS,
    _compare_factor_output,
    _make_data,
    _roll_manual,
    _ts_pct_manual,
)


class TestMomentumCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(60, 8)

    def test_ret_pct_change(self, f3d):
        """factor_mom_ret_{p}d = 逐股票 pct_change(close, p)。"""
        from seafquant.factor.momentum import compute_momentum_factors
        actual = compute_momentum_factors('test', f3d, None)
        expected = {}
        for p in [1, 3, 5, 10, 20, 40, 60, 120]:
            expected[f'factor_mom_ret_{p}d'] = _ts_pct_manual(f3d.df, 'close', p).values
        _compare_factor_output(actual, expected)

    def test_voladj(self, f3d):
        """factor_mom_voladj_{p}d = ret_pct / rolling_std(daily_ret, max(p,5))。"""
        from seafquant.factor.momentum import compute_momentum_factors
        actual = compute_momentum_factors('test', f3d, None)
        df = f3d.df.copy()
        daily_ret = df.groupby('name')['close'].pct_change(1)
        df['_ret'] = daily_ret
        expected = {}
        for p in [1, 3, 5, 10, 20, 40, 60, 120]:
            w = max(p, 5)
            vol = _roll_manual(df, '_ret', w, 'std')
            ret_pct = _ts_pct_manual(df, 'close', p)
            expected[f'factor_mom_voladj_{p}d'] = (ret_pct / (vol + EPS)).values
        _compare_factor_output(actual, expected)

    def test_reversal(self, f3d):
        """factor_rev_ret_1d / overnight_1d / intraday_1d 取负号。"""
        from seafquant.factor.momentum import compute_momentum_factors
        actual = compute_momentum_factors('test', f3d, None)
        df = f3d.df.copy()
        ret1 = _ts_pct_manual(df, 'close', 1)
        close_d1 = df.groupby('name')['close'].shift(1)
        overnight = df['open'] / (close_d1 + EPS) - 1
        intraday = df['close'] / (df['open'] + EPS) - 1
        expected = {
            'factor_rev_ret_1d': (-ret1).values,
            'factor_rev_overnight_1d': (-overnight).values,
            'factor_rev_intraday_1d': (-intraday).values,
        }
        _compare_factor_output(actual, expected)
