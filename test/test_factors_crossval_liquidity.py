"""
liquidity 因子对拍验证 — 换手率 + Amihud + 规模。
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


class TestLiquidityCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(60, 8)

    def test_turnover_mean(self, f3d):
        """factor_liq_turnover_{p}d = rolling_mean(turnover, p)。"""
        from seafquant.factor.liquidity import compute_liquidity_factors
        actual = compute_liquidity_factors('test', f3d, None)
        expected = {}
        for p in [5, 10, 20, 60]:
            expected[f'factor_liq_turnover_{p}d'] = _roll_manual(f3d.df, 'turnover', p, 'mean').values
        _compare_factor_output(actual, expected)

    def test_amihud(self, f3d):
        """factor_liq_amihud_{p}d = rolling_mean(|ret|/volume, p)。"""
        from seafquant.factor.liquidity import compute_liquidity_factors
        actual = compute_liquidity_factors('test', f3d, None)
        df = f3d.df.copy()
        ret = _ts_pct_manual(df, 'close', 1)
        df['_amihud'] = np.abs(ret) / df['volume'].replace(0, np.nan)
        expected = {}
        for p in [5, 20]:
            expected[f'factor_liq_amihud_{p}d'] = _roll_manual(df, '_amihud', p, 'mean').values
        _compare_factor_output(actual, expected)

    def test_dollar_vol(self, f3d):
        """factor_liq_dollar_vol = log(close * volume)。"""
        from seafquant.factor.liquidity import compute_liquidity_factors
        actual = compute_liquidity_factors('test', f3d, None)
        dv = f3d.df['close'] * f3d.df['volume']
        _compare_factor_output(actual, {'factor_liq_dollar_vol': np.log(dv).values})

    def test_size_basic(self, f3d):
        """factor_size_log_mcap / cs_rank / sqrt / cbrt / price。"""
        from qpipe.frame3d import Frame3D
        from seafquant.factor.liquidity import compute_liquidity_factors
        actual = compute_liquidity_factors('test', f3d, None)
        mcap = f3d.df['market_cap']
        cs_rank = Frame3D(f3d.df.copy()).cs_rank('market_cap').df['market_cap']
        expected = {
            'factor_size_log_mcap': (-np.log(mcap)).values,
            'factor_size_cs_rank': cs_rank.values,
            'factor_size_mcap_sqrt': (-np.sqrt(mcap)).values,
            'factor_size_mcap_cube_root': (-np.cbrt(mcap)).values,
            'factor_size_price': np.log(mcap / f3d.df['close']).values,
        }
        _compare_factor_output(actual, expected)

    def test_size_mcap_chg(self, f3d):
        """factor_size_mcap_chg_{p}d = pct_change(mcap, p)。"""
        from seafquant.factor.liquidity import compute_liquidity_factors
        actual = compute_liquidity_factors('test', f3d, None)
        expected = {}
        for p in [5, 20, 60]:
            expected[f'factor_size_mcap_chg_{p}d'] = _ts_pct_manual(f3d.df, 'market_cap', p).values
        _compare_factor_output(actual, expected)

    def test_size_mcap_mom(self, f3d):
        """factor_size_mcap_mom_{5,20}d = pct_change(mcap, 5/20)。"""
        from seafquant.factor.liquidity import compute_liquidity_factors
        actual = compute_liquidity_factors('test', f3d, None)
        expected = {
            'factor_size_mcap_mom_5d': _ts_pct_manual(f3d.df, 'market_cap', 5).values,
            'factor_size_mcap_mom_20d': _ts_pct_manual(f3d.df, 'market_cap', 20).values,
        }
        _compare_factor_output(actual, expected)
