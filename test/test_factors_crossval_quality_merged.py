"""
quality_merged 因子对拍验证 — 收益稳定性/Sharpe/正收益占比/最大回撤/截面动量。
每测试以 scope='class' 的 fixture params 重复 10 组随机数据。
"""

import numpy as np
import pytest
from test.crossval_helpers import (
    _compare_factor_output, _make_data, _roll_manual, _ts_pct_manual,
)


class TestQualityMergedCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(80, 8)

    def test_ret_stability(self, f3d):
        """factor_qb_ret_stability_{p}d = 1 / rolling_std(daily_ret, p)。"""
        from seafquant.factor.quality_merged import compute_quality_merged_factors
        actual = compute_quality_merged_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_ret'] = _ts_pct_manual(df, 'close', 1)
        expected = {}
        for p in [20, 60, 120]:
            std = _roll_manual(df, '_ret', p, 'std')
            expected[f'factor_qb_ret_stability_{p}d'] = (1.0 / std.replace(0, np.nan)).values
        _compare_factor_output(actual, expected)

    def test_sharpe(self, f3d):
        """factor_qb_sharpe_{p}d = mean(ret) / std(ret)。"""
        from seafquant.factor.quality_merged import compute_quality_merged_factors
        actual = compute_quality_merged_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_ret'] = _ts_pct_manual(df, 'close', 1)
        expected = {}
        for p in [20, 60, 120]:
            mean = _roll_manual(df, '_ret', p, 'mean')
            std = _roll_manual(df, '_ret', p, 'std')
            expected[f'factor_qb_sharpe_{p}d'] = (mean / std.replace(0, np.nan)).values
        _compare_factor_output(actual, expected)

    def test_pos_days(self, f3d):
        """factor_qb_pos_days_{p}d = rolling_mean(ret>0, p)。「逐股票沿时序」验证。"""
        from seafquant.factor.quality_merged import compute_quality_merged_factors
        actual = compute_quality_merged_factors('test', f3d, None)
        df = f3d.df.copy()
        ret = _ts_pct_manual(df, 'close', 1)
        df['_pos'] = (ret > 0).astype(float)
        expected = {}
        for p in [20, 60, 120]:
            expected[f'factor_qb_pos_days_{p}d'] = _roll_manual(df, '_pos', p, 'mean').values
        _compare_factor_output(actual, expected)

    def test_maxdd(self, f3d):
        """factor_qb_maxdd_{p}d = close / rolling_max(close, p) - 1。"""
        from seafquant.factor.quality_merged import compute_quality_merged_factors
        actual = compute_quality_merged_factors('test', f3d, None)
        df = f3d.df.copy()
        for p in [60, 120]:
            rmax = _roll_manual(df, 'close', p, 'max')
            val = df['close'] / rmax.replace(0, np.nan) - 1
            _compare_factor_output(actual, {f'factor_qb_maxdd_{p}d': val.values})

    def test_amp_stability(self, f3d):
        """factor_qb_amp_stability_{p}d = 1 / rolling_std((high-low)/close, p)。"""
        from seafquant.factor.quality_merged import compute_quality_merged_factors
        actual = compute_quality_merged_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_amp'] = (df['high'] - df['low']) / df['close']
        expected = {}
        for p in [20, 60]:
            std = _roll_manual(df, '_amp', p, 'std')
            expected[f'factor_qb_amp_stability_{p}d'] = (1.0 / std.replace(0, np.nan)).values
        _compare_factor_output(actual, expected)

    def test_cs_momentum(self, f3d):
        """factor_cs_momentum_{p}d = cs_zscore(pct_change(close, p))。"""
        from seafquant.factor.quality_merged import compute_quality_merged_factors
        from qpipe.frame3d import Frame3D
        actual = compute_quality_merged_factors('test', f3d, None)
        df = f3d.df.copy()
        expected = {}
        for p in [5, 20]:
            ret = _ts_pct_manual(df, 'close', p)
            z = Frame3D(df.assign(_r=ret)).cs_zscore('_r').df['_r']
            expected[f'factor_cs_momentum_{p}d'] = z.values
        _compare_factor_output(actual, expected)
