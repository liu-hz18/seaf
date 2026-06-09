"""
cross_section 因子对拍验证 — 截面排名/排名变化/Z-score/收益率分位数。
每测试以 scope='class' 的 fixture params 重复 10 组随机数据。
"""

import pandas as pd
import pytest
from test.crossval_helpers import (
    _compare_factor_output, _make_data, _ts_pct_manual,
)


def _cs_rank_pct(series):
    """截面排名百分位：(rank-1)/(n-1)，min=0, max=1，单股票返回 0.5。"""
    n = len(series)
    if n <= 1:
        return pd.Series(0.5, index=series.index)
    return (series.rank() - 1) / (n - 1)


class TestCrossSectionCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(60, 8)

    def test_cs_rank(self, f3d):
        """factor_cs_rank_close/volume = 截面排名百分位。「逐时间片沿截面」验证。"""
        from seafquant.factor.cross_section import compute_cross_section_factors
        actual = compute_cross_section_factors('test', f3d, None)
        df = f3d.df.copy()
        expected = {
            'factor_cs_rank_close': df.groupby('key')['close'].transform(_cs_rank_pct).values,
            'factor_cs_rank_volume': df.groupby('key')['volume'].transform(_cs_rank_pct).values,
        }
        _compare_factor_output(actual, expected)

    def test_rank_delta(self, f3d):
        """factor_cs_rank_delta_{p}d = cs_rank(close[t]) - shift(cs_rank(close[t]), p)。

        关键：先做截面 rank，再做时序 shift。不能先 shift 后 rank，
        否则在 IPO/退市场景下股票集合不一致会导致结果不同。
        """
        from seafquant.factor.cross_section import compute_cross_section_factors
        actual = compute_cross_section_factors('test', f3d, None)
        df = f3d.df.copy()
        # 先时序 shift：cs_rank(close)，每股票内做时序 rank 偏移
        df['_rk'] = df.groupby('key')['close'].transform(_cs_rank_pct)
        rk_now = df['_rk']
        for p in [5, 20, 60]:
            rk_prev = df.groupby('name')['_rk'].shift(p)
            _compare_factor_output(
                actual, {f'factor_cs_rank_delta_{p}d': (rk_now - rk_prev).values},
            )

    def test_rank_zscore(self, f3d):
        """factor_cs_rank_zscore_{p}d = ts_zscore(close, p)。"""
        from seafquant.factor.cross_section import compute_cross_section_factors
        from qpipe.frame3d import Frame3D
        actual = compute_cross_section_factors('test', f3d, None)
        expected = {}
        for p in [5, 20, 60]:
            z = Frame3D(f3d.df.copy()).ts_zscore('close', p).df['close']
            expected[f'factor_cs_rank_zscore_{p}d'] = z.values
        _compare_factor_output(actual, expected)

    def test_ret_rank(self, f3d):
        """factor_cs_ret_rank_{p}d = cs_rank(pct_change(close, p))。"""
        from seafquant.factor.cross_section import compute_cross_section_factors
        actual = compute_cross_section_factors('test', f3d, None)
        df = f3d.df.copy()
        ret1 = _ts_pct_manual(df, 'close', 1)
        ret20 = _ts_pct_manual(df, 'close', 20)
        expected = {
            'factor_cs_ret_rank_1d': df.assign(_r=ret1).groupby('key')['_r'].transform(_cs_rank_pct).values,
            'factor_cs_ret_rank_20d': df.assign(_r=ret20).groupby('key')['_r'].transform(_cs_rank_pct).values,
        }
        _compare_factor_output(actual, expected)
