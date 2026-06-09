"""
interaction 因子对拍验证 — 量价共振/活跃度加权/市值交互/方向振幅。
每测试以 scope='class' 的 fixture params 重复 10 组随机数据。
"""

import numpy as np
import pandas as pd
import pytest
from test.crossval_helpers import (
    _compare_factor_output, _make_data, _ts_pct_manual,
)


def _cs_rank_pct(series):
    """截面排名百分位 (0~1)。"""
    n = len(series)
    if n <= 1:
        return pd.Series(0.5, index=series.index)
    return (series.rank() - 1) / (n - 1)


class TestInteractionCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(60, 8)

    def test_ret_vol(self, f3d):
        """factor_inter_ret_vol_{p}d = ret_pct * cs_rank(volume)。"""
        from seafquant.factor.interaction import compute_interaction_factors
        actual = compute_interaction_factors('test', f3d, None)
        df = f3d.df.copy()
        vol_rank = df.groupby('key')['volume'].transform(_cs_rank_pct)
        ret1 = _ts_pct_manual(df, 'close', 1)
        ret5 = _ts_pct_manual(df, 'close', 5)
        expected = {
            'factor_inter_ret_vol_1d': (ret1 * vol_rank).values,
            'factor_inter_ret_vol_5d': (ret5 * vol_rank).values,
        }
        _compare_factor_output(actual, expected)

    def test_ret_turnover(self, f3d):
        """factor_inter_ret_turnover_{p}d = ret_pct * cs_rank(turnover)。"""
        from seafquant.factor.interaction import compute_interaction_factors
        actual = compute_interaction_factors('test', f3d, None)
        df = f3d.df.copy()
        to_rank = df.groupby('key')['turnover'].transform(_cs_rank_pct)
        ret1 = _ts_pct_manual(df, 'close', 1)
        ret20 = _ts_pct_manual(df, 'close', 20)
        expected = {
            'factor_inter_ret_turnover_1d': (ret1 * to_rank).values,
            'factor_inter_ret_turnover_20d': (ret20 * to_rank).values,
        }
        _compare_factor_output(actual, expected)

    def test_hl_vol(self, f3d):
        """factor_inter_hl_vol_1d = (high-low)/close * cs_rank(volume)。"""
        from seafquant.factor.interaction import compute_interaction_factors
        actual = compute_interaction_factors('test', f3d, None)
        df = f3d.df.copy()
        vol_rank = df.groupby('key')['volume'].transform(_cs_rank_pct)
        hl_range = (df['high'] - df['low']) / df['close']
        expected = {'factor_inter_hl_vol_1d': (hl_range * vol_rank).values}
        _compare_factor_output(actual, expected)

    def test_direction_range(self, f3d):
        """factor_inter_direction_range_{p}d = sign(ret) * (high-low)/close。"""
        from seafquant.factor.interaction import compute_interaction_factors
        actual = compute_interaction_factors('test', f3d, None)
        df = f3d.df.copy()
        ret5 = _ts_pct_manual(df, 'close', 5)
        ret20 = _ts_pct_manual(df, 'close', 20)
        hl_range = (df['high'] - df['low']) / df['close']
        expected = {
            'factor_inter_direction_range_5d': (np.sign(ret5) * hl_range).values,
            'factor_inter_direction_range_20d': (np.sign(ret20) * hl_range).values,
        }
        _compare_factor_output(actual, expected)
