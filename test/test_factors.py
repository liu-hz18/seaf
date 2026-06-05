"""
因子计算单元测试
逐一测试所有因子类别的输出形状、列数、无全 NaN 列。
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
import logging

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from seafquant.data_generator import generate_synthetic_data
from seafquant.factors import FACTOR_REGISTRY


def _make_small_data(n_times=80, n_stocks=20):
    """生成小规模测试数据，收集为单个大 Frame3D。"""
    gen = generate_synthetic_data(n_times=n_times, n_stocks=n_stocks, noise_ratio=0.3, seed=42)
    frames = []
    for f3d in gen:
        frames.append(f3d.df)
    big_df = pd.concat(frames, axis=0)
    big_df = big_df.sort_index(level=0)
    from qpipe.frame3d import Frame3D
    return Frame3D(big_df)


class TestFactorModules:
    """逐一测试所有因子模块。"""

    @pytest.fixture(scope='class')
    def f3d(self):
        return _make_small_data()

    def _test_category(self, f3d, category, expected_cols=16):
        func = FACTOR_REGISTRY[category]
        result = func(category, f3d, None)

        assert isinstance(result, type(f3d)), f"{category}: output must be Frame3D"

        cols = [c for c in result.df.columns if not c.startswith('_')]
        assert len(cols) == expected_cols, \
            f"{category}: expected {expected_cols} cols, got {len(cols)}: {cols}"

        for col in cols:
            nan_pct = result.df[col].isna().mean()
            assert nan_pct < 0.95, f"{category}/{col}: {nan_pct:.1%} NaN (too high)"

        assert len(result.df) == len(f3d.df), f"{category}: row count mismatch"

    # ---- original 8 ----
    def test_momentum(self, f3d):
        self._test_category(f3d, 'momentum')

    def test_reversal(self, f3d):
        self._test_category(f3d, 'reversal')

    def test_volatility(self, f3d):
        self._test_category(f3d, 'volatility')

    def test_liquidity(self, f3d):
        self._test_category(f3d, 'liquidity')

    def test_value(self, f3d):
        self._test_category(f3d, 'value')

    def test_trend(self, f3d):
        self._test_category(f3d, 'trend', expected_cols=8)

    def test_trend_macd(self, f3d):
        self._test_category(f3d, 'trend_macd', expected_cols=8)

    def test_size(self, f3d):
        self._test_category(f3d, 'size')

    # ---- quality split ----
    def test_quality_basic(self, f3d):
        self._test_category(f3d, 'quality_basic')

    def test_quality_advanced(self, f3d):
        self._test_category(f3d, 'quality_advanced', expected_cols=4)

    def test_quality_autocorr(self, f3d):
        self._test_category(f3d, 'quality_autocorr', expected_cols=4)

    def test_quality_pattern(self, f3d):
        self._test_category(f3d, 'quality_pattern', expected_cols=5)

    def test_quality_sign(self, f3d):
        self._test_category(f3d, 'quality_sign', expected_cols=3)

    # ---- new 4 ----
    def test_counting(self, f3d):
        self._test_category(f3d, 'counting', expected_cols=9)

    def test_counting_streak(self, f3d):
        self._test_category(f3d, 'counting_streak', expected_cols=4)

    def test_counting_nh(self, f3d):
        self._test_category(f3d, 'counting_nh', expected_cols=3)

    def test_intraday(self, f3d):
        self._test_category(f3d, 'intraday')

    def test_interaction(self, f3d):
        self._test_category(f3d, 'interaction')

    def test_cross_section(self, f3d):
        self._test_category(f3d, 'cross_section', expected_cols=10)

    def test_cross_section_neut(self, f3d):
        self._test_category(f3d, 'cross_section_neut', expected_cols=6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
