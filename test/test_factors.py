"""
因子计算单元测试 — 10 个合并后节点。
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
    gen = generate_synthetic_data(n_times=n_times, n_stocks=n_stocks, noise_ratio=0.3, seed=42)
    frames = [f3d.df for f3d in gen]
    big_df = pd.concat(frames, axis=0).sort_index(level=0)
    from qpipe.frame3d import Frame3D
    return Frame3D(big_df)


class TestFactorModules:
    """逐一测试 10 个合并后的因子模块。"""

    @pytest.fixture(scope='class')
    def f3d(self):
        return _make_small_data()

    def _test_category(self, f3d, category, expected_cols):
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

    def test_trend(self, f3d):
        self._test_category(f3d, 'trend', expected_cols=16)

    def test_momentum(self, f3d):
        self._test_category(f3d, 'momentum', expected_cols=32)

    def test_volatility(self, f3d):
        self._test_category(f3d, 'volatility', expected_cols=33)

    def test_liquidity(self, f3d):
        self._test_category(f3d, 'liquidity', expected_cols=32)

    def test_value(self, f3d):
        self._test_category(f3d, 'value', expected_cols=16)

    def test_quality_merged(self, f3d):
        self._test_category(f3d, 'quality_merged', expected_cols=25)

    def test_quality_pattern(self, f3d):
        self._test_category(f3d, 'quality_pattern', expected_cols=9)

    def test_quality_autocorr(self, f3d):
        self._test_category(f3d, 'quality_autocorr', expected_cols=4)

    def test_counting(self, f3d):
        self._test_category(f3d, 'counting', expected_cols=16)

    def test_cross_section(self, f3d):
        self._test_category(f3d, 'cross_section', expected_cols=10)

    def test_interaction(self, f3d):
        self._test_category(f3d, 'interaction', expected_cols=16)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])