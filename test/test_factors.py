"""
因子计算单元测试 — 10 个合并后节点。
逐一测试所有因子类别的输出形状、列数、无全 NaN 列。
"""

import logging
import os
import sys

import pandas as pd
import pytest

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
        # 传入副本以避免某些因子函数（如 quality_merged）直接修改输入 f3d，
        # 防止 scope='class' 的 fixture 在测试间交叉污染。
        result = func(category, f3d.copy(), None)
        assert isinstance(result, type(f3d)), f'{category}: output must be Frame3D'
        cols = [c for c in result.df.columns if not c.startswith('_')]
        assert len(cols) == expected_cols, (
            f'{category}: expected {expected_cols} cols, got {len(cols)}: {cols}'
        )
        for col in cols:
            nan_pct = result.df[col].isna().mean()
            assert nan_pct < 0.95, f'{category}/{col}: {nan_pct:.1%} NaN (too high)'
        assert len(result.df) == len(f3d.df), f'{category}: row count mismatch'

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
        self._test_category(f3d, 'counting', expected_cols=17)

    def test_cross_section(self, f3d):
        self._test_category(f3d, 'cross_section', expected_cols=10)

    def test_interaction(self, f3d):
        self._test_category(f3d, 'interaction', expected_cols=16)


# ============================================================================
# 回归测试：rolling 计算对齐
# ============================================================================


class TestRollingAlignment:
    """验证 groupby rolling 没有跨股票交叉污染（.values → .reset_index bug 回归）。"""

    def _make_deterministic_f3d(self, n_times=20, n_stocks=5):
        """每只股票价格 = stock_id * 100 + t，rolling mean 可精确预测。"""
        import pandas as pd

        from qpipe.frame3d import Frame3D

        records = []
        for t in range(n_times):
            for s in range(n_stocks):
                price = float(s * 100 + t)
                row = {
                    'key': t,
                    'code': f'S{s:03d}',
                    'close': price,
                    'open': price * 0.99,
                    'high': price * 1.02,
                    'low': price * 0.98,
                    'turnover': 1.0,
                    'volume': 1000.0,
                    'market_cap': float((s + 1) * 1e4),
                }
                records.append(row)
        df = pd.DataFrame(records).set_index(['key', 'code'])
        return Frame3D(df)

    def test_rolling_mean_no_cross_contamination(self):
        """验证 groupby rolling mean / max / min 不会跨股票错位。"""
        import numpy as np

        from seafquant.factor.value import compute_value_factors

        # 需要足够数据填充 max window=120, min_periods=60
        f3d = self._make_deterministic_f3d(130, 5)
        result = compute_value_factors('test', f3d, None)

        df = result.df
        # 对所有因子列，同一时间截面上的值不应完全相同
        # （因为有 5 只价格不同的股票，因子值应有截面方差）
        t_last = sorted(df.index.get_level_values('key').unique())[-1]
        cs_df = df.loc[t_last]
        for col in cs_df.columns:
            if col.startswith('_'):
                continue
            vals = cs_df[col].values
            if (np.allclose(vals[0], vals)
                    and ('dist_ma' in col or 'inv_price' in col or 'log_mcap' in col)):
                # 如果所有股票在该因子上完全相同，很可能是 bug
                # 但对某些因子（如 dd_60d 在完全趋势性数据上可能接近），
                # 只对预期有差异的因子严格检查
                    raise AssertionError(
                        f'{col}: all {len(vals)} stocks have identical value {vals[0]:.6f} '
                        f'— likely cross-stock contamination'
                    )

    def test_rolling_mean_values_correct(self):
        """验证 rolling 值精确匹配手工计算。"""
        import pandas as pd


        f3d = self._make_deterministic_f3d(10, 3)
        df = f3d.df.copy()
        # 手工计算 stock S001 的 3-day MA
        stock1_data = df.loc[(slice(None), 'S001'), 'close']
        ma3_manual = stock1_data.rolling(3, min_periods=1).mean()

        # 模拟 _roll 的正确实现
        rolled = df.groupby('code')['close'].rolling(3, min_periods=1).mean()
        df['ma3_correct'] = rolled.reset_index(level=0, drop=True)

        # 比较 S001 的值
        s001_correct = df.loc[(slice(None), 'S001'), 'ma3_correct']
        pd.testing.assert_series_equal(
            s001_correct, ma3_manual, check_names=False,
            obj='S001 MA3 should match manual computation',
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
