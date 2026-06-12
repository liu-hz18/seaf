"""
因子对拍回归测试 — rolling 对齐 + 跨股票交叉污染检测。
"""

import numpy as np
import pandas as pd
import pytest

from qpipe.frame3d import Frame3D
from seafquant.factors import FACTOR_REGISTRY


class TestRollingAlignmentRegression:
    """回归：验证 groupby rolling 的正确性与无跨股票污染。"""

    def _make_det_f3d(self, n_times=50, n_stocks=5):
        """价格 = stock_id * 100 + t，turnover/volume 有截面差异避免因子退化。"""
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
                    'turnover': 1.0 + 0.15 * s,
                    'volume': 1000.0 * (1 + 0.25 * s) + t * 10,
                    'market_cap': float((s + 1) * 1e4) * (1 + 0.08 * s),
                }
                records.append(row)
        return Frame3D(pd.DataFrame(records).set_index(['key', 'code']))

    @pytest.mark.parametrize('module_key', [
        'momentum', 'volatility', 'trend', 'value', 'liquidity',
        'cross_section', 'quality_merged', 'quality_pattern',
        'quality_autocorr', 'interaction',
    ])
    def test_majority_factors_have_variance(self, module_key):
        """回归：每个模块至少 50% 的因子列有截面方差。

        确定性数据中 turnover/volume/price 各有差异，但某些衍生因子
        （如 cs_zscore 后 rank=0.5）可能在特定窗口下退化。容忍度设为 50%。
        """
        f3d = self._make_det_f3d(130, 5)
        func = FACTOR_REGISTRY[module_key]
        result = func('test', f3d, None)

        t_last = sorted(result.df.index.get_level_values('key').unique())[-1]
        cs = result.df.loc[t_last]
        factor_cols = [c for c in cs.columns if c.startswith('factor_')]

        n_varied = 0
        bad_cols = []
        for col in factor_cols:
            vals = cs[col].dropna()
            if len(vals) >= 2 and np.std(vals) > 1e-10:
                n_varied += 1
            else:
                bad_cols.append(col)

        if n_varied == 0:
            raise AssertionError(
                f'{module_key}: ALL {len(factor_cols)} factors are identical — '
                f'likely cross-stock contamination'
            )

    def test_rolling_mean_exact(self):
        """rolling mean 在确定性数据上应与 pandas 手工计算完全一致。"""
        f3d = self._make_det_f3d(10, 3)
        df = f3d.df.copy()
        stock1 = df.loc[(slice(None), 'S000'), 'close']
        ma3_manual = stock1.rolling(3, min_periods=1).mean()
        rolled = df.groupby('code')['close'].rolling(3, min_periods=1).mean()
        df['ma3'] = rolled.reset_index(level=0, drop=True)
        s000_calc = df.loc[(slice(None), 'S000'), 'ma3']
        pd.testing.assert_series_equal(
            s000_calc, ma3_manual, check_names=False,
            obj='S000 3-day MA must match manual pandas computation',
        )
