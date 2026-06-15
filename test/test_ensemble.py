"""
Ensemble (bagging) 节点单元测试。

验证：
- 多模型信号 → 等权融合
- pred_signal_* 列前缀提取
- 边界情况（空帧、无预测列、NaN、单股票、单日）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qpipe.frame3d import Frame3D
from seafquant.ensemble import ensemble_fn, ensemble_epilogue


# =============================================================================
# 助函数
# =============================================================================

def _make_f3d(columns: list[str], data: np.ndarray | None = None,
              n_times: int = 1, n_stocks: int = 10) -> Frame3D:
    """构造测试 Frame3D。"""
    if data is None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (n_times * n_stocks, len(columns)))
    keys = np.repeat(np.arange(n_times), n_stocks)
    codes = np.array([f'S{i:03d}' for i in range(n_stocks)] * n_times)
    idx = pd.MultiIndex.from_arrays([keys, codes], names=['key', 'code'])
    return Frame3D(pd.DataFrame(data, columns=columns, index=idx))


# =============================================================================
# 核心融合测试
# =============================================================================

class TestEnsembleBasic:
    """等权融合基本正确性。"""

    def test_single_model_passthrough(self):
        """单模型：pred_signal_lgbm 列 → pred_signal。"""
        f3d = _make_f3d(['pred_signal_lgbm', 'close'], n_stocks=5)
        result = ensemble_fn('test', f3d)
        assert 'pred_signal' in result.df.columns
        np.testing.assert_array_almost_equal(
            result.df['pred_signal'].values,
            f3d.df['pred_signal_lgbm'].values,
        )

    def test_two_model_equal_weight(self):
        """两模型信号等权平均。"""
        rng = np.random.default_rng(42)
        s1 = rng.normal(0, 1, 10)
        s2 = rng.normal(0, 1, 10)
        f3d = _make_f3d(['pred_signal_lgbm', 'pred_signal_mlp'], n_stocks=10)
        f3d.df['pred_signal_lgbm'] = s1
        f3d.df['pred_signal_mlp'] = s2
        result = ensemble_fn('test', f3d)
        expected = (s1 + s2) / 2
        np.testing.assert_array_almost_equal(
            result.df['pred_signal'].values, expected
        )

    def test_three_model_equal_weight(self):
        """三模型等权平均。"""
        rng = np.random.default_rng(99)
        f3d = _make_f3d(
            ['pred_signal_lgbm', 'pred_signal_ridge', 'pred_signal_mlp'],
            n_stocks=8,
        )
        f3d.df['pred_signal_lgbm'] = rng.normal(0, 1, 8)
        f3d.df['pred_signal_ridge'] = rng.normal(0, 1, 8)
        f3d.df['pred_signal_mlp'] = rng.normal(0, 1, 8)
        result = ensemble_fn('test', f3d)
        expected = (
            f3d.df['pred_signal_lgbm'].values +
            f3d.df['pred_signal_ridge'].values +
            f3d.df['pred_signal_mlp'].values
        ) / 3
        np.testing.assert_array_almost_equal(
            result.df['pred_signal'].values, expected
        )

    def test_only_pred_signal_cols_used(self):
        """非 pred_signal_* 列不影响结果。"""
        rng = np.random.default_rng(7)
        f3d = _make_f3d(
            ['pred_signal_lgbm', 'close', 'stock_name', 'pred_signal_mlp'],
            n_stocks=6,
        )
        f3d.df['pred_signal_lgbm'] = rng.normal(0, 1, 6)
        f3d.df['pred_signal_mlp'] = rng.normal(0, 1, 6)
        result = ensemble_fn('test', f3d)
        expected = (
            f3d.df['pred_signal_lgbm'].values +
            f3d.df['pred_signal_mlp'].values
        ) / 2
        np.testing.assert_array_almost_equal(
            result.df['pred_signal'].values, expected
        )


# =============================================================================
# 边界情况
# =============================================================================

class TestEnsembleEdgeCases:
    """边界和异常处理。"""

    def test_empty_frame_raises(self):
        """空 Frame3D 抛出 ValueError。"""
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=['key', 'code'])
        with pytest.raises(ValueError, match='non-empty'):
            ensemble_fn('test', Frame3D(pd.DataFrame(index=empty_idx)))

    def test_no_pred_signal_cols_raises(self):
        """没有 pred_signal 或 pred_signal_* 列。"""
        f3d = _make_f3d(['close', 'volume'], n_stocks=5)
        with pytest.raises(ValueError, match='no pred_signal'):
            ensemble_fn('test', f3d)

    def test_fallback_to_plain_pred_signal(self):
        """单模型：只有 pred_signal 列（不闪退）。"""
        rng = np.random.default_rng(1)
        vals = rng.normal(0, 1, 10)
        f3d = _make_f3d(['pred_signal', 'close'], n_stocks=10)
        f3d.df['pred_signal'] = vals
        result = ensemble_fn('test', f3d)
        np.testing.assert_array_almost_equal(
            result.df['pred_signal'].values, vals
        )

    def test_nan_in_signals_ignored(self):
        """包含 NaN 的信号列：nanmean 忽略 NaN。"""
        rng = np.random.default_rng(13)
        s1 = rng.normal(0, 1, 10)
        s2 = rng.normal(0, 1, 10)
        s1[3] = np.nan
        s2[7] = np.nan
        f3d = _make_f3d(['pred_signal_lgbm', 'pred_signal_mlp'], n_stocks=10)
        f3d.df['pred_signal_lgbm'] = s1
        f3d.df['pred_signal_mlp'] = s2
        result = ensemble_fn('test', f3d)
        expected = np.nanmean(np.column_stack([s1, s2]), axis=1)
        np.testing.assert_array_almost_equal(
            result.df['pred_signal'].values, expected
        )

    def test_single_stock(self):
        """单股票不崩溃。"""
        f3d = _make_f3d(['pred_signal_lgbm'], n_stocks=1)
        result = ensemble_fn('test', f3d)
        assert len(result.df) == 1
        assert 'pred_signal' in result.df.columns

    def test_multi_day_handles_correctly(self):
        """多日数据：逐天独立等权融合。"""
        rng = np.random.default_rng(5)
        n_times, n_stocks = 3, 4
        f3d = _make_f3d(
            ['pred_signal_lgbm', 'pred_signal_mlp'],
            n_times=n_times, n_stocks=n_stocks,
        )
        f3d.df['pred_signal_lgbm'] = rng.normal(0, 1, n_times * n_stocks)
        f3d.df['pred_signal_mlp'] = rng.normal(0, 1, n_times * n_stocks)
        result = ensemble_fn('test', f3d)
        assert len(result.df) == n_times * n_stocks
        expected = (
            f3d.df['pred_signal_lgbm'].values +
            f3d.df['pred_signal_mlp'].values
        ) / 2
        np.testing.assert_array_almost_equal(
            result.df['pred_signal'].values, expected
        )


# =============================================================================
# Epilogue 测试
# =============================================================================

class TestEnsembleEpilogue:
    def test_epilogue_no_crash(self, caplog):
        """epilogue 不崩溃。"""
        import logging
        caplog.set_level(logging.INFO)
        ctx: dict = {
            'mlflow_name': 'test', 'precision': 2,
            'start_date': '2020-01-01', 'fwd': 20, 'key': 'value',
        }
        ensemble_epilogue('test', ctx)
        assert 'key' in ctx  # 未被 pop
        assert 'mlflow_name' not in ctx
