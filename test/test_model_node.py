"""
model_node / model_wrappers 单元测试
覆盖 _cs_zscore、_empty_result、_prepare_training_data、
_run_cv、model_train_predict 及 Wrapper 类（LGBM / Ridge / MLP）的
完整训练→推理流程。
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qpipe.frame3d import Frame3D
from seafquant.model_node import (
    _cs_zscore,
    _empty_result,
    _log_feature_importance,
    _prepare_training_data,
    _run_cv,
    model_train_predict,
)
from seafquant.model_wrappers import (
    WRAPPER_REGISTRY,
    BaseWrapper,
    LGBMWrapper,
    MLPWrapper,
    RidgeWrapper,
)

# ============================================================================
# 测试辅助：构造 Frame3D
# ============================================================================


def _make_f3d(
    n_times: int = 30,
    n_stocks: int = 10,
    n_features: int = 5,
    with_close: bool = True,
    seed: int = 42,
) -> Frame3D:
    """构造多截面 Frame3D（time x stock x factor + close）。"""
    rng = np.random.default_rng(seed)
    records = []
    for t in range(n_times):
        for s in range(n_stocks):
            row = {'key': t, 'name': f'S{s:03d}'}
            for f in range(n_features):
                row[f'factor_{f}'] = float(rng.normal(0, 1))
            if with_close:
                row['close'] = float(100 + rng.normal(0, 5))
            records.append(row)
    df = pd.DataFrame(records).set_index(['key', 'name'])
    return Frame3D(df)


def _make_f3d_with_signal(
    n_times: int = 30,
    n_stocks: int = 10,
    n_features: int = 5,
    seed: int = 42,
):
    """构造带有真实可预测信号的 F3D：因子 0 正比于 fwd_ret。

    Returns: (Frame3D, fwd)
    """
    rng = np.random.default_rng(seed)
    records = []
    base_close = 100 + np.cumsum(rng.normal(0, 0.5, size=n_times))
    for t in range(n_times):
        for s in range(n_stocks):
            row = {'key': t, 'name': f'S{s:03d}'}
            for f in range(n_features):
                row[f'factor_{f}'] = float(rng.normal(0, 1))
            row['close'] = float(base_close[t] + rng.normal(0, 1))
            records.append(row)
    df = pd.DataFrame(records).set_index(['key', 'name'])
    # 因子 0 = fwd_ret + noise，使模型能学到信号
    close_matrix = np.zeros((n_times, n_stocks))
    for t in range(n_times):
        close_matrix[t] = df.loc[t, 'close'].values
    fwd = 5
    for t in range(n_times - fwd):
        fwd_ret = close_matrix[t + fwd] / close_matrix[t] - 1
        fwd_ret_cs = (fwd_ret - fwd_ret.mean()) / (fwd_ret.std() + 1e-10)
        for s in range(n_stocks):
            idx = (t, f'S{s:03d}')
            df.loc[idx, 'factor_0'] = float(fwd_ret_cs[s] + rng.normal(0, 0.3))
    return Frame3D(df), fwd


# ============================================================================
# _cs_zscore 测试
# ============================================================================


class TestCsZscore:
    def test_normal_distribution(self):
        """正态分布标准化后均值≈0，标准差≈1。"""
        vals = np.random.randn(1000) * 5 + 10
        z = _cs_zscore(vals)
        assert abs(z.mean()) < 0.1
        assert abs(z.std() - 1.0) < 0.1

    def test_constant_values(self):
        """常数值应返回零向量。"""
        z = _cs_zscore(np.array([3.0, 3.0, 3.0]))
        np.testing.assert_array_equal(z, np.zeros(3))

    def test_single_value(self):
        """单元素应返回 0.0。"""
        z = _cs_zscore(np.array([42.0]))
        assert z[0] == 0.0

    def test_with_nan(self):
        """NaN 应被忽略，非 NaN 部分正常标准化。"""
        vals = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        z = _cs_zscore(vals)
        assert np.isnan(z[3])
        valid = z[~np.isnan(z)]
        assert abs(valid.mean()) < 1e-1
        assert abs(valid.std() - 1.0) < 1e-1

    def test_all_nan(self):
        """全 NaN 向量应返回零向量。"""
        z = _cs_zscore(np.array([np.nan, np.nan]))
        assert z[0] == 0.0
        assert z[1] == 0.0

    def test_zero_std(self):
        """两元素相同 -> std=0 -> 零向量。"""
        z = _cs_zscore(np.array([5.0, 5.0]))
        np.testing.assert_array_equal(z, np.zeros(2))


# ============================================================================
# _empty_result 测试
# ============================================================================


class TestEmptyResult:
    def test_shape_and_values(self):
        """返回 Frame3D 包含 n_stocks 个 pred_signal=0。"""
        idx = pd.MultiIndex.from_arrays(
            [[0, 0, 0], ['S001', 'S002', 'S003']],
            names=['key', 'name'],
        )
        result = _empty_result(3, idx)
        assert isinstance(result, Frame3D)
        assert len(result.df) == 3
        assert 'pred_signal' in result.df.columns
        np.testing.assert_array_equal(result.df['pred_signal'].values, [0.0, 0.0, 0.0])

    def test_index_preserved(self):
        """返回结果保留传入的 index。"""
        mi = pd.MultiIndex.from_arrays(
            [[10, 10, 10], ['A', 'B', 'C']], names=['key', 'name']
        )
        result = _empty_result(3, mi)
        assert result.df.index.equals(mi)
        assert 'pred_signal' in result.df.columns


# ============================================================================
# _prepare_training_data 测试
# ============================================================================


class TestPrepareTrainingData:
    def test_dimensions(self):
        """验证 X, y 的正确维度。"""
        f3d = _make_f3d(30, 10, 5, seed=42)
        df = f3d.df.copy()
        feature_cols = [c for c in df.columns if c.startswith('factor_')]
        times = sorted(df.index.get_level_values('key').unique())
        fwd = 5

        X, y, cs_stats = _prepare_training_data('test', df, feature_cols, times, fwd)

        n_train_times = len(times) - fwd - 1  # 30 - 5 - 1 = 24
        n_stocks = df.index.get_level_values('name').nunique()
        assert X.shape == (n_train_times * n_stocks, len(feature_cols))
        assert len(y) == n_train_times * n_stocks
        assert len(cs_stats) == n_train_times

    def test_label_is_cs_zscore(self):
        """label 的每个截面均值应接近 0。"""
        f3d = _make_f3d(30, 20, 5, seed=42)
        df = f3d.df.copy()
        feature_cols = [c for c in df.columns if c.startswith('factor_')]
        times = sorted(df.index.get_level_values('key').unique())
        X, y, cs_stats = _prepare_training_data('test', df, feature_cols, times, fwd=5)

        start = 0
        for stat in cs_stats:
            n = stat['n']
            section_y = y[start : start + n]
            assert abs(section_y.mean()) < 1e-10, f't={stat["t"]} mean={section_y.mean()}'
            assert abs(section_y.std() - 1.0) < 0.15, f't={stat["t"]} std={section_y.std()}'
            start += n

    def test_no_time_travel(self):
        """label 计算使用 t+fwd 的价格而非当前 t 的价格。"""
        fwd = 5
        f3d = _make_f3d(20, 10, 3, seed=42)
        df = f3d.df.copy()
        feature_cols = [c for c in df.columns if c.startswith('factor_')]
        times = sorted(df.index.get_level_values('key').unique())
        X, y, cs_stats = _prepare_training_data('test', df, feature_cols, times, fwd=fwd)

        t0 = times[0]
        t1_close = df.loc[times[1], 'close'].values

        # 第一个截面使用 t+fwd 与 t+1 的 close 做 label，而非 t
        assert cs_stats[0]['t'] == t0
        first_y = y[:len(t1_close)]
        assert not np.allclose(first_y, 0.0), 'label should not be constant zero'

    def test_n_times_insufficient(self):
        """当 n_times 不足以构造训练数据时，model_train_predict 返回空结果。"""
        f3d = _make_f3d(3, 5, 3, seed=42)
        ctx = {'model_type': 'ridge', 'fwd': 20}
        result = model_train_predict('test', f3d, ctx)
        assert 'pred_signal' in result.df.columns
        np.testing.assert_array_equal(result.df['pred_signal'].values, [0.0] * 5)


# ============================================================================
# _run_cv 测试
# ============================================================================


class MockWrapper(BaseWrapper):
    """用于测试 CV 调度的模拟 wrapper。无显式 __init__，兼容 BaseWrapper。"""

    model_type = 'mock'

    def __init__(self):
        pass

    def fit(self, X, y):
        self._is_fitted = True
        return {}

    def predict(self, X):
        return np.random.normal(0, 1, size=len(X))

    def cv_fit_predict(self, X_tr, y_tr, X_val):
        # 返回与 y_val 有弱相关的预测（产生非退化 IC）
        y_val_proxy = np.random.randn(len(X_val))
        return y_val_proxy, {}

    def get_feature_importance(self, feature_cols):
        return dict.fromkeys(feature_cols, 0.1)


class TestRunCv:
    def test_returns_scores(self):
        """CV 应返回 IC 分数列表。"""
        wrapper = MockWrapper()
        X = np.random.randn(200, 5)
        y = np.random.randn(200)
        scores, _ = _run_cv('test', wrapper, X, y, n_splits=3)
        assert isinstance(scores, list)

    def test_too_few_samples_skips_folds(self):
        """样本太少时（<10 per fold）应跳过。"""
        wrapper = MockWrapper()
        X = np.random.randn(5, 3)
        y = np.random.randn(5)
        scores, _ = _run_cv('test', wrapper, X, y, n_splits=3)
        assert len(scores) == 0

    def test_mlp_finalize_called(self):
        """MLP wrapper 在 CV 期间被调用 cv_fit_predict。"""
        wrapper = MagicMock()
        wrapper.model_type = 'mlp'
        wrapper.cv_fit_predict.return_value = (np.random.randn(50), {})

        X = np.random.randn(200, 5)
        y = np.random.randn(200)
        _run_cv('test', wrapper, X, y, n_splits=3)
        wrapper.cv_fit_predict.assert_called()


# ============================================================================
# model_train_predict 端到端测试
# ============================================================================


class TestModelTrainPredict:
    def test_ridge_training_and_inference(self):
        """Ridge 完整训练→推理流程：fit, predict, context 更新。"""
        f3d, fwd = _make_f3d_with_signal(50, 20, 5, seed=42)
        context = {
            'model_type': 'ridge',
            'fwd': fwd,
            'model_window': 50,
            'retrain_every': 20,
        }
        result = model_train_predict('test_ridge', f3d, context)
        assert isinstance(result, Frame3D)
        assert 'pred_signal' in result.df.columns
        pred = result.df['pred_signal'].values
        # 截面标准化后均值≈0
        assert abs(pred.mean()) < 1e-10
        assert abs(pred.std() - 1.0) < 0.15
        # context 应更新训练状态
        assert context['is_trained'], 'Ridge should complete training'
        assert context['trained_wrapper'] is not None
        assert context['days_since_train'] == 0

    def test_lgbm_training_and_inference(self):
        """LGBM 完整训练→推理流程。"""
        pytest.importorskip('lightgbm')
        f3d, fwd = _make_f3d_with_signal(50, 20, 5, seed=42)
        context = {
            'model_type': 'lgbm',
            'fwd': fwd,
            'model_window': 50,
            'retrain_every': 20,
        }
        result = model_train_predict('test_lgbm', f3d, context)
        assert isinstance(result, Frame3D)
        assert 'pred_signal' in result.df.columns
        pred = result.df['pred_signal'].values
        assert abs(pred.mean()) < 1e-10
        assert context['is_trained']

    def test_mlp_training_and_inference(self):
        """MLP 完整训练→推理流程（CPU mode）。"""
        pytest.importorskip('torch')
        f3d, fwd = _make_f3d_with_signal(60, 20, 5, seed=42)
        context = {
            'model_type': 'mlp',
            'fwd': fwd,
            'model_window': 60,
            'retrain_every': 60,
            'mlp_hidden': [32, 16],
            'mlp_dropout': 0.2,
            'mlp_lr': 1e-3,
            'mlp_epochs': 20,
        }
        result = model_train_predict('test_mlp', f3d, context)
        assert isinstance(result, Frame3D)
        assert 'pred_signal' in result.df.columns
        pred = result.df['pred_signal'].values
        # MLP 小样本下放宽精度
        assert abs(pred.mean()) < 1e-3, f'mean={pred.mean():.6f}'
        assert context['is_trained']

    def test_insufficient_data(self):
        """数据不足时（n_times < fwd+2）返回空结果 pred_signal=0。"""
        f3d = _make_f3d(3, 5, 3, seed=42)
        ctx = {'model_type': 'ridge', 'fwd': 20}
        result = model_train_predict('test', f3d, ctx)
        assert 'pred_signal' in result.df.columns
        np.testing.assert_array_equal(result.df['pred_signal'].values, [0.0] * 5)

    def test_context_defaults(self):
        """未传入的 context 键应有默认值，训练后会更新状态。"""
        f3d = _make_f3d(50, 5, 3, seed=42)
        context = {}
        model_train_predict('test', f3d, context)
        assert context['model_type'] == 'lgbm'
        assert context['fwd'] == 20
        assert context['retrain_every'] == 20
        assert context['model_window'] == 200
        assert context['is_trained'] is True, 'should auto-train on first call'

    def test_context_none(self):
        """context=None 时应初始化为默认值并正常返回。"""
        f3d = _make_f3d(50, 5, 3, seed=42)
        result = model_train_predict('test', f3d, None)
        assert 'pred_signal' in result.df.columns

    def test_retrain_trigger(self):
        """retrain_every 到期时才触发重训练；未到期时复用 wrapper。"""
        f3d, fwd = _make_f3d_with_signal(80, 10, 5, seed=42)
        context = {
            'model_type': 'ridge',
            'fwd': fwd,
            'retrain_every': 999,
        }
        model_train_predict('test', f3d, context)
        assert context['is_trained']
        trained_wrapper = context['trained_wrapper']

        # 第二次调用不应重训练（retrain_every 未到期）
        model_train_predict('test', f3d, context)
        assert context['trained_wrapper'] is trained_wrapper
        # 第一次训练后 days_since_train 重置为 0，第二次 +1 → 1
        assert context['days_since_train'] == 1

    def test_pred_signal_is_cs_standardized(self):
        """预测信号应经截面标准化：mean≈0, std≈1。"""
        f3d, fwd = _make_f3d_with_signal(60, 30, 5, seed=42)
        context = {'model_type': 'ridge', 'fwd': fwd, 'retrain_every': 60}
        result = model_train_predict('test', f3d, context)
        pred = result.df['pred_signal'].values
        assert abs(pred.mean()) < 1e-10
        assert 0.5 < pred.std() < 1.5

    def test_missing_close_column(self):
        """缺少 close 列时应抛出 ValueError。"""
        f3d = _make_f3d(30, 5, 3, with_close=False, seed=42)
        with pytest.raises(ValueError, match='close'):
            model_train_predict('test', f3d, {'model_type': 'ridge'})


# ============================================================================
# Wrapper 类测试：每种模型的拟合与推理
# ============================================================================


class TestRidgeWrapper:
    def test_fit_predict(self):
        """Ridge: fit → predict 返回正确形状，无 NaN。"""
        wrapper = RidgeWrapper({})
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        metrics = wrapper.fit(X, y)
        assert isinstance(metrics, dict)
        pred = wrapper.predict(X)
        assert pred.shape == (100,)
        assert not np.any(np.isnan(pred))

    def test_feature_importance(self):
        """特征重要性返回字典，值已归一化 Σ=1。"""
        wrapper = RidgeWrapper({})
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        wrapper.fit(X, y)
        fi = wrapper.get_feature_importance(['f0', 'f1', 'f2', 'f3', 'f4'])
        assert len(fi) == 5
        assert abs(sum(fi.values()) - 1.0) < 1e-10

    def test_feature_importance_untrained(self):
        """未训练时返回空字典。"""
        wrapper = RidgeWrapper({})
        fi = wrapper.get_feature_importance(['f0'])
        assert fi == {}

    def test_predict_before_fit(self):
        """未训练时 predict 应抛异常。"""
        wrapper = RidgeWrapper({})
        with pytest.raises(Exception):
            wrapper.predict(np.random.randn(10, 3))


class TestLGBMWrapper:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip('lightgbm')

    def test_fit_predict(self):
        """LGBM: fit → predict 返回正确形状，无 NaN。"""
        wrapper = LGBMWrapper({})
        X = np.random.randn(200, 5)
        y = np.random.randn(200)
        wrapper.fit(X, y)
        pred = wrapper.predict(X)
        assert pred.shape == (200,)
        assert not np.any(np.isnan(pred))

    def test_feature_importance(self):
        """LGBM 特征重要性归一化。"""
        wrapper = LGBMWrapper({})
        X = np.random.randn(200, 5)
        y = np.random.randn(200)
        wrapper.fit(X, y)
        fi = wrapper.get_feature_importance(['f0', 'f1', 'f2', 'f3', 'f4'])
        assert len(fi) <= 5
        assert len(fi) > 0

    def test_feature_importance_untrained(self):
        """未训练时返回空字典。"""
        wrapper = LGBMWrapper({})
        fi = wrapper.get_feature_importance(['f0'])
        assert fi == {}

    def test_predict_before_fit(self):
        """未训练时 predict 应抛异常。"""
        wrapper = LGBMWrapper({})
        with pytest.raises(Exception):
            wrapper.predict(np.random.randn(10, 3))


class TestMLPWrapper:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip('torch')

    def test_fit_predict(self):
        """MLP: fit → predict 返回正确形状。"""
        wrapper = MLPWrapper({'mlp_hidden': [16, 8], 'mlp_dropout': 0.2, 'mlp_epochs': 20})
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        metrics = wrapper.fit(X, y)
        pred = wrapper.predict(X)
        assert pred.shape == (100,)
        assert 'train_loss' in metrics

    def test_cv_fit_predict(self):
        """MLP CV 单折返回预测和 loss 指标。"""
        wrapper = MLPWrapper({'mlp_hidden': [16, 8], 'mlp_epochs': 10})
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        pred, metrics = wrapper.cv_fit_predict(X[:70], y[:70], X[70:])
        assert pred.shape == (30,)
        assert 'val_loss' in metrics  # metrics key 为 val_loss

    def test_finalize_epochs(self):
        """finalize_epochs 基于 CV 最优 epoch 中位数确定全量训练 epochs。"""
        wrapper = MLPWrapper({'mlp_hidden': [16, 8], 'mlp_epochs': 20})
        wrapper._cv_epoch_stats = [
            {'best_epoch': 5, 'best_val_loss': 0.1, 'final_epoch': 10},
            {'best_epoch': 8, 'best_val_loss': 0.2, 'final_epoch': 12},
            {'best_epoch': 6, 'best_val_loss': 0.15, 'final_epoch': 11},
        ]
        epochs = wrapper.finalize_epochs()
        assert epochs == 6  # median of [5, 8, 6]
        assert wrapper._cv_epoch_stats == []
        assert wrapper._model is None  # 重置网络用于全量训练

    def test_finalize_epochs_empty(self):
        """无 CV 统计时 finalize_epochs 返回默认 epochs。"""
        wrapper = MLPWrapper({'mlp_hidden': [16, 8]})
        original = wrapper._epochs
        epochs = wrapper.finalize_epochs()
        assert epochs == original

    def test_feature_importance(self):
        """MLP fit 后特征重要性归一化到 [0,1]。"""
        wrapper = MLPWrapper({'mlp_hidden': [16, 8], 'mlp_epochs': 20})
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        wrapper.fit(X, y)
        fi = wrapper.get_feature_importance(['f0', 'f1', 'f2', 'f3', 'f4'])
        assert len(fi) == 5
        assert abs(sum(fi.values()) - 1.0) < 1e-4  # float32 累积误差

    def test_feature_importance_untrained(self):
        """未训练时返回空字典。"""
        wrapper = MLPWrapper({})
        fi = wrapper.get_feature_importance(['f0'])
        assert fi == {}

    def test_predict_before_fit(self):
        """未训练时 predict 应抛异常（无 model）。"""
        wrapper = MLPWrapper({'mlp_hidden': [8]})
        with pytest.raises(Exception):
            wrapper.predict(np.random.randn(10, 3))


# ============================================================================
# WRAPPER_REGISTRY 测试
# ============================================================================


class TestWrapperRegistry:
    def test_all_types_registered(self):
        """三种模型类型均已注册。"""
        assert 'lgbm' in WRAPPER_REGISTRY
        assert 'ridge' in WRAPPER_REGISTRY
        assert 'mlp' in WRAPPER_REGISTRY

    def test_ridge_instantiable(self):
        """Ridge wrapper 可直接实例化。"""
        w = WRAPPER_REGISTRY['ridge']({})
        assert isinstance(w, RidgeWrapper)

    def test_lgbm_instantiable(self):
        """LGBM wrapper 可直接实例化。"""
        pytest.importorskip('lightgbm')
        w = WRAPPER_REGISTRY['lgbm']({})
        assert isinstance(w, LGBMWrapper)

    def test_mlp_instantiable(self):
        """MLP wrapper 可直接实例化。"""
        pytest.importorskip('torch')
        w = WRAPPER_REGISTRY['mlp']({})
        assert isinstance(w, MLPWrapper)

    def test_unknown_type_raises(self):
        """未注册的模型类型应抛出 KeyError。"""
        with pytest.raises(KeyError):
            _ = WRAPPER_REGISTRY['unknown_type']


# ============================================================================
# _log_feature_importance 测试
# ============================================================================


class TestLogFeatureImportance:
    def test_empty_fi_noop(self, caplog):
        """空特征重要性无日志输出。"""
        import logging
        caplog.set_level(logging.INFO)
        _log_feature_importance('', 'test', {}, 'ridge')
        assert 'Feature importance' not in caplog.text

    def test_logs_top_items(self, caplog):
        """应输出 top-N 特征重要性到日志。"""
        import logging
        caplog.set_level(logging.INFO)
        fi = {f'factor_{i}': 1.0 / (i + 1) for i in range(15)}
        _log_feature_importance('', 'test', fi, 'ridge')
        assert 'Feature importance top-10' in caplog.text

    def test_handles_mlflow_error(self):
        """MLflow 不可用时静默处理。"""
        fi = {'f0': 0.5, 'f1': 0.3, 'f2': 0.2}
        try:
            _log_feature_importance('fake_run_id', 'test', fi, 'ridge')
        except Exception as e:
            pytest.fail(f'_log_feature_importance raised: {e}')
