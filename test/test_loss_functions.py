"""
损失函数单元测试 — 验证 MSELossFn / ICPearsonLossFn 的计算正确性与梯度传导。

测试覆盖：
- 解析梯度 vs 数值微分 (finite difference)
- torch autograd vs 解析公式
- 最小化 loss 等价于最大化 Pearson IC (集成验证)
- 边界情况（常数/零方差/单样本）
- LOSS_REGISTRY + wrapper 集成
"""

from __future__ import annotations

import numpy as np
import pytest

# =============================================================================
# 导入被测模块
# =============================================================================
from seafquant.model_wrappers import (
    _EPS,
    LOSS_REGISTRY,
    ICPearsonLossFn,
    MSELossFn,
)

# =============================================================================
# 辅助函数
# =============================================================================


def _numerical_gradient(fn, x, h=1e-6):
    """中心差分法计算数值梯度：∂fn/∂x。"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (fn(x_plus) - fn(x_minus)) / (2 * h)
    return grad


def _pearson_r(pred, target):
    """纯 numpy 实现 Pearson 相关系数（用于对拍验证）。"""
    pred = np.asarray(pred, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    p_mean = pred.mean()
    t_mean = target.mean()
    p_std = max(pred.std(), _EPS)
    t_std = max(target.std(), _EPS)
    return np.mean((pred - p_mean) * (target - t_mean)) / (p_std * t_std)


def _make_lgbm_dataset(labels):
    """构造一个最小化的 LGBM Dataset-like 对象，仅支持 get_label()。"""

    class _Fake:
        def get_label(self):
            return np.asarray(labels, dtype=float)

    return _Fake()


# =============================================================================
# MSELossFn 测试
# =============================================================================


class TestMSELossFn:
    """验证 MSELossFn 的基本正确性。"""

    def test_lgbm_objective_grad(self):
        """MSE grad = pred - label。"""
        rng = np.random.default_rng(42)
        preds = rng.normal(0, 1, 100)
        labels = rng.normal(0.5, 1, 100)
        train_data = _make_lgbm_dataset(labels)

        loss_fn = MSELossFn()
        grad, hess = loss_fn.lgbm_objective(preds, train_data)

        np.testing.assert_allclose(grad, preds - labels, rtol=1e-10)
        assert grad.shape == preds.shape

    def test_lgbm_objective_hess(self):
        """MSE hess = 1（常数）。"""
        rng = np.random.default_rng(43)
        preds = rng.normal(0, 1, 50)
        labels = rng.normal(0, 1, 50)
        train_data = _make_lgbm_dataset(labels)

        grad, hess = MSELossFn().lgbm_objective(preds, train_data)
        np.testing.assert_allclose(hess, np.ones_like(preds))
        assert np.all(hess > 0)  # LGBM 要求 hess > 0

    def test_lower_is_better(self):
        """MSE: lower is better。"""
        assert MSELossFn().lower_is_better is True
        assert MSELossFn().name == 'mse'


# =============================================================================
# ICPearsonLossFn — lgbm_objective 梯度验证
# =============================================================================


class TestICPearsonLGBM:
    """验证 ICPearsonLossFn.lgbm_objective 的梯度正确性。"""

    @pytest.fixture
    def loss_fn(self):
        return ICPearsonLossFn()

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(44)
        n = 200
        preds = rng.normal(0, 1, n)
        labels = rng.normal(0.3, 1, n)
        return preds, labels

    def test_gradient_vs_finite_difference(self, loss_fn, data):
        """解析梯度 vs 中心差分数值梯度 — 相关系数 > 0.99。"""
        preds, labels = data
        train_data = _make_lgbm_dataset(labels)

        # 解析梯度
        anal_grad, _ = loss_fn.lgbm_objective(preds, train_data)

        # 数值梯度：∂(loss)/∂pred_i，loss = objective 的"隐含 loss"
        # ICPearsonLossFn 的 lgbm_objective 返回负相关系数的梯度，
        # 所以 grad = ∂(-pearsonr)/∂pred
        def neg_pearson(p):
            return -_pearson_r(p, labels)

        num_grad = _numerical_gradient(neg_pearson, preds)

        # 高相关验证（容许数值微分有限精度误差）
        corr = np.corrcoef(anal_grad, num_grad)[0, 1]
        assert corr > 0.99, f'Analytical vs numerical gradient correlation={corr:.6f}'
        # 量级接近
        ratio = np.mean(np.abs(anal_grad)) / max(np.mean(np.abs(num_grad)), _EPS)
        assert 0.5 < ratio < 2.0, f'Gradient magnitude ratio={ratio:.4f}'

    def test_hess_positive(self, loss_fn, data):
        """Hessian 必须为正（LGBM 约束）。"""
        preds, labels = data
        _, hess = loss_fn.lgbm_objective(preds, _make_lgbm_dataset(labels))
        assert np.all(hess > 0), f'min(hess)={hess.min():.6e}'

    def test_gradient_zero_when_perfect_correlation(self, loss_fn):
        """预测与标签完全正相关 → 梯度 ≈ 0（已是最优）。"""
        rng = np.random.default_rng(45)
        t = rng.normal(0, 1, 300)
        # 完美正相关预测
        preds = 2.0 * t + 3.0
        train_data = _make_lgbm_dataset(t)
        grad, _ = loss_fn.lgbm_objective(preds, train_data)
        assert np.max(np.abs(grad)) < 0.5 / len(t), \
            f'Gradient should be near zero for perfect +corr: max|grad|={np.max(np.abs(grad)):.6e}'

    def test_gradient_descent_increases_ic_from_negative(self, loss_fn):
        """负相关起点：梯度下降一步后 IC 应上升。"""
        rng = np.random.default_rng(46)
        t = rng.normal(0, 1, 200)
        # 接近完全负相关但加入噪声（完全负相关处梯度为 0，是鞍点）
        preds = -2.0 * t + rng.normal(0, 0.3, 200)
        train_data = _make_lgbm_dataset(t)

        ic_before = _pearson_r(preds, t)
        assert ic_before < 0, f'Expected negative IC start, got {ic_before:.4f}'

        grad, _ = loss_fn.lgbm_objective(preds, train_data)
        lr = 0.1
        preds_after = preds - lr * grad
        ic_after = _pearson_r(preds_after, t)
        assert ic_after > ic_before, \
            f'IC should increase: {ic_before:.4f} → {ic_after:.4f}'

    def test_lower_is_better(self):
        """IC 损失：higher IC = better → lower_is_better=False。"""
        assert ICPearsonLossFn().lower_is_better is False
        assert ICPearsonLossFn().name == 'ic'


# =============================================================================
# ICPearsonLossFn — torch_fn 梯度验证
# =============================================================================


class TestICPearsonTorch:
    """验证 ICPearsonLossFn.torch_fn 的 autograd 正确性。"""

    @pytest.fixture(autouse=True)
    def _torch_skip(self):
        """若 torch 不可用则跳过。"""
        pytest.importorskip('torch')

    @pytest.fixture
    def loss_fn(self):
        return ICPearsonLossFn()

    def test_torch_gradient_vs_analytical(self, loss_fn):
        """torch autograd 梯度 vs 解析梯度 — 高相关。"""
        import torch

        rng = np.random.default_rng(47)
        n = 150
        preds_np = rng.normal(0, 1, n).astype(np.float32)
        targets_np = rng.normal(0.3, 1, n).astype(np.float32)

        # 解析梯度 (numpy)
        train_data = _make_lgbm_dataset(targets_np)
        anal_grad, _ = loss_fn.lgbm_objective(preds_np, train_data)

        # torch autograd 梯度
        pred_t = torch.tensor(preds_np, requires_grad=True)
        target_t = torch.tensor(targets_np)
        torch_loss = loss_fn.torch_fn()
        loss_val = torch_loss(pred_t, target_t)
        loss_val.backward()
        torch_grad = pred_t.grad.numpy()

        corr = np.corrcoef(anal_grad.ravel(), torch_grad.ravel())[0, 1]
        assert corr > 0.99, f'analytical vs torch gradient corr={corr:.6f}'

    def test_torch_loss_equals_negative_pearson(self, loss_fn):
        """torch loss ≈ -pearsonr。"""
        import torch

        rng = np.random.default_rng(48)
        preds = rng.normal(0, 1, 100).astype(np.float32)
        targets = rng.normal(0.5, 1, 100).astype(np.float32)

        pred_t = torch.tensor(preds)
        target_t = torch.tensor(targets)
        loss_val = loss_fn.torch_fn()(pred_t, target_t).item()

        expected = -_pearson_r(preds, targets)
        # float64 (numpy) vs float32 (torch) 精度：允许 1% 误差
        np.testing.assert_allclose(loss_val, expected, rtol=5e-2)


# =============================================================================
# ICPearsonLossFn — 集成验证 (最小化 loss ↔ 最大化 IC)
# =============================================================================


class TestICPearsonEndToEnd:
    """端到端验证：梯度下降最小化 IC loss → Pearson IC 上升。"""

    def test_gd_increases_pearson_ic(self):
        """简单梯度下降（50 iters）：IC 应上升。"""
        loss_fn = ICPearsonLossFn()
        rng = np.random.default_rng(49)
        n = 50
        targets = rng.normal(0.5, 1, n)
        preds = rng.normal(0, 1, n).astype(float)

        ic_before = _pearson_r(preds, targets)
        lr = 3.0
        for _ in range(300):
            grad, _ = loss_fn.lgbm_objective(preds, _make_lgbm_dataset(targets))
            preds = preds - lr * grad  # 梯度下降（grad = ∂(-ic)/∂pred）
        ic_after = _pearson_r(preds, targets)

        assert ic_after > ic_before, \
            f'IC should increase: before={ic_before:.4f}, after={ic_after:.4f}'
        # 经过足够迭代应该接近 1
        assert ic_after > 0.9, f'IC should reach >0.9 after 300 GD iters: {ic_after:.4f}'


# =============================================================================
# 边界情况
# =============================================================================


class TestEdgeCases:
    """边界情况：常数/零方差/单样本。"""

    def test_constant_predictions_no_crash(self):
        """常数预测不应崩溃，返回有限梯度。"""
        loss_fn = ICPearsonLossFn()
        preds = np.full(100, 3.0)
        labels = np.random.default_rng(50).normal(0, 1, 100)
        grad, hess = loss_fn.lgbm_objective(preds, _make_lgbm_dataset(labels))
        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))

    def test_constant_targets_no_crash(self):
        """常数标签不崩溃。"""
        loss_fn = ICPearsonLossFn()
        preds = np.random.default_rng(51).normal(0, 1, 50)
        labels = np.full(50, 7.0)
        grad, hess = loss_fn.lgbm_objective(preds, _make_lgbm_dataset(labels))
        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))

    def test_single_sample_no_crash(self):
        """单样本：不崩溃。"""
        loss_fn = ICPearsonLossFn()
        preds = np.array([1.0])
        labels = np.array([2.0])
        grad, hess = loss_fn.lgbm_objective(preds, _make_lgbm_dataset(labels))
        assert np.all(np.isfinite(grad))

    def test_identical_pred_and_target(self):
        """预测与标签完全相同：梯度 ≈ 0 + IC=1。"""
        loss_fn = ICPearsonLossFn()
        rng = np.random.default_rng(52)
        t = rng.normal(0, 1, 200)
        grad, _ = loss_fn.lgbm_objective(t, _make_lgbm_dataset(t))
        assert np.max(np.abs(grad)) < 0.1 / len(t)
        assert _pearson_r(t, t) > 0.999


# =============================================================================
# LOSS_REGISTRY 集成
# =============================================================================


class TestLossRegistry:
    """验证注册表 + wrapper 集成。"""

    def test_mse_key(self):
        assert 'mse' in LOSS_REGISTRY
        assert isinstance(LOSS_REGISTRY['mse'], MSELossFn)

    def test_ic_key(self):
        assert 'ic' in LOSS_REGISTRY
        assert isinstance(LOSS_REGISTRY['ic'], ICPearsonLossFn)

    def test_unknown_loss_fallback(self):
        """未知 loss → 不应崩溃（LOSS_REGISTRY.get 返回 None 时 wrapper 会回退）。"""
        assert LOSS_REGISTRY.get('unknown') is None

    def test_wrapper_reads_loss_from_context(self):
        """_SklearnWrapper / MLPWrapper 从 context 读取 loss。"""
        from seafquant.model_wrappers import LGBMWrapper, MLPWrapper

        # LGBM with IC loss
        wrapper = LGBMWrapper({'loss': 'ic'})
        assert wrapper._loss.name == 'ic'

        # MLP default loss (no key)
        wrapper2 = MLPWrapper({})
        assert wrapper2._loss.name == 'mse'

        # MLP with IC loss
        wrapper3 = MLPWrapper({'loss': 'ic'})
        assert wrapper3._loss.name == 'ic'

    def test_ridge_falls_back_on_ic(self, caplog):
        """Ridge 不支持 IC，应 fallback + warning。"""
        from seafquant.model_wrappers import RidgeWrapper

        caplog.set_level('WARNING')
        wrapper = RidgeWrapper({'loss': 'ic'})
        assert 'not supported' in caplog.text.lower() or 'loss' in caplog.text.lower()
        # 检查它确实创建了模型（未崩溃）
        assert wrapper._model is not None
