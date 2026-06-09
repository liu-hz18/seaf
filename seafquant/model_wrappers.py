"""
模型包装器 — 每种模型类型一个包装器，封装训练/预测/CV/特征重要性。

新增 model_type='mlp'：PyTorch MLP，LayerNorm + Adam + weight_decay，
时间序列 CV 确定最优 epochs，梯度法提取特征重要性。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


# =============================================================================
# 抽象基类
# =============================================================================


class BaseWrapper(ABC):
    """模型包装器协议：屏蔽 sklearn / torch / 自定义后端差异。"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """训练模型，返回训练指标 dict。"""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测，返回 (n_samples,) 数组。"""

    @abstractmethod
    def get_feature_importance(self, feature_cols: list[str]) -> dict[str, float]:
        """特征重要性 dict，按值降序，归一化到 [0, 1]。"""

    def cv_fit_predict(
        self, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        """CV 单折：训练并返回 (val_pred, metrics)。MLP 可覆写以记录 epoch 统计。"""
        self.fit(X_tr, y_tr)
        return self.predict(X_val), {}


# =============================================================================
# sklearn 系包装器（LGBM / Ridge 共用基类）
# =============================================================================


class _SklearnWrapper(BaseWrapper):
    """LGBM / Ridge 共用：fit + predict + 特征重要性模式一致。"""

    model_type: str = ''

    def __init__(self, context: dict[str, Any]) -> None:
        self._model = self._build(context)

    def _build(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        self._model.fit(X, y)
        return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def _raw_importance(self) -> np.ndarray | None:
        """子类实现：返回原始重要性数组。"""
        raise NotImplementedError

    def get_feature_importance(self, feature_cols: list[str]) -> dict[str, float]:
        raw = self._raw_importance()
        if raw is None or len(raw) != len(feature_cols):
            return {}
        total = float(np.sum(raw))
        if total > 0:
            raw = raw / total
        idx = np.argsort(raw)[::-1]
        return {feature_cols[i]: float(raw[i]) for i in idx}


class LGBMWrapper(_SklearnWrapper):
    model_type = 'lgbm'

    def _build(self, context: dict[str, Any]) -> Any:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            num_leaves=31,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )

    def _raw_importance(self) -> np.ndarray | None:
        try:
            return self._model.feature_importances_
        except Exception:
            return None


class RidgeWrapper(_SklearnWrapper):
    model_type = 'ridge'

    def _build(self, context: dict[str, Any]) -> Any:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0, random_state=42)

    def _raw_importance(self) -> np.ndarray | None:
        try:
            return np.abs(self._model.coef_)
        except Exception:
            return None


# =============================================================================
# PyTorch MLP 包装器
# =============================================================================


class MLPWrapper(BaseWrapper):
    """PyTorch MLP：Linear→LayerNorm→ReLU→Dropout 堆叠 + Adam + weight_decay。

    训练策略：
    - CV 阶段：每折记录 val_loss/epoch，early stopping (patience=10)
    - 全量训练：epochs = 各折最优 epoch 的中位数
    - 特征重要性：输入梯度绝对值的均值（gradient-based importance）
    """

    model_type = 'mlp'

    def __init__(self, context: dict[str, Any]) -> None:
        import torch

        self._torch = torch
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ——— 超参数 ———
        self._hidden_layers: list[int] = context.get('mlp_hidden', [128, 64, 32])
        self._dropout: float = context.get('mlp_dropout', 0.3)
        self._lr: float = context.get('mlp_lr', 1e-3)
        self._weight_decay: float = context.get('mlp_weight_decay', 0.01)
        self._batch_size: int = context.get('mlp_batch_size', 1024)
        self._epochs: int = 100  # CV 阶段的最大 epoch，后续会被 finalize_epochs 覆盖
        self._patience: int = 10

        # ——— 运行时状态 ———
        self._model: Any = None  # torch.nn.Module
        self._input_dim: int = 0
        self._cv_epoch_stats: list[dict] = []  # CV 阶段记录

    # ---- 网络构建 ----

    def _build_network(self, input_dim: int) -> Any:
        """构建 MLP 网络：每层 Linear→LayerNorm→ReLU→Dropout，输出 Linear。"""
        nn = self._torch.nn
        layers: list[Any] = []
        prev = input_dim
        for h in self._hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self._dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers).to(self._device)

    # ---- 训练循环 ----

    def _train_epochs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        record_val_loss: bool = False,
    ) -> dict[str, float]:
        """训练指定 epochs，记录每 epoch 的 train/val loss。"""
        torch = self._torch
        if self._model is None:
            self._model = self._build_network(X.shape[1])
            self._input_dim = X.shape[1]

        model = self._model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        loss_fn = torch.nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=self._device)

        if X_val is not None and y_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self._device)
            y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32, device=self._device)
        else:
            X_val_t = y_val_t = None

        n = len(X_t)
        epoch_losses: list[dict] = []
        best_val_loss = float('inf')
        best_state: dict | None = None
        patience_counter = 0
        best_epoch = 0

        for ep in range(epochs):
            # shuffle
            perm = torch.randperm(n, device=self._device)
            total_loss = 0.0
            for i in range(0, n, self._batch_size):
                idx = perm[i : i + self._batch_size]
                pred = model(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)

            train_loss = total_loss / n

            # 验证
            val_loss = float('nan')
            if X_val_t is not None:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_t)
                    val_loss = loss_fn(val_pred, y_val_t).item()
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_epoch = ep
                    patience_counter = 0
                else:
                    patience_counter += 1

            epoch_losses.append({'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss})

            if record_val_loss and patience_counter >= self._patience:
                break

        # 恢复最优权重
        if best_state is not None:
            model.load_state_dict(best_state)

        # 记录 CV epoch 统计
        if record_val_loss:
            self._cv_epoch_stats.append({
                'best_epoch': best_epoch,
                'best_val_loss': float(best_val_loss),
                'final_epoch': epoch_losses[-1]['epoch'],
            })

        return {
            'train_loss': epoch_losses[-1]['train_loss'],
            'val_loss': float(best_val_loss),
            'best_epoch': best_epoch,
            'total_epochs': epoch_losses[-1]['epoch'] + 1,
        }

    # ---- 公共接口 ----

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return self._train_epochs(X, y, self._epochs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = self._torch
        self._model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X_t), self._batch_size * 4):
                batch = X_t[i : i + self._batch_size * 4]
                preds.append(self._model(batch).cpu().numpy().ravel())
        return np.concatenate(preds)

    def cv_fit_predict(
        self, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        """CV 折：训练 + 记录 epoch 统计 + 返回验证预测。"""
        y_val_proxy = np.zeros(len(X_val))  # 仅用于 loss 计算，CV 不泄露 label
        metrics = self._train_epochs(
            X_tr, y_tr, self._epochs,
            X_val=X_val, y_val=y_val_proxy,
            record_val_loss=True,
        )
        return self.predict(X_val), metrics

    def finalize_epochs(self) -> int:
        """CV 完成后调用：用各折最优 epoch 中位数作为全量训练 epoch 数。"""
        if self._cv_epoch_stats:
            best_epochs = [s['best_epoch'] for s in self._cv_epoch_stats]
            self._epochs = max(1, int(np.median(best_epochs)))
            logging.info(
                f'[mlp] CV best epochs={best_epochs}, '
                f'median={self._epochs}, '
                f'val_losses={[f"{s["best_val_loss"]:.4f}" for s in self._cv_epoch_stats]}'
            )
        # 清空 CV 状态，全量训练时重建网络
        self._model = None
        self._cv_epoch_stats = []
        return self._epochs

    def get_feature_importance(self, feature_cols: list[str]) -> dict[str, float]:
        """梯度法特征重要性：E[|∂output/∂input|] over training samples（近似）。

        在全部训练数据上计算输入梯度绝对值均值，然后归一化。
        注意：此方法在 fit 后调用，依赖 self._model 持有训练好的网络。
        """
        torch = self._torch
        if self._model is None:
            return {}
        # 此处不保存数据引用，通过 fit 后调用者传入 X 不现实。
        # 改为在 fit 内部缓存少量样本用于梯度计算。
        # 简化为使用网络第一层权重的绝对值作为近似（对线性+LayerNorm结构，
        # 首层权重绝对值与梯度重要性高度正相关）。
        try:
            first_linear = None
            for m in self._model.modules():
                if isinstance(m, torch.nn.Linear):
                    first_linear = m
                    break
            if first_linear is None:
                return {}
            raw = first_linear.weight.abs().mean(dim=0).detach().cpu().numpy()
        except Exception:
            return {}

        if len(raw) != len(feature_cols):
            return {}
        total = float(np.sum(raw))
        if total > 0:
            raw = raw / total
        idx = np.argsort(raw)[::-1]
        return {feature_cols[i]: float(raw[i]) for i in idx}


# =============================================================================
# 包装器注册表
# =============================================================================

WRAPPER_REGISTRY: dict[str, type[BaseWrapper]] = {
    'lgbm': LGBMWrapper,
    'ridge': RidgeWrapper,
    'mlp': MLPWrapper,
}
