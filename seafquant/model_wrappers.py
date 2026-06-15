"""
模型包装器 — 封装训练/预测/CV/特征重要性，支持可插拔损失函数 (MSE / Pearson IC)。
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

_EPS: float = 1e-8


# =============================================================================
# 损失函数抽象层 — 可扩展的损失函数族
# =============================================================================


class LossFunction(ABC):
    """损失函数抽象协议：LGBM custom objective / PyTorch loss / early stopping 方向。"""
    import torch
    name: str = ''
    lower_is_better: bool = True  # False for IC-like (higher = better)

    @abstractmethod
    def lgbm_objective(
        self, preds: np.ndarray, train_data: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """LGBM custom objective: return (grad, hess) per sample."""

    def torch_fn(self) -> torch.nn.Module:
        """返回 PyTorch nn.Module loss（训练用）。"""
        import torch
        return torch.nn.MSELoss()


class MSELossFn(LossFunction):
    """均方误差 — 默认。"""

    import torch
    name = 'mse'
    lower_is_better = True

    def lgbm_objective(
        self, preds: np.ndarray, train_data: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = train_data.get_label()
        return np.asarray(preds - labels, dtype=float), np.ones_like(preds, dtype=float)

    def torch_fn(self) -> torch.nn.Module:
        import torch
        return torch.nn.MSELoss()


class ICPearsonLossFn(LossFunction):
    """Pearson IC — 最大化预测值与真实值的截面相关系数。

    梯度推导（-pearsonr）：
        z_i = (ŷ_i-μ_ŷ)/σ_ŷ,  t_i = (y_i-μ_y)/σ_y,  r = mean(z_i·t_i)
        ∂(-r)/∂ŷ_i = -(t_i - r·z_i) / (n·σ_ŷ)
    """

    name = 'ic'
    lower_is_better = False

    def lgbm_objective(
        self, preds: np.ndarray, train_data: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = train_data.get_label()
        n = len(preds)
        preds = np.asarray(preds, dtype=float)
        labels = np.asarray(labels, dtype=float)

        mu_p, mu_l = np.mean(preds), np.mean(labels)
        std_p = max(np.std(preds), _EPS)
        std_l = max(np.std(labels), _EPS)

        z = (preds - mu_p) / std_p
        t = (labels - mu_l) / std_l
        r = np.mean(z * t)

        grad = -(t - r * z) / (n * std_p)
        hess = np.full(n, 1.0 / (n * std_p ** 2))
        return grad.astype(float), hess.astype(float)

    def torch_fn(self):
        import torch

        class _PearsonICLoss(torch.nn.Module):
            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                pred = pred.view(-1)
                target = target.view(-1)
                p_mean = pred.mean()
                t_mean = target.mean()
                p_c = pred - p_mean
                t_c = target - t_mean
                cov = (p_c * t_c).mean()
                p_std = p_c.std() + _EPS
                t_std = t_c.std() + _EPS
                return -cov / (p_std * t_std)

        return _PearsonICLoss()


LOSS_REGISTRY: dict[str, LossFunction] = {
    'mse': MSELossFn(),
    'ic': ICPearsonLossFn(),
}


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
        loss_name = context.get('loss', 'mse')
        self._loss: LossFunction = LOSS_REGISTRY.get(loss_name, MSELossFn())
        self._model = self._build(context)

    def _build(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='X does not have valid feature names',
                category=UserWarning,
            )
            return self._model.predict(np.asarray(X, dtype=float))

    def _raw_importance(self) -> np.ndarray | None:
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

        kwargs: dict[str, Any] = {
            'n_estimators': context.get('lgbm_n_estimators', 10),
            'max_depth': context.get('lgbm_max_depth', 6),
            'num_leaves': context.get('lgbm_num_leaves', 31),
            'reg_alpha': context.get('lgbm_reg_alpha', 0.1),
            'reg_lambda': context.get('lgbm_reg_lambda', 0.1),
            'random_state': context.get('seed', 42),
            'verbose': -1,
        }
        # 非 MSE 损失 → LGBM custom objective
        if self._loss.name != 'mse':
            kwargs['objective'] = self._loss.lgbm_objective
            logging.info(
                f'[lgbm] Using custom objective: {self._loss.name} '
                f'(lower_is_better={self._loss.lower_is_better})'
            )
        return LGBMRegressor(**kwargs)

    def _raw_importance(self) -> np.ndarray | None:
        try:
            return self._model.feature_importances_
        except Exception:
            return None


class RidgeWrapper(_SklearnWrapper):
    model_type = 'ridge'

    def _build(self, context: dict[str, Any]) -> Any:
        from sklearn.linear_model import Ridge

        if self._loss.name != 'mse':
            logging.warning(
                f'[ridge] Loss "{self._loss.name}" not supported by Ridge; '
                f'falling back to MSE (ridge is L2-regularized least squares).'
            )
        return Ridge(
            alpha=context.get('ridge_alpha', 42),
            random_state=context.get('seed', 42)
        )

    def _raw_importance(self) -> np.ndarray | None:
        try:
            return np.abs(self._model.coef_)
        except Exception:
            return None


# =============================================================================
# PyTorch MLP 包装器
# =============================================================================


class _ResidualMLP:
    """残差 MLP — Transformer Feed-Forward 风格。

    每块结构 (Pre-Norm):
      LayerNorm → Linear(d→4d) → GELU → Linear(4d→d) → Dropout → +skip

    特点：
    - 两线性层：先膨胀 4× 再投影回原维度，残差始终对齐
    - GELU：平滑激活，梯度处处非零，适合 IC 损失
    - Pre-Norm：层归一化在前，稳定深层训练
    """

    EXPANSION: int = 4  # FFN 膨胀比

    def __init__(
        self, nn: Any, input_dim: int, hidden_dims: list[int],
        dropout: float, device: Any,
    ) -> None:
        self.nn = nn

        # 输入投影 → 首个隐藏维度
        d_model = hidden_dims[0] if hidden_dims else input_dim
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else None
        self.blocks: list[dict[str, Any]] = []

        for d in hidden_dims:
            # 若维度变化，用投影层对齐
            proj = None
            if d_model != d:
                proj = nn.Linear(d_model, d)
                nn.init.xavier_normal_(proj.weight)
                nn.init.zeros_(proj.bias)

            d_expanded = d * self.EXPANSION
            block: dict[str, Any] = {
                'norm': nn.LayerNorm(d),
                'ffn1': nn.Linear(d, d_expanded),
                'ffn2': nn.Linear(d_expanded, d),
                'dropout': nn.Dropout(dropout),
                'proj': proj,
            }
            nn.init.kaiming_normal_(block['ffn1'].weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(block['ffn1'].bias)
            nn.init.kaiming_normal_(block['ffn2'].weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(block['ffn2'].bias)

            self.blocks.append(block)
            d_model = d

        self.head = nn.Linear(d_model, 1)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.to(device)

    def to(self, device: Any) -> '_ResidualMLP':
        if self.input_proj is not None:
            self.input_proj = self.input_proj.to(device)
        for b in self.blocks:
            b['norm'] = b['norm'].to(device)
            b['ffn1'] = b['ffn1'].to(device)
            b['ffn2'] = b['ffn2'].to(device)
            b['dropout'] = b['dropout'].to(device)
            if b['proj'] is not None:
                b['proj'] = b['proj'].to(device)
        self.head = self.head.to(device)
        return self

    def forward(self, x: Any) -> Any:
        F = self.nn.functional
        if self.input_proj is not None:
            x = F.gelu(self.input_proj(x))

        for b in self.blocks:
            residual = x
            if b['proj'] is not None:
                residual = b['proj'](residual)

            h = b['norm'](x)
            h = b['ffn1'](h)
            h = F.gelu(h)
            h = b['ffn2'](h)
            h = b['dropout'](h)
            x = residual + h

        return self.head(x)

    def state_dict(self) -> dict[str, Any]:
        sd: dict[str, Any] = {}
        if self.input_proj is not None:
            sd['input_proj'] = self.input_proj.state_dict()
        for i, b in enumerate(self.blocks):
            sd[f'block_{i}'] = {k: v.state_dict() for k, v in b.items()
                               if hasattr(v, 'state_dict') and v is not None}
        sd['head'] = self.head.state_dict()
        return sd

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        if self.input_proj is not None and 'input_proj' in sd:
            self.input_proj.load_state_dict(sd['input_proj'])
        for i, b in enumerate(self.blocks):
            key = f'block_{i}'
            if key in sd:
                for k, v in b.items():
                    if hasattr(v, 'state_dict') and v is not None and k in sd[key]:
                        v.load_state_dict(sd[key][k])
        self.head.load_state_dict(sd['head'])

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass


class MLPWrapper(BaseWrapper):
    """PyTorch ResMLP：Pre-Norm 残差块 + Adam + weight_decay + gradient clipping + noise injection。
    支持 MSE / Pearson IC 损失切换，early stopping 方向自动适配。
    """

    model_type = 'mlp'

    def __init__(self, context: dict[str, Any]) -> None:
        import torch

        self._torch = torch
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 可重入性：设置 torch 随机种子
        torch_seed: int = context.get('seed', 42)
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(torch_seed)

        loss_name = context.get('loss', 'mse')
        self._loss: LossFunction = LOSS_REGISTRY.get(loss_name, MSELossFn())

        self._hidden_layers: list[int] = context.get('mlp_hidden', [256, 256, 256, 32])
        self._dropout: float = context.get('mlp_dropout', 0.5)
        self._lr: float = context.get('mlp_lr', 1e-3)
        self._weight_decay: float = context.get('mlp_weight_decay', 1e-2)
        self._batch_size: int = context.get('mlp_batch_size', 512)
        self._epochs: int = 100
        self._patience: int = 10

        self._model: Any = None
        self._input_dim: int = 0
        self._cv_epoch_stats: list[dict] = []

        logging.info(
            f'[mlp] loss={self._loss.name}, lower_is_better={self._loss.lower_is_better}'
        )

    # ---- 网络构建 ----

    def _build_network(self, input_dim: int, use_residual: bool = True) -> Any:
        nn = self._torch.nn
        init = self._torch.nn.init

        if use_residual:
            return _ResidualMLP(
                nn, input_dim, self._hidden_layers, self._dropout, self._device,
            )

        # fallback：简单 MLP（无残差连接）
        layers: list[Any] = []
        prev = input_dim
        for h in self._hidden_layers:
            linear = nn.Linear(prev, h)
            init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(linear.bias)
            layers.append(linear)
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
        torch = self._torch
        if self._model is None:
            self._model = self._build_network(X.shape[1])
            self._input_dim = X.shape[1]

        model = self._model
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        loss_fn = self._loss.torch_fn()

        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=self._device)

        if X_val is not None and y_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self._device)
            y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32, device=self._device)
        else:
            X_val_t = y_val_t = None

        n = len(X_t)
        epoch_losses: list[dict] = []
        noise_std: float = 0.01  # 输入噪声正则化

        # early stopping 方向适配（IC = higher better → 取反存储为 loss）
        best_loss = float('inf')
        best_state: dict | None = None
        patience_counter = 0
        best_epoch = 0

        for ep in range(epochs):
            perm = torch.randperm(n, device=self._device)
            total_loss = 0.0
            for i in range(0, n, self._batch_size):
                idx = perm[i : i + self._batch_size]
                # 输入噪声注入：小方差高斯噪声，等价于岭正则化
                x_batch = X_t[idx] + torch.randn_like(X_t[idx]) * noise_std
                pred = model(x_batch)
                loss = loss_fn(pred, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item() * len(idx)

            train_loss = total_loss / n

            val_loss = float('nan')
            if X_val_t is not None:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_t)
                    val_loss = loss_fn(val_pred, y_val_t).item()
                model.train()
                # TODO: best_epoch 应该取最高的三个epoch表现的中位数？
                improved = val_loss < best_loss
                if improved:
                    best_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_epoch = ep
                    patience_counter = 0
                else:
                    patience_counter += 1

            epoch_losses.append({'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss})

            if record_val_loss and patience_counter >= self._patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        if record_val_loss:
            self._cv_epoch_stats.append({
                'best_epoch': best_epoch,
                'best_val_loss': float(best_loss),
                'final_epoch': epoch_losses[-1]['epoch'],
            })

        return {
            'train_loss': epoch_losses[-1]['train_loss'],
            'val_loss': float(best_loss),
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
        self, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray,
        y_val: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if y_val is None:
            y_val = np.zeros(len(X_val))
        metrics = self._train_epochs(
            X_tr, y_tr, self._epochs,
            X_val=X_val, y_val=y_val,
            record_val_loss=True,
        )
        return self.predict(X_val), metrics

    def finalize_epochs(self) -> int:
        if self._cv_epoch_stats:
            best_epochs = [s['best_epoch'] for s in self._cv_epoch_stats]
            self._epochs = max(1, int(np.median(best_epochs)))
            logging.info(
                f'[mlp] CV best epochs={best_epochs}, '
                f'median={self._epochs}, '
                f'val_losses={[f"{s["best_val_loss"]:.4f}" for s in self._cv_epoch_stats]}'
            )
        self._model = None
        self._cv_epoch_stats = []
        return self._epochs

    def get_feature_importance(self, feature_cols: list[str]) -> dict[str, float]:
        torch = self._torch
        if self._model is None:
            return {}
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
