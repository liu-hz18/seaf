"""
模型训练与预测节点 — 支持 LGBM / Ridge 两种模型，滚动训练，预测每日截面信号。

训练逻辑：
- 每 retrain_every 天用最近 (window - fwd) 天因子数据重新训练模型。
- Label：fwd 日后截面超额收益率（cs_zscore）。
- 时间穿越防护：只用 time [0, n_times-fwd) 训练（每个样本 label 基于 t+fwd）。
- 预测：对窗口内最后一天的因子做 predict，输出 pred_signal。

context 配置（从 pipeline 传入）：
  - model_type: 'lgbm' (默认) 或 'ridge'
  - fwd: 前向预测天数 (默认 20)
  - model_window: 模型节点总窗口大小 (factor_window + fwd)
  - retrain_every: 重训练间隔 (默认 20)
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Any
from qpipe.frame3d import Frame3D
from sklearn.model_selection import TimeSeriesSplit


def _build_model(model_type: str = 'lgbm'):
    """根据类型构建模型。"""
    if model_type == 'lgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=100, max_depth=6, num_leaves=31,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
        )
    elif model_type == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def model_train_predict(name: str, f3d: Frame3D, context: Any) -> Tuple[Frame3D, Any]:
    """模型训练与预测主函数。

    f3d 包含 window 天的数据（因子列 + close 列）。
    """
    if context is None:
        context = {}

    # 默认值（向后兼容）
    context.setdefault('trained_model', None)
    context.setdefault('is_trained', False)
    context.setdefault('retrain_every', 20)
    context.setdefault('days_since_train', 0)
    context.setdefault('feature_cols', None)
    context.setdefault('model_type', 'lgbm')
    context.setdefault('fwd', 20)
    context.setdefault('model_window', context['fwd'] + 200)  # 默认窗口

    fwd = context['fwd']

    df = f3d.df.copy()

    # 识别因子列
    raw_cols = {'open', 'high', 'low', 'close', 'turnover', 'volume', 'market_cap'}
    if context['feature_cols'] is None:
        context['feature_cols'] = [c for c in df.columns if c not in raw_cols and not c.startswith('_')]

    feature_cols = context['feature_cols']

    if 'close' not in df.columns:
        raise ValueError(f"[{name}] Model node requires 'close' column in input data")

    # ===== 提取时间维度数据 =====
    times = sorted(df.index.get_level_values('key').unique())
    n_times = len(times)

    # 需要至少 fwd 天才能计算 label，再加至少 1 个训练样本
    min_required = fwd + 1
    if n_times < min_required:
        logging.warning(f"[{name}] Insufficient data: {n_times} < {min_required} (fwd={fwd})")
        empty_result = Frame3D(pd.DataFrame({'pred_signal': [0.0] * len(df.loc[times[-1]])},
                                            index=df.loc[times[-1]].index))
        return empty_result, context

    # 检查是否需要训练
    context['days_since_train'] += 1
    should_train = (not context['is_trained']) or (context['days_since_train'] >= context['retrain_every'])

    if should_train:
        logging.info(f"[{name}] Triggering retrain (model={context.get('model_type','lgbm')}, "
                     f"fwd={fwd}). days_since_train={context['days_since_train']}")

        # 训练样本：time [0, n_times-fwd) — 每个样本在 t+fwd 有对应的 close
        n_train_times = n_times - fwd  # 可用于训练的时间点数
        if n_train_times < 10:
            logging.warning(f"[{name}] Too few training times: {n_train_times} (fwd={fwd})")
            context['days_since_train'] = 0
            empty_result = Frame3D(pd.DataFrame({'pred_signal': [0.0] * len(df.loc[times[-1]])},
                                                index=df.loc[times[-1]].index))
            return empty_result, context

        # 构建特征矩阵 X：每个训练时间点的因子值
        X_list, y_list = [], []
        for t_idx in range(n_train_times):
            t = times[t_idx]
            cs_mask = df.index.get_level_values('key') == t
            X_cs = df.loc[cs_mask, feature_cols].values.astype(float)

            # Label：t+fwd 的前瞻收益
            t_label = times[t_idx + fwd]
            cs_mask_label = df.index.get_level_values('key') == t_label
            close_fwd = df.loc[cs_mask_label, 'close'].values
            close_t = df.loc[cs_mask, 'close'].values
            fwd_ret = close_fwd / close_t - 1

            X_list.append(X_cs)
            y_list.append(fwd_ret)

        X = np.vstack(X_list)
        y = np.hstack(y_list)

        print(f"[pre-validate] X.shape={X.shape}, y.shape={y.shape}")
        # 移除 NaN 样本
        valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        X, y = X[valid], y[valid]
        print(f"[post-validate] X.shape={X.shape}, y.shape={y.shape}")

        if len(y) < 50:
            logging.warning(f"[{name}] Insufficient training samples: {len(y)}")
            context['days_since_train'] = 0
            empty_result = Frame3D(pd.DataFrame({'pred_signal': [0.0] * len(df.loc[times[-1]])},
                                                index=df.loc[times[-1]].index))
            return empty_result, context

        # 截面标准化 label
        y_xd = (y - np.mean(y)) / (np.std(y) + 1e-10)

        # 构建模型
        model_type = context.get('model_type', 'lgbm')
        model = _build_model(model_type)

        # TimeSeriesSplit 交叉验证
        cv_scores = []
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, val_idx in tscv.split(X):
            if len(val_idx) < 10:
                continue
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y_xd[train_idx], y_xd[val_idx]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            from scipy.stats import spearmanr
            try:
                ic = spearmanr(pred, y_val).correlation
                cv_scores.append(ic)
            except Exception:
                pass

        # Full training
        model = _build_model(model_type)
        model.fit(X, y_xd)

        context['trained_model'] = model
        context['is_trained'] = True
        context['days_since_train'] = 0

        cv_mean = np.mean(cv_scores) if cv_scores else 0.0
        logging.info(
            f"[{name}] Training done ({model_type}, fwd={fwd}): "
            f"samples={len(y)}, features={len(feature_cols)}, cv_ic_mean={cv_mean:.4f}"
        )

    # ===== 预测流程：对最后一天的截面因子做 predict =====
    model = context['trained_model']
    latest_t = times[-1]
    cs_mask_latest = df.index.get_level_values('key') == latest_t
    X_latest = df.loc[cs_mask_latest, feature_cols].values.astype(float)

    pred_raw = model.predict(X_latest)

    # 截面标准化
    pred_mean = np.nanmean(pred_raw)
    pred_std = np.nanstd(pred_raw)
    pred_signal = (pred_raw - pred_mean) / pred_std if pred_std > 0 else np.zeros_like(pred_raw)

    logging.info(
        f"[{name}] Predict day t={latest_t} (fwd={fwd}), "
        f"signal_mean={pred_signal.mean():.4f}, signal_std={pred_signal.std():.4f}"
    )

    result_df = pd.DataFrame({'pred_signal': pred_signal}, index=df.loc[cs_mask_latest].index)
    return Frame3D(result_df), context
