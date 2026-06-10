"""
模型训练与预测节点 — 支持 LGBM / Ridge / MLP，滚动训练，预测每日截面信号。

架构：
- model_wrappers.py：每种模型类型的封装（build / fit / predict / CV / 特征重要性）
- model_node.py（本文件）：编排层 — 数据预处理、标签构造、CV 调度、MLflow 记录

训练逻辑：
- 每 retrain_every 天用最近窗口内因子数据重新训练模型。
- Label：cs_zscore(close_{t+fwd} / close_{t+1} - 1) — 未来 (fwd-1) 日截面超额收益。
- 时间穿越防护：只用 time [0, n_times-fwd-1) 训练。

context 配置（从 pipeline 传入）：
  model_type: 'lgbm' | 'ridge' | 'mlp'
  fwd: 前向预测天数 (默认 20)
  retrain_every: 重训练间隔 (默认 20)
  — MLP 专属 —
  mlp_hidden: 隐藏层列表 (默认 [128, 64, 32])
  mlp_dropout: dropout 比例 (默认 0.3)
  mlp_lr: 学习率 (默认 1e-3)
  mlp_weight_decay: Adam weight_decay (默认 0.01)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

# scipy / sklearn 延迟导入到使用函数内，节省 ~1s 顶层导入时间
from qpipe.frame3d import Frame3D
from qpipe.utils import mlflow_log_metrics, trading_step
from seafquant.model_wrappers import WRAPPER_REGISTRY

# =============================================================================
# 工具函数
# =============================================================================


def _cs_zscore(values: np.ndarray) -> np.ndarray:
    """截面标准化：(x - mean) / std，std=0 时返回零向量。"""
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std > 0:
        return (values - mean) / std
    return np.zeros_like(values)


def _empty_result(n_stocks: int, index: pd.Index) -> Frame3D:
    """构造空预测结果（未训练或数据不足时返回）。"""
    return Frame3D(pd.DataFrame({'pred_signal': [0.0] * n_stocks}, index=index))


# =============================================================================
# 数据预处理（从 Frame3D 中提取 X, y）
# =============================================================================


def _prepare_training_data(
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    times: list,
    fwd: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """从 Frame3D DataFrame 中提取训练特征和标签。

    Returns (X, y, cs_stats)，其中：
    - X: (n_samples, n_features)
    - y: 逐截面 cs_zscore(fwd_ret)
    - cs_stats: 每个截面的统计信息
    """
    n_train_times = len(times) - fwd - 1
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    cs_stats: list[dict] = []

    for t_idx in range(n_train_times):
        t = times[t_idx]
        t_next = times[t_idx + 1]
        t_fwd = times[t_idx + fwd]

        cs_mask_t = df.index.get_level_values('key') == t
        X_cs = df.loc[cs_mask_t, feature_cols].to_numpy(dtype=float, copy=True)

        cs_mask_buy = df.index.get_level_values('key') == t_next
        cs_mask_sell = df.index.get_level_values('key') == t_fwd
        close_buy = df.loc[cs_mask_buy, 'close'].values
        close_sell = df.loc[cs_mask_sell, 'close'].values
        fwd_ret = close_sell / close_buy - 1
        label_xd = _cs_zscore(fwd_ret)

        X_list.append(X_cs)
        y_list.append(label_xd)
        cs_stats.append({
            't': t, 'n': len(fwd_ret),
            'ret_mean': float(np.nanmean(fwd_ret)),
            'ret_std': float(np.nanstd(fwd_ret)),
        })

    return np.vstack(X_list), np.hstack(y_list), cs_stats


# =============================================================================
# 交叉验证调度
# =============================================================================


def _run_cv(
    name: str,
    wrapper: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
) -> tuple[list[float], bool]:
    """时间序列交叉验证，IC 导向。

    Returns (cv_scores, is_mlp)。
    """
    from scipy.stats import spearmanr  # 延迟导入
    from sklearn.model_selection import TimeSeriesSplit  # 延迟导入

    is_mlp = hasattr(wrapper, 'finalize_epochs')
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores: list[float] = []

    logging.info(f'[{name}] CV (TimeSeriesSplit, {n_splits} folds, IC metric):')
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        if len(val_idx) < 10:
            continue
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pred, _ = wrapper.cv_fit_predict(X_tr, y_tr, X_val)

        try:
            ic = spearmanr(pred, y_val).correlation
            if not np.isnan(ic):
                cv_scores.append(ic)
                logging.info(
                    f'  Fold {fold + 1}: IC={ic:.4f}, '
                    f'n_train={len(y_tr):,}, n_val={len(y_val):,}'
                )
        except Exception:
            pass

    # MLP：CV 完成后从 epoch 统计中确定全量训练 epoch 数
    if is_mlp and hasattr(wrapper, 'finalize_epochs'):
        wrapper.finalize_epochs()

    return cv_scores, is_mlp


# =============================================================================
# MLflow 日志记录
# =============================================================================


def _log_feature_importance(
    run_id: str,
    name: str,
    fi: dict[str, float],
    model_type: str,
    step: int,
) -> None:
    """记录特征重要性到日志和 MLflow artifact。"""
    if not fi:
        return
    top_n = min(10, len(fi))
    top_items = list(fi.items())[:top_n]
    logging.info(
        f'[{name}] Feature importance top-{top_n}: '
        + ', '.join(f'{k}={v:.4f}' for k, v in top_items)
    )
    if run_id:
        try:
            import mlflow
            mlflow.set_tracking_uri('sqlite:///mlruns.db')
            mlflow.log_dict(
                fi,
                f'feature_importance_{model_type}_{step}.json',
                run_id=run_id,
            )
        except Exception:
            pass


# =============================================================================
# 主入口：model_train_predict
# =============================================================================


def model_train_predict(name: str, f3d: Frame3D, context: Any) -> Frame3D:
    """模型训练与预测主函数 — 编排层。

    f3d 包含 window 天的数据（因子列 + close 列）。
    Label = cs_zscore(close[t+fwd] / close[t+1] - 1) — 截面超额收益。
    """
    # —— context 初始化 ——
    if context is None:
        context = {}
    context.setdefault('trained_wrapper', None)
    context.setdefault('is_trained', False)
    context.setdefault('days_since_train', 0)
    context.setdefault('feature_cols', None)
    context.setdefault('model_type', 'lgbm')
    context.setdefault('fwd', 20)
    context.setdefault('model_window', 200)
    context.setdefault('retrain_every', 20)

    fwd = context['fwd']
    df = f3d.df.copy()
    model_type: str = context['model_type']

    # —— 因子列识别 ——
    raw_cols = {'open', 'high', 'low', 'close', 'turnover', 'volume', 'market_cap'}
    if context['feature_cols'] is None:
        context['feature_cols'] = [
            c for c in df.columns if c not in raw_cols and not c.startswith('_')
        ]
    feature_cols: list[str] = context['feature_cols']

    if 'close' not in df.columns:
        raise ValueError(f'[{name}] Model node requires "close" column')

    mlflow_run_id: str = context.get('mlflow_run_id', '')
    start_date: str = context.get('start_date', '')

    # —— 时间维度 ——
    times = sorted(df.index.get_level_values('key').unique())
    n_times = len(times)
    n_stocks = df.index.get_level_values('name').nunique()
    latest_t = times[-1]

    if n_times < fwd + 2:
        logging.warning(f'[{name}] Insufficient data: {n_times} < {fwd + 2}')
        cs_mask = df.index.get_level_values('key') == latest_t
        return _empty_result(n_stocks, df.loc[cs_mask].index)

    # —— 训练触发判断 ——
    context['days_since_train'] += 1
    should_train = (not context['is_trained']) or (
        context['days_since_train'] >= context['retrain_every']
    )

    # ========================================================================
    # 训练阶段
    # ========================================================================
    if should_train:
        logging.info(
            f'[{name}] ===== RETRAIN START ===== '
            f'model={model_type}, fwd={fwd}, '
            f'retrain_every={context["retrain_every"]}, '
            f'days_since_train={context["days_since_train"]}'
        )

        # 1. 准备训练数据
        n_train_times = n_times - fwd - 1
        if n_train_times < 10:
            logging.warning(f'[{name}] Too few training times: {n_train_times}')
            context['days_since_train'] = 0
            return _empty_result(n_stocks, df.loc[latest_t].index)

        X, y, cs_stats = _prepare_training_data(name, df, feature_cols, times, fwd)

        logging.info(
            f'[{name}] Training set: {len(cs_stats)} cs x ~{n_stocks}s, '
            f'{len(y)} samples, {X.shape[1]} features'
        )

        # 2. Label 统计 → MLflow
        mlflow_log_metrics(mlflow_run_id, name, {
            'label_mean': float(np.mean(y)),
            'label_std': float(np.std(y)),
            'label_min': float(np.min(y)),
            'label_max': float(np.max(y)),
        }, step=trading_step(start_date, times[n_train_times - 1]))

        # 3. NaN 处理
        #   规则：标签 y 为 NaN 的样本直接丢弃；
        #         特征 X 的 NaN：
        #           - 超过半数特征列为 NaN → 丢弃该样本
        #           - 否则 → NaN 置 0.0
        feat_nan_cnt = np.sum(np.isnan(X), axis=1)
        n_feats = X.shape[1]
        drop_mask = feat_nan_cnt > (n_feats // 2)  # 严格 > 半数
        y_nan = np.isnan(y)
        valid = ~y_nan & ~drop_mask
        nan_total = sum(~valid) + len(y)
        nan_ratio = (nan_total > 0 and (sum(~valid)) / nan_total) or 0.0
        X, y = X[valid], y[valid]
        X = np.nan_to_num(X, nan=0.0)  # 保留样本的 NaN → 0
        logging.info(
            f'[{name}] NaN handling: {sum(~valid)} removed '
            f'(y_nan={sum(y_nan)}, drop>{n_feats // 2}feat_nan={sum(drop_mask)}), '
            f'{len(y)} remain, {np.sum(np.isnan(X))} NaN filled→0'
        )

        if len(y) < 50:
            logging.warning(f'[{name}] <50 samples after NaN removal')
            context['days_since_train'] = 0
            return _empty_result(n_stocks, df.loc[latest_t].index)

        # 4. 构建 wrapper
        wrapper_cls = WRAPPER_REGISTRY[model_type]
        wrapper = wrapper_cls(context)

        # 5. 交叉验证
        cv_scores, _ = _run_cv(name, wrapper, X, y)

        # 6. 全量训练
        wrapper.fit(X, y)

        # 7. 训练集 MSE
        pred_train = wrapper.predict(X)
        train_mse = float(np.mean((pred_train - y) ** 2))

        # 8. 特征重要性
        fi = wrapper.get_feature_importance(feature_cols)
        _log_feature_importance(mlflow_run_id, name, fi, model_type, step=trading_step(start_date, times[n_train_times - 1]))

        # 9. 保存状态
        context['trained_wrapper'] = wrapper
        context['is_trained'] = True
        context['days_since_train'] = 0

        # 10. 训练指标 → MLflow
        mlflow_log_metrics(mlflow_run_id, name, {
            'train_samples': float(len(y)),
            'train_features': float(len(feature_cols)),
            'train_nan_ratio': nan_ratio,
            'train_mse': train_mse,
            'cv_ic_mean': float(np.mean(cv_scores)) if cv_scores else 0.0,
        }, step=trading_step(start_date, times[n_train_times - 1]))

        cv_mean = np.mean(cv_scores) if cv_scores else 0.0
        cv_std = np.std(cv_scores) if cv_scores else 0.0
        logging.info(
            f'[{name}] ===== RETRAIN DONE ===== '
            f'samples={len(y):,}, feats={len(feature_cols)}, '
            f'cv_ic={cv_mean:.4f}±{cv_std:.4f}, n_folds={len(cv_scores)}, '
            f'predict_day={latest_t}'
        )

    # ========================================================================
    # 预测阶段
    # ========================================================================
    wrapper = context['trained_wrapper']
    cs_mask_latest = df.index.get_level_values('key') == latest_t
    X_latest = df.loc[cs_mask_latest, feature_cols].to_numpy(dtype=float, copy=True)

    nan_rows = np.any(np.isnan(X_latest), axis=1)
    if nan_rows.any():
        logging.warning(
            f'[{name}] {nan_rows.sum()}/{len(X_latest)} stocks NaN features, filled=0'
        )
        X_latest = np.nan_to_num(X_latest, nan=0.0)

    pred_raw = wrapper.predict(X_latest)
    pred_signal = _cs_zscore(pred_raw)

    mlflow_log_metrics(mlflow_run_id, name, {
        'pred_signal_min': float(np.min(pred_signal)),
        'pred_signal_max': float(np.max(pred_signal)),
        'pred_signal_skew': float(pd.Series(pred_signal).skew()),
    }, step=trading_step(start_date, latest_t))

    logging.info(
        f'[{name}] Predict day={latest_t} (fwd={fwd}): '
        f'n={len(pred_signal)}, '
        f'mean={float(np.mean(pred_signal)):.4f}, '
        f'std={float(np.std(pred_signal)):.4f}, '
        f'min={float(np.min(pred_signal)):.4f}, '
        f'max={float(np.max(pred_signal)):.4f}'
    )

    result_df = pd.DataFrame({'pred_signal': pred_signal}, index=df.loc[cs_mask_latest].index)
    return Frame3D(result_df)
