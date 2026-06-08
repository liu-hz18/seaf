"""
模型训练与预测节点 — 支持 LGBM / Ridge 两种模型，滚动训练，预测每日截面信号。

训练逻辑：
- 每 retrain_every 天用最近窗口内因子数据重新训练模型。
- Label：cs_zscore(close_{t+fwd} / close_{t+1} - 1) — 未来 (fwd-1) 日截面超额收益。
  t+1 日买入、t+fwd 日卖出，排除信号日到买入日的隔夜收益，对齐实盘交易执行。
- 损失函数：MSE on cs_zscore labels ≈ 最大化截面 Pearson IC（数学等价）。
- 时间穿越防护：只用 time [0, n_times-fwd-1) 训练（每个样本需要 t+1 和 t+fwd 的 close）。
- 预测：对窗口内最后一天的因子做 predict，输出 cs_zscore 后的 pred_signal。

context 配置（从 pipeline 传入）：
  - model_type: 'lgbm' (默认) 或 'ridge'
  - fwd: 前向预测天数 (默认 20)
  - retrain_every: 重训练间隔 (默认 20)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit

from qpipe.frame3d import Frame3D
from qpipe.utils import mlflow_log_metrics, trading_step


def _build_model(model_type: str = 'lgbm') -> Any:
    """根据类型构建模型。"""
    if model_type == 'lgbm':
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
    if model_type == 'ridge':
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0, random_state=42)
    raise ValueError(f'Unknown model_type: {model_type}')


def _cs_zscore(values: np.ndarray) -> np.ndarray:
    """截面标准化：(x - mean) / std，std=0 时返回零向量。"""
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std > 0:
        return (values - mean) / std
    return np.zeros_like(values)


def model_train_predict(name: str, f3d: Frame3D, context: Any) -> Frame3D:
    """模型训练与预测主函数。

    f3d 包含 window 天的数据（因子列 + close 列）。
    Label = cs_zscore(close[t+fwd] / close[t+1] - 1) — 截面超额收益。
    """
    if context is None:
        context = {}

    # ——— 运行时状态（由节点自身维护，无需 pipeline 传入）———
    context.setdefault('trained_model', None)
    context.setdefault('is_trained', False)
    context.setdefault('days_since_train', 0)
    context.setdefault('feature_cols', None)

    # ——— 可配置参数（pipeline 传入或在无 context 时使用以下默认值）———
    # 注意：MultiInputNode 通过 Flow.add_node(context=...) 将 pipeline 的
    # model_context 合并到此处，setdefault 确保未传入时也有合理的 fallback。
    context.setdefault('model_type', 'lgbm')
    context.setdefault('fwd', 20)
    context.setdefault('model_window', 200)
    context.setdefault('retrain_every', 20)

    fwd = context['fwd']
    df = f3d.df.copy()

    # ---- 识别因子列 ----
    raw_cols = {'open', 'high', 'low', 'close', 'turnover', 'volume', 'market_cap'}
    if context['feature_cols'] is None:
        context['feature_cols'] = [
            c for c in df.columns if c not in raw_cols and not c.startswith('_')
        ]
    feature_cols = context['feature_cols']

    if 'close' not in df.columns:
        raise ValueError(f'[{name}] Model node requires "close" column in input data')

    mlflow_run_id = context.get('mlflow_run_id', '')
    start_date = context.get('start_date', '')

    # ---- 提取时间维度 ----
    times = sorted(df.index.get_level_values('key').unique())
    n_times = len(times)
    n_stocks = df.index.get_level_values('name').nunique()

    # 需要 fwd+1 天：t 时刻信号 + t+1 买入 + t+fwd 卖出
    min_required = fwd + 2
    if n_times < min_required:
        logging.warning(f'[{name}] Insufficient data: {n_times} < {min_required} (fwd={fwd}+2)')
        empty_result = Frame3D(
            pd.DataFrame({'pred_signal': [0.0] * n_stocks}, index=df.loc[times[-1]].index)
        )
        return empty_result

    # ---- 检查是否需要重训练 ----
    context['days_since_train'] += 1
    should_train = (not context['is_trained']) or (
        context['days_since_train'] >= context['retrain_every']
    )

    # ========================================================================
    # 训练阶段
    # ========================================================================
    if should_train:
        model_type = context.get('model_type', 'lgbm')
        logging.info(
            f'[{name}] ===== RETRAIN START ===== '
            f'model={model_type}, fwd={fwd}, retrain_every={context["retrain_every"]}, '
            f'days_since_train={context["days_since_train"]}'
        )
        logging.info(
            f'[{name}]   feature window: {n_times}d x {n_stocks}s x {len(feature_cols)}cols, '
            f'time_range=[{times[0]} .. {times[-1]}]'
        )

        # 训练样本范围：[0, n_times - fwd - 1)，每个样本需要 t+1 和 t+fwd
        n_train_times = n_times - fwd - 1
        if n_train_times < 10:
            logging.warning(
                f'[{name}] Too few training time points: {n_train_times} '
                f'(n_times={n_times}, fwd={fwd})'
            )
            context['days_since_train'] = 0
            empty_result = Frame3D(
                pd.DataFrame({'pred_signal': [0.0] * n_stocks}, index=df.loc[times[-1]].index)
            )
            return empty_result

        # label = close[t+fwd] / close[t+1] - 1，每个样本的买入/卖出各差 1 天
        logging.info(
            f'[{name}]   training X: {n_train_times} cross-sections '
            f'[{times[0]} .. {times[n_train_times - 1]}]'
        )
        logging.info(
            f'[{name}]   first label: signal@{times[0]} -> return[{times[1]}->{times[fwd]}]'
        )
        logging.info(
            f'[{name}]   last  label: signal@{times[n_train_times - 1]} '
            f'-> return[{times[n_train_times]}->{times[n_train_times - 1 + fwd]}]'
        )

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        cs_stats: list[dict] = []

        for t_idx in range(n_train_times):
            t = times[t_idx]
            t_next = times[t_idx + 1]  # t+1（买入日）
            t_fwd = times[t_idx + fwd]  # t+fwd（卖出日）

            # 特征：t 时刻截面因子
            cs_mask_t = df.index.get_level_values('key') == t
            # to_numpy(copy=True) 深拷贝剥离 pandas 列名元数据，避免 LGBM feature_names 警告
            X_cs = df.loc[cs_mask_t, feature_cols].to_numpy(dtype=float, copy=True)

            # Label：t+1 -> t+fwd 的截面超额收益
            cs_mask_buy = df.index.get_level_values('key') == t_next
            cs_mask_sell = df.index.get_level_values('key') == t_fwd
            close_buy = df.loc[cs_mask_buy, 'close'].values
            close_sell = df.loc[cs_mask_sell, 'close'].values
            fwd_ret = close_sell / close_buy - 1

            # 逐截面标准化 label（每个时间截面独立，无时间穿越）
            label_xd = _cs_zscore(fwd_ret)

            X_list.append(X_cs)
            y_list.append(label_xd)
            cs_stats.append(
                {
                    't': t,
                    'n': len(fwd_ret),
                    'ret_mean': float(np.nanmean(fwd_ret)),
                    'ret_std': float(np.nanstd(fwd_ret)),
                }
            )

        X = np.vstack(X_list)
        y = np.hstack(y_list)

        # ---- 训练数据统计 ----
        logging.info(
            f'[{name}] Training set: {len(cs_stats)} cross-sections, '
            f'{len(y)} total samples, {X.shape[1]} features, '
            f'avg stocks/cs={np.mean([cs["n"] for cs in cs_stats]):.0f}'
        )
        logging.info(
            f'[{name}] Label stats: mean={np.mean(y):.4f}, std={np.std(y):.4f}, '
            f'min={np.min(y):.4f}, max={np.max(y):.4f}, '
            f'avg cs_ret_mean={np.mean([cs["ret_mean"] for cs in cs_stats]):.6f}, '
            f'avg cs_ret_std={np.mean([cs["ret_std"] for cs in cs_stats]):.6f}'
        )
        mlflow_log_metrics(mlflow_run_id, name, {
            'label_mean': float(np.mean(y)),
            'label_std': float(np.std(y)),
            'label_min': float(np.min(y)),
            'label_max': float(np.max(y)),
        }, step=trading_step(start_date, times[n_train_times - 1]))

        # ---- 移除 NaN ----
        valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        nan_count = sum(~valid)
        X, y = X[valid], y[valid]
        nan_total = nan_count + len(y)
        nan_ratio = nan_count / nan_total if nan_total > 0 else 0.0
        logging.info(f'[{name}] NaN removal: {nan_count} removed, {len(y)} samples remain')

        if len(y) < 50:
            logging.warning(f'[{name}] Insufficient training samples after NaN removal: {len(y)}')
            context['days_since_train'] = 0
            empty_result = Frame3D(
                pd.DataFrame({'pred_signal': [0.0] * n_stocks}, index=df.loc[times[-1]].index)
            )
            return empty_result

        # ---- 交叉验证（IC 导向） ----
        model = _build_model(model_type)
        cv_scores: list[float] = []
        tscv = TimeSeriesSplit(n_splits=3)

        logging.info(f'[{name}] CV (TimeSeriesSplit, 3 folds, IC metric):')
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            if len(val_idx) < 10:
                continue
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            # sklearn/LGBM 可能在 fit 时从 numpy 元数据推断 feature_names_in_，
            # 导致后续 predict(numpy) 产生无意义警告。此处局部 suppress。
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='X does not have valid feature names')
                pred = model.predict(X_val)
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

        # ---- 全量训练 ----
        model = _build_model(model_type)
        model.fit(X, y)
        # 训练集 MSE
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='X does not have valid feature names')
            train_pred = model.predict(X)
        train_mse = float(np.mean((train_pred - y) ** 2))

        # ---- 特征重要性（仅 LGBM） ----
        if model_type == 'lgbm':
            try:
                importances = model.feature_importances_
                top_n = min(10, len(feature_cols))
                top_idx = np.argsort(importances)[-top_n:][::-1]
                top_features = [(feature_cols[i], f'{importances[i]:.4f}') for i in top_idx]
                logging.info(f'[{name}] Feature importance top-{top_n}: {top_features}')
            except Exception:
                pass

        context['trained_model'] = model
        context['is_trained'] = True
        context['days_since_train'] = 0

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
            f'samples={len(y):,}, features={len(feature_cols)}, '
            f'cv_ic_mean={cv_mean:.4f}±{cv_std:.4f}, '
            f'n_folds={len(cv_scores)}, '
            f'predict_day={times[-1]}'
        )

    # ========================================================================
    # 预测阶段
    # ========================================================================
    model = context['trained_model']
    latest_t = times[-1]
    cs_mask_latest = df.index.get_level_values('key') == latest_t
    X_latest = df.loc[cs_mask_latest, feature_cols].to_numpy(dtype=float, copy=True)

    # 检查特征缺失
    nan_rows = np.any(np.isnan(X_latest), axis=1)
    if nan_rows.any():
        logging.warning(
            f'[{name}] Prediction: {nan_rows.sum()}/{len(X_latest)} stocks have NaN features, '
            f'filling with 0'
        )
        X_latest = np.nan_to_num(X_latest, nan=0.0)

    # suppress sklearn feature_names 警告：训练和预测均使用纯 numpy，
    # 但 LGBM/sklearn 可能在 fit 时从 numpy 内部元数据推断 feature_names_in_。
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='X does not have valid feature names')
        pred_raw = model.predict(X_latest)

    # 截面标准化预测信号（保持与训练 label 一致的处理方式）
    pred_signal = _cs_zscore(pred_raw)

    mlflow_log_metrics(mlflow_run_id, name, {
        'pred_signal_min': float(np.min(pred_signal)),
        'pred_signal_max': float(np.max(pred_signal)),
        'pred_signal_skew': float(pd.Series(pred_signal).skew()),
    }, step=trading_step(start_date, latest_t))

    logging.info(
        f'[{name}] Predict signal_day={latest_t} '
        f'(target return: {latest_t}+1->{latest_t}+{fwd}, fwd={fwd}): '
        f'n={len(pred_signal)}, '
        f'mean={float(np.mean(pred_signal)):.4f}, '
        f'std={float(np.std(pred_signal)):.4f}, '
        f'min={float(np.min(pred_signal)):.4f}, '
        f'max={float(np.max(pred_signal)):.4f}'
    )

    result_df = pd.DataFrame({'pred_signal': pred_signal}, index=df.loc[cs_mask_latest].index)
    return Frame3D(result_df)