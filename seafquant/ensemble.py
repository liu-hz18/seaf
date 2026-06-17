"""
集成节点 — 多模型信号等权融合（bagging）。

输入：单个被 qpipe/node.py 合并后的 Frame3D（含多个 pred_signal_* 列）。
输出：等权平均后的单一 pred_signal Frame3D。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from qpipe.frame3d import Frame3D
from qpipe.utils import _cs_zscore, mlflow_log_metrics


def ensemble_fn(name: str, idx: int, f3d: Frame3D, context: dict[str, Any] | None = None) -> Frame3D:
    """多模型信号等权融合。

    接收单个已合并的 Frame3D，查找所有 pred_signal_ 开头的列，
    等权平均后生成单一的 pred_signal 列。
    """
    if f3d is None or f3d.df.empty:
        raise ValueError('ensemble_fn requires a non-empty Frame3D')

    df = f3d.df
    signal_cols = [c for c in df.columns if c.startswith('pred_signal_')]

    if not signal_cols:
        # 单模型回退：直接返回 pred_signal（或第一个 pred_signal_* 列）
        if 'pred_signal' in df.columns:
            signal_cols = ['pred_signal']
        else:
            raise ValueError(
                f'no pred_signal or pred_signal_* columns found in ensemble input. '
                f'Available columns: {list(df.columns)[:10]}'
            )

    # 等权平均所有信号列
    signals = df[signal_cols].values.astype(float)
    ensemble_signal = np.nanmean(signals, axis=1)
    # zscore
    ensemble_signal = _cs_zscore(ensemble_signal)

    result_df = pd.DataFrame({'pred_signal': ensemble_signal}, index=df.index)
    result = Frame3D(result_df)

    n_models = len(signal_cols)
    mlflow_run_id: str = context.get('mlflow_run_id', '')
    times = sorted(df.index.get_level_values('key').unique())
    latest_t = times[-1]
    mlflow_log_metrics(mlflow_run_id, name, {
        'pred_signal_min': float(np.min(ensemble_signal)),
        'pred_signal_max': float(np.max(ensemble_signal)),
        'pred_signal_skew': float(pd.Series(ensemble_signal).skew()),
    }, step=idx)

    logging.info(
        f'[{idx}] Ensemble {n_models} model, day={latest_t}: '
        f'n={len(ensemble_signal)}, '
        f'mean={float(np.mean(ensemble_signal)):.4f}, '
        f'std={float(np.std(ensemble_signal)):.4f}, '
        f'min={float(np.min(ensemble_signal)):.4f}, '
        f'max={float(np.max(ensemble_signal)):.4f}, '
        f'skew={float(pd.Series(ensemble_signal).skew()):.4f}'
    )

    return result


def ensemble_epilogue(name: str, context: dict[str, Any]) -> None:
    """集成 epilogue — 记录融合模型的元信息。"""
    context.pop('mlflow_name', None)
    context.pop('precision', None)
    context.pop('start_date', None)
    context.pop('fwd', None)
    if context:
        logging.info(f'ensemble context: {context}')
