"""
qpipe 工具模块 — 交易日计算、MLflow 日志等跨节点共享函数。
"""

from __future__ import annotations

import numpy as np


def trading_step(start_date: str, dt) -> int:
    """交易日序号：start_date → dt 之间的工作日天数（0 = 起始日期当天）。

    dt 可以是 datetime、Timestamp 或任何能被 np.datetime64 转换的类型。
    跳过非交易日（周末），仅计算工作日天数。
    """
    if not start_date:
        return 0
    # 统一转为 day 精度（D），避免 pandas Timestamp 的 us/ns 精度与 busday_count 不兼容
    return int(
        np.busday_count(
            np.datetime64(start_date, 'D'),
            np.datetime64(str(dt)[:10], 'D'),
        )
    )


def mlflow_log_metrics(
    mlflow_run_id: str, prefix: str, metrics: dict[str, float], step: int = 0
) -> None:
    """子进程安全的 MLflow 指标写入。

    直接使用 run_id 调用 log_metric，无需 start_run/end_run 上下文管理。
    metric 名称格式：`{prefix}.{key}`。
    沉默失败——MLflow 日志丢失不应中断主流程。
    """
    if not mlflow_run_id:
        return
    try:
        import mlflow

        mlflow.set_tracking_uri('sqlite:///mlruns.db')
        for k, v in metrics.items():
            mlflow.log_metric(f'{prefix}.{k}', v, step=step, run_id=mlflow_run_id)
    except Exception:
        pass
