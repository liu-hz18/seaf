"""
qpipe 工具模块 — 交易日计算、MLflow 日志等跨节点共享函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def snapshot_dataframe(
    run_id: str,
    node_name: str,
    df: pd.DataFrame,
    snapshot_type: str,
    time_key: str,
    artifact_subdir: str = 'snapshots',
) -> None:
    """将 DataFrame 保存为 CSV 并上传到 MLflow artifact。

    Args:
        run_id: MLflow run ID。空字符串时跳过。
        node_name: 节点名称，用于文件命名和 artifact 子目录。
        df: 待保存的 DataFrame。
        snapshot_type: 'in' 或 'out'，标识输入或输出快照。
        time_key: 最新时间片的 key（用于文件命名）。
        artifact_subdir: MLflow artifact 根目录 (默认 'snapshots')。

    文件命名规则: {node_name}_{type}_{time_key}.csv
    存储路径: {artifact_subdir}/{node_name}/{node_name}_{type}_{time_key}.csv
    """
    if not run_id:
        return
    import tempfile
    import os
    from contextlib import suppress

    try:
        import mlflow
        mlflow.set_tracking_uri('sqlite:///mlruns.db')
        filename = f'{node_name}_{snapshot_type}_{time_key}.csv'
        tmp_dir = tempfile.mkdtemp(prefix='snap_')
        tmp_path = os.path.join(tmp_dir, filename)
        try:
            df.to_csv(tmp_path)
            mlflow.log_artifact(tmp_path, artifact_path=f'{artifact_subdir}/{node_name}', run_id=run_id)
        finally:
            with suppress(Exception):
                os.unlink(tmp_path)
            with suppress(Exception):
                os.rmdir(tmp_dir)
    except Exception:
        pass


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
