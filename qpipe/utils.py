"""
qpipe 工具模块 — 交易日计算、MLflow 日志等跨节点共享函数。
"""

from __future__ import annotations

import logging
import pickle
import sqlite3
import time
from contextlib import contextmanager
from queue import Empty, Full
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import numpy as np

try:
    import psutil

    def _rss_mb() -> float:
        return psutil.Process().memory_info().rss / 1024 / 1024
except ImportError as e:
    print(e)

    def _rss_mb() -> float:
        return -1.0


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  # 每次输出后立刻刷新


# NOTE: 废弃，瓶颈不在 queue 中，因为 下游进程设置了 接收 thread, 很快就把数据获取到下游节点的 buffer 中了
class PersistentQueue:
    """数据存储在磁盘，不占用内存，支持多进程，支持 maxsize"""

    def __init__(self, name: str, db_path='queue.db', maxsize=0):
        self.db_path = db_path
        self.name = name
        self.maxsize = maxsize  # 0 表示无限大小
        self._init_db()

    @contextmanager
    def _get_conn(self):
        # 增加超时时间，防止高并发下锁等待超时
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute('PRAGMA journal_mode=WAL')  # 支持并发
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_conn() as conn:
            # 简单的防注入：确保表名仅包含合法字符（实际项目中建议使用固定表名或白名单）
            if not self.name.isidentifier():
                raise ValueError(f"Invalid table name: {self.name}")
            conn.execute(f'CREATE TABLE IF NOT EXISTS {self.name} (id INTEGER PRIMARY KEY, data BLOB)')
            conn.commit()

    def put(self, item, block=True, timeout=None):
        """
        将数据放入队列。
        如果队列已满：
        - block=True 时阻塞，直到有空位或超时。
        - block=False 时立即抛出 queue.Full 异常。
        """
        # 如果 maxsize 为 0，表示无限大小，直接插入
        if self.maxsize <= 0:
            with self._get_conn() as conn:
                conn.execute(f'INSERT INTO {self.name} (data) VALUES (?)', (pickle.dumps(item),))
                conn.commit()
            return
        start_time = time.time()
        while True:
            try:
                with self._get_conn() as conn:
                    # 关键点：使用 IMMEDIATE 事务立即获取写入锁。
                    # 这确保了 "检查大小" 和 "插入" 是一个原子操作，
                    # 防止其他进程在我们检查后、插入前抢占了位置。
                    conn.execute('BEGIN IMMEDIATE')
                    # 检查当前大小
                    current_size = conn.execute(f'SELECT COUNT(*) FROM {self.name}').fetchone()[0]
                    if current_size < self.maxsize:
                        conn.execute(f'INSERT INTO {self.name} (data) VALUES (?)', (pickle.dumps(item),))
                        conn.commit() # 提交事务，释放锁
                        return
                    # 如果代码走到这里，说明队列已满。
                    # 手动回滚以释放锁（虽然我们没有修改数据，但事务还在进行中）
                    conn.rollback()
                    if not block:
                        raise Full(f"Queue {self.name} is full")
                    # 处理超时逻辑
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        if elapsed >= timeout:
                            raise Full(f"Queue {self.name} is full, timeout reached")
                    # 短暂等待后重试，避免 CPU 空转
                    time.sleep(0.01)
            except sqlite3.OperationalError as e:
                # 如果数据库被锁定（极少数情况下的重试逻辑），稍作等待
                if "database is locked" in str(e):
                    time.sleep(0.05)
                else:
                    raise

    def get(self, block=True, timeout=None):
        """
        从队列获取数据。
        """
        start_time = time.time()
        while True:
            try:
                with self._get_conn() as conn:
                    # 使用 BEGIN IMMEDIATE 确保读取并删除的原子性
                    conn.execute('BEGIN IMMEDIATE')
                    row = conn.execute(
                        f'SELECT id, data FROM {self.name} ORDER BY id LIMIT 1'
                    ).fetchone()
                    if row:
                        conn.execute(f'DELETE FROM {self.name} WHERE id=?', (row[0],))
                        conn.commit()
                        return pickle.loads(row[1])
                    # 队列为空，回滚事务释放锁
                    conn.rollback()
                    if not block:
                        raise Empty(f"Queue {self.name} is empty")
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        if elapsed >= timeout:
                            raise Empty(f"Queue {self.name} is empty, timeout reached")
                    time.sleep(0.01)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    time.sleep(0.05)
                else:
                    raise

    def qsize(self) -> int:
        # 注意：在高并发下，这个值可能不是实时的，仅供参考
        with self._get_conn() as conn:
            return conn.execute(f'SELECT COUNT(*) FROM {self.name}').fetchone()[0]

    def __len__(self) -> int:
        return self.qsize()


def snapshot_dataframe(
    run_id: str,
    node_name: str,
    df: pd.DataFrame,
    snapshot_type: str,
    time_key: str,
    artifact_subdir: str = 'snapshots',
    gzip: bool = False,
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
    import os
    import tempfile
    from contextlib import suppress

    try:
        import mlflow
        mlflow.set_tracking_uri('sqlite:///mlruns.db')
        filename = f'{node_name}_{snapshot_type}_{time_key}.csv'
        tmp_dir = tempfile.mkdtemp(prefix='snap_')
        tmp_path = os.path.join(tmp_dir, filename)
        try:
            df.to_csv(tmp_path, compression='gzip' if gzip else None)
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


def _cs_zscore(values: np.ndarray) -> np.ndarray:
    """截面标准化：(x - mean) / std，std=0 时返回零向量。"""
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std > 0:
        return (values - mean) / std
    return np.zeros_like(values)
