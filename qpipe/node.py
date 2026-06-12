"""
Multi-process nodes: MultiInputNode and SourceNode.
V2: context + epilogue_fn support, backward compatible.
"""

from __future__ import annotations

import gc
import inspect
import logging
import multiprocessing as mp
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import pandas as pd

from .frame3d import Frame3D
from .utils import FlushStreamHandler, mlflow_log_metrics, snapshot_dataframe, trading_step

try:
    import psutil

    def _rss_mb() -> float:
        return psutil.Process().memory_info().rss / 1024 / 1024
except ImportError:
    def _rss_mb() -> float:
        return 0.0


def _to_float32(f3d: Frame3D) -> Frame3D:
    """将 Frame3D 中所有 float64 列转为 float32，减少序列化和下游缓冲内存。

    注意：仅转换 float64 列，保留 int/bool/object 等非数值列不变。
    NaN 在 float32 下语义等价，无精度风险（因子值域 ±10，7 位有效数字足够）。
    """
    import numpy as np

    df = f3d.df
    changed = False
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32, copy=False)
            changed = True
    if changed:
        return Frame3D(df)
    return f3d

# 因子节点函数签名：兼容 2 参数 (name, f3d) 和 3 参数 (name, f3d, context) 两种形式
FactorFunc = Callable[..., Frame3D]
# Epilogue 函数签名：接收 name, context，无返回值；context 为可变 dict，函数内原地修改
EpilogueFunc = Callable[[str, Any], None]
# Source 生成器函数签名：无参，yield Frame3D
GenFunc = Callable[[], Any]  # 实际是 Iterator[Frame3D]，但 pickle 兼容需要宽松类型


class MultiInputNode(mp.Process):
    """多输入节点：从多个上游 queue 接收 Frame3D，滑动窗口计算，输出到下游。"""

    HEARTBEAT_TIMEOUT: float = 10.0
    THREAD_ROUND_MAX_TIME: int = 3

    def __init__(
        self,
        name: str,
        func: FactorFunc,
        input_queues: list[mp.Queue],
        output_queues: list[mp.Queue],
        window: int = 5,
        min_periods: int = 3,
        input_columns: list[str] | None = None,
        output_columns: list[str] | None = None,
        exclude_input_columns: list[str] | None = None,
        stop_signal: Any = None,
        context: Any = None,
        epilogue_fn: EpilogueFunc | None = None,
        output_queue_names: list[str] | None = None,
        snapshot_interval: int = 0,
        log_level: str='INFO',
    ) -> None:
        super().__init__()
        self.name = name
        self.func = func
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.output_queue_names = output_queue_names if output_queue_names else []
        self.window = window
        self.min_periods = min_periods
        self.input_columns = input_columns if input_columns else []
        self.output_columns = output_columns if output_columns else []
        _exc = exclude_input_columns if exclude_input_columns else []
        # 验证 input_columns / exclude_input_columns 无交集
        _conflict = set(self.input_columns) & set(_exc)
        if _conflict:
            raise ValueError(
                f'[{name}] input_columns ∩ exclude_input_columns conflict: {_conflict}'
            )
        self.exclude_input_columns = _exc
        self.stop_signal = stop_signal
        self.context = context if context is not None else {}
        self.epilogue_fn = epilogue_fn
        self.snapshot_interval = snapshot_interval
        self.log_level = log_level
        self.buffers: list[dict[Any, Frame3D]] = [{} for _ in input_queues]

    def _call_func(self, name: str, f3d: Frame3D, ctx: Any) -> Frame3D:
        """调用节点函数，兼容 2 参数和 3 参数签名。

        只捕获 TypeError（签名参数数量不匹配时回退到 2 参数调用）。
        ValueError 等运行时错误直接向上传播，不在框架层吞没。
        """
        try:
            sig = inspect.signature(self.func)
            if len(sig.parameters) >= 3:
                return self.func(name, f3d, ctx)
            return self.func(name, f3d)
        except TypeError:
            return self.func(name, f3d)

    def receive_worker(
        self,
        queue_idx: int,
        input_queue: mp.Queue,
        ready_event: threading.Event,
        global_exit: threading.Event,
        data_lock: threading.Lock,
        heartbeat_timestamp: list[float],
        heartbeat_lock: threading.Lock,
    ) -> None:
        """接收线程：阻塞等待 queue 数据，存入 buffer。"""
        while not global_exit.is_set():
            with heartbeat_lock:
                heartbeat_timestamp[queue_idx] = time.time()
            try:
                obj = input_queue.get(timeout=0.5)
            except Exception:
                continue
            if obj == self.stop_signal:
                logging.debug(f'[{self.name}][thread-{queue_idx}] stop signal.')
                ready_event.set()
                break
            time_value = obj.df.index.get_level_values(0)[0]
            with data_lock:
                self.buffers[queue_idx][time_value] = obj
                ready_event.set()

    def run(self) -> None:
        """子进程主入口：协调接收线程和窗口计算循环。"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=f'[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d][{self.name}] %(message)s',
            handlers=[FlushStreamHandler(stream=sys.stdout)],
        )
        logging.info(f'Node {self.name} started, w={self.window}, mp={self.min_periods}.')

        num_workers = len(self.input_queues)
        data_lock = threading.Lock()
        ready_event = [threading.Event() for _ in range(num_workers)]
        heartbeat_lock = threading.Lock()
        heartbeat_timestamp = [time.time() for _ in range(num_workers)]
        global_exit = threading.Event()

        threads: list[threading.Thread] = []
        for i, inq in enumerate(self.input_queues):
            t = threading.Thread(
                target=self.receive_worker,
                args=(
                    i,
                    inq,
                    ready_event[i],
                    global_exit,
                    data_lock,
                    heartbeat_timestamp,
                    heartbeat_lock,
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)

        dead_workers: set[int] = set()
        time_order_buffer: deque[tuple[Any, Frame3D]] = deque()
        window_start_index = 0
        window_tail_index = 0
        round_time = 0
        current_context: Any = self.context
        dup_col_have_warned = False

        try:
            while True:
                alive_workers_to_wait = [i for i in range(num_workers) if i not in dead_workers]
                thread_round_time = 0
                while alive_workers_to_wait and thread_round_time < self.THREAD_ROUND_MAX_TIME:
                    ready_workers = [i for i in alive_workers_to_wait if ready_event[i].is_set()]
                    still_waiting = [i for i in alive_workers_to_wait if i not in ready_workers]
                    newly_dead: list[int] = []
                    for i in still_waiting:
                        with heartbeat_lock:
                            if time.time() - heartbeat_timestamp[i] > self.HEARTBEAT_TIMEOUT:
                                newly_dead.append(i)
                    if newly_dead:
                        for i in newly_dead:
                            dead_workers.add(i)
                            alive_workers_to_wait.remove(i)
                            still_waiting.remove(i)
                    if len(ready_workers) == len(alive_workers_to_wait):
                        break
                    time.sleep(0.2)
                    thread_round_time += 1

                submitted_workers: list[int] = []
                for i in range(num_workers):
                    if ready_event[i].is_set():
                        submitted_workers.append(i)

                with data_lock:
                    all_times_sets = [set(self.buffers[i].keys()) for i in range(num_workers)]
                shared_times = set.intersection(*all_times_sets) if all_times_sets else set()

                frame_lists: list[list[Frame3D]] = []
                if shared_times:
                    for tval in sorted(shared_times):
                        with data_lock:
                            # TODO: model 节点一般输出的key会晚一些，导致接收其他节点输出数据的buffer中可能有过早的数据永远得不到 delete
                            frame_list = [self.buffers[bi][tval] for bi in range(len(self.buffers))]
                            for bi in range(len(self.buffers)):
                                del self.buffers[bi][tval]
                        frame_lists.append(frame_list)

                for i in submitted_workers:
                    ready_event[i].clear()

                for frame_list in frame_lists:
                    df_list = [f3d.df for f3d in frame_list]
                    merged_df = pd.concat(df_list, axis=1)
                    # 去重：多路输入可能带重复列。
                    # - 数值/因子列重名 → 硬报错（防止因子冲突导致数据静默丢失）
                    # - 元数据列（stock_name 等）重名 → 日志记录 + 保留第一份
                    DEDUP_SAFE_COLS = {'stock_name'}
                    dup_cols = merged_df.columns[merged_df.columns.duplicated()].tolist()
                    if dup_cols:
                        safe_dup = [c for c in dup_cols if c in DEDUP_SAFE_COLS]
                        bad_dup = [c for c in dup_cols if c not in DEDUP_SAFE_COLS]
                        if bad_dup:
                            raise ValueError(
                                f'[{self.name}] Duplicate numeric/factor columns '
                                f'after merge: {bad_dup}'
                            )
                        if safe_dup and not dup_col_have_warned:
                            logging.warning(
                                f'[{self.name}] Deduplicating {safe_dup} '
                                f'(kept first of {dup_cols.count(safe_dup[0]) + 1} copies)'
                            )
                            dup_col_have_warned = True
                        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
                    if self.input_columns:
                        miss = [c for c in self.input_columns if c not in merged_df.columns]
                        if miss:
                            raise ValueError(f'[{self.name}] Missing input cols: {miss}')
                        # 保留非数值元数据列（stock_name 等），确保下游 CSV 可读
                        meta = [c for c in merged_df.columns
                                if c not in self.input_columns
                                and not pd.api.types.is_numeric_dtype(merged_df[c])]
                        keep = self.input_columns + meta
                        merged_df = merged_df[keep]
                    # 排除列：在入队前删除，防止泄露到下游节点
                    if self.exclude_input_columns:
                        drop_cols = [c for c in self.exclude_input_columns if c in merged_df.columns]
                        if drop_cols:
                            merged_df = merged_df.drop(columns=drop_cols)
                    time_order_buffer.append(
                        (shared_times.pop() if shared_times else None, Frame3D(merged_df))
                    )

                if len(time_order_buffer) > 0:
                    logging.debug(
                        f'time_order_buffer: {len(time_order_buffer)} '
                        f'[{time_order_buffer[0][0]}, {time_order_buffer[-1][0]}]'
                    )

                if len(time_order_buffer) < self.min_periods:
                    if len(dead_workers) == num_workers:
                        logging.info('All workers dead, insufficient data. Exiting.')
                        break
                    continue

                while window_tail_index - window_start_index <= len(time_order_buffer):
                    while window_tail_index - window_start_index < min(
                        self.min_periods, len(time_order_buffer)
                    ):
                        window_tail_index += 1
                    while window_tail_index - window_start_index > self.window:
                        window_start_index += 1
                        time_order_buffer.popleft()
                    window_length = window_tail_index - window_start_index
                    if window_length < self.min_periods:
                        continue
                    if len(time_order_buffer) < window_length:
                        break
                    window_frames = list(time_order_buffer)[:window_length]

                    start_time = time.time()
                    # 拼接窗口内所有时间片的快照
                    raw_parts = [f[-1].df for f in window_frames]
                    window_df = pd.concat(raw_parts, axis=0)
                    window_df = window_df.sort_index(level=0)

                    # === IPO/退市对齐：以最新时间片的股票集合为准 ===
                    # 最新时间片的股票集合是"当前市场"的权威集合。
                    # - 退市股票：最新片中不存在，前序时间片中应删除。
                    # - 新上市股票：最新片中存在但前序片中不存在，前序片补 NaN。
                    #   用 NaN 而非 0.0，避免 np.log(0)/0/0 等下游运算产生
                    #   "divide by zero in log" / "All-NaN slice" 警告。
                    #
                    #   优化：使用 MultiIndex.from_product 一次性 reindex，
                    #   避免逐时间片循环产生 O(T) 个中间 DataFrame。
                    latest_t = window_df.index.get_level_values(0).max()
                    latest_stocks = window_df.loc[latest_t].index.tolist()
                    all_times = sorted(window_df.index.get_level_values(0).unique())
                    full_mi = pd.MultiIndex.from_product(
                        [all_times, latest_stocks], names=window_df.index.names
                    )
                    window_df = window_df.reindex(full_mi).sort_index(level=0)
                    # === IPO/退市对齐结束 ===

                    run_input_f3d = Frame3D(window_df)

                    # ---- 快照：输入（_call_func 之前，防止原地修改） ----
                    total_calls = current_context.setdefault('_node_call_count', 0) + 1
                    current_context['_node_call_count'] = total_calls
                    run_id_s = current_context.get('mlflow_run_id', '')
                    ts = str(latest_t)[:10] if hasattr(latest_t, '__str__') else str(latest_t)
                    day_index = trading_step(current_context.get('start_date', ''), latest_t)
                    if self.snapshot_interval > 0 and day_index > 0 and day_index % self.snapshot_interval == 0:
                        snapshot_dataframe(run_id_s, self.name, run_input_f3d.df, 'in', ts)

                    output_f3d = self._call_func(self.name, run_input_f3d, current_context)
                    # 保留输入中的元数据列（name/stock_id 等字符串列），确保下游 CSV 可读
                    meta_cols = [c for c in run_input_f3d.df.columns
                                 if c not in output_f3d.df.columns
                                 and not pd.api.types.is_numeric_dtype(run_input_f3d.df[c])]
                    if meta_cols:
                        meta_df = run_input_f3d.df[meta_cols].loc[output_f3d.df.index]
                        output_df = pd.concat([output_f3d.df, meta_df], axis=1)
                        output_f3d = Frame3D(output_df)
                    if self.output_columns:
                        miss_o = [c for c in self.output_columns if c not in output_f3d.df.columns]
                        if miss_o:
                            raise ValueError(f'[{self.name}] Missing output cols: {miss_o}')
                        filtered = output_f3d.df[self.output_columns]
                    else:
                        filtered = output_f3d.df
                    result_f3d = Frame3D(filtered.copy())
                    max_key = result_f3d.df.index.get_level_values(0).max()
                    latest_df = result_f3d.df[result_f3d.df.index.get_level_values(0) == max_key]
                    latest_f3d = Frame3D(latest_df.copy())
                    # float32 输出：将队列传输数据精度降为 fp32，内存减半
                    latest_f3d = _to_float32(latest_f3d)
                    logging.debug(
                        f'window_start_index:{window_start_index}, '
                        f'window_tail_index={window_tail_index}\n'
                        f'time_order_buffer: {len(time_order_buffer)}\n'
                        f'output frame: {latest_f3d}'
                    )
                    elapsed = time.time() - start_time

                    timings: list[float] = current_context.setdefault('_node_timings', [])
                    timings.append(elapsed)
                    if len(timings) > 10:
                        timings.pop(0)
                    avg_time = sum(timings) / len(timings)

                    # ---- 快照：输出（复用 latest_f3d.df） ----
                    if self.snapshot_interval > 0 and day_index > 0 and day_index % self.snapshot_interval == 0:
                        snapshot_dataframe(run_id_s, self.name, latest_f3d.df, 'out', ts)
                        logging.info(
                            f'snapshot day_index={day_index} (call#{total_calls}): '
                            f'in={run_input_f3d.df.shape}, out={latest_f3d.df.shape}'
                        )

                    if total_calls % 10 == 0 or total_calls == 1:
                        logging.info(
                            f'call#{total_calls}: '
                            f'elapsed={elapsed:.3f}s, '
                            f'rolling_avg10={avg_time:.3f}s'
                        )

                    window_tail_index += 1
                    for outq in self.output_queues:
                        outq.put(latest_f3d)

                    # ---- 内存回收与监控 ----
                    # 每次计算后强制 GC，回收因子函数产生的临时 DataFrame。
                    # 生产环境（6000 stocks）单次迭代可能产生 100MB+ 临时对象。
                    rss = _rss_mb()
                    if total_calls % 10 == 0:
                        gc.collect()
                        buf_sizes = {f'b{i}': len(b) for i, b in enumerate(self.buffers)}
                        logging.info(
                            f'mem#{total_calls}: rss={rss:.0f}MB, '
                            f'tob_len={len(time_order_buffer)}, '
                            f'buf_sizes={buf_sizes}'
                        )

                    # ---- MLflow: 记录本节点运行时间和输出队列大小 ----
                    run_id: str = current_context.get('mlflow_run_id', '')
                    step = trading_step(current_context.get('start_date', ''), max_key)
                    if run_id:
                        queue_sizes = {}
                        for qi, q in enumerate(self.output_queues):
                            qname = (
                                self.output_queue_names[qi]
                                if qi < len(self.output_queue_names)
                                else f'q{qi}'
                            )
                            with suppress(Exception):
                                queue_sizes[f'queue_{qname}'] = float(q.qsize())
                        mlflow_log_metrics(
                            run_id,
                            self.name,
                            {
                                'elapsed_ms': elapsed * 1000,
                                'rss_mb': rss,
                                'time_order_buffer_len': float(len(time_order_buffer)),
                                **queue_sizes,
                            },
                            step=step,
                        )

                if len(dead_workers) == num_workers:
                    logging.info('All workers dead. Node process exited.')
                    break
                round_time += 1

        except Exception as e:
            logging.error(f'Exception in {self.name}: {e}', exc_info=True)
        finally:
            if self.epilogue_fn is not None:
                try:
                    self.epilogue_fn(self.name, current_context)
                except Exception as e:
                    logging.error(f'Epilogue error in {self.name}: {e}', exc_info=True)
            global_exit.set()
            for t in threads:
                t.join(timeout=1)
            for outq in self.output_queues:
                outq.put(self.stop_signal)
            logging.info(f'Node {self.name} stopped.')


class SourceNode(mp.Process):
    """数据源节点：迭代 gen_func，逐日输出 Frame3D 到下游 queue。"""

    def __init__(
        self,
        name: str,
        gen_func: GenFunc,
        output_queues: list[mp.Queue],
        stop_signal: Any = None,
        context: Any = None,
        epilogue_fn: EpilogueFunc | None = None,
        output_queue_names: list[str] | None = None,
        snapshot_interval: int = 0,
        log_level: str='INFO',
    ) -> None:
        super().__init__()
        self.name = name
        self.gen_func = gen_func
        self.output_queues = output_queues
        self.output_queue_names = output_queue_names if output_queue_names else []
        self.stop_signal = stop_signal
        self.context = context if context is not None else {}
        self.epilogue_fn = epilogue_fn
        self.snapshot_interval = snapshot_interval
        self.log_level = log_level

    def run(self) -> None:
        """子进程主入口：迭代数据生成器，逐日输出最新截面。"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=f'[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d][{self.name}] %(message)s',
            handlers=[FlushStreamHandler(stream=sys.stdout)],
        )
        logging.info('SourceNode started.')
        current_context: Any = self.context
        day_index = 0
        try:
            for frame in self.gen_func():
                max_key = frame.df.index.get_level_values(0).max()
                latest_df = frame.df[frame.df.index.get_level_values(0) == max_key]
                latest_f3d = Frame3D(latest_df.copy())
                # float32 输出：源数据精度降为 fp32，所有下游队列和缓冲受益
                latest_f3d = _to_float32(latest_f3d)

                day_index = trading_step(current_context.get('start_date', ''), max_key)
                # ---- 快照采样：按交易日序号对齐（替代私有 call_count） ----
                if self.snapshot_interval > 0 and day_index > 0 and day_index % self.snapshot_interval == 0:
                    run_id = current_context.get('mlflow_run_id', '')
                    time_str = str(max_key)[:10] if hasattr(max_key, '__str__') else str(max_key)
                    # 输入快照：数据生成器产出的当日完整 frame
                    snapshot_dataframe(
                        run_id, self.name, frame.df, 'in', time_str,
                    )
                    # 输出快照：最新时间片
                    snapshot_dataframe(
                        run_id, self.name, latest_f3d.df, 'out', time_str,
                    )
                    logging.info(
                        f'snapshot day_index={day_index}: '
                        f'in={frame.df.shape}, out={latest_f3d.df.shape}'
                    )

                for outq in self.output_queues:
                    outq.put(latest_f3d)
                # ---- MLflow: 记录源节点输出队列大小 ----
                run_id = current_context.get('mlflow_run_id', '')
                if run_id:
                    qs = {}
                    for qi, q in enumerate(self.output_queues):
                        qname = (
                            self.output_queue_names[qi]
                            if qi < len(self.output_queue_names)
                            else f'q{qi}'
                        )
                        qs[f'queue_{qname}'] = float(q.qsize())
                    step = trading_step(current_context.get('start_date', ''), max_key)
                    mlflow_log_metrics(run_id, self.name, qs, step=step)
                # 数据生成器可能在内存中持有大数组（hidden_factors / noise），
                # 但生成器自身不累积逐日数据。此处 gc 为防御性措施。
                gc.collect()
            for outq in self.output_queues:
                outq.put(self.stop_signal)
        except Exception as e:
            logging.error(f'Exception in SourceNode: {e}', exc_info=True)
        finally:
            if self.epilogue_fn is not None:
                try:
                    self.epilogue_fn(self.name, current_context)
                except Exception as e:
                    logging.error(f'Epilogue error in {self.name}: {e}', exc_info=True)
            logging.info('SourceNode stopped.')
