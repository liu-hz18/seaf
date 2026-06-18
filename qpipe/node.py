"""
Multi-process nodes: MultiInputNode and SourceNode.
V2: context + epilogue_fn support, backward compatible.
"""

from __future__ import annotations

import gc
import inspect
import logging
import multiprocessing as mp
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import suppress
from typing import Any

import numpy as np
import pandas as pd

from .frame3d import Frame3D
from .utils import FlushStreamHandler, _rss_mb, mlflow_log_metrics, snapshot_dataframe

# 因子节点函数签名：兼容 3 参数 (name, idx, f3d) 和 4 参数 (name, idx, f3d, context) 两种形式
FactorFunc = Callable[[str, int, Frame3D, dict | None], Frame3D]
# Epilogue 函数签名：接收 name, context，无返回值；context 为可变 dict，函数内原地修改
EpilogueFunc = Callable[[[str, int, dict | None], Any], None]
# Source 生成器函数签名：无参，yield Frame3D
GenFunc = Callable[[], Iterator[tuple[int, Frame3D]]]  # 实际是 Iterator[Frame3D]，但 pickle 兼容需要宽松类型

MEM_CLEARING_INTEVAL = 10
TIMEING_INTEVAL = 10


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
        log_level: str = 'INFO',
    ) -> None:
        super().__init__()
        self.name = name
        self.gen_func = gen_func
        self.output_queues = output_queues
        self.output_queue_names = output_queue_names or []
        self.stop_signal = stop_signal
        self.context = context if context is not None else {}
        self.epilogue_fn = epilogue_fn
        self.snapshot_interval = snapshot_interval
        self.log_level = log_level

    def run(self) -> None:
        """子进程主入口：迭代数据生成器，逐日输出最新截面。"""
        # ---- 日志文件：子进程也写入同一个 logs/{run_id}.txt ----
        mlflow_name = (
            self.context.get('mlflow_name', 'test') if isinstance(self.context, dict) else 'test'
        )
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=f'[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d][{self.name}] %(message)s',
            handlers=[
                FlushStreamHandler(stream=sys.stdout),
                logging.FileHandler(f'logs/{mlflow_name}.txt', encoding='utf-8'),
            ],
        )
        logging.info('SourceNode started.')
        current_context: Any = self.context
        try:
            for idx, frame in self.gen_func():
                max_key = frame.last_key()
                latest_f3d = frame.last_frame().to(np.float32)

                # ---- 快照采样 ----
                if self.snapshot_interval > 0 and idx > 0 and idx % self.snapshot_interval == 0:
                    run_id = current_context.get('mlflow_run_id', '')
                    time_str = str(max_key)[:10] if hasattr(max_key, '__str__') else str(max_key)
                    snapshot_dataframe(run_id, self.name, frame.df, 'in', time_str)
                    snapshot_dataframe(run_id, self.name, latest_f3d.df, 'out', time_str)
                    logging.info(
                        f'snapshot idx={idx}: in={frame.df.shape}, out={latest_f3d.df.shape}'
                    )

                for outq in self.output_queues:
                    outq.put((idx, latest_f3d))

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
                    mlflow_log_metrics(run_id, self.name, qs, step=idx)

                # 数据生成器可能在内存中持有大数组（hidden_factors / noise），
                # 但生成器自身不累积逐日数据。此处 gc 为防御性措施。
                if idx % MEM_CLEARING_INTEVAL == 0:
                    gc.collect()

            for outq in self.output_queues:
                with suppress(Exception):
                    outq.put(self.stop_signal, timeout=1.0)
                # with suppress(Exception):
                #     outq.cancel_join_thread()
        except Exception as e:
            logging.error(f'Exception in SourceNode: {e}', exc_info=True)
        finally:
            if self.epilogue_fn is not None:
                try:
                    self.epilogue_fn(self.name, idx, current_context)
                except Exception as e:
                    logging.error(f'Epilogue error in {self.name}: {e}', exc_info=True)
            for handler in logging.getLogger().handlers:
                with suppress(Exception):
                    handler.flush()
            sys.stdout.flush()
            sys.stderr.flush()
            logging.info('SourceNode stopped.')


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
        stop_signal: str | None = None,
        context: dict | None = None,
        epilogue_fn: EpilogueFunc | None = None,
        output_queue_names: list[str] | None = None,
        time_alignment: str = 'right',
        snapshot_interval: int = 0,
        log_level: str = 'INFO',
    ) -> None:
        super().__init__()
        self.name = name
        self.func = func
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.output_queue_names = output_queue_names or []
        self.window = window
        self.min_periods = min_periods
        self.input_columns = input_columns or []
        self.output_columns = output_columns or []
        _exc = exclude_input_columns or []
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
        self.time_alignment = time_alignment
        assert time_alignment in ['left', 'right']
        self.snapshot_interval = snapshot_interval
        self.log_level = log_level
        self.buffers: list[dict[Any, Frame3D]] = [{} for _ in input_queues]
        # 状态遍历
        self.dup_col_have_warned = False

    def run(self) -> None:
        """子进程主入口：协调接收线程和窗口计算循环。"""
        mlflow_name = (
            self.context.get('mlflow_name', 'test') if isinstance(self.context, dict) else 'test'
        )
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=f'[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d][{self.name}] %(message)s',
            handlers=[
                FlushStreamHandler(stream=sys.stdout),
                logging.FileHandler(f'logs/{mlflow_name}.txt', encoding='utf-8'),
            ],
        )
        logging.info(f'Node {self.name} started, window={self.window}, min_periods={self.min_periods}.')

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
                args=(i, inq, ready_event[i], global_exit, data_lock, heartbeat_timestamp, heartbeat_lock,),
                daemon=True,
            )
            t.start()
            threads.append(t)

        dead_workers: set[int] = set()
        time_order_buffer: deque[tuple[int, Frame3D]] = deque()
        window_start_index = 0
        window_tail_index = 0
        round_time = 0

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
                            diff = time.time() - heartbeat_timestamp[i]
                            if diff > self.HEARTBEAT_TIMEOUT:
                                newly_dead.append(i)
                    if newly_dead:
                        for i in newly_dead:
                            dead_workers.add(i)
                            alive_workers_to_wait.remove(i)
                            still_waiting.remove(i)
                    if len(ready_workers) == len(alive_workers_to_wait):
                        break
                    time.sleep(0.05)
                    thread_round_time += 1

                submitted_workers: list[int] = []
                for i in range(num_workers):
                    if ready_event[i].is_set():
                        submitted_workers.append(i)

                with data_lock:
                    idx_frame_dict = self.receive_common_idx_frames(self.buffers)

                for i in submitted_workers:
                    ready_event[i].clear()

                # [列属性维度的拼接] 遍历上游收集到的数据，进行必要的检查，合并多源上游的 dataframe，然后导入本地队列
                for tval, frame_list in idx_frame_dict.items():
                    frame = self.concat_frames(frame_list, tval)
                    frame = self.clean_frame(frame)
                    # 进入队列
                    time_order_buffer.append((tval, frame))

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

                # 滑动窗口：时间维度的拼接
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
                    # 获得最新时间片的 idx
                    day_idx, _ = window_frames[-1]
                    concated_window_frame = self.concat_and_align_frames_through_times(
                        window_frames
                    )

                    logging.debug(f'[{day_idx}] input frame:\n{concated_window_frame}')

                    # ---- 快照：输入（_call_func 之前，防止原地修改） ----
                    total_calls = self.context.setdefault('_node_call_count', 0) + 1
                    self.context['_node_call_count'] = total_calls
                    run_id_s = self.context.get('mlflow_run_id', '')
                    latest_t = concated_window_frame.last_key()
                    ts = str(latest_t)[:10] if hasattr(latest_t, '__str__') else str(latest_t)
                    if (
                        self.snapshot_interval > 0
                        and day_idx > 0
                        and day_idx % self.snapshot_interval == 0
                    ):
                        snapshot_dataframe(run_id_s, self.name, concated_window_frame.df, 'in', ts)

                    # 执行节点计算
                    output_frame = self._call_func(
                        self.name, day_idx, concated_window_frame, self.context
                    )
                    output_frame = self.clean_output_frame(output_frame, concated_window_frame)

                    # 只传输最后一个 frame, 并转换为 fp32
                    latest_frame = output_frame.last_frame().to(np.float32)
                    logging.debug(
                        f'[{day_idx}][{ts}] window_start_index:{window_start_index}, '
                        f'window_tail_index={window_tail_index}\n'
                        f'time_order_buffer: {len(time_order_buffer)}\n'
                        f'output frame: {latest_frame}'
                    )
                    elapsed = time.time() - start_time

                    timings: list[float] = self.context.setdefault('_node_timings', [])
                    timings.append(elapsed)
                    if len(timings) > TIMEING_INTEVAL:
                        timings.pop(0)
                    avg_time = sum(timings) / len(timings)

                    # ---- 快照：输出（复用 latest_f3d.df） ----
                    if (
                        self.snapshot_interval > 0
                        and day_idx > 0
                        and day_idx % self.snapshot_interval == 0
                    ):
                        snapshot_dataframe(run_id_s, self.name, latest_frame.df, 'out', ts)
                        logging.info(
                            f'[{day_idx}][{ts}][snapshot] (call#{total_calls}): '
                            f'in={concated_window_frame.df.shape}, out={latest_frame.df.shape}'
                        )

                    if total_calls % TIMEING_INTEVAL == 0 or total_calls == 1:
                        logging.info(
                            f'[{day_idx}][{ts}] call#{total_calls}: '
                            f'elapsed={elapsed:.3f}s, '
                            f'rolling_avg{TIMEING_INTEVAL}={avg_time:.3f}s'
                        )

                    window_tail_index += 1

                    # 输出 (idx, latest_frame) 元组 — idx 就是当前 frame_list 的时间 day_idx
                    for outq in self.output_queues:
                        outq.put((day_idx, latest_frame))

                    queue_sizes = {
                        f'outq_{qi}': outq.qsize() for qi in range(len(self.output_queues))
                    }
                    logging.debug(f'[{day_idx}][{ts}] Output queue: {queue_sizes}')

                    # ---- 内存回收与监控 ----
                    # 每次计算后强制 GC，回收因子函数产生的临时 DataFrame。
                    # 生产环境（6000 stocks）单次迭代可能产生 100MB+ 临时对象。
                    rss = _rss_mb()
                    if total_calls % MEM_CLEARING_INTEVAL == 0:
                        gc.collect()
                        buf_sizes = {f'b{i}': len(b) for i, b in enumerate(self.buffers)}
                        logging.info(
                            f'[{day_idx}][{ts}] mem#{total_calls}: rss={rss:.2f}MB, '
                            f'tob_len={len(time_order_buffer)}, '
                            f'buf_sizes={buf_sizes}'
                        )

                    # ---- MLflow: 记录本节点运行时间和输出队列大小 ----
                    run_id: str = self.context.get('mlflow_run_id', '')
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
                            step=day_idx,
                        )

                # === 快速退出检测：所有接收线程已退出且缓冲已空 ===
                if all(not t.is_alive() for t in threads):
                    with data_lock:
                        all_buffers_empty = all(len(b) == 0 for b in self.buffers)
                    if all_buffers_empty:
                        logging.info('All workers exited and buffers empty. Node process exiting.')
                        break

                # 上游进程有的会先结束，但是数据都已经进入了我们的缓冲。我们需要等待所有进程都结束。
                if len(dead_workers) == num_workers:
                    logging.info('All workers dead. Node process exited.')
                    break
                round_time += 1

        except Exception as e:
            logging.error(f'Exception in {self.name}: {e}', exc_info=True)
        finally:
            if self.epilogue_fn is not None:
                try:
                    self.epilogue_fn(self.name, day_idx, self.context)
                except Exception as e:
                    logging.error(f'Epilogue error in {self.name}: {e}', exc_info=True)
            # 先设置 global_exit 让可能存活的接收线程退出
            global_exit.set()
            for t in threads:
                t.join(timeout=2)
            # 向输出队列放入 stop_signal（非阻塞）。
            # 先 cancel_join_thread 防止进程退出时 queue finalizer 因 feeder 线程
            # 阻塞在管道 I/O 上而 hang（Windows spawn 模式下常见）。
            for outq in self.output_queues:
                with suppress(Exception):
                    outq.put(self.stop_signal, timeout=1.0)
                # with suppress(Exception):
                #     outq.cancel_join_thread()
            # 显式刷新日志和 stdout，防止 Windows 管道未关闭导致 WaitForSingleObject 不返回
            for handler in logging.getLogger().handlers:
                with suppress(Exception):
                    handler.flush()
            sys.stdout.flush()
            sys.stderr.flush()
            logging.info(f'Node {self.name} stopped.')

    def _call_func(self, name: str, idx: int, f3d: Frame3D, ctx: Any) -> Frame3D:
        """调用节点函数，兼容 2 参数和 3 参数签名。

        只捕获 TypeError（签名参数数量不匹配时回退到 2 参数调用）。
        ValueError 等运行时错误直接向上传播，不在框架层吞没。
        """
        sig = inspect.signature(self.func)
        if len(sig.parameters) >= 4:
            return self.func(name, idx, f3d, ctx)
        return self.func(name, idx, f3d)

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
            time.sleep(0.1)
            with heartbeat_lock:
                heartbeat_timestamp[queue_idx] = time.time()
            try:
                obj = input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception as e:
                import traceback

                logging.error(f'[thread-{queue_idx}] meet exception: {e}\n{traceback.format_exc()}')
                ready_event.set()
                break
            if obj == self.stop_signal:
                logging.info(f'[thread-{queue_idx}] receive stop signal.')
                ready_event.set()
                break
            # V3: 上游传递 (idx, Frame3D) 元组
            try:
                idx, f3d = obj
                logging.debug(f'[thread-{queue_idx}] receive data {idx}: {obj}')
            except Exception as e:
                import traceback

                logging.error(f'[thread-{queue_idx}] meet exception: {e}\n{traceback.format_exc()}')
                ready_event.set()
                break
            with data_lock:
                try:
                    self.buffers[queue_idx][idx] = f3d
                except Exception as e:
                    import traceback
                    logging.error(
                        f'[thread-{queue_idx}] meet exception: {e}\n{traceback.format_exc()}'
                    )
                finally:
                    ready_event.set()
        logging.info(f'[thread-{queue_idx}] stop.')

    def receive_common_idx_frames(
        self, worker_buffer: list[dict[Any, Frame3D]]
    ) -> dict[int, list[Frame3D]]:
        idx_frame_dict: dict[int, list[Frame3D]] = {}
        num_workers = len(worker_buffer)
        all_times_sets = [set(worker_buffer[i].keys()) for i in range(num_workers)]
        ranges = {}
        for i in range(num_workers):
            keys = list(worker_buffer[i].keys())
            if len(keys) > 0:
                ranges[i] = [keys[0], keys[-1]]
            else:
                ranges[i] = []
        logging.debug(f'ranges: {ranges}')
        shared_times = set.intersection(*all_times_sets) if all_times_sets else set()
        if shared_times:
            for tval in sorted(shared_times):
                frame_list = []  # frame3d from different upstream
                for i, buffer in enumerate(worker_buffer):
                    frame_list.append(buffer[tval])
                    logging.debug(f'[{tval}][upstream-{i}] {buffer[tval]}')
                    del buffer[tval]
                idx_frame_dict[tval] = frame_list
        return idx_frame_dict

    def concat_frames(self, frame_list: list[Frame3D], tval: int) -> pd.DataFrame:
        # frame_list: [Frame3D, ...] from buffers with same idxs
        df_list = [f3d.df for f3d in frame_list]
        # 校验多路上游 index 一致：任何节点的输出 index (key,code)
        # 必须完全相同，差异说明某节点产生了错误数据
        if len(df_list) > 1:
            base_idx = df_list[0].index
            for qi, other_df in enumerate(df_list[1:], 1):
                if not base_idx.equals(other_df.index):
                    diff_only_base = base_idx.difference(other_df.index)
                    diff_only_other = other_df.index.difference(base_idx)
                    _base_nm = base_idx.names
                    _oth_nm = other_df.index.names
                    PRINT_NUM = 10
                    _base_sample = list(base_idx)[:PRINT_NUM]
                    _oth_sample = list(other_df.index)[:PRINT_NUM]
                    _only_base_sample = list(diff_only_base)[:PRINT_NUM]
                    _only_oth_sample = list(diff_only_other)[:PRINT_NUM]
                    raise ValueError(
                        f'[{self.name}] Index mismatch at tval={tval} '
                        f'\nupstream[0] names={_base_nm} sample={_base_sample}, '
                        f'\nupstream[{qi}] names={_oth_nm} sample={_oth_sample}, '
                        f'\nbase_only({len(diff_only_base)})={_only_base_sample}, '
                        f'\nother_only({len(diff_only_other)})={_only_oth_sample}'
                    )
        return pd.concat(df_list, axis=1)

    def clean_frame(self, frame: pd.DataFrame) -> Frame3D:
        # 1. 去重：多路输入可能带重复列。
        # - 数值/因子列重名 → 硬报错（防止因子冲突导致数据静默丢失）
        # - 元数据列（stock_name 等）重名 → 日志记录 + 保留第一份
        DEDUP_SAFE_COLS = {'stock_name'}
        dup_cols = frame.columns[frame.columns.duplicated()].tolist()
        if dup_cols:
            safe_dup = [c for c in dup_cols if c in DEDUP_SAFE_COLS]
            bad_dup = [c for c in dup_cols if c not in DEDUP_SAFE_COLS]
            if bad_dup:
                raise ValueError(
                    f'[{self.name}] Duplicate numeric/factor columns after merge: {bad_dup}'
                )
            if safe_dup and not self.dup_col_have_warned:
                logging.warning(
                    f'Deduplicating {safe_dup} '
                    f'(kept first of {dup_cols.count(safe_dup[0]) + 1} copies)'
                )
                self.dup_col_have_warned = True
            frame = frame.loc[:, ~frame.columns.duplicated()]
        # 2. 保留 input_columns 列
        if self.input_columns:
            miss = [c for c in self.input_columns if c not in frame.columns]
            if miss:
                raise ValueError(f'[{self.name}] Missing input cols: {miss}')
            # 保留非数值元数据列（stock_name 等），确保下游 CSV 可读
            meta = [
                c
                for c in frame.columns
                if c not in self.input_columns and not pd.api.types.is_numeric_dtype(frame[c])
            ]
            keep = self.input_columns + meta
            frame = frame[keep]
        # 3. 排除 exclude_input_columns 列：在入队前删除，防止泄露到下游节点
        if self.exclude_input_columns:
            drop_cols = [c for c in self.exclude_input_columns if c in frame.columns]
            if drop_cols:
                frame = frame.drop(columns=drop_cols)
        return Frame3D(frame)

    def concat_and_align_frames_through_times(
        self, window_frames: list[tuple[int, Frame3D]]
    ) -> Frame3D:
        day_idx, _ = window_frames[-1]
        # 拼接窗口内所有时间片的快照
        temp = [frame.df for _, frame in window_frames]
        concated_window_frame = pd.concat(temp, axis=0)
        concated_window_frame.sort_index(level=0, inplace=True)
        logging.debug(f'[{day_idx}] concated_window_frame: \n{concated_window_frame}')
        logging.debug(f'columns: {concated_window_frame.columns} [{day_idx}]')
        # === IPO/退市对齐：以最新时间片的股票集合为准 ===
        # 最新时间片的股票集合是"当前市场"的权威集合。
        # - 退市股票：最新片中不存在，前序时间片中应删除。
        # - 新上市股票：最新片中存在但前序片中不存在，前序片补 NaN。
        #   用 NaN 而非 0.0，避免 np.log(0)/0/0 等下游运算产生
        #   "divide by zero in log" / "All-NaN slice" 警告。
        #
        #   优化：使用 MultiIndex.from_product 一次性 reindex，
        #   避免逐时间片循环产生 O(T) 个中间 DataFrame
        if self.time_alignment == 'right':
            alignment_key = concated_window_frame.index.get_level_values(0).max()
        else:
            alignment_key = concated_window_frame.index.get_level_values(0).min()
        # latest_stocks = window_df.loc[latest_t].index.unique().tolist()
        alignment_stocks = concated_window_frame.loc[alignment_key].index.tolist()
        all_times = sorted(concated_window_frame.index.get_level_values(0).unique())
        full_mi = pd.MultiIndex.from_product(
            [all_times, alignment_stocks], names=concated_window_frame.index.names
        )
        # 诊断日志：window_df 索引与期望索引的差异
        _actual = concated_window_frame.index
        _extra = _actual.difference(full_mi)
        _missing = full_mi.difference(_actual)
        error_msg = (
            f'[{day_idx}] index diagnostics: '
            f'\nexpected={len(full_mi)}, actual={len(_actual)}, '
            f'\nextra(actual-expected)={len(_extra)}, '
            f'\nmissing(expected-actual)={len(_missing)}'
        )
        logging.debug(error_msg)
        if len(_extra) > 0:
            _extra_sample = list(_extra)[:10]
            logging.error(f'[{day_idx}] extra index sample: {_extra_sample}')
            raise ValueError(error_msg)
        if len(_missing) > 0:
            _miss_sample = list(_missing)[:10]
            logging.error(f'[{day_idx}] missing index sample: {_miss_sample}')
            raise ValueError(error_msg)
        return Frame3D(concated_window_frame.reindex(full_mi).sort_index(level=0))

    def clean_output_frame(self, output_frame: Frame3D, concated_window_frame: Frame3D) -> Frame3D:
        # 保留输入中的元数据列（name/stock_id 等字符串列），确保下游 CSV 可读
        meta_cols = [
            c
            for c in concated_window_frame.df.columns
            if c not in output_frame.df.columns
            and not pd.api.types.is_numeric_dtype(concated_window_frame.df[c])
        ]
        if meta_cols:
            meta_df = concated_window_frame.df[meta_cols].loc[output_frame.df.index]
            output_df = pd.concat([meta_df, output_frame.df], axis=1)
            output_frame = Frame3D(output_df)
        if self.output_columns:
            miss_o = [c for c in self.output_columns if c not in output_frame.df.columns]
            if miss_o:
                raise ValueError(f'[{self.name}] Missing output cols: {miss_o}')
            filtered = output_frame.df[self.output_columns]
            output_frame = Frame3D(filtered)
        return output_frame
