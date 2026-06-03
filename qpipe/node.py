"""
多进程节点 — MultiInputNode（多输入计算节点）和 SourceNode（数据源节点）。

MultiInputNode: 多队列异步接收，按 time 合并多上游数据，滑动窗口计算。
SourceNode: 迭代生成 Frame3D，按天广播到下游。

v2 扩展：支持 context（有状态上下文）和 epilogue_fn（退出前回调）。
pickle 兼容：不创建闭包，在 run() 中运行时动态检查 func 签名。
"""
import multiprocessing as mp
import threading
import logging
import sys
import inspect
from typing import Callable, List, Union, Tuple, Dict, Any, Iterator, Optional
import pandas as pd
import time
from collections import deque

from .frame3d import Frame3D


class MultiInputNode(mp.Process):
    """
    多队列异步抢收，窗口滑动处理。对所有上游同一个time合并后，历史按照 window 滑动，min_periods约束。
    
    v2: func 签名支持 context：
        - 新：func(name, f3d, context) -> Frame3D | (Frame3D, context)
        - 旧：func(name, f3d) -> Frame3D（向后兼容，自动检测）
        
    epilogue_fn(name, context) 在进程退出前调用，用于汇总分析。
    """

    HEARTBEAT_TIMEOUT = 10.0
    THREAD_ROUND_MAX_TIME = 3

    def __init__(
        self,
        name: str,
        func: Callable,
        input_queues: List[mp.Queue],
        output_queues: List[mp.Queue],
        window: int = 5,
        min_periods: int = 3,
        input_columns: Optional[List[str]] = None,
        output_columns: Optional[List[str]] = None,
        stop_signal=None,
        context: Optional[Any] = None,
        epilogue_fn: Optional[Callable[[str, Any], None]] = None,
    ):
        super().__init__()
        self.name = name
        self.func = func  # 直接存储原始函数，不在 init 中包装
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.window = window
        self.min_periods = min_periods
        self.input_columns = input_columns if input_columns else []
        self.output_columns = output_columns if output_columns else []
        self.stop_signal = stop_signal
        self.context = context
        self.epilogue_fn = epilogue_fn

        self.buffers = [dict() for _ in input_queues]   # {time -> Frame3D}

    def _call_func(self, name, f3d, ctx):
        """运行时根据 func 签名动态选择调用方式。"""
        try:
            sig = inspect.signature(self.func)
            if len(sig.parameters) >= 3:
                return self.func(name, f3d, ctx)
            else:
                return self.func(name, f3d)
        except (ValueError, TypeError):
            return self.func(name, f3d)

    def receive_worker(
        self, 
        queue_idx: int, 
        input_queue: mp.Queue, 
        ready_event: threading.Event, 
        global_exit: threading.Event, 
        data_lock: threading.Lock, 
        heartbeat_timestamp: List[float],
        heartbeat_lock: threading.Lock,
    ):
        while not global_exit.is_set():
            with heartbeat_lock:
                heartbeat_timestamp[queue_idx] = time.time()
            try:
                obj = input_queue.get(timeout=0.5)
            except Exception:
                continue
            if obj == self.stop_signal:
                logging.debug(f"[{self.name}][thread-{queue_idx}] stop signal received.")
                ready_event.set()
                break
            # 某个队列新收到一个时间截面
            time_value = obj.df.index.get_level_values(0)[0]
            logging.debug(f"[{self.name}][thread-{queue_idx}] {obj.df.index.get_level_values(0)}")
            with data_lock:
                self.buffers[queue_idx][time_value] = obj
                ready_event.set()

    def run(self):
        # BEGIN: subprocess domain
        logging.basicConfig(
            level=logging.INFO,
            format=f"[%(levelname)s][{self.name}][%(asctime)s]: %(message)s",
            stream=sys.stdout,
        )
        logging.info(f"Node {self.name} started, window={self.window}, min_periods={self.min_periods}.")

        num_workers = len(self.input_queues)
        
        data_lock = threading.Lock()
        ready_event = [threading.Event() for _ in range(num_workers)]
        
        heartbeat_lock = threading.Lock()
        heartbeat_timestamp = [time.time() for _ in range(num_workers)]

        global_exit = threading.Event()

        threads: List[threading.Thread] = []
        for i, inq in enumerate(self.input_queues):
            t = threading.Thread(
                target=self.receive_worker,
                args=(i, inq, ready_event[i], global_exit, data_lock, heartbeat_timestamp, heartbeat_lock),
                daemon=True,
            )
            t.start()
            threads.append(t)

        # 接收并处理各个子线程数据
        dead_workers = set()
        time_order_buffer = deque()  # deque[(time, merged_f3d)]
        window_start_index = 0
        window_tail_index = 0
        round_time = 0
        current_context = self.context  # 有状态上下文
        try:
            while True:
                logging.debug(f"round_time: {round_time}")
                # 接收心跳数据
                alive_workers_to_wait = [i for i in range(num_workers) if i not in dead_workers]
                thread_round_time = 0
                while alive_workers_to_wait and thread_round_time < self.THREAD_ROUND_MAX_TIME:
                    ready_workers = [i for i in alive_workers_to_wait if ready_event[i].is_set()]
                    still_waiting = [i for i in alive_workers_to_wait if i not in ready_workers]
                    newly_dead = []
                    for i in still_waiting:
                        with heartbeat_lock:
                            timedelta = time.time() - heartbeat_timestamp[i]
                            logging.debug(f"[thread-{i}] {timedelta=}")
                            if timedelta > self.HEARTBEAT_TIMEOUT:
                                newly_dead.append(i)
                    if newly_dead:
                        for i in newly_dead:
                            logging.debug(f"Worker {i} heartbeat timeout for {self.HEARTBEAT_TIMEOUT} seconds.")
                            dead_workers.add(i)
                            alive_workers_to_wait.remove(i)
                            still_waiting.remove(i)
                    # 所有线程都准备好
                    logging.debug(f"{len(ready_workers)=} {len(alive_workers_to_wait)=}")
                    if len(ready_workers) == len(alive_workers_to_wait):
                        break
                    time.sleep(0.2)
                    thread_round_time += 1

                # 接收子线程收集到的数据
                nonempty_times = []
                submitted_workers = []
                for i in range(num_workers):
                    if ready_event[i].is_set():
                        with data_lock:
                            nonempty_times.append(set(self.buffers[i].keys()))
                        submitted_workers.append(i)
                shared_times = set.intersection(*nonempty_times) if nonempty_times else set()

                # 没有新数据则进入下一轮循环
                frame_lists = []
                if shared_times:
                    for tval in sorted(shared_times):
                        # 合并这一个新 time 的多路输入
                        with data_lock:
                            frame_list = [self.buffers[bi][tval] for bi in range(len(self.buffers))]
                            for bi in range(len(self.buffers)):
                                del self.buffers[bi][tval]
                        frame_lists.append(frame_list)

                # 清理不需要的数据，准备好接收下一轮
                for i in submitted_workers:
                    ready_event[i].clear()

                # 数据进本地 buffer
                for frame_list in frame_lists:
                    df_list = [f3d.df for f3d in frame_list]
                    merged_df = pd.concat(df_list, axis=1)
                    # ==== 输入列校验 ====
                    if self.input_columns:
                        miss = [col for col in self.input_columns if col not in merged_df.columns]
                        if miss:
                            raise ValueError(f"[{self.name}] Input missing columns: {miss}")
                        merged_df = merged_df[self.input_columns]
                    window_f3d = Frame3D(merged_df)
                    time_order_buffer.append((tval, window_f3d))                    
                logging.debug(f"[node] buffer size: {len(time_order_buffer)}")
                logging.debug(f"[node] buffer content:\n{time_order_buffer}")

                # 滚动取出 buffer 中的若干个时间窗口进行计算
                if len(time_order_buffer) < self.min_periods:
                    continue

                logging.debug(f"[before] data window: [{window_start_index}, {window_tail_index}]. queue length={len(time_order_buffer)}")
                while window_tail_index - window_start_index <= len(time_order_buffer):
                    while window_tail_index - window_start_index < min(self.min_periods, len(time_order_buffer)):
                        window_tail_index += 1
                    while window_tail_index - window_start_index > self.window:
                        window_start_index += 1
                        # drop old datas
                        time_order_buffer.popleft()
                    logging.debug(f"[inner] data window: [{window_start_index}, {window_tail_index}]. queue length={len(time_order_buffer)}")
                    window_length = window_tail_index - window_start_index
                    if window_length < self.min_periods:
                        continue
                    # get window data
                    window_frames = list(time_order_buffer)[:window_length]
                    logging.debug(f"window_frames: {window_frames}")
                    window_df = pd.concat([f[-1].df for f in window_frames], axis=0)
                    window_df = window_df.sort_index(level=0)
                    run_input_f3d = Frame3D(window_df)
                    # compute — v2: pass context, handle return (dynamic signature check)
                    result = self._call_func(self.name, run_input_f3d, current_context)
                    if isinstance(result, tuple):
                        output_f3d, current_context = result
                    else:
                        output_f3d = result
                    # check output
                    if self.output_columns:
                        miss_output = [col for col in self.output_columns if col not in output_f3d.df.columns]
                        if miss_output:
                            raise ValueError(f"[{self.name}] Output missing columns: {miss_output}")
                        filtered = output_f3d.df[self.output_columns]
                    else:
                        filtered = output_f3d.df
                    result_f3d = Frame3D(filtered.copy())
                    # 只取最新 key 对应的数据
                    max_key = result_f3d.df.index.get_level_values(0).max()
                    latest_df = result_f3d.df[result_f3d.df.index.get_level_values(0) == max_key]
                    latest_f3d = Frame3D(latest_df.copy())
                    # 游标移动
                    window_tail_index += 1
                    # push to queue
                    for outq in self.output_queues:
                        outq.put(latest_f3d)

                if len(dead_workers) == num_workers:
                    logging.info(f"All workers dead. Main process exited.")
                    break

                round_time += 1

        except Exception as e:
            logging.error(f"Exception in {self.name}: {e}", exc_info=True)
        finally:
            # v2: 退出前调用 epilogue_fn
            if self.epilogue_fn is not None:
                try:
                    self.epilogue_fn(self.name, current_context)
                except Exception as e:
                    logging.error(f"Epilogue error in {self.name}: {e}", exc_info=True)
            global_exit.set()
            for t in threads:
                t.join(timeout=1)
            for outq in self.output_queues:
                outq.put(self.stop_signal)
            logging.info(f"Node {self.name} stopped.")


class SourceNode(mp.Process):
    """数据源节点：迭代 gen_func() 生成 Frame3D，按天广播到下游。
    
    v2: 支持 context 和 epilogue_fn。
    """

    def __init__(
        self,
        name: str,
        gen_func: Callable[[], Iterator[Frame3D]],
        output_queues: List[mp.Queue],
        stop_signal=None,
        context: Optional[Any] = None,
        epilogue_fn: Optional[Callable[[str, Any], None]] = None,
    ):
        super().__init__()
        self.name = name
        self.gen_func = gen_func
        self.output_queues = output_queues
        self.stop_signal = stop_signal
        self.context = context
        self.epilogue_fn = epilogue_fn

    def run(self):
        logging.basicConfig(
            level=logging.INFO,
            format=f"[%(levelname)s][{self.name}][%(asctime)s]: %(message)s",
            stream=sys.stdout,
        )
        logging.info("SourceNode started.")
        current_context = self.context
        try:
            for frame in self.gen_func():
                # 只取最新 key 对应的数据
                max_key = frame.df.index.get_level_values(0).max()
                latest_df = frame.df[frame.df.index.get_level_values(0) == max_key]
                latest_f3d = Frame3D(latest_df.copy())
                for outq in self.output_queues:
                    outq.put(latest_f3d)
            for outq in self.output_queues:
                outq.put(self.stop_signal)
        except Exception as e:
            logging.error(f"Exception in SourceNode: {e}", exc_info=True)
        finally:
            # v2: 退出前调用 epilogue_fn
            if self.epilogue_fn is not None:
                try:
                    self.epilogue_fn(self.name, current_context)
                except Exception as e:
                    logging.error(f"Epilogue error in {self.name}: {e}", exc_info=True)
            logging.info("SourceNode stopped.")
