"""
Multi-process nodes: MultiInputNode and SourceNode.
V2: context + epilogue_fn support, backward compatible.
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
    HEARTBEAT_TIMEOUT = 10.0
    THREAD_ROUND_MAX_TIME = 3

    def __init__(self, name, func, input_queues, output_queues,
                 window=5, min_periods=3, input_columns=None, output_columns=None,
                 stop_signal=None, context=None, epilogue_fn=None):
        super().__init__()
        self.name = name
        self.func = func
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.window = window
        self.min_periods = min_periods
        self.input_columns = input_columns if input_columns else []
        self.output_columns = output_columns if output_columns else []
        self.stop_signal = stop_signal
        self.context = context
        self.epilogue_fn = epilogue_fn
        self.buffers = [dict() for _ in input_queues]

    def _call_func(self, name, f3d, ctx):
        try:
            sig = inspect.signature(self.func)
            if len(sig.parameters) >= 3:
                return self.func(name, f3d, ctx)
            else:
                return self.func(name, f3d)
        except (ValueError, TypeError):
            return self.func(name, f3d)

    def receive_worker(self, queue_idx, input_queue: mp.Queue, ready_event, global_exit,
                       data_lock, heartbeat_timestamp, heartbeat_lock):
        while not global_exit.is_set():
            with heartbeat_lock:
                heartbeat_timestamp[queue_idx] = time.time()
            try:
                obj = input_queue.get(timeout=0.5)
            except Exception:
                continue
            if obj == self.stop_signal:
                logging.debug(f"[{self.name}][thread-{queue_idx}] stop signal.")
                ready_event.set()
                break
            time_value = obj.df.index.get_level_values(0)[0]
            with data_lock:
                # TODO: push too fast
                logging.info(f"[queue-{queue_idx}] push {time_value}. queue size={input_queue.qsize()}. buffer size={len(self.buffers[queue_idx])}")
                self.buffers[queue_idx][time_value] = obj
                ready_event.set()

    def run(self):
        logging.basicConfig(level=logging.INFO,
                           format=f"[%(levelname)s][{self.name}][%(asctime)s]: %(message)s",
                           stream=sys.stdout)
        logging.info(f"Node {self.name} started, w={self.window}, mp={self.min_periods}.")

        num_workers = len(self.input_queues)
        data_lock = threading.Lock()
        ready_event = [threading.Event() for _ in range(num_workers)]
        heartbeat_lock = threading.Lock()
        heartbeat_timestamp = [time.time() for _ in range(num_workers)]
        global_exit = threading.Event()

        threads = []
        for i, inq in enumerate(self.input_queues):
            t = threading.Thread(target=self.receive_worker,
                                 args=(i, inq, ready_event[i], global_exit,
                                       data_lock, heartbeat_timestamp, heartbeat_lock),
                                 daemon=True)
            t.start()
            threads.append(t)

        dead_workers = set()
        time_order_buffer = deque()
        window_start_index = 0
        window_tail_index = 0
        round_time = 0
        current_context = self.context

        try:
            while True:
                alive_workers_to_wait = [i for i in range(num_workers) if i not in dead_workers]
                thread_round_time = 0
                while alive_workers_to_wait and thread_round_time < self.THREAD_ROUND_MAX_TIME:
                    ready_workers = [i for i in alive_workers_to_wait if ready_event[i].is_set()]
                    still_waiting = [i for i in alive_workers_to_wait if i not in ready_workers]
                    newly_dead = []
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

                submitted_workers = []
                for i in range(num_workers):
                    if ready_event[i].is_set():
                        submitted_workers.append(i)

                # All buffers intersection (including non-ready workers)
                with data_lock:
                    all_times_sets = [set(self.buffers[i].keys()) for i in range(num_workers)]
                shared_times = set.intersection(*all_times_sets) if all_times_sets else set()

                frame_lists = []
                if shared_times:
                    for tval in sorted(shared_times):
                        with data_lock:
                            frame_list = [self.buffers[bi][tval] for bi in range(len(self.buffers))]
                            for bi in range(len(self.buffers)):
                                del self.buffers[bi][tval]
                        frame_lists.append(frame_list)

                for i in submitted_workers:
                    ready_event[i].clear()

                for frame_list in frame_lists:
                    df_list = [f3d.df for f3d in frame_list]
                    merged_df = pd.concat(df_list, axis=1)
                    if self.input_columns:
                        miss = [c for c in self.input_columns if c not in merged_df.columns]
                        if miss:
                            raise ValueError(f"[{self.name}] Missing input cols: {miss}")
                        merged_df = merged_df[self.input_columns]
                    time_order_buffer.append((shared_times.pop() if shared_times else None,
                                              Frame3D(merged_df)))

                if len(time_order_buffer) < self.min_periods:
                    if len(dead_workers) == num_workers:
                        logging.info(f"All workers dead, insufficient data. Exiting.")
                        break
                    continue

                # Window processing
                while window_tail_index - window_start_index <= len(time_order_buffer):
                    while window_tail_index - window_start_index < min(self.min_periods,
                                                                        len(time_order_buffer)):
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

                    # 开始进行节点计算
                    start_time = time.time()
                    window_df = pd.concat([f[-1].df for f in window_frames], axis=0)
                    window_df = window_df.sort_index(level=0)
                    run_input_f3d = Frame3D(window_df)
                    # logging.info(f"input frame: {run_input_f3d}")
                    result = self._call_func(self.name, run_input_f3d, current_context)
                    if isinstance(result, tuple):
                        output_f3d, current_context = result
                    else:
                        output_f3d = result
                    if self.output_columns:
                        miss_o = [c for c in self.output_columns if c not in output_f3d.df.columns]
                        if miss_o:
                            raise ValueError(f"[{self.name}] Missing output cols: {miss_o}")
                        filtered = output_f3d.df[self.output_columns]
                    else:
                        filtered = output_f3d.df
                    result_f3d = Frame3D(filtered.copy())
                    max_key = result_f3d.df.index.get_level_values(0).max()
                    latest_df = result_f3d.df[result_f3d.df.index.get_level_values(0) == max_key]
                    latest_f3d = Frame3D(latest_df.copy())
                    logging.info(f"window_start_index:{window_start_index}, window_tail_index={window_tail_index}\ntime_order_buffer: {len(time_order_buffer)}\noutput frame: {latest_f3d}")
                    end_time = time.time()
                    time_elapsed = end_time - start_time

                    # 将耗时记录到 context（区分 node 内部字段与用户字段）
                    if current_context is None:
                        current_context = {}
                    if not isinstance(current_context, dict):
                        # 用户传入非 dict context → 包装为 dict
                        current_context = {'_user_context': current_context}
                    timings = current_context.setdefault('_node_timings', [])
                    timings.append(time_elapsed)
                    if len(timings) > 10:
                        timings.pop(0)  # 保持最近 10 次
                    avg_time = sum(timings) / len(timings)
                    total_calls = current_context.setdefault('_node_call_count', 0) + 1
                    current_context['_node_call_count'] = total_calls
                    if total_calls % 10 == 0 or total_calls == 1:
                        logging.info(
                            f"[{self.name}] call#{total_calls}: "
                            f"elapsed={time_elapsed:.3f}s, "
                            f"rolling_avg10={avg_time:.3f}s"
                        )

                    window_tail_index += 1
                    for outq in self.output_queues:
                        outq.put(latest_f3d)

                if len(dead_workers) == num_workers:
                    logging.info(f"All workers dead. Main process exited.")
                    break
                round_time += 1

        except Exception as e:
            logging.error(f"Exception in {self.name}: {e}", exc_info=True)
        finally:
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
    def __init__(self, name, gen_func, output_queues, stop_signal=None,
                 context=None, epilogue_fn=None):
        super().__init__()
        self.name = name
        self.gen_func = gen_func
        self.output_queues = output_queues
        self.stop_signal = stop_signal
        self.context = context
        self.epilogue_fn = epilogue_fn

    def run(self):
        logging.basicConfig(level=logging.INFO,
                           format=f"[%(levelname)s][{self.name}][%(asctime)s]: %(message)s",
                           stream=sys.stdout)
        logging.info("SourceNode started.")
        current_context = self.context
        try:
            for frame in self.gen_func():
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
            if self.epilogue_fn is not None:
                try:
                    self.epilogue_fn(self.name, current_context)
                except Exception as e:
                    logging.error(f"Epilogue error in {self.name}: {e}", exc_info=True)
            logging.info("SourceNode stopped.")
