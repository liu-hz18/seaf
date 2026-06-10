"""
Flow 编排器 — 管理节点拓扑、queue 创建、启停控制。
拓扑验证保证 DAG、无重复写、无孤立节点等基本约束。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any

from .node import (
    EpilogueFunc,
    FactorFunc,
    GenFunc,
    MultiInputNode,
    SourceNode,
)


class Flow:
    """流式管线编排器：声明式构建因子流水线拓扑，自动验证 DAG 约束。"""

    def __init__(self, stop_signal: Any = None, queue_maxsize: int = 0) -> None:
        """初始化流水线编排器。

        Args:
            stop_signal: 节点间传递的停止哨兵值。
            queue_maxsize: mp.Queue 最大容量。0=无限制; >0 时提供背压控制，
                           防止上游生产速率远超下游消费时队列无限膨胀导致 OOM。
                           推荐设为 max(factor_windows) 或 model_window。
        """
        self.nodes: list[mp.Process] = []
        self.stop_signal: Any = stop_signal if stop_signal is not None else '__STOP__'
        self.queues: dict[str, mp.Queue] = {}
        self._node_specs: list[dict[str, Any]] = []
        self._queue_writers: dict[str, list[str]] = {}
        self._queue_readers: dict[str, list[str]] = {}
        self.queue_maxsize = queue_maxsize

    def create_queue(self, name: str, maxsize: int | None = None) -> mp.Queue[Any]:
        """创建或获取命名管道。

        maxsize=None 时使用 Flow 实例级别的 queue_maxsize（0=无限制）。
        maxsize>0 时提供背压控制，防止队列无限膨胀。
        """
        if maxsize is None:
            maxsize = self.queue_maxsize
        if name not in self.queues:
            self.queues[name] = mp.Queue(maxsize=maxsize)
        return self.queues[name]

    def add_source(
        self,
        name: str,
        gen_func: GenFunc,
        output_to: list[str],
        context: Any = None,
        epilogue_fn: EpilogueFunc | None = None,
        snapshot_interval: int = 0,
    ) -> SourceNode:
        """添加数据源节点。gen_func 应为返回 Frame3D 迭代器的可调用对象。"""
        output_queues = [self.create_queue(qname) for qname in output_to]
        node = SourceNode(
            name,
            gen_func,
            output_queues,
            self.stop_signal,
            context=context,
            epilogue_fn=epilogue_fn,
            output_queue_names=list(output_to),
            snapshot_interval=snapshot_interval,
        )
        self.nodes.append(node)
        self._node_specs.append(
            {
                'name': name,
                'type': 'source',
                'inputs': [],
                'outputs': list(output_to),
            }
        )
        for qname in output_to:
            self._queue_writers.setdefault(qname, []).append(name)
            self._queue_readers.setdefault(qname, [])
        return node

    def add_node(
        self,
        name: str,
        func: FactorFunc,
        input_from: str | list[str],
        output_to: list[str],
        window: int = 1,
        min_periods: int = 1,
        input_columns: list[str] | None = None,
        output_columns: list[str] | None = None,
        context: Any = None,
        epilogue_fn: EpilogueFunc | None = None,
        snapshot_interval: int = 0,
    ) -> MultiInputNode:
        """添加因子/计算节点。func 接收 (name, f3d, context)，返回 Frame3D。"""
        if isinstance(input_from, str):
            input_from = [input_from]
        input_queues = [self.create_queue(qname) for qname in input_from]
        output_queues = [self.create_queue(qname) for qname in output_to] if output_to else []
        node = MultiInputNode(
            name,
            func,
            input_queues,
            output_queues,
            window=window,
            min_periods=min_periods,
            input_columns=input_columns,
            output_columns=output_columns,
            stop_signal=self.stop_signal,
            context=context,
            epilogue_fn=epilogue_fn,
            output_queue_names=list(output_to),
            snapshot_interval=snapshot_interval,
        )
        self.nodes.append(node)
        self._node_specs.append(
            {
                'name': name,
                'type': 'node',
                'inputs': list(input_from),
                'outputs': list(output_to),
            }
        )
        for qname in input_from:
            self._queue_readers.setdefault(qname, []).append(name)
        for qname in output_to:
            self._queue_writers.setdefault(qname, []).append(name)
            self._queue_readers.setdefault(qname, [])
        return node

    def validate_topology(self) -> list[str]:
        """验证拓扑约束：无重名、无多写、无孤立、无环。返回错误列表。"""
        errors: list[str] = []

        # ---- 1. 无重名节点 ----
        seen_names: set[str] = set()
        for spec in self._node_specs:
            if spec['name'] in seen_names:
                errors.append(f"Duplicate node name: '{spec['name']}'")
            seen_names.add(spec['name'])

        # ---- 2. 无重名管道（同一管道被多个节点写入） ----
        for qname, writers in self._queue_writers.items():
            if len(writers) > 1:
                errors.append(f"Queue '{qname}' has multiple producers: {writers}")

        # ---- 3. 每个管道两端都连接了节点 ----
        for qname in self.queues:
            writers = self._queue_writers.get(qname, [])
            readers = self._queue_readers.get(qname, [])
            if not writers:
                errors.append(f"Queue '{qname}' has no producer (orphan queue)")
            if not readers:
                errors.append(f"Queue '{qname}' has no consumer (dangling output)")

        # ---- 4. 无孤立节点 ----
        for spec in self._node_specs:
            name = spec['name']
            if spec['type'] == 'source':
                if not spec['outputs']:
                    errors.append(f"Source node '{name}' has no output (isolated)")
            elif not spec['inputs'] and not spec['outputs']:
                errors.append(f"Node '{name}' has no input and no output (isolated)")
            elif not spec['inputs']:
                errors.append(f"Node '{name}' has no input (orphan node)")

        # ---- 5. 无环形链路 (DFS) ----
        node_names = {spec['name'] for spec in self._node_specs}
        graph: dict[str, set[str]] = {name: set() for name in node_names}
        for qname, writers in self._queue_writers.items():
            readers = self._queue_readers.get(qname, [])
            for w in writers:
                for r in readers:
                    graph[w].add(r)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = dict.fromkeys(graph, WHITE)

        def dfs(node: str, path: list[str]) -> list[str] | None:
            color[node] = GRAY
            path.append(node)
            for neighbor in sorted(graph[node]):
                if color[neighbor] == GRAY:
                    cycle_start = path.index(neighbor)
                    return [*path[cycle_start:], neighbor]
                if color[neighbor] == WHITE:
                    result = dfs(neighbor, path)
                    if result:
                        return result
            path.pop()
            color[node] = BLACK
            return None

        for name in sorted(graph):
            if color[name] == WHITE:
                result = dfs(name, [])
                if result:
                    errors.append(f'Cycle detected: {" -> ".join(result)}')
                    break

        return errors

    def start(self) -> None:
        """验证拓扑后启动所有子进程。"""
        errors = self.validate_topology()
        if errors:
            for e in errors:
                logging.error(f'[Topology] {e}')
            raise RuntimeError(f'Topology validation failed with {len(errors)} error(s)')
        for node in self.nodes:
            node.start()
        for queue in self.queues.values():
            queue.cancel_join_thread()

    def join(self) -> None:
        """等待所有子进程结束。"""
        for node in self.nodes:
            node.join(timeout=None)
            if node.is_alive():
                logging.warning(f'Node {node.name} did not exit, terminating.')
                node.terminate()
                node.join(timeout=5)
