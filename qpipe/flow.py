import multiprocessing as mp
import logging
from typing import Callable, List, Union, Tuple, Dict, Any, Iterator, Optional

from .frame3d import Frame3D
from .node import MultiInputNode, SourceNode


class Flow:
    def __init__(self, stop_signal=None):
        self.nodes: List[mp.Process] = []
        self.stop_signal = stop_signal if stop_signal is not None else "__STOP__"
        self.queues: Dict[str, mp.Queue] = {}
        # 拓扑追踪
        self._node_specs: List[Dict] = []
        self._queue_writers: Dict[str, List[str]] = {}   # queue -> [producer nodes]
        self._queue_readers: Dict[str, List[str]] = {}   # queue -> [consumer nodes]

    def create_queue(self, name: str) -> mp.Queue:
        if name not in self.queues:
            self.queues[name] = mp.Queue()
        return self.queues[name]

    def add_source(self, name: str, gen_func: Callable[[], Iterator[Frame3D]], output_to: List[str]):
        output_queues = [self.create_queue(qname) for qname in output_to]
        node = SourceNode(name, gen_func, output_queues, self.stop_signal)
        self.nodes.append(node)
        # 记录拓扑
        self._node_specs.append({'name': name, 'type': 'source', 'inputs': [], 'outputs': list(output_to)})
        for qname in output_to:
            self._queue_writers.setdefault(qname, []).append(name)
            self._queue_readers.setdefault(qname, [])
        return node

    def add_node(
        self,
        name: str,
        func: Callable[[str, Frame3D], Frame3D],
        input_from: Union[str, List[str]],
        output_to: List[str],
        window: int = 1,
        min_periods: int = 1,
        input_columns: Optional[List[str]] = None,
        output_columns: Optional[List[str]] = None
    ):
        if isinstance(input_from, str):
            input_from = [input_from]
        input_queues = [self.create_queue(qname) for qname in input_from]
        output_queues = [self.create_queue(qname) for qname in output_to] if output_to else []
        node = MultiInputNode(
            name, func, input_queues, output_queues,
            window=window, min_periods=min_periods,
            input_columns=input_columns, output_columns=output_columns,
            stop_signal=self.stop_signal
        )
        self.nodes.append(node)
        # 记录拓扑
        self._node_specs.append({'name': name, 'type': 'node', 'inputs': list(input_from), 'outputs': list(output_to)})
        for qname in input_from:
            self._queue_readers.setdefault(qname, []).append(name)
        for qname in output_to:
            self._queue_writers.setdefault(qname, []).append(name)
            self._queue_readers.setdefault(qname, [])
        return node

    def validate_topology(self) -> List[str]:
        errors = []

        # ---- 1. 无重名节点 ----
        seen_names = set()
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
            else:
                if not spec['inputs'] and not spec['outputs']:
                    errors.append(f"Node '{name}' has no input and no output (isolated)")
                elif not spec['inputs']:
                    errors.append(f"Node '{name}' has no input (orphan node)")

        # ---- 5. 无环形链路 ----
        node_names = set(spec['name'] for spec in self._node_specs)
        graph = {name: set() for name in node_names}
        for qname, writers in self._queue_writers.items():
            readers = self._queue_readers.get(qname, [])
            for w in writers:
                for r in readers:
                    graph[w].add(r)

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in graph}

        def dfs(node, path):
            color[node] = GRAY
            path.append(node)
            for neighbor in sorted(graph[node]):
                if color[neighbor] == GRAY:
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
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
                    errors.append(f"Cycle detected: {' -> '.join(result)}")
                    break

        return errors

    def start(self):
        errors = self.validate_topology()
        if errors:
            for e in errors:
                logging.error(f"[Topology] {e}")
            raise RuntimeError(f"Topology validation failed with {len(errors)} error(s)")
        for node in self.nodes:
            node.start()
        for queue in self.queues.values():
            queue.cancel_join_thread()

    def join(self):
        for node in self.nodes:
            node.join(timeout=None)
            if node.is_alive():
                logging.warning(f"Node {node.name} did not exit, terminating.")
                node.terminate()
                node.join(timeout=5)
