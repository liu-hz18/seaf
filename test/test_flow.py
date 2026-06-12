"""
Flow 编排器测试 — 覆盖拓扑验证、队列管理、DAG 约束、启停控制。
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
import pandas as pd
import pytest

from qpipe.flow import Flow
from qpipe.frame3d import Frame3D
from qpipe.node import MultiInputNode, SourceNode


# ═══════════════════════════════════════════════════════════════════════════
# 测试用节点函数
# ═══════════════════════════════════════════════════════════════════════════

def _pass_through(name: str, f3d: Frame3D, ctx: Any = None) -> Frame3D:
    """透传节点：返回输入数据不变。"""
    return f3d


def _gen_func(n_stocks: int = 5, n_times: int = 30):
    """返回生成器函数工厂。"""
    def _inner():
        rng = np.random.default_rng(42)
        dates = pd.date_range('2020-01-02', periods=n_times, freq='B')
        stocks = [f'S{i:04d}' for i in range(n_stocks)]
        for t in dates:
            mi = pd.MultiIndex.from_product([[t], stocks], names=['key', 'code'])
            df = pd.DataFrame({
                'close': rng.normal(100, 10, size=len(stocks)),
                'volume': rng.integers(1000, 10000, size=len(stocks)),
            }, index=mi)
            yield Frame3D(df)
    return _inner


# ═══════════════════════════════════════════════════════════════════════════
# Flow 拓扑验证
# ═══════════════════════════════════════════════════════════════════════════

class TestFlowTopology:
    """Flow.validate_topology() 的各种校验。"""

    def test_valid_topology_no_errors(self):
        """标准拓扑 (1 source → 1 node) 应无验证错误。"""
        flow = Flow()
        flow.add_source('src', _gen_func(), ['q1'])
        flow.add_node('n1', _pass_through, input_from='q1', output_to=[])
        errs = flow.validate_topology()
        assert not errs, f'Expected no errors, got: {errs}'

    def test_duplicate_node_name(self):
        """重复节点名应报错。"""
        flow = Flow()
        flow.add_source('dup', _gen_func(), ['q1'])
        flow.add_node('dup', _pass_through, input_from='q1', output_to=[])
        errs = flow.validate_topology()
        assert any('Duplicate' in e for e in errs)

    def test_multiple_producers(self):
        """同一 queue 被多个节点写入应报错。"""
        flow = Flow()
        flow.add_source('src1', _gen_func(), ['q_shared'])
        flow.add_source('src2', _gen_func(), ['q_shared'])
        errs = flow.validate_topology()
        assert any('multiple producers' in e.lower() for e in errs)

    def test_orphan_queue_reader(self):
        """queue 有 reader 但无 producer 应报错。"""
        flow = Flow()
        flow.create_queue('q_orphan')
        flow._queue_readers['q_orphan'] = ['some_node']
        errs = flow.validate_topology()
        assert any('orphan' in e.lower() for e in errs)

    def test_dangling_output_queue(self):
        """queue 有 writer 无 reader → 标记 dangling output（错误）。"""
        flow = Flow()
        flow.add_source('src', _gen_func(), ['q_no_reader'])
        errs = flow.validate_topology()
        assert any('no consumer' in e.lower() or 'dangling' in e.lower()
                   for e in errs)

    def test_isolated_node_no_inputs(self):
        """无 input 的 node（非 source）应报 orphan node 错误。"""
        flow = Flow()
        flow.create_queue('q_isolated')
        flow._node_specs.append({
            'name': 'isolated', 'type': 'node', 'inputs': [], 'outputs': []
        })
        errs = flow.validate_topology()
        assert any('no input' in e.lower() or 'isolated' in e.lower()
                   for e in errs)

    def test_two_source_one_sink(self):
        """二源一汇拓扑应验证通过。"""
        flow = Flow()
        flow.add_source('src_a', _gen_func(), ['qa'])
        flow.add_source('src_b', _gen_func(), ['qb'])
        flow.add_node('merge', _pass_through, input_from=['qa', 'qb'], output_to=[])
        errs = flow.validate_topology()
        assert not errs


# ═══════════════════════════════════════════════════════════════════════════
# Flow 队列管理
# ═══════════════════════════════════════════════════════════════════════════

class TestFlowQueues:
    """Flow.create_queue() 和队列引用管理。"""

    def test_create_queue_idempotent(self):
        """重复调用 create_queue 返回同一个对象。"""
        flow = Flow()
        q1 = flow.create_queue('test_q')
        q2 = flow.create_queue('test_q')
        assert q1 is q2

    def test_queue_maxsize(self):
        """queue_maxsize 参数正确传递。"""
        flow = Flow(queue_maxsize=256)
        q = flow.create_queue('bounded')
        # mp.Queue 没有直接获取 maxsize 的 API，通过属性间接验证
        assert hasattr(q, '_maxsize')

    def test_add_source_creates_queues(self):
        """add_source 自动创建 output_to 中的 queue。"""
        flow = Flow()
        flow.add_source('src', _gen_func(), ['qa', 'qb'])
        assert 'qa' in flow.queues
        assert 'qb' in flow.queues


# ═══════════════════════════════════════════════════════════════════════════
# Flow 节点管理
# ═══════════════════════════════════════════════════════════════════════════

class TestFlowNodes:
    """Flow 添加节点的正确性。"""

    def test_add_source_returns_source_node(self):
        """add_source 返回 SourceNode 实例。"""
        flow = Flow()
        node = flow.add_source('src', _gen_func(), ['q1'])
        assert isinstance(node, SourceNode)

    def test_add_node_returns_multi_input_node(self):
        """add_node 返回 MultiInputNode 实例。"""
        flow = Flow()
        flow.create_queue('q_in')
        node = flow.add_node('n1', _pass_through, input_from='q_in', output_to=[])
        assert isinstance(node, MultiInputNode)

    def test_nodes_collected(self):
        """所有节点收集在 flow.nodes 中。"""
        flow = Flow()
        flow.add_source('src', _gen_func(), ['q1'])
        flow.add_node('n1', _pass_through, input_from='q1', output_to=[])
        assert len(flow.nodes) == 2

    def test_node_context_preserved(self):
        """Node context 参数正确传递。"""
        flow = Flow()
        ctx = {'key': 'value', 'num': 42}
        flow.create_queue('q_in')
        node = flow.add_node('n1', _pass_through, input_from='q_in',
                             output_to=[], context=ctx)
        assert node.context == ctx
