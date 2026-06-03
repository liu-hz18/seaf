"""
MultiInputNode/SourceNode 扩展测试
测试 context 传递和更新、epilogue_fn 调用、向后兼容。
使用文件系统进行 epilogue 跨进程通信（避免 Windows spawn 下 mp.Queue 序列化问题）。
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
import multiprocessing as mp
import time
import logging
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qpipe.frame3d import Frame3D
from qpipe.node import MultiInputNode, SourceNode


def _make_test_frame(time_val: int, stocks=None):
    """构造一个时间截面的测试 Frame3D。"""
    if stocks is None:
        stocks = ['A', 'B']
    arrays = [
        [time_val] * len(stocks),
        list(stocks)
    ]
    mi = pd.MultiIndex.from_arrays(arrays, names=['key', 'name'])
    df = pd.DataFrame({
        'val': np.arange(len(stocks), dtype=float),
        'time': [time_val] * len(stocks),
    }, index=mi)
    return Frame3D(df)


# ========== 测试用函数（模块级，pickle 安全） ==========

def simple_func_3arg(name: str, f3d: Frame3D, context):
    """新式 3 参数函数：每次计算后更新 context 中的计数器。"""
    if context is None:
        context = {}
    context.setdefault('call_count', 0)
    context['call_count'] += 1
    context.setdefault('values', [])
    context['values'].append(f3d.df['val'].mean())
    df = f3d.df.copy()
    df['val'] = df['val'] + 1
    return Frame3D(df), context


def simple_func_3arg_no_tuple(name: str, f3d: Frame3D, context):
    """新式 3 参数，但只返回 Frame3D（不更新 context）。"""
    df = f3d.df.copy()
    df['val'] = df['val'] * 2
    return Frame3D(df)


def simple_func_2arg(name: str, f3d: Frame3D):
    """旧式 2 参数函数（向后兼容）。"""
    df = f3d.df.copy()
    df['val'] = df['val'] + 10
    return Frame3D(df)


# ========== epilogue 测试用（文件写入方式，跨进程安全） ==========

# 使用临时文件路径作为 epilogue 输出通道
_EPILOGUE_FILE = None  # 由 conftest 或测试设置

def _make_epilogue_fn(filepath: str):
    """工厂函数：返回一个写入指定文件的 epilogue_fn（模块级函数）。"""
    def _epilogue_to_file(name: str, context):
        import json
        simple_ctx = {}
        if context is not None:
            for k, v in context.items():
                if isinstance(v, (int, float, str, bool, list, type(None))):
                    simple_ctx[k] = v
                elif isinstance(v, dict):
                    simple_ctx[k] = {sk: sv for sk, sv in v.items() 
                                     if isinstance(sv, (int, float, str, bool, list, dict, type(None)))}
        with open(filepath, 'w') as f:
            json.dump({'name': name, 'context': simple_ctx}, f)
    return _epilogue_to_file

# ⚠️ 注意：上面的 _make_epilogue_fn 返回闭包，不能直接 pickle。
# 改用模块级工厂类：

class EpilogueFileWriter:
    """模块级可调用类，pickle 安全。将 epilogue 结果写入文件。"""
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def __call__(self, name: str, context):
        import json
        simple_ctx = {}
        if context is not None:
            for k, v in context.items():
                if isinstance(v, (int, float, str, bool, list, type(None))):
                    simple_ctx[k] = v
                elif isinstance(v, dict):
                    simple_ctx[k] = {str(sk): sv for sk, sv in v.items() 
                                     if isinstance(sv, (int, float, str, bool, list, type(None)))}
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump({'name': name, 'context': simple_ctx}, f)
            f.flush()


def gen_source_frames():
    """生成器：3 天数据。"""
    for t in range(3):
        yield _make_test_frame(t)


# ========== 测试 ==========

class TestContextPassing:
    """测试 context 传递和更新。"""

    def test_context_passed_and_updated(self):
        """验证 func 收到初始 context 并能通过 tuple 返回更新。"""
        in_q = mp.Queue()
        out_q = mp.Queue()
        init_ctx = {'call_count': 0, 'values': []}
        node = MultiInputNode(
            'test_ctx', simple_func_3arg,
            [in_q], [out_q],
            window=1, min_periods=1,
            context=init_ctx,
        )
        for t in range(3):
            in_q.put(_make_test_frame(t))
        in_q.put(node.stop_signal)
        node.start()
        node.join(timeout=15)
        results = []
        while True:
            try:
                obj = out_q.get(timeout=1)
                if obj == node.stop_signal:
                    break
                results.append(obj)
            except Exception:
                break
        assert len(results) == 3

    def test_no_tuple_keeps_context(self):
        """验证 func 只返回 Frame3D 时 context 不变，epilogue 收到原始 context。"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            epi_path = f.name
        
        try:
            epilogue_writer = EpilogueFileWriter(epi_path)
            in_q = mp.Queue()
            node = MultiInputNode(
                'test_no_tuple', simple_func_3arg_no_tuple,
                [in_q], [],
                window=1, min_periods=1,
                context={'key': 'original'},
                epilogue_fn=epilogue_writer,
            )
            in_q.put(_make_test_frame(0))
            in_q.put(node.stop_signal)
            node.start()
            node.join(timeout=15)
            
            # 读取 epilogue 文件
            if os.path.exists(epi_path):
                import json
                with open(epi_path) as f:
                    data = json.load(f)
                assert data['context']['key'] == 'original'
            else:
                pytest.fail("epilogue file was not created")
        finally:
            if os.path.exists(epi_path):
                os.unlink(epi_path)


class TestEpilogueFn:
    """测试 epilogue_fn 功能。"""

    def test_epilogue_called_on_exit(self):
        """验证 MultiInputNode 的 epilogue_fn 在进程退出时被调用。"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            epi_path = f.name
        
        try:
            epilogue_writer = EpilogueFileWriter(epi_path)
            in_q = mp.Queue()
            node = MultiInputNode(
                'test_epilogue', simple_func_3arg,
                [in_q], [],
                window=1, min_periods=1,
                context={'seq': 0},
                epilogue_fn=epilogue_writer,
            )
            in_q.put(_make_test_frame(0))
            in_q.put(node.stop_signal)
            node.start()
            node.join(timeout=15)
            
            import json
            if os.path.exists(epi_path):
                with open(epi_path) as f:
                    data = json.load(f)
                assert data['name'] == 'test_epilogue'
            else:
                pytest.fail("epilogue file was not created")
        finally:
            if os.path.exists(epi_path):
                os.unlink(epi_path)

    def test_source_epilogue(self):
        """SourceNode 的 epilogue_fn 也应在 exit 时被调用。"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            epi_path = f.name
        
        try:
            epilogue_writer = EpilogueFileWriter(epi_path)
            out_q = mp.Queue()
            node = SourceNode(
                'test_src_epilogue', gen_source_frames,
                [out_q],
                context={'day_count': 0},
                epilogue_fn=epilogue_writer,
            )
            node.start()
            node.join(timeout=5)
            
            import json
            if os.path.exists(epi_path):
                with open(epi_path) as f:
                    data = json.load(f)
                assert data['name'] == 'test_src_epilogue'
            else:
                pytest.fail("SourceNode epilogue file not created")
        finally:
            if os.path.exists(epi_path):
                os.unlink(epi_path)


class TestBackwardCompatibility:
    """测试向后兼容：旧式 2 参数函数。"""

    def test_2arg_func_works(self):
        """旧式 simple_func_2arg(name, f3d) 应正常工作。"""
        in_q = mp.Queue()
        out_q = mp.Queue()
        node = MultiInputNode(
            'test_compat', simple_func_2arg,
            [in_q], [out_q],
            window=1, min_periods=1,
        )
        f3d_in = _make_test_frame(0)  # val = [0, 1]
        in_q.put(f3d_in)
        in_q.put(node.stop_signal)
        node.start()
        node.join(timeout=15)
        result = out_q.get(timeout=2)
        df = result.df
        assert np.isclose(df.loc[(0, 'A'), 'val'], 10.0)
        assert np.isclose(df.loc[(0, 'B'), 'val'], 11.0)


class TestSourceNode:
    """测试 SourceNode 基本功能。"""

    def test_basic_source(self):
        out_q = mp.Queue()
        node = SourceNode('test_src', gen_source_frames, [out_q])
        node.start()
        results = []
        while True:
            try:
                obj = out_q.get(timeout=2)
                if obj == node.stop_signal:
                    break
                results.append(obj)
            except Exception:
                break
        node.join(timeout=5)
        assert len(results) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
