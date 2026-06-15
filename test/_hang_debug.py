"""隔离 mp.Queue 清理问题的诊断脚本。"""
import multiprocessing as mp
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qpipe.node import MultiInputNode
from qpipe.frame3d import Frame3D
import pandas as pd


def _f(name, f3d, ctx=None):
    return f3d


def test_with_cleanup():
    q_in = mp.Queue()
    q_out = mp.Queue()

    node = MultiInputNode('test', _f, [q_in], [q_out],
                          window=2, min_periods=2, stop_signal='__STOP__')

    t = pd.Timestamp('2020-01-02')
    mi = pd.MultiIndex.from_product([[t], ['S0']], names=['key', 'code'])
    q_in.put(Frame3D(pd.DataFrame({'a': [1.0]}, index=mi)))
    t2 = pd.Timestamp('2020-01-03')
    mi2 = pd.MultiIndex.from_product([[t2], ['S0']], names=['key', 'code'])
    q_in.put(Frame3D(pd.DataFrame({'a': [2.0]}, index=mi2)))
    q_in.put('__STOP__')

    node.start()
    node.join(timeout=10)
    print(f'Exit code: {node.exitcode}, alive: {node.is_alive()}')

    # === 关键：清理队列 ===
    q_in.close()
    q_in.join_thread()
    q_out.close()
    q_out.join_thread()
    print('Queues cleaned up')


if __name__ == '__main__':
    print('=== Test with cleanup ===')
    test_with_cleanup()
    print('=== DONE ===')
