"""演示 Flow.__repr__ 拓扑可视化。"""
from qpipe.flow import Flow
from qpipe.frame3d import Frame3D
import pandas as pd


def _passthru(name, f3d, ctx=None):
    return f3d


def _gen():
    mi = pd.MultiIndex.from_product(
        [[pd.Timestamp('2020-01-01')], ['X']], names=['key', 'code'])
    yield Frame3D(pd.DataFrame({'a': [1.0]}, index=mi))


f = Flow()
f.add_source('data_source', _gen, ['q_raw'])
f.add_node('factors', _passthru, 'q_raw', ['q_factors', 'q_aux'])
f.add_node('model', _passthru, 'q_factors', ['q_signals'])
f.add_node('ic_analysis', _passthru, 'q_signals', [])
f.add_node('strategy', _passthru, 'q_aux', ['q_trades'])
f.add_node('risk', _passthru, 'q_trades', [])

print(repr(f))
