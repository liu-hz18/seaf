import sys
sys.path.insert(0, '.')
from qpipe.flow import Flow
from qpipe.frame3d import Frame3D
import pandas as pd

f = Flow()

def _p(n, f3d, c=None): return f3d
def _g():
    mi = pd.MultiIndex.from_product(
        [[pd.Timestamp('2020-01-01')], ['X']], names=['key', 'code'])
    yield Frame3D(pd.DataFrame({'a': [1.0]}, index=mi))

f.add_source('src', _g, ['q1', 'q2', 'q3'])
f.add_node('A', _p, 'q1', ['qA1', 'qA2'])
f.add_node('B', _p, 'q2', ['qB'])
f.add_node('C', _p, ['qA1', 'qB'], ['qC'])
f.add_node('D', _p, 'qA2', [])
f.add_node('E', _p, 'qC', [])
f.add_node('F', _p, 'q3', [])

print(repr(f))
