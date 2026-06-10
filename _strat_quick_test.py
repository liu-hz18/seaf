"""快速验证 strategy 模块基本逻辑。"""
import numpy as np, pandas as pd
from qpipe.frame3d import Frame3D
from seafquant.strategy import strategy_fn

times3 = [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-06')]
stocks = ['S{:04d}'.format(i) for i in range(20)]
np.random.seed(42)
ctx = {}

for i in range(1, len(times3)):
    ts = times3[i-1:i+1]
    mi = pd.MultiIndex.from_product([ts, stocks], names=['key', 'name'])
    df = pd.DataFrame({
        'pred_signal': np.random.randn(len(stocks) * 2),
        'close': np.abs(100 + np.random.randn(len(stocks) * 2) * 2),
        'close_uq': np.abs(100 + np.random.randn(len(stocks) * 2) * 2),
    }, index=mi)
    strategy_fn('test', Frame3D(df), ctx)
    total_trades = sum(len(g['trade_log']) for g in ctx['groups'])
    nav0 = [g['nav_log'][-1]['total_equity'] for g in ctx['groups'] if g['nav_log']]
    print(f'Call {i}: trades={total_trades}, nav_sum={sum(nav0):.0f}')

# 验证每个 group 的 cash 变化
for g in ctx['groups']:
    n_trades = len(g['trade_log'])
    n_buys = sum(1 for t in g['trade_log'] if t['action'] == 'buy')
    nav = g['nav_log'][-1]['total_equity']
    print(f'  G{g["group_id"]}: cash={g["cash"]:.0f} nav={nav:.0f} trades={n_trades}({n_buys}b)')
