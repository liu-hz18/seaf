"""
因子节点内存缩放实测 — 对比不同 n_stocks 下的 RSS。
"""
import gc, sys
import numpy as np, pandas as pd, psutil
sys.path.insert(0, '.')
from seafquant.data_generator import generate_synthetic_data
from seafquant.factor.quality_merged import compute_quality_merged_factors
from qpipe.frame3d import Frame3D

proc = psutil.Process()

def measure(n_stocks, window):
    buffer = []
    for f3d in generate_synthetic_data(300, n_stocks, noise_ratio=0.3, seed=42):
        mk = f3d.df.index.get_level_values(0).max()
        buffer.append(Frame3D(f3d.df[f3d.df.index.get_level_values(0) == mk].copy()))
        if len(buffer) >= window + 5:
            break
    gc.collect()
    # 跑 5 轮取稳态
    for i in range(5):
        win = buffer[i : i + window]
        wdf = pd.concat([f.df for f in win], axis=0).sort_index(level=0)
        lt = wdf.index.get_level_values(0).max()
        ls = wdf.loc[lt].index.tolist()
        at = sorted(wdf.index.get_level_values(0).unique())
        fmi = pd.MultiIndex.from_product([at, ls], names=wdf.index.names)
        wdf = wdf.reindex(fmi).sort_index(level=0)
        run_f3d = Frame3D(wdf)
        result = compute_quality_merged_factors('test', run_f3d, None)
    gc.collect()
    return proc.memory_info().rss / 1024 / 1024

print(f'{"n_stocks":>10s}  {"window":>7s}  {"RSS(MB)":>8s}  {"数据增量":>10s}')
prev_rss = measure(20, 130)
baseline = 84
prev_data = prev_rss - baseline
print(f'{20:>10d}  {130:>7d}  {prev_rss:>8.1f}  {"—":>10s}')

for ns in [50, 100, 200]:
    rss = measure(ns, 130)
    data = rss - baseline
    ratio = ns / 20
    print(f'{ns:>10d}  {130:>7d}  {rss:>8.1f}  +{rss-prev_rss:>6.1f}MB (理论 {prev_data*(ratio-1):.0f}MB)')
    prev_rss = rss

print(f'\n基线 (imports): {baseline}MB')
print(f'20 stocks 数据部分: {prev_data:.0f}MB')
print(f'线性缩放系数: ~{(measure(50,130)-baseline)/(measure(20,130)-baseline):.1f}x (50/20={2.5})')
