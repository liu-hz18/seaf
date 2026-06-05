"""
SEAF 因子模块全面性能基准测试
测试全部 20 个活跃因子模块，多次运行取平均时延，找出瓶颈。
"""
import time, sys, os
sys.path.insert(0, '.')
import pandas as pd
from qpipe.frame3d import Frame3D
from seafquant.data_generator import generate_synthetic_data
from seafquant.factors import FACTOR_REGISTRY

# ---- 配置 ----
N_TIMES = 200
N_STOCKS = 500
N_RUNS = 10
NOISE_RATIO = 0.3
SEED = 42

# 活跃模块列表（与 pipeline.py 一致）
ACTIVE_MODULES = [
    'momentum', 'reversal', 'volatility', 'liquidity', 'value',
    'quality_basic', 'quality_advanced', 'quality_autocorr',
    'quality_pattern', 'quality_sign',
    'trend', 'trend_macd', 'size',
    'counting', 'counting_streak', 'counting_nh',
    'intraday', 'interaction', 'cross_section', 'cross_section_neut',
]

# ---- 生成数据 ----
print(f'生成数据: {N_TIMES}t x {N_STOCKS}s ...', end=' ', flush=True)
t0 = time.perf_counter()
gen = generate_synthetic_data(n_times=N_TIMES, n_stocks=N_STOCKS,
                              noise_ratio=NOISE_RATIO, seed=SEED)
frames = [f.df for f in gen]
big_df = pd.concat(frames, axis=0).sort_index(level=0)
f3d = Frame3D(big_df)
print(f'{time.perf_counter()-t0:.1f}s  ->  {f3d}')

# ---- 预热 + 基准测试 ----
results = {}
for module_name in ACTIVE_MODULES:
    func = FACTOR_REGISTRY[module_name]
    # 预热一次
    _ = func(module_name, f3d, None)
    # 多次测量
    times = []
    for run in range(N_RUNS):
        # print(f"{module_name}-{run}")
        t_start = time.perf_counter()
        result = func(module_name, f3d, None)
        elapsed = time.perf_counter() - t_start
        times.append(elapsed)
    avg = sum(times) / len(times)
    cols = [c for c in result.df.columns]
    results[module_name] = {'avg': avg, 'min': min(times), 'max': max(times),
                            'n_cols': len(cols), 'cols': cols}
    print(f'  {module_name:25s}  avg={avg:.4f}s  min={min(times):.4f}s  '
          f'max={max(times):.4f}s  cols={len(cols)}')

# ---- 排序汇总 ----
print('\n' + '=' * 70)
print(f'{"模块":<25s} {"平均(s)":>8s}  {"列数":>5s}  相对最慢')
print('-' * 70)
sorted_mods = sorted(results.items(), key=lambda x: -x[1]['avg'])
worst = sorted_mods[0][1]['avg']
for name, info in sorted_mods:
    bar = '█' * int(info['avg'] / worst * 40) if worst > 0 else ''
    print(f'{name:<25s} {info["avg"]:>8.4f}  {info["n_cols"]:>5d}  {bar}')

print(f'\n瓶颈: {sorted_mods[0][0]} = {worst:.4f}s')
print(f'最快: {sorted_mods[-1][0]} = {sorted_mods[-1][1]["avg"]:.4f}s')
print(f'并行瓶颈/最快比: {worst / sorted_mods[-1][1]["avg"]:.1f}x')
