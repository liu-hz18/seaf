"""
因子节点全量性能基准（多进程并行版）。
每个节点在独立子进程中运行，单 CPU 核绑定，避免顺序执行时的
内存压力、缓存污染和热节流干扰。
"""

from __future__ import annotations

import gc
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from qpipe.frame3d import Frame3D
from seafquant.factors import FACTOR_REGISTRY


def _make_random_f3d(n_times: int = 200, n_stocks: int = 100, seed: int = 42) -> Frame3D:
    """生成随机 Frame3D。"""
    rng = np.random.default_rng(seed)
    times = np.repeat(np.arange(n_times), n_stocks)
    codes = np.tile([f's{i:04d}' for i in range(n_stocks)], n_times)
    mi = pd.MultiIndex.from_arrays([times, codes], names=['key', 'code'])
    log_ret = rng.normal(0.0002, 0.02, size=(n_times, n_stocks))
    close_arr = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    close_arr = np.maximum(close_arr, 0.01)
    intraday_vol = close_arr * rng.uniform(0.005, 0.03, size=(n_times, n_stocks))
    # 估值列（模拟真实 A 股分布）
    eps_arr = close_arr * rng.lognormal(-4, 1.2, size=(n_times, n_stocks))
    bvps_arr = close_arr * rng.lognormal(-2, 0.8, size=(n_times, n_stocks))
    sps_arr = close_arr * rng.lognormal(-3, 1.5, size=(n_times, n_stocks))
    cfps_arr = eps_arr * rng.lognormal(0, 0.6, size=(n_times, n_stocks))

    df = pd.DataFrame({
        'open': (close_arr * (1 + rng.normal(0, 0.005, size=(n_times, n_stocks)))).ravel(),
        'high': (close_arr + np.abs(intraday_vol)).ravel(),
        'low': (close_arr - np.abs(intraday_vol)).ravel(),
        'close': close_arr.ravel(),
        'volume': rng.lognormal(15, 1.5, size=(n_times, n_stocks)).ravel().astype(np.int64),
        'turnover': rng.lognormal(-3, 1.0, size=(n_times, n_stocks)).ravel(),
        'market_cap': (close_arr * rng.lognormal(18, 2, size=(n_times, n_stocks))).ravel(),
        'peTTM': (close_arr / np.maximum(eps_arr, 0.001)).ravel(),
        'pbMRQ': (close_arr / np.maximum(bvps_arr, 0.001)).ravel(),
        'psTTM': (close_arr / np.maximum(sps_arr, 0.001)).ravel(),
        'pcfNcfTTM': (close_arr / np.maximum(cfps_arr, 0.001)).ravel(),
    }, index=mi)
    nan_mask = rng.random(len(df)) < 0.02
    df.loc[nan_mask, :] = np.nan
    return Frame3D(df)


# ── 子进程入口（模块级别，Windows spawn 要求 pickle 可达）──

def _bench_one_module(args: tuple[str, int, int]) -> tuple[str, float, str]:
    """在子进程中运行单个模块的基准测试。

    args = (name, n_stocks, n_trials)
    返回: (name, mean_seconds, error_msg)
    """
    name, n_stocks, n_trials = args
    try:
        fn = FACTOR_REGISTRY[name]
        # 预热 (单独生成数据避免污染计时的缓存)
        warm = _make_random_f3d(n_stocks=n_stocks, seed=0)
        fn(name, 0, warm, None)
        del warm
        gc.collect()

        times = []
        for trial in range(n_trials):
            f3d = _make_random_f3d(n_stocks=n_stocks, seed=42 + trial)
            gc.collect()
            t0 = time.perf_counter()
            fn(name, 0, f3d, None)
            times.append(time.perf_counter() - t0)
            del f3d

        return name, float(np.mean(times)), ''
    except Exception as exc:
        import traceback
        print(f"{exc}\n{traceback.format_exc()}")
        return name, float('nan'), str(exc)


# ── 主进程 ──

def print_table(data: dict[str, dict[int, float]], stock_counts: list[int]) -> None:
    """打印性能表格。"""
    names = sorted(data.keys(),
                   key=lambda n: data[n].get(stock_counts[-1], float('inf')),
                   reverse=True)
    print(f'\n{"="*90}')
    header = f'{"Module":<25s}'
    for sc in stock_counts:
        header += f' {f"{sc}s":>10s}'
    print(header)
    print('-' * (25 + 11 * len(stock_counts)))
    totals = {sc: 0.0 for sc in stock_counts}
    for name in names:
        row = f'{name:<25s}'
        for sc in stock_counts:
            v = data[name].get(sc, float('nan'))
            if np.isnan(v):
                row += f' {"N/A":>10s}'
            else:
                row += f' {v:10.3f}'
                totals[sc] += v
        print(row)
    print('-' * (25 + 11 * len(stock_counts)))
    total_row = f'{"TOTAL":<25s}'
    for sc in stock_counts:
        total_row += f' {totals[sc]:10.3f}'
    print(total_row)
    print(f'{"="*90}\n')


if __name__ == '__main__':
    stock_counts = [100, 500, 1000, 2000, 5000]
    names = list(FACTOR_REGISTRY.keys())
    n_trials = 10
    max_workers = 8  # 单核串行，定时准确

    all_data: dict[str, dict[int, float]] = {}

    import random

    for sc in stock_counts:
        print(f'\n{"="*60}')
        print(f'基准测试: {sc} 股票 ({max_workers} 进程并行)...')
        print(f'{"="*60}')

        # 构造任务参数
        tasks = [(name, sc, n_trials) for name in names]
        random.shuffle(tasks)

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(_bench_one_module, t): t[0] for t in tasks}
            completed = 0
            for future in as_completed(future_map):
                name = future_map[future]
                mod_name, elapsed, err = future.result()
                completed += 1
                if err:
                    print(f'  [{completed}/{len(names)}] {mod_name}: ERROR - {err}')
                else:
                    print(f'  [{completed}/{len(names)}] {mod_name}: {elapsed:.3f}s')
                all_data.setdefault(mod_name, {})[sc] = elapsed

    print_table(all_data, stock_counts)
