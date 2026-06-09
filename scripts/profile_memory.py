"""
内存监测脚本 — 运行 pipeline 并每 10s 采样各进程 RSS / 队列大小。

用法:
    python scripts/profile_memory.py --n-times 300 --n-stocks 50 --model-type lgbm --fwd 20 --model-window 250 --no-mlflow

输出 memory_profile_{timestamp}.csv 到当前目录。
需要 psutil: pip install psutil
"""
from __future__ import annotations

import argparse
import csv
import datetime
import multiprocessing as mp
import os
import subprocess
import sys
import time

try:
    import psutil
except ImportError:
    print('请先安装 psutil: pip install psutil', file=sys.stderr)
    sys.exit(1)

# ——————————————————————————————————————————————
# 参数解析
# ——————————————————————————————————————————————
parser = argparse.ArgumentParser(description='SEAF Memory Profiler')
parser.add_argument('--n-times', type=int, default=300)
parser.add_argument('--n-stocks', type=int, default=50)
parser.add_argument('--noise-ratio', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--start-date', default='2020-01-02')
parser.add_argument('--model-type', default='ridge', choices=['lgbm', 'ridge', 'mlp'])
parser.add_argument('--fwd', type=int, default=20)
parser.add_argument('--model-window', type=int, default=250)
parser.add_argument('--no-mlflow', action='store_true', default=True)
parser.add_argument('--interval', type=float, default=5.0, help='采样间隔 (秒)')
parser.add_argument('--log-level', default='WARNING', help='减少日志噪音')
args = parser.parse_args()

# ——————————————————————————————————————————————
# 启动 pipeline 子进程
# ——————————————————————————————————————————————
pipeline_args = [
    sys.executable,
    os.path.join(os.path.dirname(__file__), '..', 'pipeline.py'),
    f'--n-times={args.n_times}',
    f'--n-stocks={args.n_stocks}',
    f'--noise-ratio={args.noise_ratio}',
    f'--seed={args.seed}',
    f'--start-date={args.start_date}',
    f'--model-type={args.model_type}',
    f'--fwd={args.fwd}',
    f'--model-window={args.model_window}',
    f'--log-level={args.log_level}',
]
if args.no_mlflow:
    pipeline_args.append('--no-mlflow')

proc = subprocess.Popen(pipeline_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
parent = psutil.Process(proc.pid)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = f'memory_profile_{timestamp}.csv'

print(f'[Profiler] Pipeline PID={proc.pid}, 采样间隔={args.interval}s, 输出={csv_path}')
print(f'[Profiler] 运行中... (n_times={args.n_times}, n_stocks={args.n_stocks})')

# ——————————————————————————————————————————————
# 采样循环
# ——————————————————————————————————————————————
fieldnames = ['elapsed_s', 'main_rss_mb', 'total_rss_mb', 'n_children',
              'max_child_rss_mb', 'avg_child_rss_mb']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    start_time = time.time()

    while proc.poll() is None:
        elapsed = time.time() - start_time
        try:
            children = parent.children(recursive=True)
            main_rss = parent.memory_info().rss / 1024 / 1024
            child_rss_list = [c.memory_info().rss / 1024 / 1024 for c in children
                              if c.pid != proc.pid]
            total_rss = main_rss + sum(child_rss_list)

            row = {
                'elapsed_s': f'{elapsed:.1f}',
                'main_rss_mb': f'{main_rss:.1f}',
                'total_rss_mb': f'{total_rss:.1f}',
                'n_children': len(child_rss_list),
                'max_child_rss_mb': f'{max(child_rss_list):.1f}' if child_rss_list else '0',
                'avg_child_rss_mb': f'{sum(child_rss_list)/len(child_rss_list):.1f}' if child_rss_list else '0',
            }
            writer.writerow(row)
            f.flush()

            now_str = datetime.datetime.now().strftime('%H:%M:%S')
            print(f'  [{now_str}] t={elapsed:6.1f}s  main={main_rss:6.1f}MB  '
                  f'total={total_rss:7.1f}MB  children={len(child_rss_list)}  '
                  f'max_child={row["max_child_rss_mb"]}MB')

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        time.sleep(args.interval)

# ——————————————————————————————————————————————
# 完成
# ——————————————————————————————————————————————
rc = proc.returncode
print(f'[Profiler] Pipeline 退出 (rc={rc}), 耗时 {time.time()-start_time:.1f}s')
print(f'[Profiler] 结果: {csv_path}')
