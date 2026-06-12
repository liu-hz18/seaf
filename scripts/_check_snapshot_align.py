"""检查最新 mlflow run 的 snapshot 对齐情况。"""
import glob
import os
import re
from collections import defaultdict

files = sorted(glob.glob(
    'mlruns/4/81498d6af46549e99c418ee7c569cc25/artifacts/snapshots/*/*.csv',
    recursive=True,
))

print(f'Total snapshot files: {len(files)}')
print()

# 按日期分组
by_date: dict[str, list[str]] = defaultdict(list)
for f in files:
    m = re.search(r'(\d{4}-\d{2}-\d{2})', f)
    if m:
        date = m.group(1)
        node_name = f.replace('\\', '/').split('/')[-2]
        by_date[date].append(node_name)

for date in sorted(by_date):
    nodes = sorted(set(by_date[date]))
    print(f'{date} ({len(nodes)} nodes): {nodes}')
