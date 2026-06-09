"""快速验证 IPO/退市机制"""
from seafquant.data_generator import generate_synthetic_data

gen = generate_synthetic_data(200, 50, seed=42, noise_ratio=0.5)
frames = list(gen)

names_day0 = sorted(frames[0].df.index.get_level_values(1).unique())
names_last = sorted(frames[-1].df.index.get_level_values(1).unique())
print(f'Day 0: {len(names_day0)} stocks')
print(f'Last day: {len(names_last)} stocks')

all_names = set()
for f in frames:
    all_names.update(f.df.index.get_level_values(1))
print(f'Total unique across all days: {len(all_names)}')

min_close = min(f.df['close'].min() for f in frames)
print(f'Min close price: {min_close:.6f}')

n_stocks = sorted([n for n in all_names if n.startswith('N')])
print(f'Newly listed (N-prefix): {len(n_stocks)} stocks')
if n_stocks:
    print(f'  Sample: {n_stocks[:10]}')

removed = set(names_day0) - set(names_last)
added = set(names_last) - set(names_day0)
print(f'Removed from day0: {len(removed)}')
print(f'Added after day0: {len(added)}')
if removed:
    print(f'  Removed: {sorted(removed)[:10]}')
if added:
    print(f'  Added: {sorted(added)[:10]}')
