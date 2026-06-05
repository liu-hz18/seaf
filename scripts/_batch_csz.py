"""批量替换 cs_zscore_batch(factor_cols) -> cs_zscore_batch(factor_cols, cp=False)"""
import glob

factor_dir = 'seafquant/factor'
changed = 0
for fpath in sorted(glob.glob(f'{factor_dir}/factors_*.py')):
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    old = 'cs_zscore_batch(factor_cols)'
    new = 'cs_zscore_batch(factor_cols, cp=False)'
    if old in content and new not in content:
        content = content.replace(old, new)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        changed += 1
        print(f'OK  {fpath}')
    elif new in content:
        print(f'DONE {fpath} (already applied)')
    else:
        print(f'NO   {fpath}')

print(f'\nChanged {changed} files.')
