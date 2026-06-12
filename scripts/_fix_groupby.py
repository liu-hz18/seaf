"""补丁：修复因子模块中遗漏的 groupby('name') → groupby('code')。"""
import pathlib
import re

FACTOR_DIR = pathlib.Path('seafquant/factor')

for f in sorted(FACTOR_DIR.glob('*.py')):
    if f.name in ('__init__.py',):
        continue
    text = f.read_text('utf-8')
    # 替换 groupby('name') 和 groupby("name")
    old_text = text
    text = re.sub(r"""groupby\((['\"])name\1\)""", r"groupby(\1code\1)", text)
    if text != old_text:
        f.write_text(text, 'utf-8')
        print(f'Fixed: {f}')
