"""Batch-add __future__ annotations to all seafquant/factor/*.py files."""

import os, re

factor_dir = os.path.join(os.path.dirname(__file__), '..', 'seafquant', 'factor')

for fname in sorted(os.listdir(factor_dir)):
    if not fname.endswith('.py') or fname == '__init__.py':
        continue
    fpath = os.path.join(factor_dir, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'from __future__' in content:
        print(f'SKIP {fname}')
        continue

    # Add after module docstring
    first = content.find('"""')
    end_doc = content.find('"""', first + 3)
    insert_pos = end_doc + 3
    content = content[:insert_pos] + '\nfrom __future__ import annotations' + content[insert_pos:]

    # Clean typing imports
    content = re.sub(r'from typing import .*', 'from typing import Any', content)

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'OK  {fname}')
