"""批量替换 factors_ 导入引用为去掉前缀的新路径"""

import glob, re

files_to_update = [
    'seafquant/factors.py',
    'pipeline.py',
    'seafquant/factor/__init__.py',
]

for fpath in files_to_update:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = re.sub(
        r'from seafquant\.factor\.factors_(\w+) import', r'from seafquant.factor.\1 import', content
    )
    new_content = re.sub(
        r'import seafquant\.factor\.factors_(\w+)', r'import seafquant.factor.\1', new_content
    )
    if new_content != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'OK  {fpath}')
    else:
        print(f'NO  {fpath}')
