"""补丁：修复 frame3d.py 中遗漏的 groupby('name') → groupby('code')。"""
import pathlib

f = pathlib.Path('qpipe/frame3d.py')
text = f.read_text('utf-8')
old = text
text = text.replace("groupby('name')", "groupby('code')")
if text != old:
    f.write_text(text, 'utf-8')
    print(f'Fixed {old.count("groupby")}→{text.count("groupby")} groupby calls in {f}')
else:
    print('No changes needed')
