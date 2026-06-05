import re
path = 'seafquant/factors.py'
t = open(path, 'r', encoding='utf-8').read()
# Remove quality old import
t = re.sub(r"from seafquant\.factor\.factors_quality import compute_quality_factors.*\n", '', t)
# Remove from registry
t = re.sub(r"\s+'quality': compute_quality_factors,\n", '', t)
# Remove from prefixes
t = re.sub(r"\s+'quality': 'factor_qual',\n", '', t)
# Remove from __all__
t = re.sub(r"\s+'compute_quality_factors',\n", '', t)
open(path, 'w', encoding='utf-8').write(t)
print('OK')
