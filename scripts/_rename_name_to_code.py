"""全局重命名 MultiIndex level 'name' → 'code' + strategy 修复。

用法: python _rename_name_to_code.py [--dry-run]
"""
import pathlib
import re
import sys

DRY_RUN = '--dry-run' in sys.argv

FILES = sorted(pathlib.Path('.').rglob('*.py'))
EXCLUDE = {'__init__.py', '_rename_name_to_code.py', 'api_test.py',
           'setup.py', 'conftest.py'}

# ═══════════════════════════════════════════════════════════
# Phase 1: MultiIndex level 'name' → 'code'
# ═══════════════════════════════════════════════════════════
# 精确替换规则：仅当 'name' 出现在 MultiIndex 上下文中
# 1. names=['key', 'name'] → names=['key', 'code']
# 2. level='name' / level="name" → level='code' / level="code"  
# 3. get_level_values('name') → get_level_values('code')
# 4. .xs(..., level='name') → .xs(..., level='code')
# 5. index.names == ['key', 'name'] → ['key', 'code']

PATTERNS_INDEX = [
    (r"names=\['key',\s*'name'\]", "names=['key', 'code']"),
    (r'names=\("key",\s*"name"\)', 'names=("key", "code")'),
    (r"level='name'", "level='code'"),
    (r'level="name"', 'level="code"'),
    (r"get_level_values\('name'\)", "get_level_values('code')"),
    (r'get_level_values\("name"\)', 'get_level_values("code")'),
    # .xs(..., level='name')
    (r"\.xs\(([^)]*?)level='name'", r".xs(\1level='code'"),
    (r'\.xs\(([^)]*?)level="name"', r'.xs(\1level="code"'),
    # index.names assertions
    (r"\.names\s*==\s*\[(['\"])key\1,\s*(['\"])name\2\]", ".names == ['key', 'code']"),
]

# ═══════════════════════════════════════════════════════════
# Phase 2: strategy 修复
# ═══════════════════════════════════════════════════════════

def fix_strategy_files():
    """修复 strategy 模块中的价格精度和 stock_name/stock_id 问题."""
    
    # --- strategy_core.py: _log_trade 价格精度 ---
    _fix_strategy_core()
    # --- strategy_daily.py: _on_bar 价格精度 + stock_name ---
    _fix_strategy_daily()
    # --- strategy.py: stock_name 传递 ---
    _fix_strategy_main()
    # --- pipeline.py: strategy 节点 input_columns 增加 stock_name ---
    _fix_pipeline()


def _fix_strategy_core():
    """_log_trade 中 price/value/commission/signal_value/hfq_price round 到 2 位小数."""
    f = pathlib.Path('seafquant/strategy_core.py')
    text = f.read_text('utf-8')
    
    # _log_trade: round price, value, commission, signal_value, hfq_price
    # 当前：无 round
    # 修改：在 trade_log 记录时 round 价格相关字段
    old_log = """    ctx['trade_log'].append({
        'date': date, 'stock_id': stock_id, 'action': action,
        'shares': shares, 'price': price, 'value': value,
        'commission': commission,
        'signal_value': signal_value,
        'hfq_price': hfq_price,
    })"""
    new_log = """    ctx['trade_log'].append({
        'date': date, 'code': stock_id, 'action': action,
        'shares': shares, 'price': round(price, 2), 'value': round(value, 2),
        'commission': round(commission, 2),
        'signal_value': round(signal_value, 4),
        'hfq_price': round(hfq_price, 2),
    })"""
    if old_log in text:
        text = text.replace(old_log, new_log)
    else:
        print('WARNING: _log_trade pattern not found in strategy_core.py')
    
    if not DRY_RUN:
        f.write_text(text, 'utf-8')
    print(f'  strategy_core.py: price precision fixed')


def _fix_strategy_daily():
    """_on_bar 中 position_log 价格精度 + stock_id→code + stock_name."""
    f = pathlib.Path('seafquant/strategy_daily.py')
    text = f.read_text('utf-8')
    
    # position_log: round market_value, p_uq, p_hfq, signal_value
    old_pos = """        ctx['position_log'].append({
            'date': date,
            'day_counter': dc,
            'stock_id': sid,"""
    new_pos = """        ctx['position_log'].append({
            'date': date,
            'day_counter': dc,
            'code': sid,"""
    if old_pos in text:
        text = text.replace(old_pos, new_pos)
    
    old_mv = "'market_value': market_value,"
    new_mv = "'market_value': round(market_value, 2),"
    text = text.replace(old_mv, new_mv)
    
    old_puq = "'p_uq': close_uq.get(sid, 0.0),"
    new_puq = "'p_uq': round(close_uq.get(sid, 0.0), 2),"
    text = text.replace(old_puq, new_puq)
    
    old_phfq = "'p_hfq': close_hfq.get(sid, 0.0),"
    new_phfq = "'p_hfq': round(close_hfq.get(sid, 0.0), 2),"
    text = text.replace(old_phfq, new_phfq)
    
    old_sv = "'signal_value': sig_info.get('v', 0.0),"
    new_sv = "'signal_value': round(sig_info.get('v', 0.0), 4),"
    text = text.replace(old_sv, new_sv)
    
    # _generate_daily_plan: stock_id → code
    old_plan_sid = "'stock_id': sid,"
    new_plan_sid = "'code': sid,"
    text = text.replace(old_plan_sid, new_plan_sid)
    
    if not DRY_RUN:
        f.write_text(text, 'utf-8')
    print(f'  strategy_daily.py: price precision + stock_id→code')


def _fix_strategy_main():
    """strategy.py: 传递 stock_name 到各层."""
    f = pathlib.Path('seafquant/strategy.py')
    text = f.read_text('utf-8')
    
    # 需要从 f3d 中提取 stock_name 映射 (code → stock_name)
    # 在提取 close_uq_t 处附近添加 stock_name 提取
    
    # 查找 "close_uq_t = df.xs(t_curr, level='key')['close_uq'].to_dict()"
    # 在其后添加 stock_name 提取
    old_extract = """    close_hfq_t = df.xs(t_curr, level='key')['close'].to_dict()

    # ---- 首次调用：用 T-1 信号为每个 group 初始化 pending_signal ----"""
    
    new_extract = """    close_hfq_t = df.xs(t_curr, level='key')['close'].to_dict()

    # 股票名映射（artifact 导出用）
    stock_name_map = {}
    if 'stock_name' in df.columns:
        stock_name_map = df.xs(t_curr, level='key')['stock_name'].to_dict()

    # ---- 首次调用：用 T-1 信号为每个 group 初始化 pending_signal ----"""
    
    if old_extract in text:
        text = text.replace(old_extract, new_extract)
    
    # 修改 _on_bar 调用，传递 stock_name_map
    # 当前: _on_bar(gctx, t_curr, sig, close_uq_t, close_hfq_t)
    old_onbar = '_on_bar(gctx, t_curr, sig, close_uq_t, close_hfq_t)'
    new_onbar = '_on_bar(gctx, t_curr, sig, close_uq_t, close_hfq_t, stock_name_map)'
    text = text.replace(old_onbar, new_onbar)
    
    # 修改 _generate_daily_plan 调用
    old_plan = '_generate_daily_plan(\n                gctx, t_curr, dc, gctx[\'pending_signal\'],\n                close_uq_t, close_hfq_t,'
    new_plan = '_generate_daily_plan(\n                gctx, t_curr, dc, gctx[\'pending_signal\'],\n                close_uq_t, close_hfq_t, stock_name_map,'
    text = text.replace(old_plan, new_plan)
    
    if not DRY_RUN:
        f.write_text(text, 'utf-8')
    print(f'  strategy.py: stock_name_map extraction')


def _fix_pipeline():
    """pipeline.py: strategy 节点 input_columns 增加 stock_name."""
    f = pathlib.Path('pipeline.py')
    text = f.read_text('utf-8')
    
    old_cols = "input_columns=['pred_signal', 'close', 'close_uq'],"
    new_cols = "input_columns=['pred_signal', 'close', 'close_uq', 'stock_name'],"
    text = text.replace(old_cols, new_cols)
    
    if not DRY_RUN:
        f.write_text(text, 'utf-8')
    print(f'  pipeline.py: strategy input_columns += stock_name')


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

print('=== Phase 1: Index level rename (name → code) ===')
for f in FILES:
    if f.name in EXCLUDE:
        continue
    text = f.read_text('utf-8')
    orig = text
    for pat, repl in PATTERNS_INDEX:
        text = re.sub(pat, repl, text)
    if text != orig:
        print(f'  {f}')
        if not DRY_RUN:
            f.write_text(text, 'utf-8')

print()
print('=== Phase 2: Strategy fixes ===')
fix_strategy_files()

if DRY_RUN:
    print('\n[DRY RUN] No files modified.')
else:
    print('\nDone. All files updated.')
