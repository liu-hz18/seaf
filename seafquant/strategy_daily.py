"""
策略日度引擎 — 逐日交易执行 + 次日交易计划生成。

_on_bar：每个交易日调用一次，执行昨日 pending_signal、存储今日信号、记录净值。
_generate_daily_plan：基于 T 日信号和收盘价，生成 T+1 日交易计划（先卖后买）。
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from seafquant.strategy_core import (
    _compute_total_equity,
    _get_actual_shares,
    _get_position_value,
    _process_close_trade,
    _process_delta_trade,
    _process_new_trade,
)

# =============================================================================
# 单日 on_bar — 执行 trading logic
# =============================================================================


def _on_bar(
    ctx: dict, date, signal: dict[str, float],
    close_uq: dict[str, float], close_hfq: dict[str, float],
) -> None:
    """逐日调用：对齐信号、交易与净值。

    T 日收盘收到 signal_T → 存储为 pending；
    同时执行 pending（signal_{T-1}），用 T 日不复权价撮合。
    """
    ctx['day_counter'] += 1
    dc = ctx['day_counter']

    # Step 1: 复权因子 F_T = hfq / uq
    f_today: dict[str, float] = {}
    for sid, puq in close_uq.items():
        phfq = close_hfq.get(sid, 0.0)
        if puq > 0 and phfq > 0:
            f_today[sid] = phfq / puq

    # Step 2: 执行昨日待执行信号
    if ctx['pending_signal'] is not None:
        n_trades_before = len(ctx['trade_log'])
        sig = ctx['pending_signal']  # {sid: {'w': weight, 'v': signal_value}}
        maturing: dict[str, list] = defaultdict(list)
        for key, pos in list(ctx['positions'].items()):
            if pos['mature_dc'] == dc:
                maturing[pos['stock_id']].append(key)

        total_equity = _compute_total_equity(ctx, close_uq, close_hfq)
        slice_capital = max(total_equity, 1.0) / ctx['fwd']

        sig_sids = set(sig.keys())
        mat_sids = set(maturing.keys())

        # 先卖后买：优先卖出释放现金，再买入分配资金
        # 1. 到期-信号 → 平仓（纯卖）
        for sid in mat_sids - sig_sids:
            _process_close_trade(
                ctx, date, dc, sid, maturing[sid], close_uq, f_today,
                close_hfq=close_hfq, signal_value=0.0,
            )
        # 2. 信号∩到期 → 差额交易（可能买卖，内部先判断方向）
        for sid in sig_sids & mat_sids:
            sw = sig[sid]['w']
            sv = sig[sid]['v']
            _process_delta_trade(
                ctx, date, dc, sid, sw, slice_capital,
                maturing[sid], close_uq, close_hfq, f_today, signal_value=sv,
            )
        # 3. 信号-到期 → 新开仓（纯买，最后执行确保现金充足）
        for sid in sig_sids - mat_sids:
            sw = sig[sid]['w']
            sv = sig[sid]['v']
            _process_new_trade(
                ctx, date, dc, sid, sw, slice_capital, close_uq, f_today,
                close_hfq=close_hfq, signal_value=sv,
            )

        # 累加当日新增手续费
        for t in ctx['trade_log'][n_trades_before:]:
            ctx['cumsum_fee'] += t['commission']

    # Step 3: 存储今日信号
    ctx['pending_signal'] = signal

    # Step 4: 净值记录
    total_equity = _compute_total_equity(ctx, close_uq, close_hfq)
    initial_cash = ctx.get('initial_cash', total_equity)
    nav = total_equity / initial_cash if initial_cash > 0 else 0.0
    # 更新历史最高净值，计算回撤
    if nav > ctx.get('peak_nav', 0):
        ctx['peak_nav'] = nav
    peak = ctx['peak_nav']
    drawdown = (nav - peak) / peak if peak > 0 else 0.0
    ctx['nav_log'].append({
        'date': date,
        'day_counter': dc,
        'cash': round(ctx['cash'], 2),
        'value': round(total_equity, 2),
        'total_equity': round(total_equity, 2),
        'nav': nav,
        'peak_nav': peak,
        'drawdown': drawdown,
        'cumsum_fee': round(ctx['cumsum_fee'], 2),
        'position_value': total_equity - ctx['cash'],
        'n_positions': len(ctx['positions']),
    })

    # Step 5: 持仓快照
    for pos in ctx['positions'].values():
        sid = pos['stock_id']
        actual_shares = (
            _get_actual_shares(pos, f_today) if sid in f_today else 0.0
        )
        market_value = _get_position_value(pos, close_hfq)
        mkt_pct = market_value / total_equity if total_equity > 0 else 0.0
        sig_info = signal.get(sid, {})
        ctx['position_log'].append({
            'date': date,
            'day_counter': dc,
            'stock_id': sid,
            'batch_dc': pos['batch_dc'],
            'n_initial': pos['n_initial'],
            'f_buy': pos['f_buy'],
            'f_today': f_today.get(sid),
            'actual_shares': actual_shares,
            'market_value': market_value,
            'market_value_pct': mkt_pct,
            'p_uq': close_uq.get(sid, 0.0),
            'p_hfq': close_hfq.get(sid, 0.0),
            'signal_value': sig_info.get('v', 0.0),
            'mature_dc': pos['mature_dc'],
            'entry_date': pos['entry_date'],
        })


# =============================================================================
# 次日交易计划生成
# =============================================================================


def _generate_daily_plan(
    ctx: dict, date, dc: int,
    signal: dict[str, dict[str, float]],
    close_uq: dict[str, float],
    close_hfq: dict[str, float],
) -> pd.DataFrame:
    """生成 T+1 日交易计划（基于 T 日收盘信息估算）。

    T 日收盘后，基于 T 日信号 + T 日收盘价，估算次日每笔交易的目标市值和占比。
    交易顺序：先卖后买（卖=0,1；买=2,3）。

    返回 DataFrame 列：
        date, group_id, stock_id, action, target_value, weight, signal_value, order
    """
    dc_tomorrow = dc + 1

    # 到期持仓
    maturing: dict[str, list] = defaultdict(list)
    for key, pos in list(ctx['positions'].items()):
        if pos['mature_dc'] == dc_tomorrow:
            maturing[pos['stock_id']].append(key)

    total_equity = _compute_total_equity(ctx, close_uq, close_hfq)
    slice_capital = max(total_equity, 1.0) / ctx['fwd']

    sig_sids = set(signal.keys())
    mat_sids = set(maturing.keys())

    plans: list[dict] = []

    # ---- 1. 纯卖：到期且不在新信号中 → 全部平仓 (order=0) ----
    for sid in mat_sids - sig_sids:
        pos_value = sum(
            _get_position_value(ctx['positions'][k], close_hfq)
            for k in maturing[sid]
        )
        plans.append({
            'date': date,
            'group_id': ctx['group_id'],
            'stock_id': sid,
            'action': 'sell',
            'target_value': round(pos_value, 2),
            'weight': round(pos_value / total_equity, 6) if total_equity > 0 else 0.0,
            'signal_value': 0.0,
            'order': 0,
        })

    # ---- 2. 差额：到期且在新信号中 → 减仓(order=1) / 补仓(order=2) ----
    for sid in sig_sids & mat_sids:
        sw = signal[sid]['w']
        sv = signal[sid]['v']
        target_val = slice_capital * sw
        current_val = sum(
            _get_position_value(ctx['positions'][k], close_hfq)
            for k in maturing[sid]
        )
        delta = target_val - current_val
        if abs(delta) < 1.0:
            continue  # 变化太小，跳过
        action = 'sell' if delta < 0 else 'buy'
        plans.append({
            'date': date,
            'group_id': ctx['group_id'],
            'stock_id': sid,
            'action': action,
            'target_value': round(abs(delta), 2),
            'weight': sw,
            'signal_value': sv,
            'order': 1 if action == 'sell' else 2,
        })

    # ---- 3. 纯买：新信号中的新开仓 (order=3) ----
    for sid in sig_sids - mat_sids:
        sw = signal[sid]['w']
        sv = signal[sid]['v']
        target_val = slice_capital * sw
        plans.append({
            'date': date,
            'group_id': ctx['group_id'],
            'stock_id': sid,
            'action': 'buy',
            'target_value': round(target_val, 2),
            'weight': sw,
            'signal_value': sv,
            'order': 3,
        })

    if not plans:
        return pd.DataFrame()

    return pd.DataFrame(plans).sort_values('order')
