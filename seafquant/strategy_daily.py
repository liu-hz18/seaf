"""
策略日度引擎 — 逐日交易执行 + 次日交易计划生成。

_on_bar：每个交易日调用一次，执行昨日交易计划、记录净值。
_generate_daily_plan：基于 T 日信号和收盘价，生成 T+1 日交易计划（先卖后买）。
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from seafquant.strategy_core import (
    _compute_total_equity,
    _get_actual_shares,
    _get_position_value,
    _log_trade,
    _process_close_trade,
    _process_delist_trade,
    _process_delta_trade,
    _process_new_trade,
)


# =============================================================================
# 单日 on_bar — 执行 trading logic
# =============================================================================
def _on_bar(
    ctx: dict,
    date: str,
    daily_plan: pd.DataFrame,
    close_uq: dict[str, float],
    close_hfq: dict[str, float],
    tradestatus_map: dict[str, str],
) -> None:
    """逐日调用：对齐信号、交易与净值。

    T 日收盘收到 signal_T → 存储为 pending；
    同时执行 pending（signal_{T-1}），用 T 日不复权价撮合。
    """
    dc = ctx['day_counter']

    # Step 1: 复权因子 F_T = hfq / uq
    f_today: dict[str, float] = {}
    for sid, puq in close_uq.items():
        phfq = close_hfq.get(sid, 0.0)
        if puq > 0 and phfq > 0:
            f_today[sid] = phfq / puq

    precision = ctx.get('precision', 2)
    total_equity = _compute_total_equity(ctx, close_uq, close_hfq, precision)
    slice_capital = max(total_equity, 1.0) / ctx['fwd']

    # Step 2: 执行昨日待执行信号
    day_trades: list[dict] = []
    buy_value = sell_value = 0.0

    # 处理退市股票
    for _, rows in daily_plan.iterrows():
        sid = rows['code']
        sname = rows['stock_name']
        signal = rows['signal']
        if tradestatus_map.get(sid) is None:
            if len(rows['positions']) > 0:
                _process_delist_trade(ctx, date, dc, sid, sname, rows['positions'])
            else:
                _log_trade(ctx, date, dc, sid, sname, 'buy', 0, 0.0,
                    0.0, 0.0,
                    signal_value=signal, hfq_price=0.0, desc='开仓遇到退市')

    # 得到实际有效的计划
    daily_plan = daily_plan[daily_plan['code'].isin(tradestatus_map.keys())]

    close_plans = daily_plan[daily_plan['type'] == '平仓']
    reduce_plans = daily_plan[daily_plan['type'] == '减仓']
    add_plans = daily_plan[daily_plan['type'] == '加仓']
    open_plans = daily_plan[daily_plan['type'] == '开仓']

    # 平仓
    for _, rows in close_plans.iterrows():
        sid = rows['code']
        sname = rows['stock_name']
        signal = rows['signal']
        if tradestatus_map[sid]:
            # target_weight = rows['target_weight']
            _process_close_trade(
                ctx, date, dc, sid, sname, rows['positions'], close_uq, f_today,
                close_hfq=close_hfq, signal_value=signal,
            )
        else:
            p_uq = close_uq[sid]
            p_hfq = close_hfq[sid]
            _log_trade(
                ctx, date, dc, sid, sname, rows['action'], 0, p_uq, 0.0, 0.0,
                signal_value=signal, hfq_price=p_hfq, desc='平仓失败, 停牌'
            )
    # 减仓
    for _, rows in reduce_plans.iterrows():
        sid = rows['code']
        sname = rows['stock_name']
        signal = rows['signal']
        if tradestatus_map[sid]:
            target_weight = rows['target_weight']
            _process_delta_trade(
                ctx, date, dc, sid, sname, target_weight, slice_capital,
                rows['positions'], close_uq, close_hfq, f_today, signal_value=signal,
            )
        else:
            p_uq = close_uq[sid]
            p_hfq = close_hfq[sid]
            _log_trade(
                ctx, date, dc, sid, sname, rows['action'], 0, p_uq, 0.0, 0.0,
                signal_value=signal, hfq_price=p_hfq, desc='减仓失败, 停牌'
            )
    # 加仓
    for _, rows in add_plans.iterrows():
        sid = rows['code']
        sname = rows['stock_name']
        signal = rows['signal']
        if tradestatus_map[sid]:
            target_weight = rows['target_weight']
            _process_delta_trade(
                ctx, date, dc, sid, sname, target_weight, slice_capital,
                rows['positions'], close_uq, close_hfq, f_today, signal_value=signal,
            )
        else:
            p_uq = close_uq[sid]
            p_hfq = close_hfq[sid]
            _log_trade(
                ctx, date, dc, sid, sname, rows['action'], 0, p_uq, 0.0, 0.0,
                signal_value=signal, hfq_price=p_hfq, desc='加仓失败, 停牌'
            )
    # 开仓
    for _, rows in open_plans.iterrows():
        sid = rows['code']
        sname = rows['stock_name']
        signal = rows['signal']
        if tradestatus_map[sid]:
            target_weight = rows['target_weight']
            _process_new_trade(
                ctx, date, dc, sid, sname, target_weight, slice_capital, close_uq, f_today,
                close_hfq=close_hfq, signal_value=signal,
            )
        else:
            p_uq = close_uq[sid]
            p_hfq = close_hfq[sid]
            _log_trade(
                ctx, date, dc, sid, sname, rows['action'], 0, p_uq, 0.0, 0.0,
                signal_value=signal, hfq_price=p_hfq, desc='开仓失败, 停牌'
            )

    # 累加当日新增手续费
    day_trades = ctx['day_trades']
    buy_value = sum(t['value'] for t in day_trades if t['action'] == 'buy')
    sell_value = sum(t['value'] for t in day_trades if t['action'] == 'sell')
    for t in day_trades:
        ctx['cumsum_fee'] += t['commission']

    # Step 4: 净值记录
    total_equity = _compute_total_equity(ctx, close_uq, close_hfq, precision)
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
        'cash': round(ctx['cash'], precision),
        'value': round(total_equity, precision),
        'total_equity': round(total_equity, precision),
        'nav': nav,
        'peak_nav': peak,
        'drawdown': drawdown,
        'cumsum_fee': round(ctx['cumsum_fee'], precision),
        'position_value': total_equity - ctx['cash'],
        'n_positions': len(ctx['positions']),
        'turnover': round((buy_value + sell_value) / total_equity, 6) if total_equity > 0 else 0.0,
    })

    # Step 5: 持仓快照
    for pos in ctx['positions'].values():
        sid = pos['stock_id']
        actual_shares = _get_actual_shares(pos, f_today) if sid in f_today else 0
        market_value = _get_position_value(pos, close_hfq)
        mkt_pct = market_value / total_equity if total_equity > 0 else 0.0
        ctx['day_positions'].append({
            'date': date,
            'day_counter': dc,
            'code': sid,
            'stock_name': pos['stock_name'],
            'entry_day_count': pos['entry_day_count'],
            'n_initial': pos['n_initial'],
            'f_buy': pos['f_buy'],
            'f_today': f_today.get(sid),
            'actual_shares': actual_shares,
            'market_value': round(market_value, precision),
            'market_value_pct': mkt_pct,
            'p_uq': close_uq.get(sid, 0.0),
            'p_hfq': close_hfq.get(sid, 0.0),
            'signal_value': pos['signal_value'],
            'mature_dc': pos['mature_dc'],
            'entry_date': pos['entry_date'],
        })


# =============================================================================
# 次日交易计划生成
# =============================================================================
def _generate_daily_plan(
    ctx: dict, date: str,
    signal: dict[str, dict[str, float]],
    close_uq: dict[str, float],
    close_hfq: dict[str, float],
    stock_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """生成 T+1 日交易计划（基于 T 日收盘信息估算）。

    T 日收盘后，基于 T 日信号 + T 日收盘价，估算次日每笔交易的目标市值和占比。
    交易顺序：先卖后买（卖=0,1；买=2,3）。

    返回 DataFrame 列：
        date, group_id, stock_id, action, target_value, weight, signal_value, order
    """
    dc_tomorrow = ctx['day_counter'] + 1

    # Step 1: 今日的复权因子 F_T = hfq / uq
    f_today: dict[str, float] = {}
    for sid, puq in close_uq.items():
        phfq = close_hfq.get(sid, 0.0)
        if puq > 0 and phfq > 0:
            f_today[sid] = phfq / puq

    # 收集所有的到期持仓 (持仓是按交易管理的)，按股票id分组
    maturing: dict[str, list] = defaultdict(list)
    for key, pos in list(ctx['positions'].items()):
        if pos['mature_dc'] <= dc_tomorrow:  # NOTE: 超出期限的都应该卖出
            maturing[pos['stock_id']].append(key)

    precision = ctx.get('precision', 2)
    total_equity = _compute_total_equity(ctx, close_uq, close_hfq, precision)
    slice_capital = max(total_equity, 1.0) / ctx['fwd']

    sig_sids = set(signal.keys())  # 目标持仓
    mat_sids = set(maturing.keys())  # 到期持仓

    plans: list[dict] = []
    sn_map = stock_name_map or {}

    # ---- 1. 纯卖：到期且不在新信号中 → 全部平仓 (order=0) ----
    for sid in mat_sids - sig_sids:
        current_share = sum(
            _get_actual_shares(ctx['positions'][k], f_today) for k in maturing[sid]
        )
        current_val = sum(
            _get_position_value(ctx['positions'][k], close_hfq) for k in maturing[sid]
        )
        current_val = round(current_val, precision)
        plans.append({
            'date': date,
            'operation_dc': dc_tomorrow,
            'group_id': ctx['group_id'],
            'code': sid,
            'stock_name': sn_map.get(sid, ''),
            'type': "平仓",
            'action': 'sell',
            'current_share': current_share,
            'delta_value_planned': current_val,
            'current_value': current_val,
            'target_value_planned': 0.0,
            'target_weight': 0.0,
            'signal': -100.0,
            'order': 3,
            'positions': maturing[sid],  # dataframe 支持 list, 在 csv 文件中以字符串形式保存
        })

    # ---- 2. 差额：到期且在新信号中 → 减仓(order=1) / 补仓(order=2) ----
    for sid in sig_sids & mat_sids:
        sw = signal[sid]['w']
        sv = signal[sid]['v']
        target_val = slice_capital * sw
        current_share = sum(
            _get_actual_shares(ctx['positions'][k], f_today) for k in maturing[sid]
        )
        current_val = sum(
            _get_position_value(ctx['positions'][k], close_hfq)
            for k in maturing[sid]
        )
        target_val = round(target_val, precision)
        current_val = round(current_val, precision)
        delta = target_val - current_val
        if abs(delta) < 1.0:
            continue  # 变化太小，跳过
        action = 'sell' if delta < 0 else 'buy'
        plans.append({
            'date': date,
            'operation_dc': dc_tomorrow,
            'group_id': ctx['group_id'],
            'code': sid,
            'stock_name': sn_map.get(sid, ''),
            'type': '减仓' if action == 'sell' else '加仓',
            'action': action,
            'current_share': current_share,
            'delta_value_planned': round(abs(delta), precision),
            'current_value': current_val,
            'target_value_planned': target_val,
            'target_weight': sw,
            'signal': sv,
            'order': 2 if action == 'sell' else 1,
            'positions': maturing[sid],
        })

    # ---- 3. 纯买：新信号中的新开仓 (order=3) ----
    for sid in sig_sids - mat_sids:
        sw = signal[sid]['w']
        sv = signal[sid]['v']
        target_val = slice_capital * sw
        target_val = round(target_val, precision)
        plans.append({
            'date': date,
            'operation_dc': dc_tomorrow,
            'group_id': ctx['group_id'],
            'code': sid,
            'stock_name': sn_map.get(sid, ''),
            'type': "开仓",
            'action': 'buy',
            'current_share': 0,
            'delta_value_planned': target_val,
            'current_value': 0.0,
            'target_value_planned': target_val,
            'target_weight': sw,
            'signal': sv,
            'order': 0,
            'positions': [],
        })

    if not plans:
        return pd.DataFrame()

    return pd.DataFrame(plans).sort_values(['order', 'signal'], ascending=False)  # 降序排序
