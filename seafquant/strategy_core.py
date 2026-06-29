"""
策略核心引擎 — 初始化、辅助函数、三种交易处理。

本模块提供 group context 初始化、佣金/复权/净值计算公式，
以及三种交易执行处理器（差额/新开仓/平仓）。
所有函数无副作用，仅操作传入的 ctx 字典。
"""

from __future__ import annotations

import math
from typing import Any

# =============================================================================
# 初始化
# =============================================================================
TICK_SIZE = 0.01


def _init_group_context(
    initial_cash: float,
    fwd: int,
    commission_rate: float,
    min_commission: float,
    group_id: int,
    slip_ticks: int,
) -> dict[str, Any]:
    """初始化单个 group 的回测上下文。"""
    return {
        'group_id': group_id,
        'initial_cash': initial_cash,
        'fwd': fwd,
        'commission_rate': commission_rate,
        'min_commission': min_commission,
        'slip_ticks': slip_ticks,
        # 核心状态
        'cash': initial_cash,
        'peak_nav': 1.0,
        'cumsum_fee': 0.0,
        'positions': {},        # (sid, batch_dc) → dict
        'pending_signal': None,  # T-1 日信号 {sid: weight}，待 T 日执行
        'day_counter': 0,
        # 输出日志
        'trade_log': [],
        'position_log': [],
        'nav_log': [],
        'daily_plans': [],      # 每日交易计划 DataFrame 列表
    }


# =============================================================================
# 内部辅助函数
# =============================================================================


def _calc_commission(trade_value: float, rate: float, min_comm: float, precision: int = 2) -> float:
    """手续费：万五，最低 5 元。"""
    raw = max(abs(trade_value) * rate, min_comm)
    return round(raw, precision)


def _get_actual_shares(pos: dict, f_today: dict[str, float]) -> float:
    """由锚定股数 + 复权因子推算当前实际股数（含送股零碎股）。"""
    ft = f_today.get(pos['stock_id'])
    if ft is None or pos['f_buy'] <= 0:
        return 0.0
    return pos['n_initial'] * (ft / pos['f_buy'])


def _get_position_value(pos: dict, close_hfq: dict[str, float]) -> float:
    """黄金公式：市值 = n_initial × (p_hfq / f_buy)。"""
    p_hfq = close_hfq.get(pos['stock_id'], 0.0)
    if p_hfq <= 0 or pos['f_buy'] <= 0:
        return 0.0
    return pos['n_initial'] * (p_hfq / pos['f_buy'])


def _compute_total_equity(
    ctx: dict, close_uq: dict, close_hfq: dict, precision: int = 2
) -> float:
    """总资产 = 现金 + 所有持仓市值（黄金公式）。"""
    total = ctx['cash']
    for pos in ctx['positions'].values():
        total += _get_position_value(pos, close_hfq)
    return round(total, precision)


def _log_trade(
    ctx: dict, date, stock_id: str, stock_name: str, action: str,
    shares: float, price: float, value: float, commission: float,
    signal_value: float = 0.0,
    hfq_price: float = 0.0,
) -> None:
    ctx['trade_log'].append({
        'date': date,
        'code': stock_id,
        'stock_name': stock_name,
        'action': action,
        'shares': shares,
        'price': round(price, ctx.get('precision', 2)),
        'value': round(value, ctx.get('precision', 2)),
        'commission': round(commission, ctx.get('precision', 2)),
        'signal_value': round(signal_value, 4),
        'hfq_price': round(hfq_price, ctx.get('precision', 2)),
    })


def _create_position(
    ctx: dict, stock_id: str, dc: int, n_initial: float,
    f_buy: float, date,
) -> None:
    ctx['positions'][(stock_id, dc)] = {
        'stock_id': stock_id,
        'batch_dc': dc,
        'n_initial': n_initial,
        'f_buy': f_buy,
        'mature_dc': dc + ctx['fwd'],
        'entry_date': date,
    }


# =============================================================================
# 三种交易处理
# =============================================================================


def _process_delta_trade(
    ctx: dict, date, dc: int, sid: str, sname: str,
    weight: float, slice_capital: float,
    maturing_keys: list, close_uq: dict, close_hfq: dict,
    f_today: dict, signal_value: float = 0.0,
) -> None:
    """差额交易：到期持仓 + 新信号继续持有 → 补仓/减仓，锚点重置。"""
    p_uq = close_uq.get(sid, 0.0)
    p_hfq = (close_hfq or {}).get(sid, p_uq)
    if p_uq <= 0:
        for key in maturing_keys:
            ctx['positions'][key]['mature_dc'] = dc + 1
        return

    rate, min_comm = ctx['commission_rate'], ctx['min_commission']
    slip_ticks = ctx['slip_ticks']
    old_shares = sum(
        _get_actual_shares(ctx['positions'][k], f_today) for k in maturing_keys
    )
    target_value = slice_capital * weight
    target_shares = math.floor(target_value / p_uq / 100) * 100
    delta = target_shares - old_shares

    if delta > 0:
        trade_price = p_uq + slip_ticks * TICK_SIZE  # 考虑滑点
        trade_value = delta * trade_price
        commission = _calc_commission(trade_value, rate, min_comm, ctx.get('precision', 2))
        if ctx['cash'] >= trade_value + commission:
            ctx['cash'] -= (trade_value + commission)
            _log_trade(ctx, date, sid, sname, 'buy', delta, trade_price, trade_value, commission,
                       signal_value=signal_value, hfq_price=p_hfq)
        else:
            # 处理资金不足的情况
            max_aff = max(0.0, ctx['cash'] - min_comm)
            buy_shares = math.floor(max_aff / trade_price / 100) * 100
            if buy_shares > 0:
                trade_value = buy_shares * trade_price
                commission = _calc_commission(trade_value, rate, min_comm, ctx.get('precision', 2))
                if ctx['cash'] >= trade_value + commission:
                    ctx['cash'] -= (trade_value + commission)
                    _log_trade(ctx, date, sid, sname, 'buy', buy_shares, trade_price,
                               trade_value, commission,
                               signal_value=signal_value, hfq_price=p_hfq)
                    target_shares = old_shares + buy_shares
                else:
                    target_shares = old_shares
            else:
                target_shares = old_shares
    elif delta < 0:
        trade_price = p_uq - slip_ticks * TICK_SIZE  # 考虑滑点
        sell_shares = min(abs(delta), old_shares)
        trade_value = sell_shares * trade_price
        commission = _calc_commission(trade_value, rate, min_comm, ctx.get('precision', 2))
        ctx['cash'] += (trade_value - commission)
        _log_trade(ctx, date, sid, sname, 'sell', sell_shares, trade_price, trade_value, commission,
                   signal_value=signal_value, hfq_price=p_hfq)
        target_shares = old_shares - sell_shares

    for key in maturing_keys:
        del ctx['positions'][key]
    if target_shares > 0 and sid in f_today:
        _create_position(ctx, sid, dc, target_shares, f_today[sid], date)


def _process_new_trade(
    ctx: dict, date, dc: int, sid: str, sname: str,
    weight: float, slice_capital: float,
    close_uq: dict, f_today: dict,
    close_hfq: dict | None = None, signal_value: float = 0.0,
) -> None:
    """新开仓：信号中的股票，当前无到期持仓。"""
    p_uq = close_uq.get(sid, 0.0)
    p_hfq = (close_hfq or {}).get(sid, p_uq)
    if p_uq <= 0 or sid not in f_today:
        return
    rate, min_comm = ctx['commission_rate'], ctx['min_commission']
    slip_ticks = ctx['slip_ticks']
    target_value = slice_capital * weight
    trade_price = p_uq + slip_ticks * TICK_SIZE  # 考虑滑点
    target_shares = math.floor(target_value / trade_price / 100) * 100
    if target_shares <= 0:
        return
    trade_value = target_shares * trade_price
    commission = _calc_commission(trade_value, rate, min_comm, ctx.get('precision', 2))
    if ctx['cash'] >= trade_value + commission:
        ctx['cash'] -= (trade_value + commission)
        _log_trade(ctx, date, sid, sname, 'buy', target_shares, trade_price, trade_value, commission,
                   signal_value=signal_value, hfq_price=p_hfq)
        _create_position(ctx, sid, dc, target_shares, f_today[sid], date)
    else:
        max_aff = max(0.0, ctx['cash'] - min_comm)
        buy_shares = math.floor(max_aff / trade_price / 100) * 100
        if buy_shares > 0:
            trade_value = buy_shares * trade_price
            commission = _calc_commission(trade_value, rate, min_comm, ctx.get('precision', 2))
            if ctx['cash'] >= trade_value + commission:
                ctx['cash'] -= (trade_value + commission)
                _log_trade(ctx, date, sid, sname, 'buy', buy_shares, trade_price,
                           trade_value, commission,
                           signal_value=signal_value, hfq_price=p_hfq)
                _create_position(ctx, sid, dc, buy_shares, f_today[sid], date)


def _process_close_trade(
    ctx: dict, date, dc: int, sid: str, sname: str,
    maturing_keys: list, close_uq: dict, f_today: dict,
    close_hfq: dict | None = None, signal_value: float = 0.0,
) -> None:
    """全部平仓：到期持仓不在新信号中。"""
    p_uq = close_uq.get(sid, 0.0)
    p_hfq = (close_hfq or {}).get(sid, p_uq)
    if p_uq <= 0:
        for key in maturing_keys:
            ctx['positions'][key]['mature_dc'] = dc + 1
        return
    rate, min_comm = ctx['commission_rate'], ctx['min_commission']
    slip_ticks = ctx['slip_ticks']
    trade_price = p_uq - slip_ticks * TICK_SIZE  # 考虑滑点
    for key in maturing_keys:
        pos = ctx['positions'][key]
        actual_shares = _get_actual_shares(pos, f_today)
        if actual_shares > 0:
            trade_value = actual_shares * trade_price
            commission = _calc_commission(trade_value, rate, min_comm, ctx.get('precision', 2))
            ctx['cash'] += (trade_value - commission)
            _log_trade(ctx, date, sid, sname, 'sell', actual_shares, trade_price,
                       trade_value, commission,
                       signal_value=signal_value, hfq_price=p_hfq)
        del ctx['positions'][key]


def _process_delist_trade(ctx: dict, date, dc: int, sid: str, sname: str, maturing_keys: list) -> None:
    """退市处理：所有持仓全部平仓，信号无效。"""
    for key in maturing_keys:
        pos = ctx['positions'][key]
        actual_shares = pos['n_initial']  # 退市时不考虑复权，按锚定股数卖出
        trade_value = 0.0
        commission = 0.0
        ctx['cash'] += (trade_value - commission)
        _log_trade(ctx, date, sid, sname, 'sell', actual_shares, 0.0,
                    trade_value, commission,
                    signal_value=0.0, hfq_price=0.0)
        del ctx['positions'][key]
