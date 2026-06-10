"""
截面分组选股策略与绩效计算节点。

架构：
- 按 pred_signal 排名将股票分为 num_groups 组，每组独立运作。
- T 日收盘产生信号 → T+1 日收盘执行交易（防时间穿越）。
- 不复权 close_uq 用于撮合计算股数；后复权 close 用于净值（黄金公式）。
- fwd 日滚动：资金均分为 fwd 份，每日轮换到期批次。
- 手续费万五（最低 5 元），A 股 100 股整数倍。
- 每 group 独立维护 cash / positions / 净值 / 回撤日志，最终导出 MLflow。

context 配置（从 pipeline 传入）：
    num_groups: 分组数（默认 10）
    fwd: 持仓周期（默认 20）
    initial_cash: 初始资金（默认 10_000_000）
    commission_rate: 手续费率（默认 0.0005）
    min_commission: 最低手续费（默认 5.0）
    mlflow_run_id: MLflow run ID
    start_date: 起始日期
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from qpipe.frame3d import Frame3D
from qpipe.utils import mlflow_log_metrics, trading_step

# =============================================================================
# 初始化
# =============================================================================


def _init_group_context(
    initial_cash: float,
    fwd: int,
    commission_rate: float,
    min_commission: float,
    group_id: int,
) -> dict[str, Any]:
    """初始化单个 group 的回测上下文。"""
    return {
        'group_id': group_id,
        'initial_cash': initial_cash,
        'fwd': fwd,
        'commission_rate': commission_rate,
        'min_commission': min_commission,
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
    }


# =============================================================================
# 内部辅助函数
# =============================================================================


def _calc_commission(trade_value: float, rate: float, min_comm: float) -> float:
    """手续费：万五，最低 5 元。"""
    return max(abs(trade_value) * rate, min_comm)


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
    ctx: dict, close_uq: dict, close_hfq: dict
) -> float:
    """总资产 = 现金 + 所有持仓市值（黄金公式）。"""
    total = ctx['cash']
    for pos in ctx['positions'].values():
        total += _get_position_value(pos, close_hfq)
    return total


def _log_trade(
    ctx: dict, date, stock_id: str, action: str,
    shares: float, price: float, value: float, commission: float,
    signal_value: float = 0.0,
    hfq_price: float = 0.0,
) -> None:
    ctx['trade_log'].append({
        'date': date, 'stock_id': stock_id, 'action': action,
        'shares': shares, 'price': price, 'value': value,
        'commission': commission,
        'signal_value': signal_value,
        'hfq_price': hfq_price,
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
    ctx: dict, date, dc: int, sid: str,
    weight: float, slice_capital: float,
    maturing_keys: list, close_uq: dict, close_hfq: dict,
    f_today: dict, signal_value: float = 0.0,
) -> None:
    """差额交易：到期持仓 + 新信号继续持有 → 补仓/减仓，锚点重置。"""
    p_uq = close_uq.get(sid, 0.0)
    p_hfq = close_hfq.get(sid, p_uq)
    if p_uq <= 0:
        for key in maturing_keys:
            ctx['positions'][key]['mature_dc'] = dc + 1
        return

    rate, min_comm = ctx['commission_rate'], ctx['min_commission']
    old_shares = sum(
        _get_actual_shares(ctx['positions'][k], f_today) for k in maturing_keys
    )
    target_value = slice_capital * weight
    target_shares = math.floor(target_value / p_uq / 100) * 100
    delta = target_shares - old_shares

    if delta > 0:
        trade_value = delta * p_uq
        commission = _calc_commission(trade_value, rate, min_comm)
        if ctx['cash'] >= trade_value + commission:
            ctx['cash'] -= (trade_value + commission)
            _log_trade(ctx, date, sid, 'buy', delta, p_uq, trade_value, commission,
                       signal_value=signal_value, hfq_price=p_hfq)
        else:
            max_aff = max(0.0, ctx['cash'] - min_comm)
            buy_shares = math.floor(max_aff / p_uq / 100) * 100
            if buy_shares > 0:
                trade_value = buy_shares * p_uq
                commission = _calc_commission(trade_value, rate, min_comm)
                if ctx['cash'] >= trade_value + commission:
                    ctx['cash'] -= (trade_value + commission)
                    _log_trade(ctx, date, sid, 'buy', buy_shares, p_uq,
                               trade_value, commission,
                               signal_value=signal_value, hfq_price=p_hfq)
                    target_shares = old_shares + buy_shares
                else:
                    target_shares = old_shares
            else:
                target_shares = old_shares
    elif delta < 0:
        sell_shares = min(abs(delta), old_shares)
        trade_value = sell_shares * p_uq
        commission = _calc_commission(trade_value, rate, min_comm)
        ctx['cash'] += (trade_value - commission)
        _log_trade(ctx, date, sid, 'sell', sell_shares, p_uq, trade_value, commission,
                   signal_value=signal_value, hfq_price=p_hfq)
        target_shares = old_shares - sell_shares

    for key in maturing_keys:
        del ctx['positions'][key]
    if target_shares > 0 and sid in f_today:
        _create_position(ctx, sid, dc, target_shares, f_today[sid], date)


def _process_new_trade(
    ctx: dict, date, dc: int, sid: str,
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
    target_value = slice_capital * weight
    target_shares = math.floor(target_value / p_uq / 100) * 100
    if target_shares <= 0:
        return
    trade_value = target_shares * p_uq
    commission = _calc_commission(trade_value, rate, min_comm)
    if ctx['cash'] >= trade_value + commission:
        ctx['cash'] -= (trade_value + commission)
        _log_trade(ctx, date, sid, 'buy', target_shares, p_uq, trade_value, commission,
                   signal_value=signal_value, hfq_price=p_hfq)
        _create_position(ctx, sid, dc, target_shares, f_today[sid], date)
    else:
        max_aff = max(0.0, ctx['cash'] - min_comm)
        buy_shares = math.floor(max_aff / p_uq / 100) * 100
        if buy_shares > 0:
            trade_value = buy_shares * p_uq
            commission = _calc_commission(trade_value, rate, min_comm)
            if ctx['cash'] >= trade_value + commission:
                ctx['cash'] -= (trade_value + commission)
                _log_trade(ctx, date, sid, 'buy', buy_shares, p_uq,
                           trade_value, commission,
                           signal_value=signal_value, hfq_price=p_hfq)
                _create_position(ctx, sid, dc, buy_shares, f_today[sid], date)


def _process_close_trade(
    ctx: dict, date, dc: int, sid: str,
    maturing_keys: list, close_uq: dict, f_today: dict,
    close_hfq: dict | None = None, signal_value: float = 0.0,
) -> None:
    """全部平仓：到期持仓不在新信号中。"""
    p_uq = close_uq.get(sid, 0.0)
    p_hfq = close_hfq.get(sid, p_uq)
    if p_uq <= 0:
        for key in maturing_keys:
            ctx['positions'][key]['mature_dc'] = dc + 1
        return
    rate, min_comm = ctx['commission_rate'], ctx['min_commission']
    for key in maturing_keys:
        pos = ctx['positions'][key]
        actual_shares = _get_actual_shares(pos, f_today)
        if actual_shares > 0:
            trade_value = actual_shares * p_uq
            commission = _calc_commission(trade_value, rate, min_comm)
            ctx['cash'] += (trade_value - commission)
            _log_trade(ctx, date, sid, 'sell', actual_shares, p_uq,
                       trade_value, commission,
                       signal_value=signal_value, hfq_price=p_hfq)
        del ctx['positions'][key]


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
    for sid in close_uq:
        puq, phfq = close_uq[sid], close_hfq.get(sid, 0.0)
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
    drawdown = (peak - nav) / peak if peak > 0 else 0.0
    ctx['nav_log'].append({
        'date': date,
        'day_counter': dc,
        'cash': ctx['cash'],
        'value': total_equity,
        'total_equity': total_equity,
        'nav': nav,
        'peak_nav': peak,
        'drawdown': drawdown,
        'cumsum_fee': ctx['cumsum_fee'],
        'position_value': total_equity - ctx['cash'],
        'n_positions': len(ctx['positions']),
    })

    # Step 5: 持仓快照
    for key, pos in ctx['positions'].items():
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
# 信号 ranking → 分组字典
# =============================================================================


def _rank_into_groups(
    signal_series: pd.Series, num_groups: int,
) -> dict[int, dict[str, float]]:
    """按截面 signal 排名分为 num_groups 组，等权分配。

    group 0 = top 信号分位，group N-1 = bottom。
    信号分组格式::
        {stock_id: {'w': equal_weight, 'v': float(signal_value)}}"""
    n = len(signal_series)
    if n < num_groups:
        return {}
    # rank 从 0（最小）到 n-1（最大），除以 n 得 [0, 1) 分位数
    ranks = signal_series.rank(method='min') - 1
    quantiles = (ranks / n).clip(0, 1 - 1e-12)
    results: dict[int, dict[str, dict[str, float]]] = {}
    for g in range(num_groups):
        lo, hi = g / num_groups, (g + 1) / num_groups
        mask = (quantiles >= lo) & (quantiles < hi)
        members = signal_series.index[mask].tolist()
        if not members:
            continue
        w = 1.0 / len(members)
        results[g] = {m: {'w': w, 'v': float(signal_series[m])} for m in members}
    return results


# =============================================================================
# 主入口：strategy_fn
# =============================================================================


def strategy_fn(name: str, f3d: Frame3D, context: Any) -> Frame3D:
    """策略节点主函数 — 每个 frame 包含 window=2 天的数据。

    f3d 包含：
      - T-1 日：pred_signal（来自 model）+ close / close_uq（来自 source）
      - T 日：  close / close_uq（来自 source）

    工作流程：
      1. 提取 T-1 的信号 + T 的价格
      2. 按信号排名分 num_groups 组
      3. 每组独立 on_bar

    返回：空 Frame3D（策略节点无下游输出）。
    """
    if context is None:
        context = {}

    # ---- context 初始化 ----
    context.setdefault('num_groups', 10)
    context.setdefault('fwd', 20)
    context.setdefault('initial_cash', 10_000_000.0)
    context.setdefault('commission_rate', 0.0005)
    context.setdefault('min_commission', 5.0)
    context.setdefault('groups', None)
    context.setdefault('first_date', None)
    context.setdefault('last_date', None)

    if context['groups'] is None:
        num_groups = context['num_groups']
        fwd = context['fwd']
        ic = context['initial_cash'] / num_groups
        cr = context['commission_rate']
        mc = context['min_commission']
        context['groups'] = [
            _init_group_context(ic, fwd, cr, mc, g)
            for g in range(num_groups)
        ]

    df = f3d.df.copy()
    times = sorted(df.index.get_level_values('key').unique())

    if len(times) < 2:
        return Frame3D(pd.DataFrame(index=df.index[:0]))

    # T-1 和 T
    t_prev, t_curr = times[-2], times[-1]
    if context['first_date'] is None:
        context['first_date'] = t_curr
    context['last_date'] = t_curr

    # T 日的价格（纯 name 索引 Series → dict ）
    close_uq_t = df.xs(t_curr, level='key')['close_uq'].to_dict()
    close_hfq_t = df.xs(t_curr, level='key')['close'].to_dict()

    # ---- 首次调用：用 T-1 信号为每个 group 初始化 pending_signal ----
    if context.get('_primed') is None:
        signal_first = df.xs(t_prev, level='key')['pred_signal']
        first_groups = _rank_into_groups(signal_first, context['num_groups'])
        for gctx in context['groups']:
            gid = gctx['group_id']
            sig = first_groups.get(gid, {})
            if sig:
                gctx['pending_signal'] = sig
        context['_primed'] = True

    # ---- T 日信号分组 + 每组独立 on_bar ----
    signal_curr = df.xs(t_curr, level='key')['pred_signal']
    group_signals = _rank_into_groups(signal_curr, context['num_groups'])
    for gctx in context['groups']:
        gid = gctx['group_id']
        sig = group_signals.get(gid, {})
        _on_bar(gctx, t_curr, sig, close_uq_t, close_hfq_t)

    # ---- MLflow 逐日指标 ----
    run_id = context.get('mlflow_run_id', '')
    if run_id:
        step = trading_step(context.get('start_date', ''), t_curr)
        for gctx in context['groups']:
            if gctx['nav_log']:
                last_nav = gctx['nav_log'][-1]
                ic = gctx.get('initial_cash', 1.0)
                mlflow_log_metrics(run_id, f'{name}.g{gctx["group_id"]}', {
                    'nav': last_nav.get('nav', last_nav.get('total_equity', 0) / ic),
                    'value': last_nav.get('value', last_nav['total_equity']),
                    'cash': last_nav['cash'],
                    'drawdown': last_nav.get('drawdown', 0.0),
                    'cumsum_fee': last_nav.get('cumsum_fee', 0.0),
                    'position_value': last_nav['position_value'],
                    'n_positions': float(last_nav['n_positions']),
                }, step=step)
        # top-bottom NAV spread
        ng = context['num_groups']
        if ng >= 2:
            nav0 = context['groups'][0]['nav_log']
            nav1 = context['groups'][ng - 1]['nav_log']
            if nav0 and nav1:
                top_nav = nav1[-1].get('nav', 0)
                bot_nav = nav0[-1].get('nav', 0)
                top_val = nav1[-1].get('value', nav1[-1]['total_equity'])
                bot_val = nav0[-1].get('value', nav0[-1]['total_equity'])
                mlflow_log_metrics(run_id, f'{name}.spread', {
                    'top_nav': top_nav,
                    'bottom_nav': bot_nav,
                    'top_bottom_nav_spread': top_nav / bot_nav - 1.0,
                    'top_value': top_val,
                    'bottom_value': bot_val,
                    'top_bottom_value_gap': top_val - bot_val,
                }, step=step)

    dc_global = context['groups'][0]['day_counter'] if context['groups'] else 0
    if dc_global % 50 == 0:
        navs = [g['nav_log'][-1]['total_equity'] if g['nav_log'] else 0
                for g in context['groups']]
        logging.info(
            f'[{name}] day#{dc_global} date={t_curr}: '
            f'navs=[{min(navs):.0f}..{max(navs):.0f}] '
            f'total_nav={sum(navs):.0f}'
        )

    # 返回非空 Frame3D（含 t_curr），保证框架中 max_key 有效 → trading_step 正确计算 step
    latest_mi = pd.MultiIndex.from_tuples([(t_curr, '_strategy_')], names=['key', 'name'])
    return Frame3D(pd.DataFrame({'_dummy': [0.0]}, index=latest_mi))


# =============================================================================
# Epilogue：汇总统计 & MLflow 导出
# =============================================================================


def strategy_epilogue(name: str, context: dict[str, Any] | None) -> None:
    """退出前汇总：每个 group 的绩效指标 + 数据导出到 MLflow artifact。"""
    if context is None or not context.get('groups'):
        logging.warning(f'[{name}] Epilogue: No group data.')
        return

    run_id = context.get('mlflow_run_id', '')
    first_date = context.get('first_date', 'N/A')
    last_date = context.get('last_date', 'N/A')
    logging.info(f'[{name}] ===== Strategy Summary [{first_date}..{last_date}] =====')

    for gctx in context['groups']:
        gid = gctx['group_id']
        if not gctx['nav_log']:
            continue

        nav_df = pd.DataFrame(gctx['nav_log'])
        values = nav_df['value'].values if 'value' in nav_df.columns else nav_df['total_equity'].values
        nav_series = nav_df['nav'].values if 'nav' in nav_df.columns else values / gctx['initial_cash']
        start_value = gctx['initial_cash']
        final_value = values[-1]
        final_nav = nav_series[-1]
        total_return = (final_value / start_value - 1) if start_value > 0 else 0.0
        # value-based risk metrics
        rets_v = np.diff(values) / values[:-1]
        rets_v = rets_v[np.isfinite(rets_v)]
        ann_vol = float(np.std(rets_v) * np.sqrt(252)) if len(rets_v) > 1 else 0.0
        ann_ret = total_return * (252 / max(len(values), 1))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        # nav-based risk metrics
        rets_n = np.diff(nav_series) / nav_series[:-1]
        rets_n = rets_n[np.isfinite(rets_n)]
        nav_ann_vol = float(np.std(rets_n) * np.sqrt(252)) if len(rets_n) > 1 else 0.0

        cummax = np.maximum.accumulate(nav_series)
        dd = (cummax - nav_series) / cummax
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        n_trades = len(gctx['trade_log'])
        n_buys = sum(1 for t in gctx['trade_log'] if t['action'] == 'buy')

        logging.info(
            f'[{name}]   Group {gid}: N={len(values)}d, '
            f'final_value={final_value:,.0f}, final_nav={final_nav:.4f}, '
            f'return={total_return:.2%}, ann_ret={ann_ret:.2%}, '
            f'ann_vol={ann_vol:.2%}, nav_ann_vol={nav_ann_vol:.2%}, '
            f'sharpe={sharpe:.2f}, max_dd={max_dd:.2%}, '
            f'trades={n_trades} ({n_buys} buys)'
        )

        # ---- MLflow 指标 ----
        if run_id:
            mlflow_log_metrics(run_id, f'{name}_summary.g{gid}', {
                'final_value': final_value,
                'final_nav': final_nav,
                'total_return': total_return,
                'ann_return': ann_ret,
                'ann_volatility_value': ann_vol,
                'ann_volatility_nav': nav_ann_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'n_trades': float(n_trades),
                'n_buys': float(n_buys),
            }, step=0)

        # ---- MLflow artifact: nav / trade / position CSV ----
        if run_id:
            _export_artifact(run_id, f'{name}_g{gid}', 'nav', pd.DataFrame(gctx['nav_log']))
            if gctx['trade_log']:
                _export_artifact(run_id, f'{name}_g{gid}', 'trades',
                                 pd.DataFrame(gctx['trade_log']))
            if gctx['position_log']:
                _export_artifact(run_id, f'{name}_g{gid}', 'positions',
                                 pd.DataFrame(gctx['position_log']))

    logging.info(f'[{name}] ==========================================')


def _export_artifact(
    run_id: str, prefix: str, kind: str, df: pd.DataFrame,
) -> None:
    """将 DataFrame 写入 CSV 并上传到 MLflow artifact。"""
    import os
    import tempfile
    from contextlib import suppress

    try:
        import mlflow
        mlflow.set_tracking_uri('sqlite:///mlruns.db')
        tmp_dir = tempfile.mkdtemp(prefix='strat_')
        tmp_path = os.path.join(tmp_dir, f'{prefix}_{kind}.csv')
        try:
            df.to_csv(tmp_path, index=False)
            mlflow.log_artifact(tmp_path, artifact_path=f'strategy/{prefix}', run_id=run_id)
        finally:
            with suppress(Exception):
                os.unlink(tmp_path)
            with suppress(Exception):
                os.rmdir(tmp_dir)
    except Exception:
        pass