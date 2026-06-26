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
from typing import Any

import numpy as np
import pandas as pd

from qpipe.frame3d import Frame3D
from qpipe.utils import mlflow_log_metrics
from seafquant.strategy_core import _init_group_context
from seafquant.strategy_daily import _generate_daily_plan, _on_bar

# =============================================================================
# 信号 ranking → 分组字典
# =============================================================================


def _rank_into_groups(
    signal_series: pd.Series, num_groups: int,
) -> dict[int, dict[str, dict[str, float]]]:
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


def strategy_fn(name: str, idx: int, f3d: Frame3D, context: Any) -> Frame3D:
    """策略节点主函数 — 每个 frame 包含 window=1 天的数据。

    f3d 包含：
      - T 日：  pred_signal（来自 model）+ close / close_uq（来自 source）

    工作流程：
      1. 缓存 T 的信号，按T日价格计算净值，在T+1调用时，按T+1 的价格进行交易
      2. 按信号排名分 num_groups 组
      3. 每组独立 on_bar

    返回：非空 Frame3D（含 t_curr），保证框架中 trading_step 正确计算 step。
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
    context.setdefault('include_star', False)

    if context['groups'] is None:
        num_groups = context['num_groups']
        fwd = context['fwd']
        ic = context['initial_cash']
        cr = context['commission_rate']
        mc = context['min_commission']
        context['groups'] = [
            _init_group_context(ic, fwd, cr, mc, g)
            for g in range(num_groups)
        ]

    df = f3d.df.copy()

    # 过滤 ST 和停牌股票（仅策略节点排除，模型训练/推理保留这些样本）
    if 'tradestatus' in df.columns:
        df = df[df['tradestatus'] == 1]
    if 'isST' in df.columns:
        df = df[df['isST'] == 0]
    # 指定 prefix 以方便排除创业板、科创版
    if not context['include_star']:
        CODE_PREFIXS = (
            'sh.600', 'sh.601', 'sh.603', 'sh.605',  # 沪市主板
            'sz.000', 'sz.001', 'sz.002', 'sz.003', 'sz.004'  # 深市主板
        )  # tuple type
        code_series = pd.Series(df.index.get_level_values('code'), index=df.index)
        df = df[code_series.str.startswith(CODE_PREFIXS, na=False)]

    times = sorted(df.index.get_level_values('key').unique())

    assert len(times) == 1

    # T-1 和 T
    t_curr = times[-1]
    if context['first_date'] is None:
        context['first_date'] = t_curr
    context['last_date'] = t_curr

    # T 日的价格（纯 name 索引 Series → dict ）
    close_uq_t = df.xs(t_curr, level='key')['close_uq'].to_dict()
    close_hfq_t = df.xs(t_curr, level='key')['close'].to_dict()

    # 股票名映射（artifact 导出用）
    stock_name_map = {}
    if 'stock_name' in df.columns:
        stock_name_map = df.xs(t_curr, level='key')['stock_name'].to_dict()

    # ---- 首次调用：用 T-1 信号为每个 group 初始化 pending_signal ----
    # if context.get('_primed') is None:
    #     signal_first = df.xs(t_prev, level='key')['pred_signal']
    #     first_groups = _rank_into_groups(signal_first, context['num_groups'])
    #     for gctx in context['groups']:
    #         gid = gctx['group_id']
    #         sig = first_groups.get(gid, {})
    #         if sig:
    #             gctx['pending_signal'] = sig
    #     context['_primed'] = True

    # ---- T 日信号分组 + 每组独立 on_bar ----
    signal_curr = df.xs(t_curr, level='key')['pred_signal']
    group_signals = _rank_into_groups(signal_curr, context['num_groups'])
    for gctx in context['groups']:
        gid = gctx['group_id']
        sig = group_signals.get(gid, {})
        _on_bar(gctx, t_curr, sig, close_uq_t, close_hfq_t, stock_name_map)

    # ---- 生成次日交易计划（T 日收盘后可立即给出） ----
    for gctx in context['groups']:
        gid = gctx['group_id']
        sig = group_signals.get(gid, {})
        if sig and gctx['pending_signal']:
            dc = gctx['day_counter']
            plan_df = _generate_daily_plan(
                gctx, t_curr, dc, gctx['pending_signal'],
                close_uq_t, close_hfq_t, stock_name_map,
            )
            if not plan_df.empty:
                gctx['daily_plans'].append(plan_df)

    # ---- MLflow 逐日指标 ----
    run_id = context.get('mlflow_run_id', '')
    if run_id:
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
                    'turnover': last_nav.get('turnover', 0.0),
                }, step=idx)
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
                    'top_bottom_log_nav_spread': np.log(top_nav) - np.log(bot_nav),
                    'top_value': top_val,
                    'bottom_value': bot_val,
                    'top_bottom_value_gap': top_val - bot_val,
                }, step=idx)

    dc_global = context['groups'][0]['day_counter'] if context['groups'] else 0
    if dc_global % 50 == 0:
        navs = [g['nav_log'][-1]['total_equity'] if g['nav_log'] else 0
                for g in context['groups']]
        logging.info(
            f'[{idx}] day#{dc_global} date={t_curr}: '
            f'navs=[{min(navs):.2f}..{max(navs):.2f}] '
            f'mean_nav={np.mean(navs):.2f}'
        )

    # 返回逐股逐组持仓市值 Frame3D（与 factor 节点格式对齐）
    stocks_sorted = sorted(df.xs(t_curr, level='key').index)
    ng = context['num_groups']
    data: dict[str, list[float]] = {}
    for g in range(ng):
        gctx = context['groups'][g]
        col = f'g{g}_mv'
        group_mv: dict[str, float] = {}
        for (sid, _), pos in gctx['positions'].items():
            phfq = close_hfq_t.get(sid, 0.0)
            if phfq > 0 and pos['f_buy'] > 0:
                group_mv[sid] = group_mv.get(sid, 0.0) + pos['n_initial'] * (phfq / pos['f_buy'])
        data[col] = [group_mv.get(s, 0.0) for s in stocks_sorted]
    latest_mi = pd.MultiIndex.from_product([[t_curr], stocks_sorted], names=['key', 'code'])
    return Frame3D(pd.DataFrame(data, index=latest_mi))


# =============================================================================
# Epilogue：汇总统计 & MLflow 导出
# =============================================================================


def strategy_epilogue(name: str, idx: int, context: dict[str, Any] | None) -> None:
    """退出前汇总：每个 group 的绩效指标 + 数据导出到 MLflow artifact。"""
    if context is None or not context.get('groups'):
        logging.warning(f'[{idx}] Epilogue: No group data.')
        return

    run_id = context.get('mlflow_run_id', '')
    first_date = context.get('first_date', 'N/A')
    last_date = context.get('last_date', 'N/A')
    logging.info(f'[{idx}] ===== Strategy Summary [{first_date}..{last_date}] =====')

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

        # ---- 平均换手率 ----
        turnovers = [nl.get('turnover', 0.0) for nl in gctx['nav_log'] if nl.get('turnover') is not None]
        avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

        logging.info(
            f'[{idx}] Group {gid}: N={len(values)}d, '
            f'final_value={final_value:,.0f}, final_nav={final_nav:.4f}, '
            f'return={total_return:.2%}, ann_ret={ann_ret:.2%}, '
            f'ann_vol={ann_vol:.2%}, nav_ann_vol={nav_ann_vol:.2%}, '
            f'sharpe={sharpe:.2f}, max_dd={max_dd:.2%}, '
            f'avg_turnover={avg_turnover:.4%}, '
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

    logging.info(f'[{idx}] ==========================================')

    for gctx in context['groups']:
        # ---- MLflow artifact: 每日交易计划 CSV ----
        if run_id and gctx['daily_plans']:
            all_plans = pd.concat(gctx['daily_plans'], ignore_index=True)
            if not all_plans.empty:
                for plan_date, day_df in all_plans.groupby('date', sort=True):
                    date_str = str(plan_date)[:10]  # Timestamp → YYYY-MM-DD，避免冒号
                    _export_artifact(
                        run_id, f'{name}_g{gid}', f'daily_plan_{date_str}',
                        day_df, artifact_subdir=f'strategy/{name}_g{gid}/daily_plans',
                    )
                logging.info(
                    f'[{idx}] Group {gid}: {len(all_plans)} plan rows '
                    f'({all_plans["date"].nunique()} days) exported'
                )

    logging.info(f'[{idx}] ==========================================')


def _export_artifact(
    run_id: str, prefix: str, kind: str, df: pd.DataFrame,
    artifact_subdir: str = '',
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
            art_path = artifact_subdir or f'strategy/{prefix}'
            mlflow.log_artifact(tmp_path, artifact_path=art_path, run_id=run_id)
        finally:
            with suppress(Exception):
                os.unlink(tmp_path)
            with suppress(Exception):
                os.rmdir(tmp_dir)
    except Exception as e:
        logging.error(f"mlflow artifact Exception: {e}")
