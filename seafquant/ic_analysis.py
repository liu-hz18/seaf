"""
IC 分析节点 — 每日计算截面 Rank IC，累积统计，退出时输出汇总。

输入：
- 来自 model 的 pipeline：pred_signal（每日截面向量）
- 来自 data source 的 pipeline：close 数据，window=fwd+1 天

功能：
- 每日计算 Spearman rank IC(pred_signal, fwd_ret_xd)
- 记录 ic_history，累积 cumsum_ic
- 退出时（epilogue_fn）输出汇总统计：mean IC, ICIR, winrate, max drawdown
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from qpipe.frame3d import Frame3D


def ic_analysis_fn(name: str, f3d: Frame3D, context: Any) -> tuple[Frame3D, Any]:
    """每日 IC 计算函数。

    f3d 包含 window 天的数据（pred_signal 列 + close 列）。
    context 可包含 'fwd'（默认 20）控制前瞻天数。
    """
    if context is None:
        context = {}
    context.setdefault('ic_history', [])
    context.setdefault('cumsum_ic', 0.0)
    context.setdefault('day_count', 0)
    context.setdefault('first_signal_day', None)
    context.setdefault('last_signal_day', None)
    fwd = context.get('fwd', 20)

    df = f3d.df.copy()
    times = sorted(df.index.get_level_values('key').unique())

    if len(times) < fwd + 1:
        return f3d

    # 时间对齐：当前 frame 最早一天 times[0] 的预测信号 pred_signal，
    # 对齐未来 (fwd-1) 日的截面超额收益率（cs_zscore(close[t+fwd]/close[t+1]-1)）。
    # 窗口设计：IC_WINDOW == fwd+1，保证 times[fwd] 在窗口内。
    pred_t = times[0]          # 信号产生日
    buy_t = times[1]            # t+1 买入日
    sell_t = times[fwd]         # t+fwd 卖出日（fwd 由 context 传入）

    # 记录首次和末次信号日，供 epilogue 汇总使用
    if context['first_signal_day'] is None:
        context['first_signal_day'] = pred_t
    context['last_signal_day'] = pred_t

    cs_pred = df.index.get_level_values('key') == pred_t
    cs_buy = df.index.get_level_values('key') == buy_t
    cs_sell = df.index.get_level_values('key') == sell_t

    pred_signal = df.loc[cs_pred, 'pred_signal'].values.astype(float)
    close_buy = df.loc[cs_buy, 'close'].values.astype(float)
    close_sell = df.loc[cs_sell, 'close'].values.astype(float)

    # cs_zscore(close[t+fwd] / close[t+1] - 1)：截面超额收益率
    fwd_ret = close_sell / close_buy - 1

    valid_mask = ~np.isnan(fwd_ret) & ~np.isnan(pred_signal)

    if valid_mask.sum() >= 10:
        pred_valid = pred_signal[valid_mask]
        fwd_valid = fwd_ret[valid_mask]

        # 截面标准化（与 model label 定义一致）
        cs_mean = np.nanmean(fwd_valid)
        cs_std = np.nanstd(fwd_valid)
        cs_excess = (fwd_valid - cs_mean) / cs_std if cs_std > 0 else np.zeros_like(fwd_valid)

        try:
            ic = spearmanr(pred_valid, cs_excess).correlation
            if np.isnan(ic):
                ic = 0.0
        except Exception:
            ic = 0.0

        context['ic_history'].append(ic)
        context['cumsum_ic'] += ic
        context['day_count'] += 1

        if context['day_count'] % 10 == 0 or context['day_count'] == 1:
            recent = context['ic_history'][-10:]
            logging.info(
                f'[{name}] IC#{context["day_count"]} '
                f'signal_day={pred_t} buy={buy_t} sell={sell_t} '
                f'ic={ic:.4f} cumsum={context["cumsum_ic"]:.4f} '
                f'recent10_mean={np.mean(recent):.4f}'
            )
    else:
        context['ic_history'].append(np.nan)
        context['day_count'] += 1

    return f3d


def ic_epilogue(name: str, context: dict[str, Any] | None) -> None:
    """退出前汇总：计算 mean IC, ICIR, winrate, max drawdown。"""
    if context is None or not context.get('ic_history'):
        logging.warning(f'[{name}] Epilogue: No IC data to summarize.')
        return

    ics = [x for x in context['ic_history'] if not np.isnan(x)]

    if len(ics) < 10:
        logging.warning(f'[{name}] Epilogue: Insufficient IC data ({len(ics)} points).')
        return

    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    winrate = sum(1 for x in ics if x > 0) / len(ics)

    cumsum = np.cumsum(ics)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    first_day = context.get('first_signal_day', 'N/A')
    last_day = context.get('last_signal_day', 'N/A')
    logging.info(f'[{name}] ========== IC Summary ==========')
    logging.info(f'[{name}]   Signal range: [{first_day} .. {last_day}]')
    logging.info(f'[{name}]   N={len(ics)}, Mean IC={mean_ic:.4f}, ICIR={icir:.4f}')
    logging.info(f'[{name}]   WinRate={winrate:.2%}, CumSum IC={cumsum[-1]:.4f}')
    logging.info(f'[{name}]   IC Std={std_ic:.4f}, IC Skew={pd.Series(ics).skew():.4f}')
    logging.info(f'[{name}]   Max CumSum DD={max_dd:.4f}')
    logging.info(f'[{name}] ======================================')