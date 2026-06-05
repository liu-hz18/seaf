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
import numpy as np
import logging
from typing import Tuple, Any
from scipy.stats import spearmanr
import pandas as pd
from qpipe.frame3d import Frame3D


def ic_analysis_fn(name: str, f3d: Frame3D, context: Any) -> Tuple[Frame3D, Any]:
    """每日 IC 计算函数。

    f3d 包含 window 天的数据（pred_signal 列 + close 列）。
    context 可包含 'fwd'（默认 20）控制前瞻天数。
    """
    if context is None:
        context = {}
    context.setdefault('ic_history', [])
    context.setdefault('cumsum_ic', 0.0)
    context.setdefault('day_count', 0)
    fwd = context.get('fwd', 20)

    df = f3d.df.copy()
    times = sorted(df.index.get_level_values('key').unique())

    if len(times) < fwd + 1:
        return f3d, context

    # 取最新一天预测信号
    latest_t = times[-1]
    cs_mask = df.index.get_level_values('key') == latest_t
    pred_signal = df.loc[cs_mask, 'pred_signal'].values.astype(float)

    # 计算 fwd 日前瞻收益：close[latest] / close[t-fwd] - 1
    t_past = times[-(fwd + 1)]  # fwd 天前
    cs_mask_past = df.index.get_level_values('key') == t_past
    close_now = df.loc[cs_mask, 'close'].values
    close_then = df.loc[cs_mask_past, 'close'].values

    fwd_ret = close_now / close_then - 1

    # 截面标准化 fwd_ret
    valid_mask = ~np.isnan(fwd_ret) & ~np.isnan(pred_signal)

    if valid_mask.sum() >= 10:
        pred_valid = pred_signal[valid_mask]
        fwd_valid = fwd_ret[valid_mask]

        fwd_mean = np.nanmean(fwd_valid)
        fwd_std = np.nanstd(fwd_valid)
        fwd_xd = (fwd_valid - fwd_mean) / fwd_std if fwd_std > 0 else np.zeros_like(fwd_valid)

        try:
            ic = spearmanr(pred_valid, fwd_xd).correlation
            if np.isnan(ic):
                ic = 0.0
        except Exception:
            ic = 0.0

        context['ic_history'].append(ic)
        context['cumsum_ic'] += ic
        context['day_count'] += 1

        if context['day_count'] % 20 == 0:
            recent = context['ic_history'][-20:]
            logging.info(
                f"[{name}] day={context['day_count']}, ic={ic:.4f}, "
                f"cumsum_ic={context['cumsum_ic']:.4f}, "
                f"recent_mean_ic={np.mean(recent):.4f}"
            )
    else:
        context['ic_history'].append(np.nan)
        context['day_count'] += 1

    return f3d, context


def ic_epilogue(name: str, context: dict) -> None:
    """退出前汇总：计算 mean IC, ICIR, winrate, max drawdown。"""
    if context is None or not context.get('ic_history'):
        logging.warning(f"[{name}] Epilogue: No IC data to summarize.")
        return

    ics = [x for x in context['ic_history'] if not np.isnan(x)]

    if len(ics) < 10:
        logging.warning(f"[{name}] Epilogue: Insufficient IC data ({len(ics)} points).")
        return

    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    winrate = sum(1 for x in ics if x > 0) / len(ics)

    cumsum = np.cumsum(ics)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    logging.info(f"[{name}] ========== IC Summary ==========")
    logging.info(f"[{name}]   N={len(ics)}, Mean IC={mean_ic:.4f}, ICIR={icir:.4f}")
    logging.info(f"[{name}]   WinRate={winrate:.2%}, CumSum IC={cumsum[-1]:.4f}")
    logging.info(f"[{name}]   IC Std={std_ic:.4f}, IC Skew={pd.Series(ics).skew():.4f}")
    logging.info(f"[{name}]   Max CumSum DD={max_dd:.4f}")
    logging.info(f"[{name}] ======================================")
