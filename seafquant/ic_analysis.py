"""
IC 分析节点 — 每日计算截面 Pearson IC / Rank IC，MLflow 逐日记录。

输入：
- 来自 model 的 pipeline：pred_signal（每日截面向量）
- 来自 data source 的 pipeline：close 数据，window=fwd+1 天

功能：
- 每日计算 Pearson IC (corrcoef) + Spearman rank IC
- 记录 raw return std、cumsum IC
- 逐日推送到 MLflow
- 退出时（epilogue_fn）输出汇总统计
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# scipy 延迟导入到 ic_analysis_fn 内，节省顶层导入时间
from qpipe.utils import mlflow_log_metrics, trading_step

if TYPE_CHECKING:
    from qpipe.frame3d import Frame3D


def ic_analysis_fn(name: str, f3d: Frame3D, context: Any) -> Frame3D:
    """每日 IC 计算函数。

    f3d 包含 window 天的数据（pred_signal 列 + close 列）。
    context 可包含 'fwd'（默认 20）控制前瞻天数。
    """
    if context is None:
        context = {}
    from scipy.stats import pearsonr, spearmanr  # 延迟导入

    context.setdefault('pearson_ic_history', [])
    context.setdefault('rank_ic_history', [])
    context.setdefault('raw_ret_std_history', [])
    context.setdefault('raw_ret_skew_history', [])
    context.setdefault('cumsum_pearson_ic', 0.0)
    context.setdefault('cumsum_vol_pearson_ic', 0.0)
    context.setdefault('cumsum_raw_ret_std', 0.0)
    context.setdefault('cumsum_rank_ic', 0.0)
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
    pred_t = times[0]      # 信号产生日
    buy_t = times[1]        # t+1 买入日
    sell_t = times[fwd]     # t+fwd 卖出日

    if context['first_signal_day'] is None:
        context['first_signal_day'] = pred_t
    context['last_signal_day'] = pred_t

    cs_pred = df.index.get_level_values('key') == pred_t
    cs_buy = df.index.get_level_values('key') == buy_t
    cs_sell = df.index.get_level_values('key') == sell_t

    pred_signal = df.loc[cs_pred, 'pred_signal'].values.astype(float)
    close_buy = df.loc[cs_buy, 'close'].values.astype(float)
    close_sell = df.loc[cs_sell, 'close'].values.astype(float)

    fwd_ret = np.log(close_sell) - np.log(close_buy)

    valid_mask = ~np.isnan(fwd_ret) & ~np.isnan(pred_signal)

    if valid_mask.sum() >= 10:
        pred_valid = pred_signal[valid_mask]
        fwd_valid = fwd_ret[valid_mask]

        # Raw return std（未经 cs_zscore 的截面收益率标准差）
        raw_ret_std = float(np.nanstd(fwd_valid))
        context['raw_ret_std_history'].append(raw_ret_std)
        # Raw return skew（截面收益率偏度）
        context['cumsum_raw_ret_std'] += raw_ret_std
        raw_ret_skew = float(pd.Series(fwd_valid).skew())
        context['raw_ret_skew_history'].append(raw_ret_skew)

        # 截面标准化（与 model label 定义一致）
        cs_mean = np.nanmean(fwd_valid)
        cs_std = np.nanstd(fwd_valid)
        cs_excess = (fwd_valid - cs_mean) / cs_std if cs_std > 0 else np.zeros_like(fwd_valid)

        # Pearson IC
        try:
            pearson_ic = pearsonr(pred_valid, cs_excess).correlation
            if np.isnan(pearson_ic):
                pearson_ic = 0.0
        except Exception:
            pearson_ic = 0.0

        # Rank (Spearman) IC
        try:
            rank_ic = spearmanr(pred_valid, cs_excess).correlation
            if np.isnan(rank_ic):
                rank_ic = 0.0
        except Exception:
            rank_ic = 0.0

        context['pearson_ic_history'].append(pearson_ic)
        context['rank_ic_history'].append(rank_ic)
        context['cumsum_pearson_ic'] += pearson_ic
        context['cumsum_vol_pearson_ic'] += pearson_ic * raw_ret_std
        context['cumsum_rank_ic'] += rank_ic
        context['day_count'] += 1

        # ---- MLflow 逐日记录 ----
        mlflow_run_id = context.get('mlflow_run_id', '')
        # 理论 top-bottom 对数净值差（逐日）NOTE: 这个数字比真实的持仓净值滞后 fwd 天
        num_groups = context.get('num_groups', 10)
        cc = context['day_count']
        theo_log_nav_spread = 0.0
        if cc > 0 and num_groups >= 2:
            from scipy.stats import norm  # 延迟导入
            phi_inv = float(norm.ppf(1.0 / num_groups))
            phi_val = float(norm.pdf(phi_inv))
            mean_ret_std = context['cumsum_raw_ret_std'] / cc
            theo_log_nav_spread = (
                2.0 * num_groups * phi_val * mean_ret_std * context['cumsum_pearson_ic'] / (fwd-1)  # NOTE: we actually hold fwd-1 days
            )
        mlflow_log_metrics(mlflow_run_id, name, {
            'pearson_ic': pearson_ic,
            'rank_ic': rank_ic,
            'raw_ret_std': raw_ret_std,
            'raw_ret_skew': raw_ret_skew,
            'theo_log_nav_spread': theo_log_nav_spread,
            'cumsum_pearson_ic': context['cumsum_pearson_ic'],
            'cumsum_vol_pearson_ic': context['cumsum_vol_pearson_ic'],
            'cumsum_rank_ic': context['cumsum_rank_ic'],
        }, step=trading_step(context.get('start_date', ''), pred_t))

        if context['day_count'] % 10 == 0 or context['day_count'] == 1:
            recent_p = context['pearson_ic_history'][-10:]
            recent_r = context['rank_ic_history'][-10:]
            logging.info(
                f'[{name}] IC#{context["day_count"]} '
                f'signal_day={pred_t} buy={buy_t} sell={sell_t} '
                f'pearson_ic={pearson_ic:.4f} rank_ic={rank_ic:.4f} '
                f'raw_ret_std={raw_ret_std:.6f} '
                f'raw_ret_skew={raw_ret_skew:.4f} '
                f'theo_log_nav_spread={theo_log_nav_spread:.6f} '
                f'cumsum_pearson_ic={context["cumsum_pearson_ic"]:.4f} '
                f'cumsum_rank_ic={context["cumsum_rank_ic"]:.4f} '
                f'cumsum_vol_pearson_ic={context["cumsum_vol_pearson_ic"]:.4f} '
                f'recent10_pic_mean={np.mean(recent_p):.4f} '
                f'recent10_ric_mean={np.mean(recent_r):.4f}'
            )
    else:
        context['pearson_ic_history'].append(np.nan)
        context['rank_ic_history'].append(np.nan)
        context['raw_ret_std_history'].append(np.nan)
        context['day_count'] += 1

    return f3d


def ic_epilogue(name: str, context: dict[str, Any] | None) -> None:
    """退出前汇总：计算 mean IC, ICIR, winrate, max drawdown（基于 rank IC）。"""
    if context is None or not context.get('rank_ic_history'):
        logging.warning(f'[{name}] Epilogue: No IC data to summarize.')
        return

    ics = [x for x in context['rank_ic_history'] if not np.isnan(x)]

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

    # Pearson IC stats too
    p_ics = [x for x in context.get('pearson_ic_history', []) if not np.isnan(x)]
    p_mean = np.mean(p_ics) if p_ics else 0.0
    p_icir = p_mean / np.std(p_ics) if p_ics and np.std(p_ics) > 0 else 0.0

    first_day = context.get('first_signal_day', 'N/A')
    last_day = context.get('last_signal_day', 'N/A')
    logging.info('========== IC Summary ==========')
    logging.info(f' Signal range: [{first_day} .. {last_day}]')
    logging.info(f' N={len(ics)}, Rank IC: mean={mean_ic:.4f}, ICIR={icir:.4f}')
    logging.info(f' Pearson IC: mean={p_mean:.4f}, ICIR={p_icir:.4f}')
    logging.info(f' WinRate={winrate:.2%}, CumSum Rank IC={cumsum[-1]:.4f}')
    logging.info(f' IC Std={std_ic:.4f}, IC Skew={pd.Series(ics).skew():.4f}')
    logging.info(f' Max CumSum DD={max_dd:.4f}')

    # ---- 理论 top-bottom 对数净值差（逐日已记录，此处仅汇总日志） ----
    cc = context.get('day_count', 0)
    cumsum_raw = context.get('cumsum_raw_ret_std', 0.0)
    cumsum_p = context.get('cumsum_pearson_ic', 0.0)
    num_groups = context.get('num_groups', 10)
    if cc > 0 and cumsum_raw > 0 and num_groups >= 2:
        from scipy.stats import norm  # 延迟导入
        mean_ret_std = cumsum_raw / cc
        phi_val = float(norm.pdf(norm.ppf(1.0 / num_groups)))
        final_theo = 2.0 * num_groups * phi_val * mean_ret_std * cumsum_p
        logging.info(
            f' Final theoretical log NAV spread: {final_theo:.6f} '
            f'(N={num_groups}, mean_ret_std={mean_ret_std:.6f}, '
            f'cumsum_pearson_ic={cumsum_p:.4f})'
        )

    logging.info('======================================')
