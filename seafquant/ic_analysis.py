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
import warnings
from typing import Any

import numpy as np
import pandas as pd

# scipy 延迟导入到 ic_analysis_fn 内，节省顶层导入时间
from qpipe.frame3d import Frame3D
from qpipe.utils import mlflow_log_metrics

CLIP_PERCENT = 1
MIN_VALID_CODES = 2
LOG_EVERY = 20
CODE_PREFIXS = (
    'sh.600', 'sh.601', 'sh.603', 'sh.605',  # 沪市主板
    'sz.000', 'sz.001', 'sz.002', 'sz.003', 'sz.004',  # 深市主板
)  # tuple type


def compute_stats(context: dict, pred_df: pd.DataFrame, buy_df: pd.DataFrame, sell_df: pd.DataFrame) -> dict | None:
    codes = pred_df.index.intersection(buy_df.index).intersection(sell_df.index)
    n_codes = len(codes)
    if n_codes <= MIN_VALID_CODES:
        return None

    signal_col = context.get('signal_col', 'pred_signal')
    pred_signal = pred_df.loc[codes, signal_col].astype(float).values
    close_buy = buy_df.loc[codes, 'close'].astype(float).values
    close_sell = sell_df.loc[codes, 'close'].astype(float).values

    close_uq_vec = sell_df.loc[codes, 'close_uq'].astype(float).values
    stock_name_vec = sell_df.loc[codes, 'stock_name'].values

    # close 为 0（退市/停牌）时 log 报除零
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     log_sell = np.log(close_sell)
    #     log_buy = np.log(close_buy)
    # log_sell[close_sell <= 0] = np.nan
    # log_buy[close_buy <= 0] = np.nan
    # fwd_ret = log_sell - log_buy
    # NOTE: 使用简单收益率符合截面预期超额收益率的定义
    with np.errstate(divide='ignore', invalid='ignore'):
        fwd_ret = close_sell / close_buy - 1.0
    fwd_ret[close_sell <= 0] = np.nan
    fwd_ret[close_buy <= 0] = np.nan

    # 全量 cs_excess（默认全 NaN；valid 子集在 IC 计算块中填充）
    cs_excess_full = np.full(n_codes, np.nan, dtype=float)

    valid_mask = ~np.isnan(fwd_ret) & ~np.isnan(pred_signal)
    if valid_mask.sum() > MIN_VALID_CODES:
        from scipy.stats import pearsonr, spearmanr  # 延迟导入

        pred_valid = pred_signal[valid_mask]
        fwd_valid = fwd_ret[valid_mask]

        # 截面标准化（与 model label 定义一致）
        cs_mean, cs_std = np.nanmean(fwd_valid), np.nanstd(fwd_valid)
        cs_excess = (fwd_valid - cs_mean) / cs_std if cs_std > 0 else np.zeros_like(fwd_valid)
        # 全量 cs_excess（含 NaN 股票，供输出帧使用）
        cs_excess_full[valid_mask] = cs_excess

        # 记录统计指标
        # Raw return std（未经 cs_zscore 的截面收益率标准差）
        raw_ret_min, raw_ret_max = float(np.nanmin(fwd_valid)), float(np.nanmax(fwd_valid))
        raw_ret_p01, raw_ret_p99 = np.nanpercentile(fwd_valid, [CLIP_PERCENT, 100 - CLIP_PERCENT])
        raw_ret_mean, raw_ret_std = float(cs_mean), float(cs_std)
        raw_ret_skew, raw_ret_kurt = float(pd.Series(fwd_valid).skew()), float(pd.Series(fwd_valid).kurt())

        # Pearson IC
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            try:
                pearson_ic = pearsonr(pred_valid, cs_excess).correlation
                if np.isnan(pearson_ic):
                    pearson_ic = 0.0
            except Exception:
                pearson_ic = 0.0

        # Rank (Spearman) IC
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            try:
                rank_ic = spearmanr(pred_valid, cs_excess).correlation
                if np.isnan(rank_ic):
                    rank_ic = 0.0
            except Exception:
                rank_ic = 0.0

        return {
            'raw_ret_min': raw_ret_min,
            'raw_ret_max': raw_ret_max,
            'raw_ret_p01': raw_ret_p01,
            'raw_ret_p99': raw_ret_p99,
            'raw_ret_mean': raw_ret_mean,
            'raw_ret_std': raw_ret_std,
            'raw_ret_skew': raw_ret_skew,
            'raw_ret_kurt': raw_ret_kurt,
            'pearson_ic': pearson_ic,
            'rank_ic': rank_ic,

            'codes': codes,
            'stock_name': stock_name_vec,
            'close_buy': close_buy,
            'close_sell': close_sell,
            'close_uq': close_uq_vec,
            'pred_signal': pred_signal,
            'fwd_ret': fwd_ret,
            'cs_excess_fwd_ret': cs_excess_full,
        }

    return None


def filter_df(codes: tuple[str], df: pd.DataFrame) -> pd.DataFrame:
    code_series = pd.Series(df.index.get_level_values('code'), index=df.index)
    return df[code_series.str.startswith(codes, na=False)]


def ic_analysis_fn(name: str, idx: int, f3d: Frame3D, context: dict) -> Frame3D:
    """每日 IC 计算函数。

    f3d 包含 window 天的数据（pred_signal 列 + close 列）。
    context 可包含 'fwd'（默认 20）控制前瞻天数。
    """
    if context is None:
        context = {}

    context.setdefault('day_count', 0)
    context.setdefault('first_signal_day', None)
    context.setdefault('last_signal_day', None)
    # 全指统计量
    context.setdefault('pearson_ic_history', [])
    context.setdefault('rank_ic_history', [])
    context.setdefault('cumsum_pearson_ic', 0.0)
    context.setdefault('cumsum_rank_ic', 0.0)
    context.setdefault('cumsum_vol_pearson_ic', 0.0)
    # 主板统计量
    context.setdefault('pearson_ic_history_main', [])
    context.setdefault('rank_ic_history_main', [])
    context.setdefault('cumsum_pearson_ic_main', 0.0)
    context.setdefault('cumsum_rank_ic_main', 0.0)
    context.setdefault('cumsum_vol_pearson_ic_main', 0.0)
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

    # ---- 按股票代码对齐，一次计算服务 IC 统计 + 输出帧两个用途 ----
    pred_df = df.xs(pred_t, level='key')
    buy_df = df.xs(buy_t, level='key')
    sell_df = df.xs(sell_t, level='key')

    stats_all = compute_stats(context, pred_df, buy_df, sell_df)
    stats_main = compute_stats(
        context,
        filter_df(CODE_PREFIXS, pred_df),
        filter_df(CODE_PREFIXS, buy_df),
        filter_df(CODE_PREFIXS, sell_df),
    )
    if not stats_all or not stats_main:
        return f3d

    context['day_count'] += 1
    # 全指统计量
    context['pearson_ic_history'].append(stats_all['pearson_ic'])
    context['rank_ic_history'].append(stats_all['rank_ic'])
    context['cumsum_pearson_ic'] += stats_all['pearson_ic']
    context['cumsum_rank_ic'] += stats_all['rank_ic']
    simple_excess_ret = stats_all['pearson_ic'] * stats_all['raw_ret_std']
    context['cumsum_vol_pearson_ic'] += np.log(1. + simple_excess_ret)  # 简单收益率转化为对数收益率，可以累加
    # 主板统计量
    context['pearson_ic_history_main'].append(stats_main['pearson_ic'])
    context['rank_ic_history_main'].append(stats_main['rank_ic'])
    context['cumsum_pearson_ic_main'] += stats_main['pearson_ic']
    context['cumsum_rank_ic_main'] += stats_main['rank_ic']
    simple_excess_ret = stats_main['pearson_ic'] * stats_main['raw_ret_std']
    context['cumsum_vol_pearson_ic_main'] += np.log(1. + simple_excess_ret)  # 简单收益率转化为对数收益率，可以累加

    # 理论 top-bottom 对数净值差（逐日）NOTE: 这个数字比真实的持仓净值滞后 fwd 天
    num_groups = context.get('num_groups', 10)
    theo_log_nav_spread, theo_log_nav_spread_main = 0.0, 0.0
    if num_groups >= 2:
        from scipy.stats import norm  # 延迟导入
        phi_inv = float(norm.ppf(1.0 / num_groups))
        phi_val = float(norm.pdf(phi_inv))
        theo_log_nav_spread = (
            2.0 * num_groups * phi_val * context['cumsum_vol_pearson_ic'] / (fwd-1)  # NOTE: we actually hold fwd-1 days
        )
        theo_log_nav_spread_main = (
            2.0 * num_groups * phi_val * context['cumsum_vol_pearson_ic_main'] / (fwd-1)  # NOTE: we actually hold fwd-1 days
        )

    # ---- MLflow 逐日记录 ----
    mlflow_run_id = context.get('mlflow_run_id', '')
    if mlflow_run_id:
        mlflow_log_metrics(mlflow_run_id, name, {
            # 全指
            'all.pearson_ic': stats_all['pearson_ic'],
            'all.rank_ic': stats_all['rank_ic'],
            'all.raw_ret_min': stats_all['raw_ret_min'],
            'all.raw_ret_max': stats_all['raw_ret_max'],
            'all.raw_ret_p01': stats_all['raw_ret_p01'],
            'all.raw_ret_p99': stats_all['raw_ret_p99'],
            'all.raw_ret_mean': stats_all['raw_ret_mean'],
            'all.raw_ret_std': stats_all['raw_ret_std'],
            'all.raw_ret_skew': stats_all['raw_ret_skew'],
            'all.raw_ret_kurt': stats_all['raw_ret_kurt'],
            'all.theo_log_nav_spread': theo_log_nav_spread,
            'all.cumsum_pearson_ic': context['cumsum_pearson_ic'],
            'all.cumsum_vol_pearson_ic': context['cumsum_vol_pearson_ic'],
            'all.cumsum_rank_ic': context['cumsum_rank_ic'],
            # 主板
            'main.pearson_ic': stats_main['pearson_ic'],
            'main.rank_ic': stats_main['rank_ic'],
            'main.raw_ret_std': stats_main['raw_ret_std'],
            'main.raw_ret_skew': stats_main['raw_ret_skew'],
            'main.raw_ret_kurt': stats_main['raw_ret_kurt'],
            'main.theo_log_nav_spread': theo_log_nav_spread_main,
            'main.cumsum_pearson_ic': context['cumsum_pearson_ic_main'],
            'main.cumsum_vol_pearson_ic': context['cumsum_vol_pearson_ic_main'],
            'main.cumsum_rank_ic': context['cumsum_rank_ic_main'],
        }, step=idx)

    if context['day_count'] % LOG_EVERY == 0 or context['day_count'] == 1:
        recent_p = context['pearson_ic_history'][-LOG_EVERY:]
        recent_r = context['rank_ic_history'][-LOG_EVERY:]
        logging.info(
            f"[{idx}][all] IC#{context["day_count"]} "
            f"signal_day={pred_t} buy={buy_t} sell={sell_t} "
            f"pearson_ic={stats_all['pearson_ic']:.4f} rank_ic={stats_all['rank_ic']:.4f} "
            f"raw_ret_std={stats_all['raw_ret_std']:.6f} "
            f"raw_ret_skew={stats_all['raw_ret_skew']:.4f} "
            f"theo_log_nav_spread={theo_log_nav_spread:.6f} "
            f"cumsum_pearson_ic={context['cumsum_pearson_ic']:.4f} "
            f"cumsum_rank_ic={context['cumsum_rank_ic']:.4f} "
            f"recent10_pic_mean={np.mean(recent_p):.4f} "
            f"recent10_ric_mean={np.mean(recent_r):.4f}"
        )
        recent_p = context['pearson_ic_history_main'][-LOG_EVERY:]
        recent_r = context['rank_ic_history_main'][-LOG_EVERY:]
        logging.info(
            f"[{idx}][main] IC#{context["day_count"]} "
            f"signal_day={pred_t} buy={buy_t} sell={sell_t} "
            f"pearson_ic={stats_main['pearson_ic']:.4f} rank_ic={stats_main['rank_ic']:.4f} "
            f"raw_ret_std={stats_main['raw_ret_std']:.6f} "
            f"raw_ret_skew={stats_main['raw_ret_skew']:.4f} "
            f"theo_log_nav_spread={theo_log_nav_spread_main:.6f} "
            f"cumsum_pearson_ic={context['cumsum_pearson_ic_main']:.4f} "
            f"cumsum_rank_ic={context['cumsum_rank_ic_main']:.4f} "
            f"recent10_pic_mean={np.mean(recent_p):.4f} "
            f"recent10_ric_mean={np.mean(recent_r):.4f}"
        )

    mi = pd.MultiIndex.from_product([[sell_t], stats_all['codes']], names=['key', 'code'])
    result_df = pd.DataFrame({
        'stock_name': stats_all['stock_name'],
        'close_buy': stats_all['close_buy'],
        'close_sell': stats_all['close_sell'],
        'close_uq': stats_all['close_uq'],
        'fwd_ret': stats_all['fwd_ret'],
        'cs_excess_fwd_ret': stats_all['cs_excess_fwd_ret'],
        'pred_signal': stats_all['pred_signal'],
    }, index=mi)
    return Frame3D(result_df)


def print_ic_summary(idx: int, prefix: str, first_day: str, last_day: str, rank_ic_history: list[float], pearson_ic_history: list[float]):
    ics = [x for x in rank_ic_history if not np.isnan(x)]
    if len(ics) < 10:
        logging.warning(f'[{idx}] Epilogue: Insufficient IC data ({len(ics)} points).')
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
    p_ics = [x for x in pearson_ic_history if not np.isnan(x)]
    p_mean = np.mean(p_ics) if p_ics else 0.0
    p_icir = p_mean / np.std(p_ics) if p_ics and np.std(p_ics) > 0 else 0.0

    logging.info(f'[{idx}][{prefix}] ========== IC Summary ==========')
    logging.info(f'[{idx}][{prefix}]  Signal range: [{first_day} .. {last_day}]')
    logging.info(f'[{idx}][{prefix}]  N={len(ics)}, Rank IC: mean={mean_ic:.4f}, ICIR={icir:.4f}')
    logging.info(f'[{idx}][{prefix}]  Pearson IC: mean={p_mean:.4f}, ICIR={p_icir:.4f}')
    logging.info(f'[{idx}][{prefix}]  WinRate={winrate:.2%}, CumSum Rank IC={cumsum[-1]:.4f}')
    logging.info(f'[{idx}][{prefix}]  IC Std={std_ic:.4f}, IC Skew={pd.Series(ics).skew():.4f}')
    logging.info(f'[{idx}][{prefix}]  Max CumSum DD={max_dd:.4f}')


def ic_epilogue(name: str, idx: int, context: dict[str, Any] | None) -> None:
    """退出前汇总：计算 mean IC, ICIR, winrate, max drawdown（基于 rank IC）。"""
    if context is None or not context.get('rank_ic_history'):
        logging.warning(f'[{idx}] Epilogue: No IC data to summarize.')
        return

    first_day = context.get('first_signal_day', 'N/A')
    last_day = context.get('last_signal_day', 'N/A')
    fwd = context.get('fwd', 20)

    print_ic_summary(idx, 'all', first_day, last_day, context['rank_ic_history'], context['pearson_ic_history'])
    print_ic_summary(idx, 'main', first_day, last_day, context['rank_ic_history_main'], context['pearson_ic_history_main'])

    # ---- 全指 理论 top-bottom 对数净值差（逐日已记录，此处仅汇总日志） ----
    cumsum_vol_p = context.get('cumsum_vol_pearson_ic', 0.0)
    cumsum_pic = context.get('cumsum_pearson_ic', 0.0)
    num_groups = context.get('num_groups', 10)
    if num_groups >= 2:
        from scipy.stats import norm  # 延迟导入
        phi_val = float(norm.pdf(norm.ppf(1.0 / num_groups)))
        final_theo = 2.0 * num_groups * phi_val * cumsum_vol_p / (fwd-1)  # NOTE: we actually hold fwd-1 days
        logging.info(
            f'[{idx}][all]  Final theoretical log NAV spread: {final_theo:.6f} '
            f'(N={num_groups}, cumsum_vol_pearson_ic={cumsum_vol_p:.6f}, '
            f'cumsum_pearson_ic={cumsum_pic:.4f})'
        )

    # ---- 主板 理论 top-bottom 对数净值差（逐日已记录，此处仅汇总日志） ----
    cumsum_vol_p = context.get('cumsum_vol_pearson_ic_main', 0.0)
    cumsum_pic = context.get('cumsum_pearson_ic_main', 0.0)
    num_groups = context.get('num_groups', 10)
    if num_groups >= 2:
        from scipy.stats import norm  # 延迟导入
        phi_val = float(norm.pdf(norm.ppf(1.0 / num_groups)))
        final_theo = 2.0 * num_groups * phi_val * cumsum_vol_p / (fwd-1)  # NOTE: we actually hold fwd-1 days
        logging.info(
            f'[{idx}][main]  Final theoretical log NAV spread: {final_theo:.6f} '
            f'(N={num_groups}, cumsum_vol_pearson_ic={cumsum_vol_p:.6f}, '
            f'cumsum_pearson_ic={cumsum_pic:.4f})'
        )

    logging.info(f'[{idx}] ======================================')
