"""
质量因子（基础）— 16 个因子，基于收益稳定性/Sharpe/正向占比等简单滚动统计。
优化：使用直接 pandas groupby rolling 减少 f3d.copy() 深拷贝开销。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_quality_basic_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个质量基础因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    high = f3d.df['high']
    low = f3d.df['low']
    df = result.df

    ret = f3d.ts_pct_change('close', 1).df['close']
    df['_ret'] = ret

    def _roll(src, dst, window, agg):
        df[dst] = df.groupby('name')[src].rolling(window, min_periods=max(1, window // 2)).agg(agg).values

    # ---- 1-3: 收益稳定性 (1/std) ----
    for p in [20, 60, 120]:
        _roll('_ret', f'_ret_std{p}', p, 'std')
        df[f'factor_qb_ret_stability_{p}d'] = 1.0 / df[f'_ret_std{p}'].replace(0, np.nan)

    # ---- 4-6: Sharpe-like ratio ----
    for p in [20, 60, 120]:
        _roll('_ret', f'_ret_mean{p}', p, 'mean')
        df[f'factor_qb_sharpe_{p}d'] = df[f'_ret_mean{p}'] / df[f'_ret_std{p}'].replace(0, np.nan)

    # ---- 7-9: 正收益天数占比 ----
    df['_pos'] = (ret > 0).astype(float)
    for p in [20, 60, 120]:
        _roll('_pos', f'factor_qb_pos_days_{p}d', p, 'mean')

    # ---- 10-11: 收益衰减比 ----
    df['factor_qb_stability_decay'] = (
        df['factor_qb_ret_stability_20d'] / df['factor_qb_ret_stability_120d'].replace(0, np.nan)
    )
    df['factor_qb_sharpe_decay'] = (
        df['factor_qb_sharpe_20d'] / df['factor_qb_sharpe_120d'].replace(0, np.nan)
    )

    # ---- 12-13: 振幅稳定性 ----
    df['_amp'] = (high - low) / close
    for p in [20, 60]:
        _roll('_amp', f'_amp_std{p}', p, 'std')
        df[f'factor_qb_amp_stability_{p}d'] = 1.0 / df[f'_amp_std{p}'].replace(0, np.nan)

    # ---- 14-15: 最大回撤代理 ----
    for p in [60, 120]:
        _roll('close', f'_max{p}', p, 'max')
        df[f'factor_qb_maxdd_{p}d'] = close / df[f'_max{p}'].replace(0, np.nan) - 1

    # ---- 16: 复合 ----
    df['factor_qb_composite'] = (
        df['factor_qb_sharpe_60d'] + df['factor_qb_pos_days_60d'] +
        df['factor_qb_maxdd_60d'] + df['factor_qb_ret_stability_60d']
    ) / 4

    factor_cols = [c for c in df.columns if c.startswith('factor_qb_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] QualityBasic NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
