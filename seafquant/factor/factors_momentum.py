"""
动量因子 — 16 个因子，基于多周期收益率和波动率调整收益率。
优化：预计算 daily_ret + 唯一窗口 rolling_std，消除 8→6 次计算。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D

EPS = 1e-8


def compute_momentum_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个动量因子。"""
    result = f3d.copy()
    df = result.df
    periods = [1, 3, 5, 10, 20, 40, 60, 120]
    # 唯一波动率窗口: [5, 10, 20, 40, 60, 120]
    vol_windows = sorted(set(max(p, 5) for p in periods))

    # 预计算每日收益率（一次）
    df['_daily_ret'] = df.groupby('name')['close'].pct_change(1)

    # 预计算所有唯一窗口的 rolling std（每个窗口只算一次）
    for w in vol_windows:
        df[f'_vol_{w}'] = df.groupby('name')['_daily_ret'].rolling(
            w, min_periods=max(1, w // 2)).std().values

    # 批量 pct_change (一次 GroupBy 循环代替 8 次独立调用)
    result = result.ts_pct_change_multi('close', periods, prefix='factor_mom_ret', cp=False)
    for p in periods:
        df[f'factor_mom_voladj_{p}d'] = (
            df[f'factor_mom_ret_{p}d'] / (df[f'_vol_{max(p, 5)}'] + EPS))

    factor_cols = [c for c in df.columns if c.startswith('factor_mom_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] Momentum NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())