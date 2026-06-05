"""
动量/反转因子（合并） — 32 个因子。
动量：多周期收益率 ×8, 波动率调整收益率 ×8。
反转：短期反转 ×3, 隔夜反转 ×2, 日内反转 ×2, 量价反转 ×2,
      Z-score反转 ×3, 缺口反转 ×1, 振幅反转 ×1, 换手反转 ×1, 复合 ×1。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D

EPS = 1e-8


def compute_momentum_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 32 个动量+反转因子。"""
    result = f3d.copy()
    close, open_p, high, low = (f3d.df['close'], f3d.df['open'],
                                 f3d.df['high'], f3d.df['low'])
    volume = f3d.df['volume']
    df = result.df

    # ===== 动量：16 cols (prefix factor_mom_) =====
    periods = [1, 3, 5, 10, 20, 40, 60, 120]
    vol_windows = sorted(set(max(p, 5) for p in periods))
    df['_daily_ret'] = df.groupby('name')['close'].pct_change(1)
    for w in vol_windows:
        df[f'_vol_{w}'] = df.groupby('name')['_daily_ret'].rolling(
            w, min_periods=max(1, w // 2)).std().values
    result = result.ts_pct_change_multi('close', periods, prefix='factor_mom_ret', cp=False)
    for p in periods:
        df[f'factor_mom_voladj_{p}d'] = (
            df[f'factor_mom_ret_{p}d'] / (df[f'_vol_{max(p, 5)}'] + EPS))

    # ===== 反转：16 cols (prefix factor_rev_) =====
    ret1 = f3d.ts_pct_change('close', 1).df['close']
    df['_ret1'] = ret1
    df['_ret3'] = f3d.ts_pct_change('close', 3).df['close']
    df['_ret5'] = f3d.ts_pct_change('close', 5).df['close']
    vol_rank = f3d.cs_rank('volume').df['volume']

    df['factor_rev_ret_1d'] = -df['_ret1']
    df['factor_rev_ret_3d'] = -df['_ret3']
    df['factor_rev_ret_5d'] = -df['_ret5']

    df['_close_d1'] = df.groupby('name')['close'].shift(1)
    overnight = open_p / (df['_close_d1'] + EPS) - 1
    df['factor_rev_overnight_1d'] = -overnight
    df['_ovn'] = overnight
    df['factor_rev_overnight_5d'] = -df.groupby('name')['_ovn'].rolling(
        5, min_periods=2).mean().values

    intraday = close / (open_p + EPS) - 1
    df['factor_rev_intraday_1d'] = -intraday
    df['_intra'] = intraday
    df['factor_rev_intraday_5d'] = -df.groupby('name')['_intra'].rolling(
        5, min_periods=2).mean().values

    df['factor_rev_volrev_1d'] = -df['_ret1'] * vol_rank
    df['factor_rev_volrev_3d'] = -df['_ret3'] * vol_rank

    z20 = f3d.copy().ts_zscore('close', 20).df['close']
    df['factor_rev_zscore_1d'] = -z20
    df['factor_rev_zscore_3d'] = -z20
    df['factor_rev_zscore_5d'] = -z20
    df['factor_rev_gap_1d'] = -overnight * z20.fillna(0)
    df['factor_rev_range_rev'] = -(high - low) / (close + EPS)
    to_rank = f3d.cs_rank('turnover').df['turnover']
    df['factor_rev_turnover_1d'] = -ret1 * to_rank
    df['factor_rev_composite'] = (
        -ret1 * 0.25 - overnight * 0.25 - intraday * 0.25 - ret1 * vol_rank * 0.25)

    # 联合截面标准化
    factor_cols = [c for c in df.columns
                   if c.startswith('factor_mom_') or c.startswith('factor_rev_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)
    return Frame3D(result.df[factor_cols].copy())
