"""
反转因子 — 16 个因子，基于短期反转效应和极端值检测。
优化：消除重复 ts_zscore 调用，添加 epsilon 防 NaN。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D

EPS = 1e-8


def compute_reversal_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个反转因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    open_p = f3d.df['open']
    volume = f3d.df['volume']
    high = f3d.df['high']
    low = f3d.df['low']
    df = result.df

    ret1 = f3d.ts_pct_change('close', 1).df['close']
    df['_ret1'] = ret1
    df['_ret3'] = f3d.ts_pct_change('close', 3).df['close']
    df['_ret5'] = f3d.ts_pct_change('close', 5).df['close']

    vol_rank = f3d.cs_rank('volume').df['volume']

    # ---- 1-3: 短期反转 ----
    df['factor_rev_ret_1d'] = -df['_ret1']
    df['factor_rev_ret_3d'] = -df['_ret3']
    df['factor_rev_ret_5d'] = -df['_ret5']

    # ---- 4-5: 隔夜反转 ----
    df['_close_d1'] = df.groupby('name')['close'].shift(1)
    overnight = open_p / (df['_close_d1'] + EPS) - 1  # epsilon 防除零
    df['factor_rev_overnight_1d'] = -overnight
    df['_ovn'] = overnight
    df['factor_rev_overnight_5d'] = -df.groupby('name')['_ovn'].rolling(
        5, min_periods=2).mean().values

    # ---- 6-7: 日内反转 ----
    intraday = close / (open_p + EPS) - 1
    df['factor_rev_intraday_1d'] = -intraday
    df['_intra'] = intraday
    df['factor_rev_intraday_5d'] = -df.groupby('name')['_intra'].rolling(
        5, min_periods=2).mean().values

    # ---- 8-9: 成交量反转 ----
    df['factor_rev_volrev_1d'] = -df['_ret1'] * vol_rank
    df['factor_rev_volrev_3d'] = -df['_ret3'] * vol_rank

    # ---- 10-12: Z-score 反转（只算一次，复用三次） ----
    z20 = f3d.copy().ts_zscore('close', 20).df['close']
    df['factor_rev_zscore_1d'] = -z20
    df['factor_rev_zscore_3d'] = -z20   # 复用！之前重复算了 ts_zscore
    df['factor_rev_zscore_5d'] = -z20   # 复用！之前重复算了 ts_zscore

    # ---- 13: 缺口反转 ----
    df['factor_rev_gap_1d'] = -overnight * z20.fillna(0)

    # ---- 14: 高低价范围反转 ----
    df['factor_rev_range_rev'] = -(high - low) / (close + EPS)

    # ---- 15: 换手率反转 ----
    to_rank = f3d.cs_rank('turnover').df['turnover']
    df['factor_rev_turnover_1d'] = -ret1 * to_rank

    # ---- 16: 复合 ----
    df['factor_rev_composite'] = (
        -ret1 * 0.25 - overnight * 0.25 - intraday * 0.25 - ret1 * vol_rank * 0.25
    )

    factor_cols = [c for c in df.columns if c.startswith('factor_rev_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] Reversal NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
