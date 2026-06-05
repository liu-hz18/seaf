"""
日内特征因子 — 17 个因子，基于开盘-收盘价差、高低价范围、日内波动等。
优化：使用直接 pandas groupby rolling 减少 f3d.copy() 深拷贝开销。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_intraday_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 17 个日内特征因子。"""
    result = f3d.copy()
    open_p = f3d.df['open']
    high = f3d.df['high']
    low = f3d.df['low']
    close = f3d.df['close']
    df = result.df

    def _roll(src, dst, window, agg):
        df[dst] = df.groupby('name')[src].rolling(window, min_periods=max(1, window // 2)).agg(agg).values

    # ---- 1-2: 日内收益 ----
    df['_itra'] = close / open_p - 1
    _roll('_itra', 'factor_intra_ret_mean_5d', 5, 'mean')
    _roll('_itra', 'factor_intra_ret_mean_20d', 20, 'mean')

    # ---- 3-4: 隔夜跳空 ----
    df['_close_d1'] = df.groupby('name')['close'].shift(1)
    df['_gap'] = open_p / df['_close_d1'] - 1
    _roll('_gap', 'factor_intra_overnight_gap_5d', 5, 'mean')
    _roll('_gap', 'factor_intra_overnight_gap_20d', 20, 'mean')

    # ---- 5-7: 高低价范围 ----
    df['_hl'] = (high - low) / close
    for p in [5, 20, 60]:
        _roll('_hl', f'factor_intra_hl_range_{p}d', p, 'mean')

    # ---- 8-9: 收盘价相对位置 ----
    df['_rpos'] = (close - low) / (high - low).replace(0, np.nan)
    _roll('_rpos', 'factor_intra_close_position_5d', 5, 'mean')
    _roll('_rpos', 'factor_intra_close_position_20d', 20, 'mean')

    # ---- 10-11: 开盘价相对位置 ----
    df['_opos'] = (open_p - low) / (high - low).replace(0, np.nan)
    _roll('_opos', 'factor_intra_open_position_5d', 5, 'mean')

    # ---- 12-13: 日内波动率 ----
    df['_loghl'] = np.log(high / low)
    _roll('_loghl', 'factor_intra_hl_vol_5d', 5, 'std')
    _roll('_loghl', 'factor_intra_hl_vol_20d', 20, 'std')

    # ---- 14-15: 方向持续性 ----
    df['_dir'] = (close - open_p) / (high - low).replace(0, np.nan)
    _roll('_dir', 'factor_intra_directionality_5d', 5, 'mean')
    _roll('_dir', 'factor_intra_directionality_std_5d', 5, 'std')

    # ---- 16: 开盘跳空衰减率 ----
    df['_gr'] = df['_gap'] / (df['_hl'] + 1e-6)
    _roll('_gr', 'factor_intra_gap_efficiency_5d', 5, 'mean')

    # ---- 17: 复合 ----
    df['factor_intra_composite'] = (
        df['factor_intra_close_position_5d'] + df['factor_intra_directionality_5d'] -
        df['factor_intra_hl_range_5d'] + df['factor_intra_gap_efficiency_5d']
    ) / 4

    factor_cols = [c for c in df.columns if c.startswith('factor_intra_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] Intraday NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
