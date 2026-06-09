"""
波动/日内因子（合并） — 33 个因子。
波动率：已实现波动 ×4, 下行波动 ×4, 波动率波动 ×2, Parkinson ×2,
        GK ×1, 波动率趋势 ×2, 高低价范围 ×1。
日内：日内收益 ×2, 隔夜跳空 ×2, HL范围 ×3, 收盘位置 ×2, 开盘位置 ×1,
      日内波动率 ×2, 方向持续性 ×2, 跳空效率 ×1, 复合 ×1。
"""

from __future__ import annotations

import numpy as np

from qpipe.frame3d import Frame3D


def compute_volatility_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 33 个波动率+日内因子。"""
    result = f3d.copy()
    close, high, low, open_p, _volume = (
        f3d.df['close'],
        f3d.df['high'],
        f3d.df['low'],
        f3d.df['open'],
        f3d.df['volume'],
    )
    df = result.df
    ret = f3d.ts_pct_change('close', 1).df['close']
    df['_ret'] = ret

    def _roll(src, dst, window, agg):
        df[dst] = (
            df.groupby('name')[src].rolling(window, min_periods=max(1, window // 2)).agg(agg).reset_index(level=0, drop=True)
        )

    # ===== 波动率：16 cols (prefix factor_vol_) =====
    for p in [5, 10, 20, 60]:
        _roll('_ret', f'factor_vol_realized_{p}d', p, 'std')

    df['_neg_ret'] = ret.clip(upper=0)
    for p in [5, 10, 20, 60]:
        _roll('_neg_ret', f'_ds_mean{p}', p, 'mean')
        df[f'factor_vol_downside_{p}d'] = np.sqrt(np.abs(df[f'_ds_mean{p}']))

    _roll('factor_vol_realized_5d', 'factor_vol_of_vol_20d', 20, 'std')
    _roll('factor_vol_realized_20d', 'factor_vol_of_vol_60d', 60, 'std')

    df['_loghl'] = np.log(high / low)
    park_factor = 1.0 / (4 * np.log(2))
    df['_park_sq'] = park_factor * df['_loghl'] ** 2
    _roll('_park_sq', '_park5_mean', 5, 'mean')
    df['factor_vol_parkinson_5d'] = np.sqrt(np.abs(df['_park5_mean']))
    _roll('_park_sq', '_park20_mean', 20, 'mean')
    df['factor_vol_parkinson_20d'] = np.sqrt(np.abs(df['_park20_mean']))

    log_co = np.log(close / open_p)
    df['_gk'] = 0.5 * df['_loghl'] ** 2 - (2 * np.log(2) - 1) * log_co**2
    _roll('_gk', '_gk5_mean', 5, 'mean')
    df['factor_vol_gk_5d'] = np.sqrt(np.abs(df['_gk5_mean']))

    df['factor_vol_trend_20d'] = (
        df['factor_vol_realized_20d'] / df['factor_vol_realized_60d'].replace(0, np.nan) - 1
    )
    _roll('_ret', '_rv120', 120, 'std')
    df['factor_vol_trend_60d'] = df['factor_vol_realized_60d'] / df['_rv120'].replace(0, np.nan) - 1
    df['_range'] = high / low - 1
    _roll('_range', 'factor_vol_range_20d', 20, 'mean')

    # ===== 日内：17 cols (prefix factor_intra_) =====
    df['_itra'] = close / open_p - 1
    _roll('_itra', 'factor_intra_ret_mean_5d', 5, 'mean')
    _roll('_itra', 'factor_intra_ret_mean_20d', 20, 'mean')

    df['_close_d1'] = df.groupby('name')['close'].shift(1)
    df['_gap'] = open_p / df['_close_d1'] - 1
    _roll('_gap', 'factor_intra_overnight_gap_5d', 5, 'mean')
    _roll('_gap', 'factor_intra_overnight_gap_20d', 20, 'mean')

    df['_hl'] = (high - low) / close
    for p in [5, 20, 60]:
        _roll('_hl', f'factor_intra_hl_range_{p}d', p, 'mean')

    df['_rpos'] = (close - low) / (high - low).replace(0, np.nan)
    _roll('_rpos', 'factor_intra_close_position_5d', 5, 'mean')
    _roll('_rpos', 'factor_intra_close_position_20d', 20, 'mean')

    df['_opos'] = (open_p - low) / (high - low).replace(0, np.nan)
    _roll('_opos', 'factor_intra_open_position_5d', 5, 'mean')
    _roll('_opos', 'factor_intra_open_position_20d', 20, 'mean')

    _roll('_loghl', 'factor_intra_hl_vol_5d', 5, 'std')
    _roll('_loghl', 'factor_intra_hl_vol_20d', 20, 'std')

    df['_dir'] = (close - open_p) / (high - low).replace(0, np.nan)
    _roll('_dir', 'factor_intra_directionality_5d', 5, 'mean')
    _roll('_dir', 'factor_intra_directionality_std_5d', 5, 'std')

    df['_gr'] = df['_gap'] / (df['_hl'] + 1e-6)
    _roll('_gr', 'factor_intra_gap_efficiency_5d', 5, 'mean')

    df['factor_intra_composite'] = (
        df['factor_intra_close_position_5d']
        + df['factor_intra_directionality_5d']
        - df['factor_intra_hl_range_5d']
        + df['factor_intra_gap_efficiency_5d']
    ) / 4

    # 联合截面标准化
    factor_cols = [c for c in df.columns if c.startswith(('factor_vol_', 'factor_intra_'))]
    result = result.cs_zscore_batch(factor_cols, cp=False)
    return Frame3D(result.df[factor_cols].copy())
