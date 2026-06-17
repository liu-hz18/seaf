"""
趋势因子（合并：MA偏离/交叉 + MACD/动量） — 16 个因子。
趋势家族：price vs MA ×5, MA交叉 ×3, MACD ×2, 通道突破 ×2,
          时序动量 ×2, 成交量确认 ×1, 复合 ×1。
"""

from __future__ import annotations

import logging

import numpy as np

from qpipe.frame3d import Frame3D


def compute_trend_factors(name: str, idx: int, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个趋势因子（MA + MACD/动量）。"""
    result = f3d.copy()
    close = f3d.df['close']
    df = result.df
    grp = df.index.get_level_values('code')

    def _roll(src, dst, window, agg):
        df[dst] = (
            df.groupby('code')[src].rolling(window, min_periods=max(1, window // 2)).agg(agg).reset_index(level=0, drop=True)
        )

    # ===== MA 偏离 + 交叉 (8 cols) =====
    for p in [5, 10, 20, 60, 120]:
        df[f'_ma{p}'] = close
        _roll(f'_ma{p}', f'_ma{p}', p, 'mean')
        df[f'factor_trend_ma_{p}d'] = close / df[f'_ma{p}'].replace(0, np.nan) - 1

    df['factor_trend_ma_cross_5_20'] = df['_ma5'] / df['_ma20'].replace(0, np.nan) - 1
    df['factor_trend_ma_cross_10_60'] = df['_ma10'] / df['_ma60'].replace(0, np.nan) - 1
    df['factor_trend_ma_cross_20_120'] = df['_ma20'] / df['_ma120'].replace(0, np.nan) - 1

    # ===== MACD (2 cols) =====
    def _ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    ema12 = close.groupby(grp).transform(lambda x: _ema(x, 12))
    ema26 = close.groupby(grp).transform(lambda x: _ema(x, 26))
    macd = ema12 - ema26
    macd_signal = macd.groupby(grp).transform(lambda x: _ema(x, 9))
    df['factor_trend_macd'] = macd
    df['factor_trend_macd_signal'] = macd - macd_signal

    # ===== 通道突破 (2 cols) =====
    for p in [20, 60]:
        df[f'_min{p}'] = close
        df[f'_max{p}'] = close
        _roll(f'_min{p}', f'_min{p}', p, 'min')
        _roll(f'_max{p}', f'_max{p}', p, 'max')
        df[f'factor_trend_channel_{p}d'] = (close - df[f'_min{p}']) / (
            df[f'_max{p}'] - df[f'_min{p}']
        ).replace(0, np.nan)

    # ===== 时序动量强度 (2 cols) =====
    df['factor_trend_mom_strength_20d'] = f3d.ts_zscore('close', 20).df['close']
    df['factor_trend_mom_strength_60d'] = f3d.ts_zscore('close', 60).df['close']

    # ===== 成交量确认 (1 col) =====
    vol_z = f3d.cs_zscore('volume').df['volume']
    df['_ma20'] = close
    _roll('_ma20', '_ma20', 20, 'mean')
    df['factor_trend_vol_confirm'] = (close / df['_ma20'].replace(0, np.nan) - 1) * vol_z

    # ===== 复合 (1 col) =====
    df['factor_trend_composite'] = (
        df['factor_trend_macd_signal']
        + df['factor_trend_channel_20d']
        + df['factor_trend_mom_strength_20d']
        + df['factor_trend_vol_confirm']
    ) / 4

    factor_cols = [c for c in df.columns if c.startswith('factor_trend_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f'[{idx}] Factor NaN: { {c: result.df[c].isna().sum() for c in factor_cols} }')
    return Frame3D(result.df[factor_cols].copy())
