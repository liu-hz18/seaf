"""
趋势因子（MACD/动量）— 8 个因子：MACD/价格通道/动量强度/成交量确认/复合。
从 trend 拆分而来，与 trend（MA偏离/MA交叉）并行执行。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_trend_macd_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 8 个趋势 MACD/动量因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    grp = f3d.df.index.get_level_values('name')

    # ---- 辅助：直接 pandas groupby rolling ----
    def _add_rolling(df, src_col, dst_col, window, agg):
        rolled = df.groupby('name')[src_col].rolling(window, min_periods=max(1, window // 2)).agg(agg)
        df[dst_col] = rolled.values

    df = result.df

    # ---- 1-2: MACD ----
    def _ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    ema12 = close.groupby(grp).transform(lambda x: _ema(x, 12))
    ema26 = close.groupby(grp).transform(lambda x: _ema(x, 26))
    macd = ema12 - ema26
    macd_signal = macd.groupby(grp).transform(lambda x: _ema(x, 9))
    df['factor_trend_macd'] = macd
    df['factor_trend_macd_signal'] = macd - macd_signal

    # ---- 3-4: 价格通道突破 ----
    for p in [20, 60]:
        df[f'_min{p}'] = close
        df[f'_max{p}'] = close
        _add_rolling(df, f'_min{p}', f'_min{p}', p, 'min')
        _add_rolling(df, f'_max{p}', f'_max{p}', p, 'max')
        df[f'factor_trend_channel_{p}d'] = (
            (close - df[f'_min{p}']) / (df[f'_max{p}'] - df[f'_min{p}']).replace(0, np.nan)
        )

    # ---- 5-6: 时序动量强度 (zscore) ----
    df['factor_trend_mom_strength_20d'] = f3d.ts_zscore('close', 20).df['close']
    df['factor_trend_mom_strength_60d'] = f3d.ts_zscore('close', 60).df['close']

    # ---- 7: 成交量确认趋势 ----
    vol_z = f3d.cs_zscore('volume').df['volume']
    df['_ma20'] = close
    _add_rolling(df, '_ma20', '_ma20', 20, 'mean')
    df['factor_trend_vol_confirm'] = (close / df['_ma20'].replace(0, np.nan) - 1) * vol_z

    # ---- 8: 趋势复合 ----
    df['factor_trend_composite'] = (
        df['factor_trend_macd_signal'] +
        df['factor_trend_channel_20d'] +
        df['factor_trend_mom_strength_20d'] +
        df['factor_trend_vol_confirm']
    ) / 4

    # 截面标准化
    factor_cols = [c for c in df.columns if c.startswith('factor_trend_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] TrendMACD NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
