"""
趋势因子 — 16 个因子，基于移动均线、MACD、价格通道突破等趋势指标。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_trend_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个趋势因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    volume = f3d.df['volume']
    
    # 1-5: 价格 vs 移动均线偏离
    ma_periods = [5, 10, 20, 60, 120]
    for p in ma_periods:
        ma = f3d.copy().add_column('_c', close).ts_rolling('_c', p, 'mean').df['_c']
        dist = close / ma.replace(0, np.nan) - 1
        result = result.add_column(f'factor_trend_ma_{p}d', dist)
    
    # 6-8: MA 交叉
    ma5 = f3d.copy().add_column('_c', close).ts_rolling('_c', 5, 'mean').df['_c']
    ma20 = f3d.copy().add_column('_c', close).ts_rolling('_c', 20, 'mean').df['_c']
    ma10 = f3d.copy().add_column('_c', close).ts_rolling('_c', 10, 'mean').df['_c']
    ma60 = f3d.copy().add_column('_c', close).ts_rolling('_c', 60, 'mean').df['_c']
    ma120 = f3d.copy().add_column('_c', close).ts_rolling('_c', 120, 'mean').df['_c']
    
    result = result.add_column('factor_trend_ma_cross_5_20', ma5 / ma20.replace(0, np.nan) - 1)
    result = result.add_column('factor_trend_ma_cross_10_60', ma10 / ma60.replace(0, np.nan) - 1)
    result = result.add_column('factor_trend_ma_cross_20_120', ma20 / ma120.replace(0, np.nan) - 1)
    
    # 9-10: MACD (EMA(12) - EMA(26), signal=EMA(9) of MACD)
    def _ema(series, span):
        return series.ewm(span=span, adjust=False).mean()
    
    ema12 = close.groupby(f3d.df.index.get_level_values('name')).transform(
        lambda x: _ema(x, 12))
    ema26 = close.groupby(f3d.df.index.get_level_values('name')).transform(
        lambda x: _ema(x, 26))
    macd = ema12 - ema26
    macd_signal = macd.groupby(f3d.df.index.get_level_values('name')).transform(
        lambda x: _ema(x, 9))
    
    result = result.add_column('factor_trend_macd', macd)
    result = result.add_column('factor_trend_macd_signal', macd - macd_signal)
    
    # 11-12: 价格通道突破
    for p in [20, 60]:
        min_p = f3d.copy().add_column('_c', close).ts_rolling('_c', p, 'min').df['_c']
        max_p = f3d.copy().add_column('_c', close).ts_rolling('_c', p, 'max').df['_c']
        channel = (close - min_p) / (max_p - min_p).replace(0, np.nan)
        result = result.add_column(f'factor_trend_channel_{p}d', channel)
    
    # 13-14: 时序动量强度 (zscore)
    z20 = f3d.ts_zscore('close', 20).df['close']
    result = result.add_column('factor_trend_mom_strength_20d', z20)
    z60 = f3d.ts_zscore('close', 60).df['close']
    result = result.add_column('factor_trend_mom_strength_60d', z60)
    
    # 15: 成交量确认趋势
    vol_z = f3d.cs_zscore('volume').df['volume']
    dist_20 = close / ma20.replace(0, np.nan) - 1
    result = result.add_column('factor_trend_vol_confirm', dist_20 * vol_z)
    
    # 16: 趋势复合
    trend_comp = (result.df['factor_trend_ma_20d'] + result.df['factor_trend_ma_60d'] + 
                  result.df['factor_trend_ma_cross_5_20'] + result.df['factor_trend_macd_signal'] + 
                  result.df['factor_trend_channel_20d']) / 5
    result = result.add_column('factor_trend_composite', trend_comp)
    
    # 截面标准化
    factor_cols = [c for c in result.df.columns if c.startswith('factor_trend_')]
    for col in factor_cols:
        result = result.cs_zscore(col)
    
    nan_counts = {col: result.df[col].isna().sum() for col in factor_cols}
    logging.debug(f"[{name}] Trend factor NaN counts: {nan_counts}")
    
    return Frame3D(result.df[factor_cols].copy())
