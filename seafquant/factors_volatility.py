"""
波动率因子 — 16 个因子，基于已实现波动、下行波动、Parkinson、GK 估计等。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_volatility_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个波动率因子。"""
    result = f3d.copy()
    high = f3d.df['high']
    low = f3d.df['low']
    close = f3d.df['close']
    open_p = f3d.df['open']
    
    # 日收益率
    ret = f3d.ts_pct_change('close', 1).df['close']
    result = result.add_column('_ret', ret)
    
    periods = [5, 10, 20, 60]
    
    # 1-4: 已实现波动率
    for p in periods:
        rv = f3d.copy().add_column('_ret', ret)
        rv_val = rv.ts_rolling('_ret', p, 'std').df['_ret']
        result = result.add_column(f'factor_vol_realized_{p}d', rv_val)
    
    # 5-8: 下行波动率 (semivariance sqrt)
    neg_ret = ret.clip(upper=0)
    result = result.add_column('_neg_ret', neg_ret)
    for p in periods:
        ds = f3d.copy().add_column('_neg_ret', neg_ret)
        ds_val = ds.ts_rolling('_neg_ret', p, 'mean').df['_neg_ret']
        result = result.add_column(f'factor_vol_downside_{p}d', np.sqrt(np.abs(ds_val)))
    
    # 9-10: 波动率的波动率
    rv5 = result.df.get('factor_vol_realized_5d', rv_val)
    vol_of_vol_20 = f3d.copy().add_column('_rv5', rv5).ts_rolling('_rv5', 20, 'std').df['_rv5']
    result = result.add_column('factor_vol_of_vol_20d', vol_of_vol_20)
    
    rv20 = result.df.get('factor_vol_realized_20d', rv_val)
    vol_of_vol_60 = f3d.copy().add_column('_rv20', rv20).ts_rolling('_rv20', 60, 'std').df['_rv20']
    result = result.add_column('factor_vol_of_vol_60d', vol_of_vol_60)
    
    # 11-12: Parkinson 波动率估计
    log_hl = np.log(high / low)
    parkinson_factor = 1.0 / (4 * np.log(2))
    park_sq = parkinson_factor * log_hl ** 2
    result = result.add_column('_park_sq', park_sq)
    park5 = f3d.copy().add_column('_park_sq', park_sq).ts_rolling('_park_sq', 5, 'mean').df['_park_sq']
    result = result.add_column('factor_vol_parkinson_5d', np.sqrt(np.abs(park5)))
    park20 = f3d.copy().add_column('_park_sq', park_sq).ts_rolling('_park_sq', 20, 'mean').df['_park_sq']
    result = result.add_column('factor_vol_parkinson_20d', np.sqrt(np.abs(park20)))
    
    # 13: Garman-Klass 估计 (5d)
    log_co = np.log(close / open_p)
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    result = result.add_column('_gk', gk)
    gk5_mean = f3d.copy().add_column('_gk', gk).ts_rolling('_gk', 5, 'mean').df['_gk']
    result = result.add_column('factor_vol_gk_5d', np.sqrt(np.abs(gk5_mean)))
    
    # 14-15: 波动率趋势
    trend_20 = result.df['factor_vol_realized_20d'] / result.df['factor_vol_realized_60d'].replace(0, np.nan) - 1
    result = result.add_column('factor_vol_trend_20d', trend_20)
    
    rv120 = f3d.copy().add_column('_ret', ret).ts_rolling('_ret', 120, 'std').df['_ret']
    trend_60 = result.df['factor_vol_realized_60d'] / rv120.replace(0, np.nan) - 1
    result = result.add_column('factor_vol_trend_60d', trend_60)
    
    # 16: 高低价范围
    range_ratio = high / low - 1
    result = result.add_column('_range', range_ratio)
    range20 = f3d.copy().add_column('_range', range_ratio).ts_rolling('_range', 20, 'mean').df['_range']
    result = result.add_column('factor_vol_range_20d', range20)
    
    # 截面标准化
    factor_cols = [c for c in result.df.columns if c.startswith('factor_vol_')]
    for col in factor_cols:
        result = result.cs_zscore(col)
    
    nan_counts = {col: result.df[col].isna().sum() for col in factor_cols}
    logging.debug(f"[{name}] Volatility factor NaN counts: {nan_counts}")
    
    return Frame3D(result.df[factor_cols].copy())
