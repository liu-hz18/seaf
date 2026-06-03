"""
流动性因子 — 16 个因子，基于换手率、成交量、Amihud 非流动性指标。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_liquidity_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个流动性因子。"""
    result = f3d.copy()
    turnover = f3d.df['turnover']
    volume = f3d.df['volume']
    close = f3d.df['close']
    
    periods = [5, 10, 20, 60]
    
    # 1-4: 日均换手率
    for p in periods:
        to_mean = f3d.copy().add_column('_to', turnover).ts_rolling('_to', p, 'mean').df['_to']
        result = result.add_column(f'factor_liq_turnover_{p}d', to_mean)
    
    # 5-8: 换手率变化
    for p in periods:
        to_mean = f3d.copy().add_column('_to', turnover).ts_rolling('_to', p, 'mean').df['_to']
        to_chg = turnover / to_mean.replace(0, np.nan) - 1
        result = result.add_column(f'factor_liq_turnover_chg_{p}d', to_chg)
    
    # 9-10: 成交量变化
    vol_mean_5 = f3d.copy().add_column('_vol', volume).ts_rolling('_vol', 5, 'mean').df['_vol']
    vol_chg_5 = volume / vol_mean_5.replace(0, np.nan) - 1
    result = result.add_column('factor_liq_volume_chg_5d', vol_chg_5)
    
    vol_mean_20 = f3d.copy().add_column('_vol', volume).ts_rolling('_vol', 20, 'mean').df['_vol']
    vol_chg_20 = volume / vol_mean_20.replace(0, np.nan) - 1
    result = result.add_column('factor_liq_volume_chg_20d', vol_chg_20)
    
    # 11-12: Amihud 非流动性 |ret| / volume
    ret = f3d.ts_pct_change('close', 1).df['close']
    amihud_raw = np.abs(ret) / volume.replace(0, np.nan)
    result = result.add_column('_amihud', amihud_raw)
    ami5 = f3d.copy().add_column('_amihud', amihud_raw).ts_rolling('_amihud', 5, 'mean').df['_amihud']
    result = result.add_column('factor_liq_amihud_5d', ami5)
    ami20 = f3d.copy().add_column('_amihud', amihud_raw).ts_rolling('_amihud', 20, 'mean').df['_amihud']
    result = result.add_column('factor_liq_amihud_20d', ami20)
    
    # 13-14: 成交额 (dollar volume)
    dollar_vol = close * volume
    result = result.add_column('factor_liq_dollar_vol', np.log(dollar_vol))
    dv_chg = f3d.copy().add_column('_dv', dollar_vol)
    dv_chg_val = dv_chg.ts_pct_change('_dv', 20).df['_dv']
    result = result.add_column('factor_liq_dollar_vol_chg', dv_chg_val)
    
    # 15: 换手率波动
    to_vol_20 = f3d.copy().add_column('_to', turnover).ts_rolling('_to', 20, 'std').df['_to']
    result = result.add_column('factor_liq_turnover_vol_20d', to_vol_20)
    
    # 16: 复合流动性（高Amihud+高换手波动 = 低流动性 → negated）
    composite = -ami20 - to_vol_20
    result = result.add_column('factor_liq_composite', composite)
    
    # 截面标准化
    factor_cols = [c for c in result.df.columns if c.startswith('factor_liq_')]
    for col in factor_cols:
        result = result.cs_zscore(col)
    
    nan_counts = {col: result.df[col].isna().sum() for col in factor_cols}
    logging.debug(f"[{name}] Liquidity factor NaN counts: {nan_counts}")
    
    return Frame3D(result.df[factor_cols].copy())
