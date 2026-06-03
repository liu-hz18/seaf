"""
价值因子 — 16 个因子，基于价格/规模比值、历史偏离、市值中性化等代理指标。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_value_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个价值因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    mcap = f3d.df['market_cap']
    turnover = f3d.df['turnover']
    
    # 1-3: 价格倒数、对数市值、市值/价格比
    result = result.add_column('factor_val_inv_price', 1.0 / close)
    result = result.add_column('factor_val_log_mcap', -np.log(mcap))
    result = result.add_column('factor_val_mcap_to_price', mcap / close)
    
    # 4-6: 价格相对历史均值偏离（距离越大 = 越便宜 → 反转）
    ma_periods = [20, 60, 120]
    for p in ma_periods:
        ma = f3d.copy().add_column('_close', close).ts_rolling('_close', p, 'mean').df['_close']
        dist = close / ma.replace(0, np.nan) - 1
        result = result.add_column(f'factor_val_dist_ma_{p}d', -dist)
    
    # 7-8: 价格回撤（距高点距离）
    for p in [60, 120]:
        max_p = f3d.copy().add_column('_close', close).ts_rolling('_close', p, 'max').df['_close']
        dd = close / max_p.replace(0, np.nan) - 1
        result = result.add_column(f'factor_val_dd_{p}d', dd)
    
    # 9: 换手率倒数（低换手 = 持有型 = 价值）
    result = result.add_column('factor_val_turnover_yield', 1.0 / turnover)
    
    # 10: Z-score 逆（便宜 = Z低）
    z60 = f3d.ts_zscore('close', 60).df['close']
    result = result.add_column('factor_val_sharpe_inv', -z60)
    
    # 11-12: 复合价值得分
    inv_p = 1.0 / close
    ma20 = f3d.copy().add_column('_c', close).ts_rolling('_c', 20, 'mean').df['_c']
    d20 = -(close / ma20 - 1)
    max60 = f3d.copy().add_column('_c', close).ts_rolling('_c', 60, 'max').df['_c']
    dd60 = close / max60 - 1
    comp1 = (inv_p + d20 + dd60) / 3
    result = result.add_column('factor_val_composite_short', comp1)
    
    ma60 = f3d.copy().add_column('_c', close).ts_rolling('_c', 60, 'mean').df['_c']
    d60 = -(close / ma60 - 1)
    max120 = f3d.copy().add_column('_c', close).ts_rolling('_c', 120, 'max').df['_c']
    dd120 = close / max120 - 1
    comp2 = (d60 + result.df.get(f'factor_val_dist_ma_120d', d60) + dd120) / 3
    result = result.add_column('factor_val_composite_long', comp2)
    
    # 13-14: 市值中性化价值
    inv_p_s = result.df['factor_val_inv_price']
    result = result.add_column('_inv_p', inv_p_s)
    result = result.add_column('_mcap', mcap)
    neut_f3d = Frame3D(result.df[['_inv_p', '_mcap']].copy())
    neut_f3d = neut_f3d.cs_neutralize('_inv_p', by=['_mcap'])
    result = result.add_column('factor_val_inv_price_neut', neut_f3d.df['_inv_p'])
    
    d60_col = result.df.get(f'factor_val_dist_ma_60d', d60)
    result = result.add_column('_d60', d60_col)
    neut2_f3d = Frame3D(result.df[['_d60', '_mcap']].copy())
    neut2_f3d = neut2_f3d.cs_neutralize('_d60', by=['_mcap'])
    result = result.add_column('factor_val_dist_ma_60d_neut', neut2_f3d.df['_d60'])
    
    # 15: 便宜且低换手
    to_z = f3d.cs_zscore('turnover').df['turnover']
    result = result.add_column('factor_val_cheap_low_turn', d60_col - to_z)
    
    # 16: 价格在 120d 范围内的位置（逆向 = 便宜）
    min120 = f3d.copy().add_column('_c', close).ts_rolling('_c', 120, 'min').df['_c']
    max120v = f3d.copy().add_column('_c', close).ts_rolling('_c', 120, 'max').df['_c']
    range_pos = (close - min120) / (max120v - min120).replace(0, np.nan)
    result = result.add_column('factor_val_price_to_range', -range_pos)
    
    # 截面标准化
    factor_cols = [c for c in result.df.columns if c.startswith('factor_val_')]
    for col in factor_cols:
        result = result.cs_zscore(col)
    
    nan_counts = {col: result.df[col].isna().sum() for col in factor_cols}
    logging.debug(f"[{name}] Value factor NaN counts: {nan_counts}")
    
    return Frame3D(result.df[factor_cols].copy())
