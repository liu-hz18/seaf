"""
规模因子 — 16 个因子，基于市值变换、变化率和交互效应。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_size_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个规模因子。"""
    result = f3d.copy()
    mcap = f3d.df['market_cap']
    close = f3d.df['close']
    
    # 1-2: 对数市值、截面排名
    result = result.add_column('factor_size_log_mcap', -np.log(mcap))
    result = result.add_column('factor_size_cs_rank', 
                                f3d.cs_rank('market_cap').df['market_cap'])
    
    # 3-5: 市值变化率
    for p in [5, 20, 60]:
        chg = f3d.copy().add_column('_mcap', mcap).ts_pct_change('_mcap', p).df['_mcap']
        result = result.add_column(f'factor_size_mcap_chg_{p}d', chg)
    
    # 6-7: 市值波动率
    mcap_ret = f3d.copy().add_column('_mcap', mcap).ts_pct_change('_mcap', 1).df['_mcap']
    result = result.add_column('_mcap_ret', mcap_ret)
    mcap_vol_20 = f3d.copy().add_column('_mcap_ret', mcap_ret).ts_rolling('_mcap_ret', 20, 'std').df['_mcap_ret']
    result = result.add_column('factor_size_mcap_vol_20d', mcap_vol_20)
    mcap_vol_60 = f3d.copy().add_column('_mcap_ret', mcap_ret).ts_rolling('_mcap_ret', 60, 'std').df['_mcap_ret']
    result = result.add_column('factor_size_mcap_vol_60d', mcap_vol_60)
    
    # 8-9: 市值动量
    mcap_mom_5 = mcap_ret  # 1d ret = 5d? Wait, mcap_chg_5d already covered.
    # Actually use pct_change directly
    mcap_ret_s = f3d.copy().add_column('_mcap', mcap).ts_pct_change('_mcap', 5).df['_mcap']
    result = result.add_column('factor_size_mcap_mom_5d', mcap_ret_s)
    mcap_ret_l = f3d.copy().add_column('_mcap', mcap).ts_pct_change('_mcap', 20).df['_mcap']
    result = result.add_column('factor_size_mcap_mom_20d', mcap_ret_l)
    
    # 10-11: 非线性规模变换
    result = result.add_column('factor_size_mcap_sqrt', -np.sqrt(mcap))
    result = result.add_column('factor_size_mcap_cube_root', -np.cbrt(mcap))
    
    # 12: 对数股本 (log shares outstanding proxy)
    shares = mcap / close
    result = result.add_column('factor_size_price', np.log(shares))
    
    # 13: 市值五分位
    result = result.add_column('factor_size_quintile', 
                                f3d.cs_rank('market_cap').df['market_cap'])
    
    # 14: 小市值且上涨
    small = -np.log(mcap)
    rising = f3d.copy().add_column('_mcap', mcap).ts_pct_change('_mcap', 20).df['_mcap']
    result = result.add_column('factor_size_small_and_rising', small * rising)
    
    # 15: 规模中性化收益
    ret20 = f3d.ts_pct_change('close', 20).df['close']
    result = result.add_column('_ret20', ret20)
    result = result.add_column('_mcap_raw', mcap)
    neut_f3d = Frame3D(result.df[['_ret20', '_mcap_raw']].copy())
    neut_f3d = neut_f3d.cs_neutralize('_ret20', by=['_mcap_raw'])
    result = result.add_column('factor_size_residual_ret', neut_f3d.df['_ret20'])
    
    # 16: 规模复合
    size_comp = (result.df['factor_size_log_mcap'] + result.df['factor_size_cs_rank'] + 
                 result.df['factor_size_mcap_vol_20d'] + result.df['factor_size_price']) / 4
    result = result.add_column('factor_size_composite', size_comp)
    
    # 截面标准化
    factor_cols = [c for c in result.df.columns if c.startswith('factor_size_')]
    for col in factor_cols:
        result = result.cs_zscore(col)
    
    nan_counts = {col: result.df[col].isna().sum() for col in factor_cols}
    logging.debug(f"[{name}] Size factor NaN counts: {nan_counts}")
    
    return Frame3D(result.df[factor_cols].copy())
