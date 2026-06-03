"""
动量因子 — 16 个因子，基于多周期收益率和波动率调整收益率。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_momentum_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个动量因子，返回仅包含因子列的 Frame3D，已做截面标准化。"""
    periods = [1, 3, 5, 10, 20, 40, 60, 120]
    result_f3d = f3d.copy()
    
    for p in periods:
        # 1. 原始收益率
        ret_col = f'factor_mom_ret_{p}d'
        ret_f3d = f3d.ts_pct_change('close', p)
        result_f3d = result_f3d.add_column(ret_col, ret_f3d.df['close'])
        
        # 2. 波动率调整收益率 = 收益率 / 滚动标准差
        voladj_col = f'factor_mom_voladj_{p}d'
        # 计算每日收益率
        daily_ret = f3d.ts_pct_change('close', 1).df['close']
        vol_f3d = f3d.copy().add_column('_daily_ret', daily_ret)
        vol = vol_f3d.ts_rolling('_daily_ret', max(p, 5), 'std').df['_daily_ret']
        voladj = ret_f3d.df['close'] / vol.replace(0, np.nan)
        result_f3d = result_f3d.add_column(voladj_col, voladj)
    
    # 截面标准化所有因子列
    factor_cols = [c for c in result_f3d.df.columns if c.startswith('factor_mom_')]
    for col in factor_cols:
        result_f3d = result_f3d.cs_zscore(col)
    
    # NaN 统计
    nan_counts = {col: result_f3d.df[col].isna().sum() for col in factor_cols}
    logging.debug(f"[{name}] Momentum factor NaN counts: {nan_counts}")
    
    # 只返回因子列
    return Frame3D(result_f3d.df[factor_cols].copy())
