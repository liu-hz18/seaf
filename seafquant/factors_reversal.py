"""
反转因子 — 16 个因子，基于短期反转效应和极端值检测。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_reversal_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个反转因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    open_p = f3d.df['open']
    turnover = f3d.df['turnover']
    volume = f3d.df['volume']
    high = f3d.df['high']
    low = f3d.df['low']
    
    # 辅助：日收益率
    ret1 = f3d.ts_pct_change('close', 1).df['close']
    result = result.add_column('_ret1', ret1)
    result = result.add_column('_ret3', f3d.ts_pct_change('close', 3).df['close'])
    result = result.add_column('_ret5', f3d.ts_pct_change('close', 5).df['close'])
    
    # 成交量截面排名
    vol_rank = f3d.cs_rank('volume').df['volume']
    result = result.add_column('_vol_rank', vol_rank)
    
    # 1-3: 短期反转（负动量）
    result = result.add_column('factor_rev_ret_1d', -result.df['_ret1'])
    result = result.add_column('factor_rev_ret_3d', -result.df['_ret3'])
    result = result.add_column('factor_rev_ret_5d', -result.df['_ret5'])
    
    # 4-5: 隔夜反转
    delayed_close = f3d.ts_delay('close', 1).df['close']
    overnight = open_p / delayed_close - 1
    result = result.add_column('factor_rev_overnight_1d', -overnight)
    ovn5 = f3d.copy().add_column('_ovn', overnight)
    ovn5_mean = ovn5.ts_rolling('_ovn', 5, 'mean').df['_ovn']
    result = result.add_column('factor_rev_overnight_5d', -ovn5_mean)
    
    # 6-7: 日内反转
    intraday = close / open_p - 1
    result = result.add_column('factor_rev_intraday_1d', -intraday)
    intra_f3d = f3d.copy().add_column('_intra', intraday)
    intra5 = intra_f3d.ts_rolling('_intra', 5, 'mean').df['_intra']
    result = result.add_column('factor_rev_intraday_5d', -intra5)
    
    # 8-9: 成交量反转
    result = result.add_column('factor_rev_volrev_1d', -result.df['_ret1'] * vol_rank)
    result = result.add_column('factor_rev_volrev_3d', -result.df['_ret3'] * vol_rank)
    
    # 10-12: Z-score 反转
    z20 = f3d.ts_zscore('close', 20).df['close']
    z20_delayed = f3d.copy().add_column('_z20', z20).ts_delay('_z20', 1).df['_z20']
    result = result.add_column('factor_rev_zscore_1d', -z20_delayed)
    
    # 3d ret zscore
    ret3_z = f3d.copy().add_column('_ret3', result.df['_ret3'])
    ret3_z = ret3_z.ts_zscore('_ret3', 20).df['_ret3']
    result = result.add_column('factor_rev_zscore_3d', -ret3_z)
    
    ret5_z = f3d.copy().add_column('_ret5', result.df['_ret5'])
    ret5_z = ret5_z.ts_zscore('_ret5', 20).df['_ret5']
    result = result.add_column('factor_rev_zscore_5d', -ret5_z)
    
    # 13: 缺口反转
    result = result.add_column('factor_rev_gap_1d', -overnight * z20.fillna(0))
    
    # 14: 高低价范围反转
    hl_range = (high - low) / close
    result = result.add_column('factor_rev_range_rev', -hl_range)
    
    # 15: 换手率反转
    to_rank = f3d.cs_rank('turnover').df['turnover']
    result = result.add_column('factor_rev_turnover_1d', -ret1 * to_rank)
    
    # 16: 复合反转
    composite = (-ret1 * 0.25 - overnight * 0.25 - intraday * 0.25 - ret1 * vol_rank * 0.25)
    result = result.add_column('factor_rev_composite', composite)
    
    # 截面标准化
    factor_cols = [c for c in result.df.columns if c.startswith('factor_rev_')]
    for col in factor_cols:
        result = result.cs_zscore(col)
    
    nan_counts = {col: result.df[col].isna().sum() for col in factor_cols}
    logging.debug(f"[{name}] Reversal factor NaN counts: {nan_counts}")
    
    return Frame3D(result.df[factor_cols].copy())
