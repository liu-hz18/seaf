"""
质量因子 — 16 个因子，基于收益稳定性、Sharpe比、回撤等收益代理指标。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_quality_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个质量因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    high = f3d.df['high']
    low = f3d.df['low']
    
    ret = f3d.ts_pct_change('close', 1).df['close']
    result = result.add_column('_ret', ret)
    
    # 1-3: 收益稳定性 (1/std)
    for p in [20, 60, 120]:
        std = f3d.copy().add_column('_ret', ret).ts_rolling('_ret', p, 'std').df['_ret']
        stability = 1.0 / std.replace(0, np.nan)
        result = result.add_column(f'factor_qual_ret_stability_{p}d', stability)
    
    # 4-6: Sharpe-like
    for p in [20, 60, 120]:
        mean_r = f3d.copy().add_column('_ret', ret).ts_rolling('_ret', p, 'mean').df['_ret']
        std_r = f3d.copy().add_column('_ret', ret).ts_rolling('_ret', p, 'std').df['_ret']
        sharpe = mean_r / std_r.replace(0, np.nan)
        result = result.add_column(f'factor_qual_sharpe_{p}d', sharpe)
    
    # 7-8: 正收益天数占比
    pos = (ret > 0).astype(float)
    result = result.add_column('_pos', pos)
    pos20 = f3d.copy().add_column('_pos', pos).ts_rolling('_pos', 20, 'mean').df['_pos']
    result = result.add_column('factor_qual_pos_days_20d', pos20)
    pos60 = f3d.copy().add_column('_pos', pos).ts_rolling('_pos', 60, 'mean').df['_pos']
    result = result.add_column('factor_qual_pos_days_60d', pos60)
    
    # 9-10: 最大回撤代理（距高点距离，越高越好）
    for p in [60, 120]:
        max_p = f3d.copy().add_column('_c', close).ts_rolling('_c', p, 'max').df['_c']
        dd = close / max_p.replace(0, np.nan) - 1
        result = result.add_column(f'factor_qual_maxdd_{p}d', dd)
    
    # 11-12: 高低价差稳定性
    hl_range = high / low - 1
    result = result.add_column('_hl', hl_range)
    hl_std_20 = f3d.copy().add_column('_hl', hl_range).ts_rolling('_hl', 20, 'std').df['_hl']
    result = result.add_column('factor_qual_hl_stability_20d', 1.0 / hl_std_20.replace(0, np.nan))
    hl_std_60 = f3d.copy().add_column('_hl', hl_range).ts_rolling('_hl', 60, 'std').df['_hl']
    result = result.add_column('factor_qual_hl_stability_60d', 1.0 / hl_std_60.replace(0, np.nan))
    
    # 13-14: 收益率偏度
    skew60 = f3d.copy().add_column('_ret', ret).ts_rolling('_ret', 60, 'skew').df['_ret']
    result = result.add_column('factor_qual_skew_60d', skew60)
    skew120 = f3d.copy().add_column('_ret', ret).ts_rolling('_ret', 120, 'skew').df['_ret']
    result = result.add_column('factor_qual_skew_120d', skew120)
    
    # 15: Up/Down capture
    def _up_down_ratio(series):
        pos_ret = series[series > 0]
        neg_ret = series[series < 0]
        pos_mean = pos_ret.mean() if len(pos_ret) > 0 else 0
        neg_mean = abs(neg_ret.mean()) if len(neg_ret) > 0 else 1e-6
        return pos_mean / neg_mean if neg_mean > 0 else 0
    
    up_down = ret.groupby(f3d.df.index.get_level_values('name')).transform(
        lambda x: x.rolling(60, min_periods=20).apply(_up_down_ratio, raw=False)
    )
    result = result.add_column('factor_qual_up_down_60d', up_down)
    
    # 16: 质量复合
    sh20 = result.df['factor_qual_sharpe_20d']
    pos20_v = result.df['factor_qual_pos_days_20d']
    dd60_v = result.df['factor_qual_maxdd_60d']
    stab20 = result.df['factor_qual_ret_stability_20d']
    quality_comp = (sh20 + pos20_v + dd60_v + stab20) / 4
    result = result.add_column('factor_qual_composite', quality_comp)
    
    # 截面标准化
    factor_cols = [c for c in result.df.columns if c.startswith('factor_qual_')]
    for col in factor_cols:
        result = result.cs_zscore(col)
    
    nan_counts = {col: result.df[col].isna().sum() for col in factor_cols}
    logging.debug(f"[{name}] Quality factor NaN counts: {nan_counts}")
    
    return Frame3D(result.df[factor_cols].copy())
