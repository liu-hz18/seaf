"""
交互特征因子 — 16 个因子，基于现有量价特征的乘积/比值/差分。
捕捉因子间的非线性交互关系，与线性加权因子组合正交。
"""

from __future__ import annotations

import logging

import numpy as np

from qpipe.frame3d import Frame3D


def compute_interaction_factors(name: str, idx: int, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个交互类因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    f3d.df['volume']
    turnover = f3d.df['turnover']
    f3d.df['market_cap']

    ret1 = f3d.ts_pct_change('close', 1).df['close']
    ret5 = f3d.ts_pct_change('close', 5).df['close']
    ret20 = f3d.ts_pct_change('close', 20).df['close']

    # ---- 1-2: 收益率 × 成交量排名（量价共振） ----
    vol_rank = f3d.cs_rank('volume').df['volume']
    result = result.add_column('factor_inter_ret_vol_1d', ret1 * vol_rank)
    result = result.add_column('factor_inter_ret_vol_5d', ret5 * vol_rank)

    # ---- 3-4: 收益率 × 换手率（活跃度加权利率） ----
    to_rank = f3d.cs_rank('turnover').df['turnover']
    result = result.add_column('factor_inter_ret_turnover_1d', ret1 * to_rank)
    result = result.add_column('factor_inter_ret_turnover_20d', ret20 * to_rank)

    # ---- 5-6: 绝对收益 × 市值（大票动量效应） ----
    mcap_rank = f3d.cs_rank('market_cap').df['market_cap']
    result = result.add_column('factor_inter_ret_mcap_5d', ret5 * mcap_rank)
    result = result.add_column('factor_inter_ret_mcap_20d', ret20 * mcap_rank)

    # ---- 7-8: 波动率 × 市值交互 ----
    ret_vol_20 = f3d.copy().add_column('_r', ret1).ts_rolling('_r', 20, 'std').df['_r']
    result = result.add_column('factor_inter_vol_mcap_20d', ret_vol_20 * mcap_rank)
    ret_vol_60 = f3d.copy().add_column('_r', ret1).ts_rolling('_r', 60, 'std').df['_r']
    result = result.add_column('factor_inter_vol_mcap_60d', ret_vol_60 * mcap_rank)

    # ---- 9-10: 日内振幅 × 成交量交互 ----
    hl_range = (f3d.df['high'] - f3d.df['low']) / close
    result = result.add_column('factor_inter_hl_vol_1d', hl_range * vol_rank)
    result = result.add_column('factor_inter_hl_vol_5d', hl_range * vol_rank * ret5.fillna(0))

    # ---- 11-12: 收益方向 × 振幅 —— 确认性交互 ----
    ret_sign = np.sign(ret5)
    result = result.add_column('factor_inter_direction_range_5d', ret_sign * hl_range)

    ret_sign_20 = np.sign(ret20)
    result = result.add_column('factor_inter_direction_range_20d', ret_sign_20 * hl_range)

    # ---- 13-14: 换手率变化 × 收益 — 流动性驱动效应检测 ----
    to_chg = f3d.copy().add_column('_to', turnover).ts_pct_change('_to', 5).df['_to']
    result = result.add_column('factor_inter_to_chg_ret_5d', to_chg * ret5)

    to_chg_20 = f3d.copy().add_column('_to', turnover).ts_pct_change('_to', 20).df['_to']
    result = result.add_column('factor_inter_to_chg_ret_20d', to_chg_20 * ret20)

    # ---- 15-16: 复合交互 + 规模交互 ----
    result = result.add_column(
        'factor_inter_mom_vol_conflict', ret20 * ret_vol_20
    )  # 动量×波动：趋势确认或反转信号

    # 复合
    comp = (
        result.df['factor_inter_ret_vol_5d']
        + result.df['factor_inter_ret_mcap_20d']
        + result.df['factor_inter_direction_range_5d']
        + result.df['factor_inter_to_chg_ret_5d']
    ) / 4
    result = result.add_column('factor_inter_composite', comp)

    # 截面标准化
    factor_cols = [c for c in result.df.columns if c.startswith('factor_inter_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f'[{idx}] Factor NaN: { {c: result.df[c].isna().sum() for c in factor_cols} }')
    return Frame3D(result.df[factor_cols].copy())
