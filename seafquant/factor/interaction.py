"""
交互特征因子 — 16 个因子。优化 v3：ts_rolling/ts_pct_change → 2D-array。
"""

from __future__ import annotations

import numpy as np

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import rolling_std_2d


def compute_interaction_factors(name: str, idx: int, f3d: Frame3D, context: dict) -> Frame3D:
    """计算 16 个交互因子 — 向量化 v3。"""
    result = f3d.copy()
    close = f3d.df['close']
    turnover = f3d.df['turnover']
    df = result.df

    # ── 提取 2D ──
    close_2d = close.unstack(level='code').values
    to_2d = turnover.unstack(level='code').values

    # ret1 = (C[t] - C[t-1]) / C[t-1]
    shifted1 = np.roll(close_2d, 1, axis=0); shifted1[0] = np.nan
    ret1_2d = (close_2d - shifted1) / np.where(shifted1 != 0, shifted1, np.nan)
    df['_ret1'] = ret1_2d.ravel()

    # ret5
    shifted5 = np.roll(close_2d, 5, axis=0); shifted5[:5] = np.nan
    ret5_2d = (close_2d - shifted5) / np.where(shifted5 != 0, shifted5, np.nan)
    df['_ret5'] = ret5_2d.ravel()

    # ret20
    shifted20 = np.roll(close_2d, 20, axis=0); shifted20[:20] = np.nan
    ret20_2d = (close_2d - shifted20) / np.where(shifted20 != 0, shifted20, np.nan)
    df['_ret20'] = ret20_2d.ravel()

    # ── 截面排名 ──
    vol_rank = f3d.cs_rank('volume').df['volume']
    to_rank = f3d.cs_rank('turnover').df['turnover']
    mcap_rank = f3d.cs_rank('market_cap').df['market_cap']

    # ---- 1-2: ret × vol_rank (量价共振) ----
    # f = r_t · rank(V)
    result = result.add_column('factor_inter_ret_vol_1d', df['_ret1'] * vol_rank)
    result = result.add_column('factor_inter_ret_vol_5d', df['_ret5'] * vol_rank)

    # ---- 3-4: ret × turnover_rank (活跃度加权利率) ----
    # f = r_t · rank(TO)
    result = result.add_column('factor_inter_ret_turnover_1d', df['_ret1'] * to_rank)
    result = result.add_column('factor_inter_ret_turnover_20d', df['_ret20'] * to_rank)

    # ---- 5-6: ret × mcap_rank (大票动量效应) ----
    # f = r_t · rank(MC)
    result = result.add_column('factor_inter_ret_mcap_5d', df['_ret5'] * mcap_rank)
    result = result.add_column('factor_inter_ret_mcap_20d', df['_ret20'] * mcap_rank)

    # ---- 7-8: σ(ret) × mcap_rank (波动率×市值) ----
    # f = σ_20(ret) · rank(MC),  f = σ_60(ret) · rank(MC)
    vol_stds = rolling_std_2d(ret1_2d, [20, 60])
    result = result.add_column('factor_inter_vol_mcap_20d',
                               vol_stds[20].ravel() * mcap_rank)
    result = result.add_column('factor_inter_vol_mcap_60d',
                               vol_stds[60].ravel() * mcap_rank)

    # ---- 9-10: HL_range × vol_rank (日内振幅×量) ----
    # f = (H-L)/C · rank(V)
    hl_range = (f3d.df['high'] - f3d.df['low']) / close.replace(0, np.nan)
    result = result.add_column('factor_inter_hl_vol_1d', hl_range * vol_rank)
    # f = (H-L)/C · rank(V) · r5
    result = result.add_column('factor_inter_hl_vol_5d',
                               hl_range * vol_rank * df['_ret5'].fillna(0))

    # ---- 11-12: sign(ret) × HL_range (方向×振幅确认) ----
    # f = sign(r5) · (H-L)/C
    result = result.add_column('factor_inter_direction_range_5d',
                               np.sign(df['_ret5']) * hl_range)
    # f = sign(r20) · (H-L)/C
    result = result.add_column('factor_inter_direction_range_20d',
                               np.sign(df['_ret20']) * hl_range)

    # ---- 13-14: ΔTO × ret (流动性驱动效应) ----
    # f = (TO[t]/TO[t-5]-1) · r5
    to_shift5 = np.roll(to_2d, 5, axis=0); to_shift5[:5] = np.nan
    to_chg5 = (to_2d - to_shift5) / np.where(to_shift5 != 0, to_shift5, np.nan)
    result = result.add_column('factor_inter_to_chg_ret_5d',
                               to_chg5.ravel() * df['_ret5'])

    # f = (TO[t]/TO[t-20]-1) · r20
    to_shift20 = np.roll(to_2d, 20, axis=0); to_shift20[:20] = np.nan
    to_chg20 = (to_2d - to_shift20) / np.where(to_shift20 != 0, to_shift20, np.nan)
    result = result.add_column('factor_inter_to_chg_ret_20d',
                               to_chg20.ravel() * df['_ret20'])

    # ---- 15: 动量×波动冲突 ----
    # f = r20 · σ_20(ret)  (趋势确认 vs 反转信号)
    result = result.add_column('factor_inter_mom_vol_conflict',
                               df['_ret20'] * vol_stds[20].ravel())

    # ---- 16: 复合 ----
    # f = avg( ret_vol_5d, ret_mcap_20d, direction_range_5d, to_chg_ret_5d )
    comp = (result.df['factor_inter_ret_vol_5d']
            + result.df['factor_inter_ret_mcap_20d']
            + result.df['factor_inter_direction_range_5d']
            + result.df['factor_inter_to_chg_ret_5d']) / 4
    result = result.add_column('factor_inter_composite', comp)

    # 截面标准化
    factor_cols = [c for c in result.df.columns if c.startswith('factor_inter_')]
    return Frame3D(result.df[factor_cols].copy())
