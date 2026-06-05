"""
截面特征因子 — 10 个因子：截面排名/排名变化/排名Z-score/收益率分位数。
动量/中性化/复合已拆分至 cross_section_neut 节点。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_cross_section_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 10 个截面排名因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    df = result.df

    ret1 = f3d.ts_pct_change('close', 1).df['close']
    ret20 = f3d.ts_pct_change('close', 20).df['close']

    # ---- 1-2: 截面排名 ----
    df['factor_cs_rank_close'] = f3d.cs_rank('close').df['close']
    df['factor_cs_rank_volume'] = f3d.cs_rank('volume').df['volume']

    # ---- 3-5: 排名变化 ----
    df['_rk_now'] = df['factor_cs_rank_close']
    for period in [5, 20, 60]:
        df[f'_rk_d{period}'] = df.groupby('name')['_rk_now'].shift(period)
        df[f'factor_cs_rank_delta_{period}d'] = df['_rk_now'] - df[f'_rk_d{period}']

    # ---- 6-8: 截面偏离度（时序Z-score） ----
    for period in [5, 20, 60]:
        df[f'factor_cs_rank_zscore_{period}d'] = f3d.ts_zscore('close', period).df['close']

    # ---- 9-10: 收益率截面分位数 ----
    df['_ret1'] = ret1
    df['factor_cs_ret_rank_1d'] = result.cs_rank('_ret1').df['_ret1']
    df['_ret20'] = ret20
    df['factor_cs_ret_rank_20d'] = result.cs_rank('_ret20').df['_ret20']

    result = Frame3D(df.copy())
    factor_cols = [c for c in df.columns if c.startswith('factor_cs_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] CrossSection NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
