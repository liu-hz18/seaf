"""
截面中性化因子 — 6 个因子：截面动量/市值中性化/回归残差/复合。
从 cross_section 拆分而来，包含 cs_neutralize（OLS）重操作。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_cross_section_neut_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 6 个截面中性化因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    df = result.df

    ret5 = f3d.ts_pct_change('close', 5).df['close']
    ret20 = f3d.ts_pct_change('close', 20).df['close']
    df['_ret5'] = ret5
    df['_ret20'] = ret20

    # ---- 1-2: 截面动量 ----
    df['factor_cs_momentum_5d'] = result.cs_zscore('_ret5').df['_ret5']
    df['factor_cs_momentum_20d'] = result.cs_zscore('_ret20').df['_ret20']

    # ---- 3: 截面 Z-score ----
    df['factor_cs_close_zscore'] = f3d.cs_zscore('close').df['close']

    # ---- 4: 市值中性化 ----
    result2 = Frame3D(df.copy())
    neut = result2.cs_neutralize('close', by=['market_cap'])
    df['factor_cs_close_neut_mcap'] = neut.cs_zscore('close').df['close']

    # ---- 5: 量价回归残差 ----
    neut2 = result2.cs_neutralize('_ret20', by=['volume'])
    result3 = Frame3D(neut2.df.copy())
    df['factor_cs_ret_neut_volume'] = result3.cs_zscore('_ret20').df['_ret20']

    # ---- 6: 复合（仅用本节点因子） ----
    df['factor_cs_composite'] = (
        df['factor_cs_momentum_20d'] + df['factor_cs_close_zscore'] +
        df['factor_cs_close_neut_mcap'] + df['factor_cs_ret_neut_volume']
    ) / 4

    result = Frame3D(df.copy())
    factor_cols = [c for c in df.columns if c.startswith('factor_cs_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] CS-Neut NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
