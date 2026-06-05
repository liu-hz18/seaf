"""
规模因子 — 16 个因子，基于市值变换、变化率和交互效应。
优化：使用直接 pandas groupby rolling 减少 f3d.copy() 深拷贝开销。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_size_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个规模因子。"""
    result = f3d.copy()
    mcap = f3d.df['market_cap']
    close = f3d.df['close']
    df = result.df

    def _roll(src, dst, window, agg):
        df[dst] = df.groupby('name')[src].rolling(window, min_periods=max(1, window // 2)).agg(agg).values

    # ---- 1-2: 对数市值、截面排名 ----
    df['factor_size_log_mcap'] = -np.log(mcap)
    df['factor_size_cs_rank'] = f3d.cs_rank('market_cap').df['market_cap']

    # ---- 3-5: 市值变化率 ----
    for p in [5, 20, 60]:
        df[f'factor_size_mcap_chg_{p}d'] = df.groupby('name')['market_cap'].pct_change(p)

    # ---- 6-7: 市值波动率 ----
    df['_mcap_ret'] = df.groupby('name')['market_cap'].pct_change(1)
    _roll('_mcap_ret', 'factor_size_mcap_vol_20d', 20, 'std')
    _roll('_mcap_ret', 'factor_size_mcap_vol_60d', 60, 'std')

    # ---- 8-9: 市值动量 ----
    df['factor_size_mcap_mom_5d'] = df.groupby('name')['market_cap'].pct_change(5)
    df['factor_size_mcap_mom_20d'] = df.groupby('name')['market_cap'].pct_change(20)

    # ---- 10-11: 非线性规模变换 ----
    df['factor_size_mcap_sqrt'] = -np.sqrt(mcap)
    df['factor_size_mcap_cube_root'] = -np.cbrt(mcap)

    # ---- 12: 对数股本 ----
    df['factor_size_price'] = np.log(mcap / close)

    # ---- 13: 市值五分位 ----
    df['factor_size_quintile'] = f3d.cs_rank('market_cap').df['market_cap']

    # ---- 14: 小市值且上涨 ----
    df['factor_size_small_and_rising'] = -np.log(mcap) * df['factor_size_mcap_mom_20d']

    # ---- 15: 规模中性化收益 ----
    df['_ret20'] = f3d.ts_pct_change('close', 20).df['close']
    df['_mcap_raw'] = mcap
    result = Frame3D(df.copy())
    neut = result.cs_neutralize('_ret20', by=['_mcap_raw'])
    df['factor_size_residual_ret'] = neut.df['_ret20']

    # ---- 16: 规模复合 ----
    df['factor_size_composite'] = (
        df['factor_size_log_mcap'] + df['factor_size_cs_rank'] +
        df['factor_size_mcap_vol_20d'] + df['factor_size_price']
    ) / 4

    result = Frame3D(df.copy())
    factor_cols = [c for c in df.columns if c.startswith('factor_size_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] Size NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
