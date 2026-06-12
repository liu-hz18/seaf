"""
价值因子 — 16 个因子，基于价格/规模比值、历史偏离、市值中性化等代理指标。
优化：使用直接 pandas groupby rolling 减少 f3d.copy() 深拷贝开销；
cs_neutralize 保留不变。
"""

from __future__ import annotations

import logging

import numpy as np

from qpipe.frame3d import Frame3D


def compute_value_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个价值因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    mcap = f3d.df['market_cap']
    turnover = f3d.df['turnover']
    df = result.df

    def _roll(src, dst, window, agg):
        df[dst] = (
            df.groupby('code')[src].rolling(window, min_periods=max(1, window // 2)).agg(agg).reset_index(level=0, drop=True)
        )

    # ---- 1-3: 基础价值 ----
    df['factor_val_inv_price'] = 1.0 / close
    with np.errstate(divide='ignore'):
        df['factor_val_log_mcap'] = -np.where(mcap > 0, np.log(mcap), np.nan)
    df['factor_val_mcap_to_price'] = mcap / close

    # ---- 4-6: 价格相对 MA 偏离 ----
    for p in [20, 60, 120]:
        df[f'_ma{p}'] = close
        _roll(f'_ma{p}', f'_ma{p}', p, 'mean')
        df[f'factor_val_dist_ma_{p}d'] = -(close / df[f'_ma{p}'].replace(0, np.nan) - 1)

    # ---- 7-8: 价格回撤 ----
    for p in [60, 120]:
        df[f'_max{p}'] = close
        _roll(f'_max{p}', f'_max{p}', p, 'max')
        df[f'factor_val_dd_{p}d'] = close / df[f'_max{p}'].replace(0, np.nan) - 1

    # ---- 9: 换手率倒数 ----
    df['factor_val_turnover_yield'] = 1.0 / turnover

    # ---- 10: Z-score 逆 ----
    df['factor_val_sharpe_inv'] = -f3d.ts_zscore('close', 60).df['close']

    # ---- 11-12: 复合价值得分 ----
    df['factor_val_composite_short'] = (
        df['factor_val_inv_price'] + df['factor_val_dist_ma_20d'] + df['factor_val_dd_60d']
    ) / 3
    df['factor_val_composite_long'] = (
        df['factor_val_dist_ma_60d'] + df['factor_val_dist_ma_120d'] + df['factor_val_dd_120d']
    ) / 3

    # ---- 13-14: 市值中性化（保留 cs_neutralize） ----
    df['_inv_p'] = df['factor_val_inv_price']
    df['_d60'] = df['factor_val_dist_ma_60d']
    result = Frame3D(df.copy())  # 同步 result 到当前 df
    neut1 = result.cs_neutralize('_inv_p', by=['market_cap'])
    df['factor_val_inv_price_neut'] = neut1.df['_inv_p']
    neut2 = result.cs_neutralize('_d60', by=['market_cap'])
    df['factor_val_dist_ma_60d_neut'] = neut2.df['_d60']

    # ---- 15: 便宜且低换手 ----
    to_z = f3d.cs_zscore('turnover').df['turnover']
    df['factor_val_cheap_low_turn'] = df['_d60'] - to_z

    # ---- 16: 价格在范围内位置 ----
    df['_min120'] = close
    df['_max120'] = close
    _roll('_min120', '_min120', 120, 'min')
    _roll('_max120', '_max120', 120, 'max')
    df['factor_val_price_to_range'] = -(
        (close - df['_min120']) / (df['_max120'] - df['_min120']).replace(0, np.nan)
    )

    # 同步 result
    result = Frame3D(df.copy())
    factor_cols = [c for c in df.columns if c.startswith('factor_val_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f'Value NaN: { {c: result.df[c].isna().sum() for c in factor_cols} }')
    return Frame3D(result.df[factor_cols].copy())
