"""
价值因子 — 16 个因子。优化 v2：_roll 替换为 2D-array + ravel()。
"""

from __future__ import annotations

import numpy as np

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import rolling_max_2d, rolling_mean_2d


def compute_value_factors(name: str, idx: int, f3d: Frame3D, context: dict) -> Frame3D:
    """计算 16 个价值因子 — 向量化 v2。"""
    result = f3d.copy()
    close = f3d.df['close']
    mcap = f3d.df['market_cap']
    turnover = f3d.df['turnover']
    df = result.df

    # ── 提取 2D-array ──
    close_2d = close.unstack(level='code').values

    # ---- 1-3: 基础价值 ----
    df['factor_val_inv_price'] = np.where(close != 0, 1.0 / close, np.nan)
    df['factor_val_log_mcap'] = -np.log(np.where(mcap > 0, mcap, np.nan))
    df['factor_val_mcap_to_price'] = np.where(close != 0, mcap / close, np.nan)

    # ---- 4-6: 价格相对 MA 偏离 ----
    mas = rolling_mean_2d(close_2d, [20, 60, 120])
    for p in [20, 60, 120]:
        df[f'_ma{p}'] = mas[p].ravel()
        df[f'factor_val_dist_ma_{p}d'] = -(close / df[f'_ma{p}'].replace(0, np.nan) - 1)

    # ---- 7-8: 价格回撤 ----
    close_maxs = rolling_max_2d(close_2d, [60, 120])
    for p in [60, 120]:
        df[f'_max{p}'] = close_maxs[p].ravel()
        df[f'factor_val_dd_{p}d'] = close / df[f'_max{p}'].replace(0, np.nan) - 1

    # ---- 9: 换手率倒数 ----
    df['factor_val_turnover_yield'] = np.where(turnover != 0, 1.0 / turnover, np.nan)

    # ---- 10: Z-score 逆 ----
    df['factor_val_sharpe_inv'] = -f3d.ts_zscore('close', 60).df['close']

    # ---- 11-12: 复合价值得分 ----
    df['factor_val_composite_short'] = (
        df['factor_val_inv_price'] + df['factor_val_dist_ma_20d'] + df['factor_val_dd_60d']
    ) / 3
    df['factor_val_composite_long'] = (
        df['factor_val_dist_ma_60d'] + df['factor_val_dist_ma_120d'] + df['factor_val_dd_120d']
    ) / 3

    # ---- 13-14: 市值中性化 ----
    df['_inv_p'] = df['factor_val_inv_price']
    df['_d60'] = df['factor_val_dist_ma_60d']
    result = Frame3D(df.copy())
    neut1 = result.cs_neutralize('_inv_p', by=['market_cap'])
    df['factor_val_inv_price_neut'] = neut1.df['_inv_p']
    neut2 = result.cs_neutralize('_d60', by=['market_cap'])
    df['factor_val_dist_ma_60d_neut'] = neut2.df['_d60']

    # ---- 15: 便宜且低换手 ----
    to_z = f3d.cs_zscore('turnover').df['turnover']
    df['factor_val_cheap_low_turn'] = df['_d60'] - to_z

    # ---- 16: 价格在范围内位置 ----
    # Actually need min, not max of negative
    from seafquant.factor._perf import rolling_min_2d
    mins_120 = rolling_min_2d(close_2d, [120])
    df['_min120'] = mins_120[120].ravel()
    df['_max120'] = close_maxs[120].ravel()
    df['factor_val_price_to_range'] = -(
        (close - df['_min120']) / (df['_max120'] - df['_min120']).replace(0, np.nan)
    )

    result = Frame3D(df.copy())
    factor_cols = [c for c in df.columns if c.startswith('factor_val_')]
    return Frame3D(result.df[factor_cols].copy())
