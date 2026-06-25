"""
流动性/规模因子 — 32 个因子。优化 v2：_roll 替换为 2D-array + ravel()。
"""

from __future__ import annotations

import numpy as np

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import rolling_mean_2d, rolling_std_2d


def compute_liquidity_factors(name: str, idx: int, f3d: Frame3D, context) -> Frame3D:
    """计算 32 个流动性+规模因子 — 向量化 v2。"""
    result = f3d.copy()
    turnover, volume, close, mcap = (
        f3d.df['turnover'], f3d.df['volume'],
        f3d.df['close'], f3d.df['market_cap'],
    )
    df = result.df

    # ── 提取 2D-array ──
    to_2d = turnover.unstack(level='code').values
    vol_2d = volume.unstack(level='code').values
    close_2d = close.unstack(level='code').values
    mcap_2d = mcap.unstack(level='code').values

    # ===== 流动性：16 cols =====
    to_means = rolling_mean_2d(to_2d, [5, 10, 20, 60])
    for p in [5, 10, 20, 60]:
        df[f'factor_liq_turnover_{p}d'] = to_means[p].ravel()
        df[f'factor_liq_turnover_chg_{p}d'] = (
            turnover / df[f'factor_liq_turnover_{p}d'].replace(0, np.nan) - 1
        )

    vol_means_5 = rolling_mean_2d(vol_2d, [5])[5]
    vol_means_20 = rolling_mean_2d(vol_2d, [20])[20]
    df['_vol5_mean'] = vol_means_5.ravel()
    df['_vol20_mean'] = vol_means_20.ravel()
    df['factor_liq_volume_chg_5d'] = volume / df['_vol5_mean'].replace(0, np.nan) - 1
    df['factor_liq_volume_chg_20d'] = volume / df['_vol20_mean'].replace(0, np.nan) - 1

    ret_2d = ((close_2d[1:] - close_2d[:-1])
              / np.where(close_2d[:-1] != 0, close_2d[:-1], np.nan))
    ret_2d = np.vstack([np.full((1, vol_2d.shape[1]), np.nan), ret_2d])
    amihud_2d = np.abs(ret_2d) / np.where(vol_2d != 0, vol_2d, np.nan)
    amihud_means = rolling_mean_2d(amihud_2d, [5, 20])
    df['factor_liq_amihud_5d'] = amihud_means[5].ravel()
    df['factor_liq_amihud_20d'] = amihud_means[20].ravel()

    dollar_vol_2d = close_2d * vol_2d
    with np.errstate(divide='ignore'):
        df['factor_liq_dollar_vol'] = np.where(dollar_vol_2d > 0,
                                               np.log(dollar_vol_2d), np.nan).ravel()
    df['_dv'] = dollar_vol_2d.ravel()
    df['factor_liq_dollar_vol_chg'] = df.groupby('code')['_dv'].pct_change(20, fill_method=None)

    to_vol = rolling_std_2d(to_2d, [20])[20]
    df['factor_liq_turnover_vol_20d'] = to_vol.ravel()
    df['factor_liq_composite'] = (-df['factor_liq_amihud_20d']
                                  - df['factor_liq_turnover_vol_20d'])

    # ===== 规模：16 cols =====
    df['factor_size_log_mcap'] = -np.where(mcap_2d > 0, np.log(mcap_2d), np.nan).ravel()
    df['factor_size_cs_rank'] = f3d.cs_rank('market_cap').df['market_cap']

    for p in [5, 20, 60]:
        df[f'factor_size_mcap_chg_{p}d'] = df.groupby('code')['market_cap'].pct_change(p, fill_method=None)

    mcap_ret_2d = np.empty_like(mcap_2d)
    mcap_ret_2d[0] = np.nan
    mcap_ret_2d[1:] = (mcap_2d[1:] - mcap_2d[:-1]) / np.where(mcap_2d[:-1] != 0, mcap_2d[:-1], np.nan)
    df['_mcap_ret'] = mcap_ret_2d.ravel()
    mcap_vols = rolling_std_2d(mcap_ret_2d, [20, 60])
    df['factor_size_mcap_vol_20d'] = mcap_vols[20].ravel()
    df['factor_size_mcap_vol_60d'] = mcap_vols[60].ravel()

    df['factor_size_mcap_mom_5d'] = df.groupby('code')['market_cap'].pct_change(5, fill_method=None)
    df['factor_size_mcap_mom_20d'] = df.groupby('code')['market_cap'].pct_change(20, fill_method=None)
    df['factor_size_mcap_sqrt'] = -np.sqrt(mcap)
    df['factor_size_mcap_cube_root'] = -np.cbrt(mcap)

    ratio_2d = mcap_2d / np.where(close_2d != 0, close_2d, np.nan)
    df['factor_size_price'] = np.where(ratio_2d > 0, np.log(ratio_2d), np.nan).ravel()
    df['factor_size_quintile'] = f3d.cs_rank('market_cap').df['market_cap']
    df['factor_size_small_and_rising'] = (-np.where(mcap_2d > 0, np.log(mcap_2d), np.nan).ravel()
                                          * df['factor_size_mcap_mom_20d'])

    ret20_2d = np.roll(close_2d, 20, axis=0)
    ret20_2d[:20] = np.nan
    ret20_2d = (close_2d - ret20_2d) / np.where(ret20_2d != 0, ret20_2d, np.nan)
    df['_ret20'] = ret20_2d.ravel()
    df['_mcap_raw'] = mcap

    df['factor_size_composite'] = (
        df['factor_size_log_mcap'] + df['factor_size_cs_rank']
        + df['factor_size_mcap_vol_20d'] + df['factor_size_price']
    ) / 4

    result = Frame3D(df.copy())
    neut = result.cs_neutralize('_ret20', by=['_mcap_raw'])
    df['factor_size_residual_ret'] = neut.df['_ret20']
    result = Frame3D(df.copy())
    factor_cols = [c for c in df.columns if c.startswith(('factor_liq_', 'factor_size_'))]
    return Frame3D(result.df[factor_cols].copy())
