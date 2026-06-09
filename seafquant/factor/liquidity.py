"""
流动性/规模因子（合并） — 32 个因子。
流动性：换手率均值 ×4, 换手率变化 ×4, 成交量变化 ×2, Amihud ×2,
        成交额 ×2, 换手率波动 ×1, 复合 ×1。
规模：对数市值 ×2, 市值变化率 ×3, 市值波动率 ×2, 市值动量 ×2,
      非线性变换 ×2, 对数股本 ×1, 五分位 ×1, 小市值交互 ×1,
      规模中性化收益 ×1, 复合 ×1。
"""

from __future__ import annotations

import numpy as np

from qpipe.frame3d import Frame3D


def compute_liquidity_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 32 个流动性+规模因子。"""
    result = f3d.copy()
    turnover, volume, close, mcap = (
        f3d.df['turnover'],
        f3d.df['volume'],
        f3d.df['close'],
        f3d.df['market_cap'],
    )
    df = result.df

    def _roll(src, dst, window, agg):
        df[dst] = (
            df.groupby('name')[src].rolling(window, min_periods=max(1, window // 2)).agg(agg).reset_index(level=0, drop=True)
        )

    # ===== 流动性：16 cols (prefix factor_liq_) =====
    for p in [5, 10, 20, 60]:
        df[f'_to{p}'] = turnover
        _roll(f'_to{p}', f'factor_liq_turnover_{p}d', p, 'mean')
    for p in [5, 10, 20, 60]:
        df[f'factor_liq_turnover_chg_{p}d'] = (
            turnover / df[f'factor_liq_turnover_{p}d'].replace(0, np.nan) - 1
        )

    df['_vol5'] = volume
    _roll('_vol5', '_vol5_mean', 5, 'mean')
    df['factor_liq_volume_chg_5d'] = volume / df['_vol5_mean'].replace(0, np.nan) - 1
    df['_vol20'] = volume
    _roll('_vol20', '_vol20_mean', 20, 'mean')
    df['factor_liq_volume_chg_20d'] = volume / df['_vol20_mean'].replace(0, np.nan) - 1

    ret = f3d.ts_pct_change('close', 1).df['close']
    df['_amihud'] = np.abs(ret) / volume.replace(0, np.nan)
    _roll('_amihud', 'factor_liq_amihud_5d', 5, 'mean')
    _roll('_amihud', 'factor_liq_amihud_20d', 20, 'mean')

    dollar_vol = close * volume
    with np.errstate(divide='ignore'):
        df['factor_liq_dollar_vol'] = np.where(dollar_vol > 0, np.log(dollar_vol), np.nan)
    df['_dv'] = dollar_vol
    df['factor_liq_dollar_vol_chg'] = df.groupby('name')['_dv'].pct_change(20)
    df['_to_vol'] = turnover
    _roll('_to_vol', 'factor_liq_turnover_vol_20d', 20, 'std')
    df['factor_liq_composite'] = -df['factor_liq_amihud_20d'] - df['factor_liq_turnover_vol_20d']

    # ===== 规模：16 cols (prefix factor_size_) =====
    with np.errstate(divide='ignore'):
        df['factor_size_log_mcap'] = -np.where(mcap > 0, np.log(mcap), np.nan)
    df['factor_size_cs_rank'] = f3d.cs_rank('market_cap').df['market_cap']

    for p in [5, 20, 60]:
        df[f'factor_size_mcap_chg_{p}d'] = df.groupby('name')['market_cap'].pct_change(p)

    df['_mcap_ret'] = df.groupby('name')['market_cap'].pct_change(1)
    _roll('_mcap_ret', 'factor_size_mcap_vol_20d', 20, 'std')
    _roll('_mcap_ret', 'factor_size_mcap_vol_60d', 60, 'std')

    df['factor_size_mcap_mom_5d'] = df.groupby('name')['market_cap'].pct_change(5)
    df['factor_size_mcap_mom_20d'] = df.groupby('name')['market_cap'].pct_change(20)

    df['factor_size_mcap_sqrt'] = -np.sqrt(mcap)
    df['factor_size_mcap_cube_root'] = -np.cbrt(mcap)
    ratio = mcap / close
    with np.errstate(divide='ignore'):
        df['factor_size_price'] = np.where(ratio > 0, np.log(ratio), np.nan)
    df['factor_size_quintile'] = f3d.cs_rank('market_cap').df['market_cap']
    with np.errstate(divide='ignore'):
        df['factor_size_small_and_rising'] = -np.where(mcap > 0, np.log(mcap), np.nan) * df['factor_size_mcap_mom_20d']

    df['_ret20'] = f3d.ts_pct_change('close', 20).df['close']
    df['_mcap_raw'] = mcap
    # 复合必须在下一次 result 重建前计算
    df['factor_size_composite'] = (
        df['factor_size_log_mcap']
        + df['factor_size_cs_rank']
        + df['factor_size_mcap_vol_20d']
        + df['factor_size_price']
    ) / 4

    # 中性化需要 Frame3D
    result = Frame3D(df.copy())
    neut = result.cs_neutralize('_ret20', by=['_mcap_raw'])
    df['factor_size_residual_ret'] = neut.df['_ret20']
    result = Frame3D(df.copy())
    # 联合截面标准化
    factor_cols = [c for c in df.columns if c.startswith(('factor_liq_', 'factor_size_'))]
    result = result.cs_zscore_batch(factor_cols, cp=False)
    return Frame3D(result.df[factor_cols].copy())
