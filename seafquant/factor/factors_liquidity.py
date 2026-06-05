"""
流动性因子 — 16 个因子，基于换手率、成交量、Amihud 非流动性指标。
优化：使用直接 pandas groupby rolling 减少 f3d.copy() 深拷贝开销。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_liquidity_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个流动性因子。"""
    result = f3d.copy()
    turnover = f3d.df['turnover']
    volume = f3d.df['volume']
    close = f3d.df['close']
    df = result.df

    def _roll(src, dst, window, agg):
        df[dst] = df.groupby('name')[src].rolling(window, min_periods=max(1, window // 2)).agg(agg).values

    # ---- 1-4: 日均换手率 ----
    for p in [5, 10, 20, 60]:
        df[f'_to{p}'] = turnover
        _roll(f'_to{p}', f'factor_liq_turnover_{p}d', p, 'mean')

    # ---- 5-8: 换手率变化 ----
    for p in [5, 10, 20, 60]:
        df[f'factor_liq_turnover_chg_{p}d'] = (
            turnover / df[f'factor_liq_turnover_{p}d'].replace(0, np.nan) - 1
        )

    # ---- 9-10: 成交量变化 ----
    df['_vol5'] = volume; _roll('_vol5', '_vol5_mean', 5, 'mean')
    df['factor_liq_volume_chg_5d'] = volume / df['_vol5_mean'].replace(0, np.nan) - 1
    df['_vol20'] = volume; _roll('_vol20', '_vol20_mean', 20, 'mean')
    df['factor_liq_volume_chg_20d'] = volume / df['_vol20_mean'].replace(0, np.nan) - 1

    # ---- 11-12: Amihud 非流动性 ----
    ret = f3d.ts_pct_change('close', 1).df['close']
    df['_amihud'] = np.abs(ret) / volume.replace(0, np.nan)
    _roll('_amihud', 'factor_liq_amihud_5d', 5, 'mean')
    _roll('_amihud', 'factor_liq_amihud_20d', 20, 'mean')

    # ---- 13-14: 成交额 ----
    dollar_vol = close * volume
    df['factor_liq_dollar_vol'] = np.log(dollar_vol)
    df['_dv'] = dollar_vol
    df['factor_liq_dollar_vol_chg'] = df.groupby('name')['_dv'].pct_change(20)

    # ---- 15: 换手率波动 ----
    df['_to_vol'] = turnover
    _roll('_to_vol', 'factor_liq_turnover_vol_20d', 20, 'std')

    # ---- 16: 复合 ----
    df['factor_liq_composite'] = (
        -df['factor_liq_amihud_20d'] - df['factor_liq_turnover_vol_20d']
    )

    factor_cols = [c for c in df.columns if c.startswith('factor_liq_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] Liquidity NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
