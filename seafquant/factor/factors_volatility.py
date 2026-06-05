"""
波动率因子 — 16 个因子，基于已实现波动、下行波动、Parkinson、GK 估计等。
优化：使用直接 pandas groupby rolling 减少 f3d.copy() 深拷贝开销。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_volatility_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 16 个波动率因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    high = f3d.df['high']
    low = f3d.df['low']
    open_p = f3d.df['open']
    df = result.df

    ret = f3d.ts_pct_change('close', 1).df['close']
    df['_ret'] = ret

    def _roll(src, dst, window, agg):
        df[dst] = df.groupby('name')[src].rolling(window, min_periods=max(1, window // 2)).agg(agg).values

    # ---- 1-4: 已实现波动率 ----
    for p in [5, 10, 20, 60]:
        _roll('_ret', f'factor_vol_realized_{p}d', p, 'std')

    # ---- 5-8: 下行波动率 ----
    df['_neg_ret'] = ret.clip(upper=0)
    for p in [5, 10, 20, 60]:
        _roll('_neg_ret', f'_ds_mean{p}', p, 'mean')
        df[f'factor_vol_downside_{p}d'] = np.sqrt(np.abs(df[f'_ds_mean{p}']))

    # ---- 9-10: 波动率的波动率 ----
    _roll('factor_vol_realized_5d', 'factor_vol_of_vol_20d', 20, 'std')
    _roll('factor_vol_realized_20d', 'factor_vol_of_vol_60d', 60, 'std')

    # ---- 11-12: Parkinson 波动率 ----
    df['_loghl'] = np.log(high / low)
    park_factor = 1.0 / (4 * np.log(2))
    df['_park_sq'] = park_factor * df['_loghl'] ** 2
    _roll('_park_sq', '_park5_mean', 5, 'mean')
    df['factor_vol_parkinson_5d'] = np.sqrt(np.abs(df['_park5_mean']))
    _roll('_park_sq', '_park20_mean', 20, 'mean')
    df['factor_vol_parkinson_20d'] = np.sqrt(np.abs(df['_park20_mean']))

    # ---- 13: Garman-Klass 估计 ----
    log_co = np.log(close / open_p)
    df['_gk'] = 0.5 * df['_loghl']**2 - (2 * np.log(2) - 1) * log_co**2
    _roll('_gk', '_gk5_mean', 5, 'mean')
    df['factor_vol_gk_5d'] = np.sqrt(np.abs(df['_gk5_mean']))

    # ---- 14-15: 波动率趋势 ----
    df['factor_vol_trend_20d'] = (
        df['factor_vol_realized_20d'] / df['factor_vol_realized_60d'].replace(0, np.nan) - 1
    )
    _roll('_ret', '_rv120', 120, 'std')
    df['factor_vol_trend_60d'] = (
        df['factor_vol_realized_60d'] / df['_rv120'].replace(0, np.nan) - 1
    )

    # ---- 16: 高低价范围 ----
    df['_range'] = high / low - 1
    _roll('_range', 'factor_vol_range_20d', 20, 'mean')

    factor_cols = [c for c in df.columns if c.startswith('factor_vol_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] Volatility NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
