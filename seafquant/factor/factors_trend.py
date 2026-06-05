"""
趋势因子（MA）— 8 个因子：价格相对 MA 偏离 + MA 交叉。
MACD/动量/复合已拆分至 trend_macd 节点并行执行。

优化：使用直接 pandas groupby rolling 减少 f3d.copy() 深拷贝开销。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_trend_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 8 个趋势 MA 因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    df = result.df

    # ---- 辅助 ----
    def _add_rolling(src_col, dst_col, window, agg):
        rolled = df.groupby('name')[src_col].rolling(window, min_periods=max(1, window // 2)).agg(agg)
        df[dst_col] = rolled.values

    # ---- 1-5: 价格 vs 移动均线偏离 ----
    for p in [5, 10, 20, 60, 120]:
        df[f'_ma{p}'] = close
        _add_rolling(f'_ma{p}', f'_ma{p}', p, 'mean')
        df[f'factor_trend_ma_{p}d'] = close / df[f'_ma{p}'].replace(0, np.nan) - 1

    # ---- 6-8: MA 交叉 ----
    df['factor_trend_ma_cross_5_20'] = (
        df['_ma5'] / df['_ma20'].replace(0, np.nan) - 1
    )
    df['factor_trend_ma_cross_10_60'] = (
        df['_ma10'] / df['_ma60'].replace(0, np.nan) - 1
    )
    df['factor_trend_ma_cross_20_120'] = (
        df['_ma20'] / df['_ma120'].replace(0, np.nan) - 1
    )

    # 截面标准化
    factor_cols = [c for c in df.columns if c.startswith('factor_trend_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] Trend MA NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
