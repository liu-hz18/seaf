"""
精度相关因子 — 12 个因子。优化 v2：_roll + groupby diff/pct_change 替换为 2D-array。
"""

from __future__ import annotations

import logging

import numpy as np

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import rolling_mean_2d, rolling_std_2d

EPS: float = 1e-8


def compute_precision_factors(name: str, idx: int, f3d: Frame3D, context) -> Frame3D:
    """计算 12 个精度相关因子 — 向量化 v2。"""
    result = f3d.copy()
    close = f3d.df['close']
    high = f3d.df['high']
    low = f3d.df['low']
    df = result.df

    # ── 提取 2D-array ──
    close_piv = close.unstack(level='code')
    close_2d = close_piv.values  # (T, S)
    vwap_2d = (high.unstack(level='code').values
               + low.unstack(level='code').values
               + close_2d) / 3.0

    # ===== 1-4: VWAP 因子 =====
    df['_vwap'] = vwap_2d.ravel()

    # VWAP pct_change (向量化 shift)
    for p in [1, 5, 20]:
        shifted = np.roll(vwap_2d, p, axis=0)
        shifted[:p] = np.nan
        ret_arr = (vwap_2d - shifted) / np.where(shifted != 0, shifted, np.nan)
        df[f'_vwap_ret{p}'] = ret_arr.ravel()
        if p > 1:
            df[f'factor_vwap_ret_{p}d'] = ret_arr.ravel()

    # VWAP deviation
    vwap_mas = rolling_mean_2d(vwap_2d, [5, 20])
    for p in [5, 20]:
        df[f'_vwap_ma{p}'] = vwap_mas[p].ravel()
        df[f'factor_vwap_deviation_{p}d'] = close / df[f'_vwap_ma{p}'].replace(0, np.nan) - 1

    # ===== 5-10: 价格梯度 =====
    # 一阶梯度
    grad1_2d = np.empty_like(close_2d)
    grad1_2d[0] = np.nan
    grad1_2d[1:] = close_2d[1:] - close_2d[:-1]
    df['_grad1'] = grad1_2d.ravel()

    grad1_mas = rolling_mean_2d(grad1_2d, [5, 20, 60])
    close_mas = rolling_mean_2d(close_2d, [5, 20, 60])
    for w in [5, 20, 60]:
        df[f'_grad1_ma{w}'] = grad1_mas[w].ravel()
        df[f'_close_ma{w}'] = close_mas[w].ravel()
        df[f'factor_grad_momentum_{w}d'] = (
            grad1_mas[w].ravel() / df[f'_close_ma{w}'].replace(0, np.nan)
        )

    # 二阶梯度 (加速度) — 保持 groupby diff (操作简单, 开销可接受)
    accel_strides = [1, 2, 5]
    for stride in accel_strides:
        df[f'_grad2_s{stride}'] = (
            df.groupby('code')['close'].diff(2 * stride)
            - df.groupby('code')['close'].diff(stride).shift(stride)
        )
        for w in [5, 20]:
            # 提取到 2D 做 rolling
            grad2_piv = df[f'_grad2_s{stride}'].unstack(level='code')
            grad2_2d = grad2_piv.values
            grad2_mas = rolling_mean_2d(grad2_2d, [5, 20])
            price_stds = rolling_std_2d(close_2d, [5, 20])
            df[f'_grad2_s{stride}_ma{w}'] = grad2_mas[w].ravel()
            df[f'_price_std{w}'] = price_stds[w].ravel()
            df[f'factor_grad_accel_{w}d_s{stride}'] = (
                grad2_mas[w].ravel() / df[f'_price_std{w}'].replace(0, np.nan)
            )

    factor_cols = [c for c in df.columns if c.startswith(('factor_vwap_', 'factor_grad_'))]
    return Frame3D(result.df[factor_cols].copy())
