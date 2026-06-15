"""
精度相关因子 — 12 个因子：VWAP 均价 / 价格梯度 / 对数精度损失。

设计意图：
- VWAP：O/H/L/C 均值天然压制单价格舍入噪声，1/√3 方差缩减
- 梯度（一阶/二阶差分）：将价格序列变换到频域，突出动量和拐点
- 对数精度损失：log(p) 与 log(round(p,2)) 的差值反映舍入信息丢失量
"""

from __future__ import annotations

import numpy as np

from qpipe.frame3d import Frame3D

EPS: float = 1e-8


def compute_precision_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 12 个精度相关因子。"""
    result = f3d.copy()
    close = f3d.df['close']
    high = f3d.df['high']
    low = f3d.df['low']
    open_p = f3d.df['open']
    volume = f3d.df['volume']
    df = result.df

    def _roll(src, dst, window, agg):
        df[dst] = (
            df.groupby('code')[src]
            .rolling(window, min_periods=max(1, window // 2))
            .agg(agg)
            .reset_index(level=0, drop=True)
        )

    # ===== 1-4: VWAP 因子 =====
    # 简化 VWAP ≡ (H+L+C)/3，比单个 OHLC 舍入方差低 1/3
    df['_vwap'] = (high + low + close) / 3.0
    df['_vwap_ret1'] = df.groupby('code')['_vwap'].pct_change(1)

    for p in [5, 20]:
        df[f'_vwap_ret{p}'] = df.groupby('code')['_vwap'].pct_change(p)
        df[f'factor_vwap_ret_{p}d'] = df[f'_vwap_ret{p}']

    for p in [5, 20]:
        _roll('_vwap', f'_vwap_ma{p}', p, 'mean')
        df[f'factor_vwap_deviation_{p}d'] = (
            close / df[f'_vwap_ma{p}'].replace(0, np.nan) - 1
        )

    # ===== 5-10: 价格梯度（一阶 + 二阶差分）=====
    # 一阶梯度 Δp[t] = p[t] - p[t-1] （动量方向）
    df['_grad1'] = df.groupby('code')['close'].diff(1)

    for w in [5, 20, 60]:
        # 梯度均线：平滑的一阶变化
        _roll('_grad1', f'_grad1_ma{w}', w, 'mean')
        grad_smooth = df[f'_grad1_ma{w}']
        # 梯度强度 = 梯度 / 价格量级，归一化
        _roll('close', f'_close_ma{w}', w, 'mean')
        df[f'factor_grad_momentum_{w}d'] = (
            grad_smooth / df[f'_close_ma{w}'].replace(0, np.nan)
        )

    # 二阶梯度（加速度）Δ²p[t] = p[t] - 2p[t-1] + p[t-2]
    df['_grad2_raw'] = df.groupby('code')['close'].diff(2) - df.groupby('code')['_grad1'].shift(1)

    for w in [5, 20]:
        _roll('_grad2_raw', f'_grad2_ma{w}', w, 'mean')
        _roll('close', f'_price_std{w}', w, 'std')
        df[f'factor_grad_accel_{w}d'] = (
            df[f'_grad2_ma{w}'] / df[f'_price_std{w}'].replace(0, np.nan)
        )

    # ===== 11-12: 对数精度损失 =====
    # log(p) 与 log(round(p, 2)) 的差值 ≈ 舍入误差相对量级
    # 这等价于度量 "价格变化是否大到足以跨越舍入阈值"
    for p in [5, 20]:
        log_p = np.log(close.replace(0, np.nan))
        log_p_round = np.log(np.round(close, 2))
        df[f'_log_diff_{p}'] = np.abs(log_p - log_p_round)
        _roll(f'_log_diff_{p}', f'factor_prec_sig_loss_{p}d', p, 'mean')

    # ==== 联合截面标准化 ====
    factor_cols = [c for c in df.columns if c.startswith(('factor_vwap_', 'factor_grad_', 'factor_prec_'))]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    return Frame3D(result.df[factor_cols].copy())
