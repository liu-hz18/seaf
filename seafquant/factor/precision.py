"""
精度相关因子 — 12 个因子。优化 v2：_roll + groupby diff/pct_change 替换为 2D-array。
"""

from __future__ import annotations

import numpy as np

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import rolling_mean_2d, rolling_std_2d

EPS: float = 1e-8


def compute_precision_factors(name: str, idx: int, f3d: Frame3D, context) -> Frame3D:
    """计算 12 个精度相关因子 — 向量化 v2。"""
    result = f3d.copy()
    close = f3d.df['close']
    vwap = f3d.df['vwap']
    df = result.df

    # ── 提取 2D-array ──
    close_2d = close.unstack(level='code').values
    vwap_2d = vwap.unstack(level='code').values

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
        df[f'factor_vwap_deviation_{p}d'] = vwap / df[f'_vwap_ma{p}'].replace(0, np.nan) - 1

    # ===== 5-10: 价格梯度 =====
    # 一阶梯度
    grad1_2d = np.empty_like(vwap_2d)
    grad1_2d[0] = np.nan
    grad1_2d[1:] = vwap_2d[1:] - vwap_2d[:-1]
    df['_grad1'] = grad1_2d.ravel()

    grad1_mas = rolling_mean_2d(grad1_2d, [5, 20, 60])
    vwap_mas = rolling_mean_2d(vwap_2d, [5, 20, 60])
    for w in [5, 20, 60]:
        df[f'_grad1_ma{w}'] = grad1_mas[w].ravel()
        df[f'_vwap_ma{w}'] = vwap_mas[w].ravel()
        df[f'factor_grad_momentum_{w}d'] = (
            grad1_mas[w].ravel() / df[f'_vwap_ma{w}'].replace(0, np.nan)
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
            price_stds = rolling_std_2d(vwap_2d, [5, 20])
            df[f'_grad2_s{stride}_ma{w}'] = grad2_mas[w].ravel()
            df[f'_price_std{w}'] = price_stds[w].ravel()
            df[f'factor_grad_accel_{w}d_s{stride}'] = (
                grad2_mas[w].ravel() / df[f'_price_std{w}'].replace(0, np.nan)
            )

    # 1) 瞬时偏离度: (close - vwap) / vwap
    dev_2d = (close_2d - vwap_2d) / np.where(np.abs(vwap_2d) > EPS, vwap_2d, np.nan)
    df[f'factor_cv_dev_inst'] = dev_2d.ravel()

    # 2) 偏离度均值 / 标准差 / 偏度 (滚动窗口)
    dev_mas = rolling_mean_2d(dev_2d, [5, 20, 60])
    dev_stds = rolling_std_2d(dev_2d, [5, 20, 60])
    for w in [5, 20, 60]:
        df[f'factor_cv_dev_mean_{w}d'] = dev_mas[w].ravel()
        df[f'factor_cv_dev_std_{w}d'] = dev_stds[w].ravel()

        # z-score: 当前偏离度相对窗口均值的标准化 (先计算2D, 再ravel)
        zscore_2d = (dev_2d - dev_mas[w]) / np.where(dev_stds[w] > EPS, dev_stds[w], np.nan)
        df[f'factor_cv_dev_zscore_{w}d'] = zscore_2d.ravel()

    # 3) close > vwap 占比 (上行强势度), 滚动窗口求和符号均值
    sign_2d = (close_2d > vwap_2d).astype(np.float64)  # 1 if close > vwap else 0
    for w in [5, 20, 60]:
        # 用滚动均值近似 (sum / w)
        sign_ma = rolling_mean_2d(sign_2d, [w])[w]
        df[f'factor_cv_above_ratio_{w}d'] = sign_ma.ravel()

    # 4) 偏离度动量: 当前 dev 与 t-w 前 dev 的差分
    for w in [5, 20]:
        shifted_dev = np.roll(dev_2d, w, axis=0)
        shifted_dev[:w] = np.nan
        df[f'factor_cv_dev_diff_{w}d'] = (dev_2d - shifted_dev).ravel()

    # 5) 偏离度加速度: dev 的二阶差分
    dev_diff1 = np.empty_like(dev_2d)
    dev_diff1[0] = np.nan
    dev_diff1[1:] = dev_2d[1:] - dev_2d[:-1]
    dev_diff2 = np.empty_like(dev_2d)
    dev_diff2[:2] = np.nan
    dev_diff2[2:] = dev_diff1[2:] - dev_diff1[1:-1]
    dev_diff2_mas = rolling_mean_2d(dev_diff2, [5, 20])
    for w in [5, 20]:
        accel_2d = dev_diff2_mas[w] / np.where(dev_stds[w] > EPS, dev_stds[w], np.nan)
        df[f'factor_cv_dev_accel_{w}d'] = accel_2d.ravel()

    # 6) close 与 vwap 的滚动相关性 (经度系数)
    # 用 Pearson 相关: corr = cov(c,v) / (std_c * std_v)
    close_means = rolling_mean_2d(close_2d, [5, 20, 60])
    vwap_means = rolling_mean_2d(vwap_2d, [5, 20, 60])
    close_stds = rolling_std_2d(close_2d, [5, 20, 60])
    vwap_stds = rolling_std_2d(vwap_2d, [5, 20, 60])
    for w in [5, 20, 60]:
        c_demean = close_2d - close_means[w]
        v_demean = vwap_2d - vwap_means[w]
        # 滚动 cov: rolling_mean(c_demean * v_demean)
        cv_prod_ma = rolling_mean_2d(c_demean * v_demean, [w])[w]
        denom = close_stds[w] * vwap_stds[w]
        corr = cv_prod_ma / np.where(denom > EPS, denom, np.nan)
        df[f'factor_cv_corr_{w}d'] = corr.ravel()

    # 7) close/vwap 比值的滚动均值 (相对价格水位)
    ratio_2d = close_2d / np.where(np.abs(vwap_2d) > EPS, vwap_2d, np.nan)
    ratio_mas = rolling_mean_2d(ratio_2d, [5, 20, 60])
    for w in [5, 20, 60]:
        df[f'factor_cv_ratio_ma_{w}d'] = ratio_mas[w].ravel()

    # 8) 比值的反转: 当前 ratio 与窗口均值的偏离
    for w in [5, 20, 60]:
        df[f'factor_cv_ratio_reversal_{w}d'] = (ratio_2d - ratio_mas[w]).ravel()

    # 9) close-vwap 累积面积 — 趋势持续性
    for w in [5, 20, 60]:
        df[f'factor_cv_dev_cumsum_{w}d'] = (dev_mas[w] * w).ravel()

    # 10) 上穿/下穿 vwap 的频率 (rolling 窗口内符号变化次数)
    sign_change = np.zeros_like(close_2d, dtype=np.float64)
    sign_change[1:] = (sign_2d[1:] != sign_2d[:-1]).astype(np.float64)
    sign_change[0] = np.nan
    for w in [5, 20]:
        cross_count_ma = rolling_mean_2d(sign_change, [w])[w]
        df[f'factor_cv_cross_freq_{w}d'] = cross_count_ma.ravel()

    factor_cols = [c for c in df.columns if c.startswith(('factor_vwap_', 'factor_grad_', 'factor_cv_'))]
    return Frame3D(result.df[factor_cols].copy())
