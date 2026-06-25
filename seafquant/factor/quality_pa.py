"""
质量形态+自相关合并 — 13 因子。共享 ret_2d。
"""

from __future__ import annotations

import numpy as np

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import (
    njit,
    rolling_autocorr_2d,
    rolling_mean_2d,
    rolling_std_2d,
    rolling_tail_risk_2d,
)


@njit
def _mcp(binary, window):
    T, S = binary.shape
    out = np.full((T - window + 1, S), np.nan)
    for i in range(T - window + 1):
        for s in range(S):
            cur = 0; best = 0
            for j in range(window):
                if binary[i + j, s]: cur += 1
                if cur > best: best = cur
                else: cur = 0
            out[i, s] = best
    return out / window


@njit
def _udr(arr, window):
    T, S = arr.shape
    out = np.full((T - window + 1, S), np.nan)
    for i in range(T - window + 1):
        for s in range(S):
            ps = 0.0; pc = 0; ns = 0.0; nc = 0
            for j in range(window):
                v = arr[i + j, s]
                if np.isnan(v): continue
                if v > 0: ps += v; pc += 1
                elif v < 0: ns += abs(v); nc += 1
            if pc > 0 and nc > 0: out[i, s] = (ps / pc) / max(ns / nc, 1e-6)
    return out


def compute_quality_pa_factors(name: str, idx: int, f3d: Frame3D, context) -> Frame3D:
    result = f3d.copy()
    high, low = f3d.df['high'], f3d.df['low']
    ret = f3d.ts_pct_change('close', 1).df['close']
    df = result.df

    ret_2d = ret.unstack(level='code').values

    # ── HL 稳定性 (2) ──
    hl_2d = (high.unstack(level='code').values
             / np.where(low.unstack(level='code').values != 0, low.unstack(level='code').values, np.nan) - 1)
    hl_stds = rolling_std_2d(hl_2d, [20, 60])
    df['factor_qa_hl_stability_20d'] = 1.0 / np.where(hl_stds[20].ravel() != 0, hl_stds[20].ravel(), np.nan)
    df['factor_qa_hl_stability_60d'] = 1.0 / np.where(hl_stds[60].ravel() != 0, hl_stds[60].ravel(), np.nan)

    # ── 峰度+偏度 (4) ──
    r2, r3, r4 = ret_2d ** 2, ret_2d ** 3, ret_2d ** 4
    qw = [60, 120]
    m1 = rolling_mean_2d(ret_2d, qw); m2 = rolling_mean_2d(r2, qw)
    m3 = rolling_mean_2d(r3, qw); m4 = rolling_mean_2d(r4, qw)
    for w in qw:
        mu = m1[w]; var = np.maximum(m2[w] - mu * mu, 1e-15)
        mu3 = m3[w] - 3 * mu * m2[w] + 2 * mu ** 3
        mu4 = m4[w] - 4 * mu * m3[w] + 6 * mu ** 2 * m2[w] - 3 * mu ** 4
        df[f'factor_qa_kurt_{w}d'] = (mu4 / (var * var)).ravel()
        df[f'factor_qa_skew_{w}d'] = (mu3 / (var * np.sqrt(var))).ravel()

    # ── 最大连续正 (1) ──
    bin_2d = (ret_2d > 0).astype(np.int8)
    mcp_full = np.full_like(ret_2d, np.nan); mcp_full[59:] = _mcp(bin_2d, 60)
    df['factor_qa_max_consec_pos_60d'] = mcp_full.ravel()

    # ── Up/Down (1) ──
    udr_full = np.full_like(ret_2d, np.nan); udr_full[59:] = _udr(ret_2d, 60)
    df['factor_qa_up_down_60d'] = udr_full.ravel()

    # ── 复合 (1) ──
    df['factor_qa_composite'] = (df['factor_qa_skew_60d'] + df['factor_qa_up_down_60d']) / 2

    # ── 自相关+尾部 (4) ──
    ac = rolling_autocorr_2d(ret_2d, [20, 60])
    df['factor_qa_autocorr_20d'] = ac[20].ravel()
    df['factor_qa_autocorr_60d'] = ac[60].ravel()
    tr = rolling_tail_risk_2d(ret_2d, [60, 120])
    df['factor_qa_tail_risk_60d'] = tr[60].ravel()
    df['factor_qa_tail_risk_120d'] = tr[120].ravel()

    result = Frame3D(df.copy())
    fc = [c for c in df.columns if c.startswith('factor_qa_')]
    result = result.cs_zscore_batch(fc, cp=False)
    return Frame3D(result.df[fc].copy())
