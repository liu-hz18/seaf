"""
估值因子 — 32 个因子。基于 PE/PB/PS/PCF + ROE 代理 + TSPCT 历史分位。
仅 baostock 真实数据可用；synthetic 模式下列为 NaN 自动跳过。
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from qpipe.frame3d import Frame3D
from seafquant.factor._perf import rolling_mean_2d, rolling_std_2d

# ── 时序百分位排名 (TSPCT) 辅助 ──

def _ts_rank_pct(arr: np.ndarray, window: int) -> np.ndarray:
    """单窗口时序百分位排名 (0~1)，2D 向量化。"""
    T, S = arr.shape
    out = np.full((T, S), np.nan)
    if window > T:
        return out
    swv = sliding_window_view(arr, window, axis=0)  # (T-w+1, S, w)
    win = swv[:, :, -window:]  # 确保窗口大小正确
    last = win[:, :, -1]        # (T-w+1, S)
    valid = ~np.isnan(win)
    vc = valid.sum(axis=2)
    last_nan = np.isnan(last)
    le = (win <= last[:, :, np.newaxis]) & valid
    le_cnt = le.sum(axis=2)
    mask = (vc >= max(2, window // 2)) & (~last_nan)
    rp = np.full((T - window + 1, S), np.nan)
    rp[mask] = (le_cnt[mask] - 1.0) / np.maximum(vc[mask] - 1.0, 1.0)
    out[window - 1:] = rp
    return out


# ── 主函数 ──

def compute_valuation_factors(name: str, idx: int, f3d: Frame3D, context: dict) -> Frame3D:
    """计算 32 个估值因子。"""
    result = f3d.copy()
    df = result.df

    # ── 估值原始列 → 2D-array ──
    n_times = f3d.df.index.get_level_values('key').nunique()
    # 估值乘数可能为 0（如亏损股 PE=0）→ 1/0 触发 divide by zero
    _ep = df['peTTM'].values.astype(np.float64)
    _bp = df['pbMRQ'].values.astype(np.float64)
    _sp = df['psTTM'].values.astype(np.float64)
    _cfp = df['pcfNcfTTM'].values.astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        ep_2d  = np.where(_ep  != 0, 1.0 / _ep,  np.nan).reshape(n_times, -1)
        bp_2d  = np.where(_bp  != 0, 1.0 / _bp,  np.nan).reshape(n_times, -1)
        sp_2d  = np.where(_sp  != 0, 1.0 / _sp,  np.nan).reshape(n_times, -1)
        cfp_2d = np.where(_cfp != 0, 1.0 / _cfp, np.nan).reshape(n_times, -1)
    pe_2d = df['peTTM'].values.reshape(n_times, -1)
    pb_2d = df['pbMRQ'].values.reshape(n_times, -1)
    close_2d = df['close'].values.reshape(n_times, -1)

    # ═══════════════ 1-4: 原始估值 yield ═══════════════
    # f = 1 / PE  (E/P — Earnings Yield)
    df['factor_est_ep_1d'] = ep_2d.ravel()
    # f = 1 / PB  (B/P — Book-to-Price)
    df['factor_est_bp_1d'] = bp_2d.ravel()
    # f = 1 / PS  (S/P — Sales Yield)
    df['factor_est_sp_1d'] = sp_2d.ravel()
    # f = 1 / PCF (CF/P — Cash Flow Yield)
    df['factor_est_cfp_1d'] = cfp_2d.ravel()

    # ═══════════════ 5-8: 相对自身历史偏离 (MA 偏离) ═══════════════
    # f = yield / MA_w(yield) - 1,  w=20,60
    ep_mas = rolling_mean_2d(ep_2d, [20, 60])
    bp_mas = rolling_mean_2d(bp_2d, [20, 60])
    df['factor_est_ep_ma20'] = (ep_2d / np.where(ep_mas[20] != 0, ep_mas[20], np.nan) - 1).ravel()
    df['factor_est_bp_ma20'] = (bp_2d / np.where(bp_mas[20] != 0, bp_mas[20], np.nan) - 1).ravel()
    df['factor_est_ep_ma60'] = (ep_2d / np.where(ep_mas[60] != 0, ep_mas[60], np.nan) - 1).ravel()
    df['factor_est_bp_ma60'] = (bp_2d / np.where(bp_mas[60] != 0, bp_mas[60], np.nan) - 1).ravel()

    # ═══════════════ 9-12: 时序 Z-score ═══════════════
    # f = (yield - μ_w) / σ_w,  w=60,120
    ep_stds = rolling_std_2d(ep_2d, [60, 120])
    bp_stds = rolling_std_2d(bp_2d, [60, 120])
    df['factor_est_ep_zscore_60d'] = ((ep_2d - ep_mas[60])
                                       / np.where(ep_stds[60] != 0, ep_stds[60], np.nan)).ravel()
    df['factor_est_bp_zscore_60d'] = ((bp_2d - bp_mas[60])
                                       / np.where(bp_stds[60] != 0, bp_stds[60], np.nan)).ravel()
    ep_m120 = rolling_mean_2d(ep_2d, [120])
    bp_m120 = rolling_mean_2d(bp_2d, [120])
    ep_s120 = rolling_std_2d(ep_2d, [120])
    bp_s120 = rolling_std_2d(bp_2d, [120])
    df['factor_est_ep_zscore_120d'] = ((ep_2d - ep_m120[120])
                                        / np.where(ep_s120[120] != 0, ep_s120[120], np.nan)).ravel()
    df['factor_est_bp_zscore_120d'] = ((bp_2d - bp_m120[120])
                                        / np.where(bp_s120[120] != 0, bp_s120[120], np.nan)).ravel()

    # ═══════════════ 13-14: 估值变化率 ═══════════════
    # f = yield[t] / yield[t-20] - 1  (positive = getting cheaper)
    ep_s20 = np.roll(ep_2d, 20, axis=0); ep_s20[:20] = np.nan
    df['factor_est_ep_chg_20d'] = (ep_2d / np.where(ep_s20 != 0, ep_s20, np.nan) - 1).ravel()
    bp_s20 = np.roll(bp_2d, 20, axis=0); bp_s20[:20] = np.nan
    df['factor_est_bp_chg_20d'] = (bp_2d / np.where(bp_s20 != 0, bp_s20, np.nan) - 1).ravel()

    # ═══════════════ 15-16: 截面排名 ═══════════════
    df['_ep_raw'] = ep_2d.ravel(); df['_bp_raw'] = bp_2d.ravel()
    tmp = Frame3D(df.copy())
    df['factor_est_ep_rank'] = tmp.cs_rank('_ep_raw').df['_ep_raw']
    df['factor_est_bp_rank'] = tmp.cs_rank('_bp_raw').df['_bp_raw']

    # ═══════════════ 17-18: 复合估值 ═══════════════
    # f = avg(ep_zs60, bp_zs60)
    df['factor_est_composite_2'] = (df['factor_est_ep_zscore_60d']
                                     + df['factor_est_bp_zscore_60d']) / 2
    # f = avg(4 yield z-scores)
    sp_s60 = rolling_std_2d(sp_2d, [60])[60]
    sp_m60 = rolling_mean_2d(sp_2d, [60])[60]
    sp_zs = (sp_2d - sp_m60) / np.where(sp_s60 != 0, sp_s60, np.nan)
    cfp_s60 = rolling_std_2d(cfp_2d, [60])[60]; cfp_m60 = rolling_mean_2d(cfp_2d, [60])[60]
    cfp_zs = (cfp_2d - cfp_m60) / np.where(cfp_s60 != 0, cfp_s60, np.nan)
    df['factor_est_composite_4'] = (df['factor_est_ep_zscore_60d']
                                     + df['factor_est_bp_zscore_60d']
                                     + sp_zs.ravel() + cfp_zs.ravel()) / 4

    # ═══════════════ 19: 便宜且稳定 ═══════════════
    # f = ep_zs60 * (1 - cs_zscore(σ_20(ret)))
    ret_2d = np.empty_like(close_2d); ret_2d[0] = np.nan
    ret_2d[1:] = (close_2d[1:] - close_2d[:-1]) / np.where(close_2d[:-1] != 0, close_2d[:-1], np.nan)
    rv = rolling_std_2d(ret_2d, [20])[20]
    df['_ret_vol'] = rv.ravel()
    tmp2 = Frame3D(df.copy())
    df['factor_est_cheap_quality'] = df['factor_est_ep_zscore_60d'] * (1.0 - tmp2.cs_zscore('_ret_vol').df['_ret_vol'])

    # ═══════════════ 20: 便宜且有动量 ═══════════════
    # f = ep_rank * cs_rank(r_20)
    r20 = np.roll(close_2d, 20, axis=0); r20[:20] = np.nan
    r20 = (close_2d - r20) / np.where(r20 != 0, r20, np.nan)
    df['_r20'] = r20.ravel()
    tmp3 = Frame3D(df.copy())
    df['factor_est_value_momentum'] = df['factor_est_ep_rank'] * tmp3.cs_rank('_r20').df['_r20']

    # ═══════════════ 21-24: TSPCT 历史分位 ═══════════════
    # f = percentile(ep in rolling window) — 当前估值在历史中的分位 (高=便宜)
    df['factor_est_ep_tspct_60d'] = _ts_rank_pct(ep_2d, 60).ravel()
    df['factor_est_ep_tspct_120d'] = _ts_rank_pct(ep_2d, 120).ravel()
    df['factor_est_bp_tspct_60d'] = _ts_rank_pct(bp_2d, 60).ravel()
    df['factor_est_bp_tspct_120d'] = _ts_rank_pct(bp_2d, 120).ravel()

    # ═══════════════ 25-26: ROE 代理 ═══════════════
    # ROE ≈ E/B = (P/PE) / (P/PB) = PB/PE = ep/bp
    # f = ep / bp  (ROE proxy)
    roe_2d = ep_2d / np.where(bp_2d != 0, bp_2d, np.nan)
    df['factor_est_roe'] = roe_2d.ravel()
    # f = roe[t] / roe[t-20] - 1  (ROE improvement)
    roe_s20 = np.roll(roe_2d, 20, axis=0); roe_s20[:20] = np.nan
    df['factor_est_roe_chg_20d'] = (roe_2d / np.where(roe_s20 != 0, roe_s20, np.nan) - 1).ravel()

    # ═══════════════ 27-28: 估值波动率 ═══════════════
    # f = σ_60(ep)  — 估值稳定性 (低波动 = 估值锚定清晰)
    ev = rolling_std_2d(ep_2d, [60])
    bv = rolling_std_2d(bp_2d, [60])
    df['factor_est_ep_vol_60d'] = ev[60].ravel()
    df['factor_est_bp_vol_60d'] = bv[60].ravel()

    # ═══════════════ 29-30: 原始 PE/PB 的 TSPCT (反向) ═══════════════
    # f = 1 - percentile(PE) — 低 PE = 高排名 = 便宜
    df['factor_est_pe_tspct_60d'] = (1.0 - _ts_rank_pct(pe_2d, 60)).ravel()
    df['factor_est_pb_tspct_60d'] = (1.0 - _ts_rank_pct(pb_2d, 60)).ravel()

    # ═══════════════ 31: 估值分歧 ═══════════════
    # f = ep_rank - bp_rank — PE和PB给出的信号分歧
    df['factor_est_spread'] = df['factor_est_ep_rank'] - df['factor_est_bp_rank']

    # ═══════════════ 32: ROE vs 自身历史 ═══════════════
    # f = (roe - μ_60(roe)) / σ_60(roe)
    roe_m = rolling_mean_2d(roe_2d, [60])[60]
    roe_s = rolling_std_2d(roe_2d, [60])[60]
    df['factor_est_roe_zscore_60d'] = ((roe_2d - roe_m)
                                        / np.where(roe_s != 0, roe_s, np.nan)).ravel()

    # 截面标准化
    factor_cols = [c for c in df.columns if c.startswith('factor_est_')]
    result = Frame3D(df.copy())
    return Frame3D(result.df[factor_cols].copy())
