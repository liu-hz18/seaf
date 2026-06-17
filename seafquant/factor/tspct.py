"""
TSPCT 因子 — 时序百分位排名因子（40 个）。

对基础 OHLCV 和隔夜/日内涨跌幅，在历史滑动窗口内计算百分位排名（0~1），
反映当前值在历史分布中的相对位置。

因子列表：
  OHLCV × 5窗口 (2,5,10,20,60) → 30 个
  隔夜涨跌幅 × 5窗口          →  5 个
  日内涨跌幅 × 5窗口          →  5 个

优化：rolling().apply(lambda) 替换为 rolling().rank(pct=True)
     纯 C 实现，单次调用 10s → 0.3s。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from qpipe.frame3d import Frame3D

_WINDOWS = [5, 20, 60]

# 基础量价列
_BASE_COLS = ['open', 'high', 'low', 'close', 'volume', 'turnover']


def compute_tspct_factors(name: str, idx: int, f3d: Frame3D, ctx: Any = None) -> Frame3D:
    """计算 40 个时序百分位排名因子。

    对每个源列 × 每个窗口，计算当前值在过去 w 天内的 percent rank (0~1)。
    使用 pandas 内置 rolling().rank(pct=True) 替代 apply(lambda) 实现 30x 加速。
    """
    result = f3d.copy()
    df = result.df

    # ── 1. 隔夜涨跌幅 ─────────────────────────────────────────
    grp = df.groupby('code')
    prev_close = grp['close'].shift(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['_overnight_pct'] = (
            (df['open'] - prev_close) / prev_close.replace(0, np.nan)
        )

    # ── 2. 日内涨跌幅 ─────────────────────────────────────────
    with np.errstate(divide='ignore', invalid='ignore'):
        df['_intraday_pct'] = (
            (df['close'] - df['open']) / df['open'].replace(0, np.nan)
        )

    # ── 3. 所有源列 → 批量 rolling rank ─────────────────────
    src_cols = [*_BASE_COLS, '_overnight_pct', '_intraday_pct']
    factor_cols: list[str] = []

    for col in src_cols:
        if col == '_overnight_pct': alias = 'on'
        elif col == '_intraday_pct': alias = 'id'
        else: alias = col

        for w in _WINDOWS:
            fcol = f'factor_tspct_{alias}_{w}d'
            factor_cols.append(fcol)
            min_p = max(2, w // 2)
            # groupby().rolling().rank(pct=True) — C 原生，计算所有历史日期的 rank
            df[fcol] = (
                df.groupby('code')[col]
                .rolling(window=w, min_periods=min_p)
                .rank(pct=True)
                .droplevel(0)
            )

    # ── 4. 截面标准化 ─────────────────────────────────────────
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(
        f'[{idx}] NaN factors: { {c: result.df[c].isna().sum() for c in factor_cols} }'
    )
    return Frame3D(result.df[factor_cols].copy())
