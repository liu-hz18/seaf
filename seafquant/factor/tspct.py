"""
TSPCT 因子 — 时序百分位排名因子（40 个）。

对基础 OHLCV 和隔夜/日内涨跌幅，在历史滑动窗口内计算百分位排名（0~1），
反映当前值在历史分布中的相对位置。

因子列表：
  OHLCV × 5窗口 (2,5,10,20,60) → 30 个
  隔夜涨跌幅 × 5窗口          →  5 个
  日内涨跌幅 × 5窗口          →  5 个
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from qpipe.frame3d import Frame3D

_WINDOWS = [2, 5, 10, 20, 60]

# 基础量价列
_BASE_COLS = ['open', 'high', 'low', 'close', 'volume', 'turnover']


def compute_tspct_factors(name: str, f3d: Frame3D, context: Any = None) -> Frame3D:
    """计算 40 个时序百分位排名因子。"""
    result = f3d.copy()
    df = result.df

    # ── 1. 隔夜涨跌幅：(open - prev_close) / prev_close ──
    grp = df.groupby('code')
    prev_close = grp['close'].shift(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['_overnight_pct'] = (df['open'] - prev_close) / prev_close.replace(0, np.nan)

    # ── 2. 日内涨跌幅：(close - open) / open ──
    with np.errstate(divide='ignore', invalid='ignore'):
        df['_intraday_pct'] = (df['close'] - df['open']) / df['open'].replace(0, np.nan)

    # ── 3. 所有源列汇总 ──
    src_cols = [*_BASE_COLS, '_overnight_pct', '_intraday_pct']

    # ── 4. 对每个源列 × 窗口，计算时序排名 ──
    factor_cols: list[str] = []
    for col in src_cols:
        for w in _WINDOWS:
            # 简短别名：overnight → on, intraday → id
            if col == '_overnight_pct':
                alias = 'on'
            elif col == '_intraday_pct':
                alias = 'id'
            else:
                alias = col
            fcol = f'factor_tspct_{alias}_{w}d'
            factor_cols.append(fcol)

            df[fcol] = df[col].copy()
            # 使用 groupby + rolling rank (百分位)，min_periods = max(2, w//2)
            min_p = max(2, w // 2)
            df[fcol] = df.groupby('code')[fcol].transform(
                lambda x: x.rolling(window=w, min_periods=min_p).apply(  # noqa: B023
                    lambda r: (r.rank().iloc[-1] - 1) / (len(r) - 1)
                    if len(r) > 1 else np.nan,
                    raw=False,
                )
            )

    # ── 5. 截面标准化 ──
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(
        f'[tspct] {len(factor_cols)} factors, '
        f'NaN: { {c: result.df[c].isna().sum() for c in factor_cols} }'
    )
    return Frame3D(result.df[factor_cols].copy())
