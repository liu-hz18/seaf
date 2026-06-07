"""
质量自相关/尾部因子 — 4 个因子：自相关×2 + 尾部风险×2。
pandas rolling.corr/quantile 为 C 实现，不做额外 numpy 重写。
"""

from __future__ import annotations

import logging

from qpipe.frame3d import Frame3D


def compute_quality_autocorr_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 4 个自相关/尾部风险因子。"""
    result = f3d.copy()
    df = result.df
    df['_ret'] = df.groupby('name')['close'].pct_change(1)
    df.index.get_level_values('name')

    df['factor_qa_autocorr_60d'] = df.groupby('name')['_ret'].transform(
        lambda x: x.rolling(60, min_periods=20).corr(x.shift(1))
    )
    df['factor_qa_autocorr_20d'] = df.groupby('name')['_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).corr(x.shift(1))
    )

    df['factor_qa_tail_risk_60d'] = df.groupby('name')['_ret'].transform(
        lambda x: -x.where(x < 0).rolling(60, min_periods=20).quantile(0.05)
    )
    df['factor_qa_tail_risk_120d'] = df.groupby('name')['_ret'].transform(
        lambda x: -x.where(x < 0).rolling(120, min_periods=40).quantile(0.05)
    )

    factor_cols = [c for c in result.df.columns if c.startswith('factor_qa_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(
        f'[{name}] QA-Autocorr NaN: { {c: result.df[c].isna().sum() for c in factor_cols} }'
    )
    return Frame3D(result.df[factor_cols].copy())
