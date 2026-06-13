"""
因子对拍验证公共辅助函数。

使用方式：
    from test.crossval_helpers import _make_data, _cs_zscore_manual, _roll_manual, \
        _ts_pct_manual, _compare_factor_output
"""

import os
import sys
import time as _time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qpipe.frame3d import Frame3D
from seafquant.data_generator import generate_synthetic_data

EPS = 1e-8
_SEED_COUNTER = 0


def _make_data(n_times: int = 60, n_stocks: int = 8) -> Frame3D:
    """每次调用用全局递增 seed 生成随机数据。测试中 fixture scope='class' 复用。"""
    global _SEED_COUNTER
    seed = _SEED_COUNTER
    _SEED_COUNTER += 1
    # 掐掉 generate_synthetic_data 里的 time.sleep(0.2)，测试不需要
    _orig_sleep = _time.sleep
    _time.sleep = lambda _: None
    try:
        gen = generate_synthetic_data(
            n_times=n_times, n_stocks=n_stocks, noise_ratio=0.3, seed=seed,
        )
        frames = [f3d.df for f3d in gen]
    finally:
        _time.sleep = _orig_sleep
    big_df = pd.concat(frames, axis=0).sort_index(level=0)
    return Frame3D(big_df)


def _cs_zscore_manual(f3d: Frame3D, cols: list[str]) -> Frame3D:
    """独立实现截面标准化：(x - cs_mean) / cs_std，std=0 → 0。"""
    result = f3d.copy()
    df = result.df
    for col in cols:
        grp = df.groupby('key')[col]
        cs_mean = grp.transform('mean')
        cs_std = grp.transform('std')
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (df[col] - cs_mean) / cs_std.replace(0, np.nan)
        df[col] = z.fillna(0.0)
    return result


def _roll_manual(df: pd.DataFrame, col: str, window: int, agg: str) -> pd.Series:
    """逐股票时序滚动聚合。"""
    return df.groupby('code')[col].transform(  # pyright: ignore[reportReturnType]
        lambda x: x.rolling(window=window, min_periods=max(1, window // 2)).agg(agg)
    )


def _ts_pct_manual(df: pd.DataFrame, col: str, period: int) -> pd.Series:
    """逐股票时序百分比变化。"""
    shifted = df.groupby('code')[col].shift(period)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (df[col] - shifted) / shifted.replace(0, np.nan)


def _compare_factor_output(
    actual_f3d: Frame3D,
    expected_cols: dict[str, np.ndarray],
    atol: float = 1e-10,
):
    """对比实际因子输出与期望值（先对期望值做 cs_zscore 再对比）。"""
    expect_df = actual_f3d.df.copy()
    for col, vals in expected_cols.items():
        expect_df[col] = vals
    expect_f3d = _cs_zscore_manual(Frame3D(expect_df), list(expected_cols.keys()))

    for col in expected_cols:
        actual_vals = actual_f3d.df[col].values
        expected_vals = expect_f3d.df[col].values
        diff = np.abs(actual_vals - expected_vals)  # pyright: ignore[reportOperatorIssue]
        max_diff = np.nanmax(diff)
        if max_diff >= atol:
            n_bad = int(np.sum(diff >= atol))
            raise AssertionError(
                f'{col}: max_abs_diff={max_diff:.2e} >= {atol:.0e}, '
                f'{n_bad}/{len(diff)} rows differ\n'
                f'  actual[:5]   = {actual_vals[~np.isnan(actual_vals)][:5]}\n'
                f'  expected[:5] = {expected_vals[~np.isnan(expected_vals)][:5]}'
            )
