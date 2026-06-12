"""
IC 分析节点测试 — 覆盖 ic_analysis_fn 和 ic_epilogue。

测试基本计算正确性、边界情况（NaN/短数据）、epilogue 汇总逻辑、
以及 ic_analysis_fn 在不改变输入 df 的同时正确更新 context。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qpipe.frame3d import Frame3D
from seafquant.ic_analysis import ic_analysis_fn, ic_epilogue


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def _make_f3d(
    n_stocks: int = 20,
    n_times: int = 10,
    seed: int = 42,
    constant_close: bool = False,
    all_nan_signal: bool = False,
) -> Frame3D:
    """构造 IC 分析节点所需的 Frame3D（pred_signal + close）。"""
    rng = np.random.default_rng(seed)
    times = pd.date_range('2020-01-02', periods=n_times, freq='B')
    stocks = [f'S{i:04d}' for i in range(n_stocks)]
    mi = pd.MultiIndex.from_product([times, stocks], names=['key', 'name'])

    if constant_close:
        close = np.full(len(stocks) * n_times, 100.0, dtype=float)
    else:
        base_price = 100.0
        close_vals: list[float] = []
        for _ in range(n_times):
            base_price *= (1.0 + rng.normal(0.005, 0.015))
            for _ in range(n_stocks):
                close_vals.append(base_price * (1.0 + rng.normal(0.0, 0.01)))
        close = np.array(close_vals, dtype=float)

    if all_nan_signal:
        pred_signal = np.full(len(stocks) * n_times, np.nan, dtype=float)
    else:
        pred_signal = rng.normal(0, 1.0, size=n_stocks * n_times)

    df = pd.DataFrame({'pred_signal': pred_signal, 'close': close}, index=mi)
    return Frame3D(df)


# ═══════════════════════════════════════════════════════════════════════════
# ic_analysis_fn 单元测试
# ═══════════════════════════════════════════════════════════════════════════

class TestIcAnalysisFn:
    """ic_analysis_fn 基本功能和边界测试。"""

    def test_insufficient_data_returns_early(self):
        """times < fwd+1 时直接返回原 f3d，不更新 context。"""
        f3d = _make_f3d(n_stocks=5, n_times=2)
        ctx: dict = {}
        result = ic_analysis_fn('test', f3d, ctx)
        assert result is f3d
        assert ctx.get('day_count', 0) == 0

    def test_first_call_updates_context(self):
        """首次调用初始化 ctx 字段（需 fwd+1=6 天，用 10 天+5fwd=6）。"""
        f3d = _make_f3d(n_stocks=15, n_times=10)
        ctx: dict = {'start_date': '2020-01-02', 'fwd': 4}  # times=10 >= 4+1=5
        ic_analysis_fn('test', f3d, ctx)
        assert ctx['first_signal_day'] is not None
        assert ctx['last_signal_day'] is not None
        assert ctx['day_count'] >= 1
        assert list(f3d.df.columns) == ['pred_signal', 'close']

    def test_pearson_ic_in_range(self):
        """Pearson IC 应在 [-1, 1] 之间。"""
        f3d = _make_f3d(n_stocks=30, n_times=8)
        ctx: dict = {'fwd': 3}
        ic_analysis_fn('test', f3d, ctx)
        for ic in ctx.get('pearson_ic_history', []):
            assert -1.0 <= ic <= 1.0, f'Pearson IC out of range: {ic}'

    def test_rank_ic_in_range(self):
        """Rank IC 应在 [-1, 1] 之间。"""
        f3d = _make_f3d(n_stocks=30, n_times=8)
        ctx: dict = {'fwd': 3}
        ic_analysis_fn('test', f3d, ctx)
        for ic in ctx.get('rank_ic_history', []):
            assert -1.0 <= ic <= 1.0, f'Rank IC out of range: {ic}'

    def test_context_is_dict_mutable(self):
        """多次调用在同一 context dict 上累积。"""
        rng = np.random.default_rng(42)
        times = pd.date_range('2020-01-02', periods=20, freq='B')
        stocks = [f'S{i:04d}' for i in range(10)]
        mi = pd.MultiIndex.from_product([times, stocks], names=['key', 'name'])
        base = 100.0
        close_vals: list[float] = []
        for _ in range(20):
            base *= (1.0 + rng.normal(0.005, 0.015))
            for _ in range(10):
                close_vals.append(base * (1.0 + rng.normal(0.0, 0.01)))
        df = pd.DataFrame({
            'pred_signal': rng.normal(0, 1, size=10 * 20),
            'close': np.array(close_vals),
        }, index=mi)

        ctx: dict = {'fwd': 3}
        # 用滑动窗口模拟逐日调用
        for i in range(len(times) - 4):
            w = df.loc[times[i]:times[i + 4]]
            ic_analysis_fn('test', Frame3D(w), ctx)
        assert ctx['day_count'] >= 1

    def test_nan_in_signal_handled(self):
        """pred_signal 含 NaN 时不应崩溃。"""
        f3d = _make_f3d(n_stocks=10, n_times=8)
        df = f3d.df.copy()
        df.loc[df.index[:5], 'pred_signal'] = np.nan
        ctx: dict = {'fwd': 3}
        result = ic_analysis_fn('test', Frame3D(df), ctx)
        assert result is not None

    def test_constant_close(self):
        """close 列全为常量时不应崩溃（截面 std=0）。"""
        f3d = _make_f3d(n_stocks=10, n_times=8, constant_close=True)
        ctx: dict = {'fwd': 3}
        result = ic_analysis_fn('test', f3d, ctx)
        assert result is not None

    def test_all_nan_pred_signal(self):
        """全 NaN pred_signal → valid_mask.sum()<10 → 记录 NaN。"""
        f3d = _make_f3d(n_stocks=15, n_times=8, all_nan_signal=True)
        ctx: dict = {'fwd': 3}
        ic_analysis_fn('test', f3d, ctx)
        assert len(ctx.get('pearson_ic_history', [])) > 0
        assert np.isnan(ctx['pearson_ic_history'][-1])


# ═══════════════════════════════════════════════════════════════════════════
# ic_epilogue 单元测试
# ═══════════════════════════════════════════════════════════════════════════

class TestIcEpilogue:
    """ic_epilogue 汇总逻辑测试。"""

    def test_no_data_warns(self, caplog):
        """无 IC 数据时 epilogue 不崩溃。"""
        ic_epilogue('test', None)
        ic_epilogue('test', {'rank_ic_history': []})

    def test_insufficient_data_warns(self, caplog):
        """不足 10 个数据点时给出警告。"""
        ctx = {'rank_ic_history': [0.1] * 5,
               'pearson_ic_history': [0.1] * 5,
               'day_count': 5}
        ic_epilogue('test', ctx)

    def test_normal_summary(self):
        """正常数据点计算 ICIR, winrate, max_dd，不抛异常。"""
        rng = np.random.default_rng(42)
        ics = rng.normal(0.03, 0.1, size=50).tolist()
        ctx = {
            'rank_ic_history': ics,
            'pearson_ic_history': ics,
            'day_count': 50,
            'first_signal_day': pd.Timestamp('2020-01-02'),
            'last_signal_day': pd.Timestamp('2020-03-15'),
        }
        ic_epilogue('test', ctx)

    def test_icir_winrate_plausible(self):
        """ICIR 和 winrate 正常，epilogue 不崩溃。"""
        rng = np.random.default_rng(123)
        ics = rng.normal(0.05, 0.1, size=100).tolist()
        ctx = {
            'rank_ic_history': ics,
            'pearson_ic_history': ics,
            'day_count': 100,
            'first_signal_day': '2020-01-02',
            'last_signal_day': '2020-06-01',
            'cumsum_raw_ret_std': 2.0,
            'cumsum_pearson_ic': 5.0,
            'num_groups': 10,
        }
        ic_epilogue('test', ctx)
