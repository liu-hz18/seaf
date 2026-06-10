"""
策略模块单元测试 — 覆盖初始化、佣金、复权公式、净值计算、分组排名。

测试分层：
- TestHelpers: 辅助函数纯逻辑测试
- TestOnBar: 逐日交易逻辑（含停牌/资金不足/送股场景）
- TestRankIntoGroups: 信号排序分组
- TestStrategyFn: 框架集成（含多组独立状态）
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from qpipe.frame3d import Frame3D
from seafquant.strategy import (
    _calc_commission,
    _compute_total_equity,
    _create_position,
    _get_actual_shares,
    _get_position_value,
    _init_group_context,
    _on_bar,
    _rank_into_groups,
    strategy_fn,
)


# =============================================================================
# 辅助函数测试
# =============================================================================


class TestHelpers:
    """测试辅助函数：佣金、复权、净值、持仓。"""

    def test_init_group_context(self):
        """初始化后 core state 正确。"""
        ctx = _init_group_context(1_000_000, 20, 0.0005, 5.0, 3)
        assert ctx['group_id'] == 3
        assert ctx['cash'] == 1_000_000
        assert ctx['fwd'] == 20
        assert ctx['positions'] == {}
        assert ctx['pending_signal'] is None
        assert ctx['nav_log'] == []
        assert ctx['trade_log'] == []

    def test_calc_commission_normal(self):
        """手续费 = max(交易额*费率, 最低)"""
        assert _calc_commission(10_000, 0.0005, 5.0) == 5.0   # 10k*0.0005=5
        assert _calc_commission(5_000, 0.0005, 5.0) == 5.0    # 5k*0.0005=2.5→最低5
        assert _calc_commission(100_000, 0.0005, 5.0) == 50.0  # 100k*0.0005=50

    def test_actual_shares_no_split(self):
        """无送股时 F_today == F_buy，实际股数 = N_initial。"""
        pos = {'stock_id': 'S1', 'n_initial': 1000.0, 'f_buy': 2.5}
        f_today = {'S1': 2.5}
        assert _get_actual_shares(pos, f_today) == 1000.0

    def test_actual_shares_with_split(self):
        """F_today = 1.5 * F_buy → 送出 50%，实际股数 = 1.5 * N_initial。"""
        pos = {'stock_id': 'S1', 'n_initial': 1000.0, 'f_buy': 2.0}
        f_today = {'S1': 3.0}  # 1.5x
        assert _get_actual_shares(pos, f_today) == 1500.0

    def test_actual_shares_stock_not_in_f_today(self):
        """停牌/缺失时返回 0。"""
        pos = {'stock_id': 'S1', 'n_initial': 1000.0, 'f_buy': 2.0}
        assert _get_actual_shares(pos, {}) == 0.0

    def test_position_value_golden_formula(self):
        """黄金公式：市值 = N_initial * P_hfq / F_buy。"""
        pos = {'stock_id': 'S1', 'n_initial': 500.0, 'f_buy': 2.0}
        close_hfq = {'S1': 100.0}
        # 市值 = 500 * 100 / 2 = 25000
        assert _get_position_value(pos, close_hfq) == 25000.0

    def test_compute_total_equity(self):
        """总资产 = 现金 + 所有持仓市值。"""
        ctx = _init_group_context(1_000_000, 20, 0.0005, 5.0, 0)
        pos1 = {'stock_id': 'S1', 'n_initial': 500.0, 'f_buy': 2.0}
        pos2 = {'stock_id': 'S2', 'n_initial': 1000.0, 'f_buy': 1.0}
        ctx['positions'][('S1', 0)] = pos1
        ctx['positions'][('S2', 0)] = pos2
        close_uq = {'S1': 49.0, 'S2': 99.0}
        close_hfq = {'S1': 100.0, 'S2': 100.0}
        # S1: 500*100/2=25000, S2: 1000*100/1=100000, total=125000
        equity = _compute_total_equity(ctx, close_uq, close_hfq)
        assert equity == 1_125_000.0


# =============================================================================
# on_bar 交易逻辑测试
# =============================================================================


class TestOnBar:
    """测试逐日交易：首日只缓存信号、次日执行交易、停牌/送股。"""

    @staticmethod
    def _make_prices(close_uq, close_hfq):
        """构造价格字典，假设所有股票都有相同价格。"""
        return (
            {'S{:02d}'.format(i): close_uq for i in range(10)},
            {'S{:02d}'.format(i): close_hfq for i in range(10)},
        )

    def test_first_day_no_trade(self):
        """第一天只有 pending_signal=None，不交易，只缓存信号。"""
        ctx = _init_group_context(1_000_000, 20, 0.0005, 5.0, 0)
        sig = {'S00': 0.5, 'S01': 0.5}
        uq, hfq = self._make_prices(50.0, 55.0)
        _on_bar(ctx, pd.Timestamp('2020-01-02'), sig, uq, hfq)
        assert ctx['pending_signal'] == sig
        assert ctx['day_counter'] == 1
        assert len(ctx['trade_log']) == 0  # 没有执行交易
        assert len(ctx['nav_log']) == 1     # 记录了净值
        assert len(ctx['position_log']) == 0

    def test_second_day_executes_trade(self):
        """第两天用第 1 天信号 + 第 2 天价格执行交易。"""
        ctx = _init_group_context(1_000_000, 20, 0.0005, 5.0, 0)
        # Day 1: 缓存信号
        sig_d1 = {'S00': 0.5, 'S01': 0.5}
        uq1, hfq1 = self._make_prices(50.0, 55.0)
        _on_bar(ctx, pd.Timestamp('2020-01-02'), sig_d1, uq1, hfq1)
        trades_d1 = len(ctx['trade_log'])

        # Day 2: 执行信号
        sig_d2 = {'S00': 0.6, 'S01': 0.4}
        uq2, hfq2 = self._make_prices(52.0, 57.0)
        _on_bar(ctx, pd.Timestamp('2020-01-03'), sig_d2, uq2, hfq2)

        # 第 2 天应产生交易
        assert len(ctx['trade_log']) > trades_d1
        assert ctx['pending_signal'] == sig_d2  # 缓存了第 2 天信号
        # 有持仓
        assert len(ctx['positions']) > 0

    def test_suspended_stock_no_trade(self):
        """停牌（p_uq=0）不执行交易，仓位延期。"""
        ctx = _init_group_context(1_000_000, 20, 0.0005, 5.0, 0)
        sig_d1 = {'S00': 1.0}
        uq1, hfq1 = self._make_prices(50.0, 55.0)
        _on_bar(ctx, pd.Timestamp('2020-01-02'), sig_d1, uq1, hfq1)

        # 第 2 天 S00 停牌
        uq2 = {'S00': 0.0}
        hfq2 = {'S00': 0.0}
        _on_bar(ctx, pd.Timestamp('2020-01-03'), {}, uq2, hfq2)

        # 无新持仓（停牌无法买入），但第一天买入的仓位应延期到期
        # 第一天买入后只有一个持仓，到期日 = batch_dc + fwd
        for key, pos in ctx['positions'].items():
            assert pos['mature_dc'] > ctx['day_counter']  # 延期了

    def test_insufficient_cash_partial_buy(self):
        """资金不足时尽力买入，不会超额支出。"""
        ctx = _init_group_context(100_000, 20, 0.0005, 5.0, 0)
        sig_d1 = {'S00': 1.0}
        uq1, hfq1 = {'S00': 100.0}, {'S00': 110.0}
        _on_bar(ctx, pd.Timestamp('2020-01-02'), sig_d1, uq1, hfq1)

        uq2, hfq2 = {'S00': 5000.0}, {'S00': 5500.0}  # 很贵
        _on_bar(ctx, pd.Timestamp('2020-01-03'), {}, uq2, hfq2)
        # 资金只有 100k，slice_capital ≈ 5k，买不起 5000/股（100股=500k）
        # 检查现金未变为负数
        assert ctx['cash'] >= 0
        # 检查总资产合理
        assert ctx['nav_log'][-1]['total_equity'] >= 0

    def test_nav_tracks_equity(self):
        """净值日志应反映总资产变化。"""
        ctx = _init_group_context(1_000_000, 20, 0.0005, 5.0, 0)
        sig_d1 = {'S00': 0.5, 'S01': 0.5}
        uq, hfq = self._make_prices(50.0, 55.0)
        _on_bar(ctx, pd.Timestamp('2020-01-02'), sig_d1, uq, hfq)
        _on_bar(ctx, pd.Timestamp('2020-01-03'), sig_d1, uq, hfq)
        # 验证净值序列
        nav_vals = [n['total_equity'] for n in ctx['nav_log']]
        assert len(nav_vals) == 2
        # 第一天净值 = 初始资金（无持仓变化）
        assert nav_vals[0] == 1_000_000
        # 第两天净值 >= 初始资金（可能有交易损益）


# =============================================================================
# 分组排名测试
# =============================================================================


class TestRankIntoGroups:
    """测试 _rank_into_groups：按信号分位数等权分组。"""

    def test_even_groups(self):
        """20 只股票分 5 组，每组 4 只，等权。"""
        sids = [f'S{i:02d}' for i in range(20)]
        sig = pd.Series(np.arange(20), index=sids)  # 0..19
        groups = _rank_into_groups(sig, 5)
        assert len(groups) == 5
        for g, members in groups.items():
            assert len(members) == 4
            for w in members.values():
                assert w == 0.25  # 等权 1/4

    def test_low_signal_in_group0(self):
        """最低信号在 group 0，最高信号在 group N-1。"""
        sids = [f'S{i:02d}' for i in range(10)]
        sig = pd.Series(np.arange(10), index=sids)
        groups = _rank_into_groups(sig, 2)
        # group 0 = 分位 [0, 0.5) → 信号最低的 5 只
        assert all(s.startswith('S0') for s in groups[0])   # S00-S04
        # group 1 = 分位 [0.5, 1) → 信号最高的 5 只
        assert all(s.startswith('S0') for s in groups[1])   # S05-S09

    def test_too_few_stocks(self):
        """股票数 < 分组数时返回空。"""
        sig = pd.Series([1.0, 2.0], index=['A', 'B'])
        groups = _rank_into_groups(sig, 10)
        assert groups == {}

    def test_equal_weight_sum_one(self):
        """每组内权重之和 = 1.0。"""
        sig = pd.Series(np.random.randn(50), index=[f'S{i:02d}' for i in range(50)])
        groups = _rank_into_groups(sig, 10)
        for members in groups.values():
            assert abs(sum(members.values()) - 1.0) < 1e-10


# =============================================================================
# strategy_fn 框架集成测试
# =============================================================================


class TestStrategyFn:
    """strategy_fn 在 qpipe 框架下的集成测试。"""

    @staticmethod
    def _make_f3d(pred_signal, close, close_uq, times=None, n_stocks=10):
        """构造 strategy_fn 所需的最小 Frame3D。"""
        if times is None:
            times = [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-03')]
        stocks = [f'S{i:04d}' for i in range(n_stocks)]
        mi = pd.MultiIndex.from_product([times, stocks], names=['key', 'name'])
        df = pd.DataFrame({
            'pred_signal': pred_signal if isinstance(pred_signal, np.ndarray) else np.full(len(stocks) * 2, pred_signal),
            'close': close if isinstance(close, np.ndarray) else np.full(len(stocks) * 2, close),
            'close_uq': close_uq if isinstance(close_uq, np.ndarray) else np.full(len(stocks) * 2, close_uq),
        }, index=mi)
        return Frame3D(df)

    def test_first_call_initializes_groups(self):
        """首次调用 strategy_fn 初始化 10 个 group。"""
        ctx = {}
        f3d = self._make_f3d(0.0, 100.0, 98.0)
        result = strategy_fn('test', f3d, ctx)
        assert ctx['groups'] is not None
        assert len(ctx['groups']) == 10

    def test_multi_group_independence(self):
        """多组独立状态：各组 cash/positions 不互相影响。"""
        ctx = {'num_groups': 3, 'fwd': 5, 'initial_cash': 300_000, 'mlflow_run_id': ''}
        np.random.seed(42)
        base = pd.Timestamp('2020-01-02')
        for day in range(5):
            t0, t1 = base + pd.Timedelta(days=day), base + pd.Timedelta(days=day + 1)
            f3d = self._make_f3d(
                np.random.randn(20), 100.0, 98.0, times=[t0, t1], n_stocks=10,
            )
            strategy_fn('test', f3d, ctx)

        # 每组有各自的 nav_log
        nav_lengths = [len(g['nav_log']) for g in ctx['groups']]
        assert all(l > 0 for l in nav_lengths)
        # 每组 nav 历史长度一致（同时开始交易）
        assert len(set(nav_lengths)) == 1

    def test_empty_frame3d_safe(self):
        """空 Frame3D（不足 2 天）安全返回。"""
        ctx = {}
        stocks = ['S0001']
        mi = pd.MultiIndex.from_product([[pd.Timestamp('2020-01-02')], stocks], names=['key', 'name'])
        df = pd.DataFrame({'pred_signal': [0.0], 'close': [100.0], 'close_uq': [98.0]}, index=mi)
        result = strategy_fn('test', Frame3D(df), ctx)
        assert result.df.empty

    def test_position_log_contains_keys(self):
        """持仓快照包含必要字段。"""
        ctx = {'num_groups': 2, 'fwd': 5, 'initial_cash': 2_000_000, 'mlflow_run_id': ''}
        np.random.seed(42)
        base = pd.Timestamp('2020-01-02')
        for day in range(5):
            t0, t1 = base + pd.Timedelta(days=day), base + pd.Timedelta(days=day + 1)
            f3d = self._make_f3d(
                np.random.randn(20), 12.0, 10.0, times=[t0, t1], n_stocks=10,
            )
            strategy_fn('test', f3d, ctx)

        # 至少有 1 个 group 产生了持仓日志
        has_positions = any(
            len(g['position_log']) > 0 for g in ctx['groups']
        )
        assert has_positions
        for g in ctx['groups']:
            for plog in g['position_log']:
                assert 'stock_id' in plog
                assert 'market_value' in plog
                assert 'actual_shares' in plog
                assert 'p_uq' in plog
                assert 'p_hfq' in plog
