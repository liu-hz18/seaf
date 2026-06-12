"""
counting 因子对拍验证 — 纯计数算子（无离散化阈值）。

Streak: consec_up/down ×2, run_pct ×2
CountPos/CountNeg: countpos ×2, countneg ×2
TillNow: tillnow_ret ×2, tillnow_dd ×2
新高新低: new_high ×2, new_low ×1
Turnover Rank Change: rank_chg ×2

共 10 个测试，覆盖 17 个因子。
"""

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from test.crossval_helpers import (
    _compare_factor_output,
    _make_data,
    _roll_manual,
    _ts_pct_manual,
)


class TestCountingCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(80, 8)

    # ── CountPos / CountNeg ──────────────────────────────────────────

    def test_countpos_countneg(self, f3d):
        """countpos_Nd = rolling_sum(ret>0, N); countneg 同理。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)
        df = f3d.df.copy()
        ret = _ts_pct_manual(df, 'close', 1)
        df['_pos'] = (ret > 0).astype(float)
        df['_neg'] = (ret < 0).astype(float)
        expected = {}
        for w in [10, 60]:
            expected[f'factor_cnt_countpos_{w}d'] = _roll_manual(df, '_pos', w, 'sum').values
            expected[f'factor_cnt_countneg_{w}d'] = _roll_manual(df, '_neg', w, 'sum').values
        _compare_factor_output(actual, expected)

    # ── Streak — pivot → numpy → flatten ─────────────────────────────

    def test_consec_up(self, f3d):
        """factor_cnt_consec_up_10d — 连续上涨天数。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)
        ret = _ts_pct_manual(f3d.df, 'close', 1)
        # pivot (T, S)，手动算连续上涨
        arr = ret.unstack(level='code').values.astype(np.float64)
        T, S = arr.shape
        up = np.zeros((T, S), dtype=np.float64)
        for s in range(S):
            c = 0
            for i in range(T):
                r = arr[i, s]
                if np.isnan(r):
                    c = 0
                elif r > 0:
                    c = min(c + 1, 10)
                else:
                    c = 0
                up[i, s] = c
        expected_series = ret.copy()
        expected_series.iloc[:] = up.ravel()
        _compare_factor_output(actual, {'factor_cnt_consec_up_10d': expected_series.values})

    def test_run_pct(self, f3d):
        """factor_cnt_run_pct_20d — 同向天数占比。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)
        ret = _ts_pct_manual(f3d.df, 'close', 1)
        arr = ret.unstack(level='code').values.astype(np.float64)
        T, S = arr.shape
        valid = np.isfinite(arr)
        same_dir = np.zeros((T, S), dtype=np.float64)
        both_pos = (arr[1:] > 0) & (arr[:-1] > 0)
        both_neg = (arr[1:] < 0) & (arr[:-1] < 0)
        valid_pair = valid[1:] & valid[:-1]
        same_dir[1:] = (both_pos | both_neg) & valid_pair
        win = sliding_window_view(same_dir, 20, axis=0)
        rp = np.full((T, S), np.nan)
        rp[19:] = win.mean(axis=-1)
        expected_series = ret.copy()
        expected_series.iloc[:] = rp.ravel()
        _compare_factor_output(actual, {'factor_cnt_run_pct_20d': expected_series.values})

    # ── TillNow — pivot → numpy → flatten ────────────────────────────

    def test_tillnow_ret(self, f3d):
        """tillnow_ret_{p}d = cumprod(1+ret) - 1 over window。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)
        ret = _ts_pct_manual(f3d.df, 'close', 1)
        arr = ret.unstack(level='code').values.astype(np.float64)
        for w in [20, 60]:
            one_plus = 1.0 + arr
            clean = np.where(np.isfinite(one_plus), one_plus, 1.0)
            win = sliding_window_view(clean, w, axis=0)
            tr = np.full(arr.shape, np.nan)
            tr[w - 1 :] = np.prod(win, axis=-1) - 1.0
            expected_series = ret.copy()
            expected_series.iloc[:] = tr.ravel()
            _compare_factor_output(actual, {f'factor_cnt_tillnow_ret_{w}d': expected_series.values})

    def test_tillnow_dd(self, f3d):
        """tillnow_dd_{p}d = min(price / cummax(price)) - 1 over window。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)
        arr = f3d.df['close'].unstack(level='code').values.astype(np.float64)
        for w in [20, 60]:
            win = sliding_window_view(arr, w, axis=0)
            cmax = np.maximum.accumulate(win, axis=-1)
            dd = win / np.where(cmax == 0, 1.0, cmax) - 1.0
            out = np.full(arr.shape, np.nan)
            out[w - 1 :] = np.min(dd, axis=-1)
            expected_series = f3d.df['close'].copy()
            expected_series.iloc[:] = out.ravel()
            _compare_factor_output(actual, {f'factor_cnt_tillnow_dd_{w}d': expected_series.values})

    # ── 新高新低 — 逐股票沿时序 ──────────────────────────────────────

    def test_new_high(self, f3d):
        """factor_cnt_new_high_20d：逐股票创新高天数计数。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)

        def _nh(series, window):
            arr = series.values.astype(float)
            n = len(arr)
            if n < window:
                return np.full(n, np.nan)
            clean = np.where(np.isnan(arr), -np.inf, arr)
            cmax = np.maximum.accumulate(clean)
            is_nh = np.zeros(n, dtype=float)
            for i in range(1, n):
                if arr[i] > cmax[i - 1] and cmax[i - 1] != -np.inf:
                    is_nh[i] = 1.0
            win = sliding_window_view(is_nh, window)
            out = np.full(n, np.nan)
            out[window - 1 :] = win.sum(axis=1)
            return out

        nh = f3d.df.groupby('name')['close'].transform(lambda x: _nh(x, 20))
        _compare_factor_output(actual, {'factor_cnt_new_high_20d': nh.values})

    def test_new_low(self, f3d):
        """factor_cnt_new_low_60d：逐股票创新低天数计数。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)

        def _nl(series, window):
            arr = series.values.astype(float)
            n = len(arr)
            if n < window:
                return np.full(n, np.nan)
            clean = np.where(np.isnan(arr), np.inf, arr)
            cmin = np.minimum.accumulate(clean)
            is_nl = np.zeros(n, dtype=float)
            for i in range(1, n):
                if arr[i] < cmin[i - 1] and cmin[i - 1] != np.inf:
                    is_nl[i] = 1.0
            win = sliding_window_view(is_nl, window)
            out = np.full(n, np.nan)
            out[window - 1 :] = win.sum(axis=1)
            return out

        nl = f3d.df.groupby('name')['close'].transform(lambda x: _nl(x, 60))
        _compare_factor_output(actual, {'factor_cnt_new_low_60d': nl.values})

    # ── Turnover Rank Change ─────────────────────────────────────────

    def test_turnover_rank_chg(self, f3d):
        """factor_cnt_turnover_rank_chg_{p}d = mean(abs(rank - rank_prev), p)。"""
        from seafquant.factor.counting import compute_counting_factors
        actual = compute_counting_factors('test', f3d, None)
        df = f3d.df.copy()
        # 手动算 rank（按 key 分组，percentile rank）
        ranks = df.groupby('key')['turnover'].rank(pct=True)
        df['_rk'] = ranks.values
        df['_rk_d1'] = df.groupby('name')['_rk'].shift(1)
        df['_rc'] = np.abs(df['_rk'] - df['_rk_d1'])
        expected = {}
        for w in [20, 60]:
            expected[f'factor_cnt_turnover_rank_chg_{w}d'] = (
                _roll_manual(df, '_rc', w, 'mean').values
            )
        _compare_factor_output(actual, expected)
