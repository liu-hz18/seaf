"""
quality_autocorr 因子对拍验证 — 自相关 + 尾部风险。
每测试以 scope='class' 的 fixture params 重复 10 组随机数据。
"""

import pytest

from test.crossval_helpers import _compare_factor_output, _make_data


class TestQualityAutocorrCrossVal:

    @pytest.fixture(scope='class', params=range(10))
    def f3d(self, request):
        return _make_data(80, 8)

    def test_autocorr(self, f3d):
        """factor_qa_autocorr_{p}d = rolling_corr(ret, ret.shift(1), p)，min_periods=10/20。"""
        from seafquant.factor.quality_autocorr import compute_quality_autocorr_factors
        actual = compute_quality_autocorr_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_ret'] = df.groupby('name')['close'].pct_change(1)
        mp_map = {20: 10, 60: 20}
        expected = {}
        for p in [20, 60]:
            expected[f'factor_qa_autocorr_{p}d'] = df.groupby('name')['_ret'].transform(
                lambda x, _p=p, _mp=mp_map[p]: x.rolling(_p, min_periods=_mp).corr(x.shift(1))
            ).values
        _compare_factor_output(actual, expected)

    def test_tail_risk(self, f3d):
        """factor_qa_tail_risk_{p}d = -rolling_quantile(ret|ret<0, 0.05, p)，min_periods=20/40。"""
        from seafquant.factor.quality_autocorr import compute_quality_autocorr_factors
        actual = compute_quality_autocorr_factors('test', f3d, None)
        df = f3d.df.copy()
        df['_ret'] = df.groupby('name')['close'].pct_change(1)
        mp_map = {60: 20, 120: 40}
        expected = {}
        for p in [60, 120]:
            expected[f'factor_qa_tail_risk_{p}d'] = df.groupby('name')['_ret'].transform(
                lambda x, _p=p, _mp=mp_map[p]: -x.where(x < 0).rolling(_p, min_periods=_mp).quantile(0.05)
            ).values
        _compare_factor_output(actual, expected)
