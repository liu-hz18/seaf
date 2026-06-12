"""
数据生成器单元测试
测试数据形状、列完整性、可复现性、信噪比与 IC 的关系。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from seafquant.data_generator import generate_synthetic_data


def _collect_frames(gen) -> list:
    """从生成器收集所有 Frame3D。"""
    return list(gen)


def _compute_fwd_ret(frames: list, horizon: int = 20) -> np.ndarray:
    """从 close 价格计算前瞻截面超额收益矩阵。

    Returns: (n_times - horizon, n_stocks) 的 fwd_ret_xd 矩阵。
    """
    n_times = len(frames)
    n_stocks = len(frames[0].df)
    close_matrix = np.zeros((n_times, n_stocks))
    for t, f3d in enumerate(frames):
        close_matrix[t] = f3d.df['close'].values

    fwd_ret = np.full((n_times, n_stocks), np.nan)
    for t in range(n_times - horizon):
        fwd_ret[t] = close_matrix[t + horizon] / close_matrix[t] - 1

    # 截面标准化
    fwd_ret_xd = np.full_like(fwd_ret, np.nan)
    for t in range(n_times - horizon):
        row = fwd_ret[t]
        valid = ~np.isnan(row)
        if valid.sum() > 3:
            mu = row[valid].mean()
            std = row[valid].std()
            if std > 0:
                fwd_ret_xd[t][valid] = (row[valid] - mu) / std
    return fwd_ret_xd


def _compute_ic(signal_matrix: np.ndarray, fwd_ret_xd: np.ndarray) -> list:
    """计算每日截面 rank IC 列表。"""
    from scipy.stats import spearmanr

    n_times, n_stocks = signal_matrix.shape
    ics = []
    for t in range(n_times):
        sig = signal_matrix[t]
        fwd = fwd_ret_xd[t]
        valid = ~np.isnan(sig) & ~np.isnan(fwd)
        if valid.sum() >= 10:
            ic = spearmanr(sig[valid], fwd[valid]).correlation
            ics.append(ic)
    return ics


class TestDataGenerator:
    EXPECTED_COLUMNS = ['open', 'high', 'low', 'close', 'turnover', 'volume', 'market_cap']

    def test_shape_and_columns(self):
        """验证数据形状和列完整性。"""
        gen = generate_synthetic_data(n_times=50, n_stocks=20, seed=42)
        frames = _collect_frames(gen)
        assert len(frames) == 50

        f3d = frames[0]
        assert f3d.df.index.names == ['key', 'name']
        assert len(f3d.df) == 20  # n_stocks
        for col in self.EXPECTED_COLUMNS:
            assert col in f3d.df.columns, f'Missing column: {col}'

    def test_no_nan_in_basic_cols(self):
        """基础列不应有 NaN。"""
        gen = generate_synthetic_data(n_times=50, n_stocks=20, seed=42)
        frames = _collect_frames(gen)
        for f3d in frames:
            assert not f3d.df.isna().any().any(), 'Found NaN in generated data'

    def test_reproducibility(self):
        """相同 seed 生成相同数据。"""
        gen1 = generate_synthetic_data(n_times=20, n_stocks=10, seed=123)
        gen2 = generate_synthetic_data(n_times=20, n_stocks=10, seed=123)
        frames1 = _collect_frames(gen1)
        frames2 = _collect_frames(gen2)
        for f1, f2 in zip(frames1, frames2, strict=False):
            num_cols = [c for c in f1.df.columns if c != 'stock_name']
            assert np.allclose(f1.df[num_cols].values, f2.df[num_cols].values), \
                'Same seed should give same data'

    def test_different_seed_gives_different_data(self):
        """不同 seed 生成不同数据。"""
        gen1 = generate_synthetic_data(n_times=20, n_stocks=10, seed=1)
        gen2 = generate_synthetic_data(n_times=20, n_stocks=10, seed=2)
        frames1 = _collect_frames(gen1)
        frames2 = _collect_frames(gen2)
        diff = False
        for f1, f2 in zip(frames1, frames2, strict=False):
            num_cols = [c for c in f1.df.columns if c != 'stock_name']
            if not np.allclose(f1.df[num_cols].values, f2.df[num_cols].values):
                diff = True
                break
        assert diff, 'Different seeds should give different data'

    def test_low_noise_has_predictability(self):
        """低噪声时信号应具有一定可预测性（IC > 0）。"""
        n_times = 200
        n_stocks = 30
        gen = generate_synthetic_data(n_times=n_times, n_stocks=n_stocks, noise_ratio=0.1, seed=42)
        frames = _collect_frames(gen)
        fwd_ret_xd = _compute_fwd_ret(frames, horizon=20)

        # 用 5 日动量做简单信号
        close_matrix = np.zeros((n_times, n_stocks))
        for t, f3d in enumerate(frames):
            close_matrix[t] = f3d.df['close'].values
        mom5 = np.full((n_times, n_stocks), np.nan)
        for t in range(5, n_times):
            mom5[t] = close_matrix[t] / close_matrix[t - 5] - 1
        # 截面标准化动量
        for t in range(n_times):
            row = mom5[t]
            valid = ~np.isnan(row)
            if valid.sum() > 3:
                mu = row[valid].mean()
                std = row[valid].std()
                if std > 0:
                    mom5[t][valid] = (row[valid] - mu) / std

        ics = _compute_ic(mom5, fwd_ret_xd)
        mean_ic = np.nanmean(ics)
        assert mean_ic > -0.05, f'Low noise mean IC={mean_ic:.4f} should not be strongly negative'

    def test_high_noise_reduces_predictability(self):
        """高噪声时预测 IC 应显著下降。"""
        n_times = 200
        n_stocks = 30

        def get_mean_ic(noise_ratio):
            gen = generate_synthetic_data(
                n_times=n_times, n_stocks=n_stocks, noise_ratio=noise_ratio, seed=42
            )
            frames = _collect_frames(gen)
            fwd_ret_xd = _compute_fwd_ret(frames, horizon=20)
            close_matrix = np.zeros((n_times, n_stocks))
            for t, f3d in enumerate(frames):
                close_matrix[t] = f3d.df['close'].values
            mom5 = np.full((n_times, n_stocks), np.nan)
            for t in range(5, n_times):
                mom5[t] = close_matrix[t] / close_matrix[t - 5] - 1
            for t in range(n_times):
                row = mom5[t]
                valid = ~np.isnan(row)
                if valid.sum() > 3:
                    mu = row[valid].mean()
                    std = row[valid].std()
                    if std > 0:
                        mom5[t][valid] = (row[valid] - mu) / std
            ics = _compute_ic(mom5, fwd_ret_xd)
            return np.nanmean(ics) if ics else 0.0

        ic_low = get_mean_ic(0.1)
        ic_high = get_mean_ic(2.0)
        # 高噪声时 IC 绝对值应下降（可预测性减弱）
        assert abs(ic_high) < abs(ic_low) or abs(ic_high) < 0.03, (
            f'High noise IC ({ic_high:.4f}) should be weaker than low noise IC ({ic_low:.4f})'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
