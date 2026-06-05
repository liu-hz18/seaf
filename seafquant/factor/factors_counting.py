"""
计数类因子（Volume/Rank）— 9 个因子，基于成交量放量/换手率排名变化/振幅突破等。
streak/新高新低已拆分至 counting_streak 节点并行执行。

优化：使用直接 pandas groupby rolling 减少 f3d.copy() 深拷贝开销。
"""
import numpy as np
import logging
from qpipe.frame3d import Frame3D


def compute_counting_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 9 个计数类因子（vol/rank/amp）。"""
    result = f3d.copy()
    close = f3d.df['close']
    ret = f3d.ts_pct_change('close', 1).df['close']

    # ---- 辅助：直接 pandas groupby rolling，避免 Frame3D 不可变 copy ----
    def _add_rolling(df, src_col, dst_col, window, agg):
        """在 df 上直接做 groupby rolling 聚合，然后赋值到 dst_col。"""
        rolled = df.groupby('name')[src_col].rolling(window, min_periods=max(1, window // 2)).agg(agg)
        df[dst_col] = rolled.values

    # ================================================================
    # 因子 1-3: 成交量放量/缩量
    # ================================================================
    volume = f3d.df['volume']
    df = result.df
    df['_vol'] = volume
    _add_rolling(df, '_vol', '_vol_ma20', 20, 'mean')
    vol_ma20 = df['_vol_ma20']

    df['factor_cnt_vol_spike_20d'] = (volume > 1.5 * vol_ma20).astype(float)
    _add_rolling(df, 'factor_cnt_vol_spike_20d', 'factor_cnt_vol_spike_20d', 20, 'sum')

    df['factor_cnt_vol_spike_60d'] = (volume > 1.5 * vol_ma20).astype(float)
    _add_rolling(df, 'factor_cnt_vol_spike_60d', 'factor_cnt_vol_spike_60d', 60, 'sum')

    df['factor_cnt_vol_shrink_20d'] = (volume < 0.5 * vol_ma20).astype(float)
    _add_rolling(df, 'factor_cnt_vol_shrink_20d', 'factor_cnt_vol_shrink_20d', 20, 'sum')

    # ================================================================
    # 因子 4-5: 换手率排名变化
    # ================================================================
    to_rank = result.cs_rank('turnover').df['turnover']
    df['_rk'] = to_rank
    df['_rk_d1'] = df.groupby('name')['_rk'].shift(1)
    df['_rc'] = np.abs(df['_rk'] - df['_rk_d1'])
    _add_rolling(df, '_rc', 'factor_cnt_turnover_rank_chg_20d', 20, 'mean')
    _add_rolling(df, '_rc', 'factor_cnt_turnover_rank_chg_60d', 60, 'mean')

    # ================================================================
    # 因子 6-7: 涨跌幅超过 2% 的次数
    # ================================================================
    df['factor_cnt_big_move_20d'] = (np.abs(ret) > 0.02).astype(float)
    _add_rolling(df, 'factor_cnt_big_move_20d', 'factor_cnt_big_move_20d', 20, 'sum')
    df['factor_cnt_big_move_60d'] = (np.abs(ret) > 0.02).astype(float)
    _add_rolling(df, 'factor_cnt_big_move_60d', 'factor_cnt_big_move_60d', 60, 'sum')

    # ================================================================
    # 因子 8: 振幅突破
    # ================================================================
    amp = (f3d.df['high'] - f3d.df['low']) / close
    df['_amp'] = amp
    _add_rolling(df, '_amp', '_amp_ma20', 20, 'mean')
    df['factor_cnt_amp_break_20d'] = (amp > 1.5 * df['_amp_ma20']).astype(float)
    _add_rolling(df, 'factor_cnt_amp_break_20d', 'factor_cnt_amp_break_20d', 20, 'sum')

    # ================================================================
    # 因子 9: 复合因子（仅用本节点内的因子）
    # ================================================================
    # composite 原依赖 new_high_20d/consec_down（在 streak 节点），这里用可用因子代替
    df['factor_cnt_composite'] = (
        df['factor_cnt_vol_spike_20d'] / 20 +
        df['factor_cnt_big_move_20d'] / 20 -
        df['factor_cnt_vol_shrink_20d'] / 20
    ) / 3

    # 截面标准化（仍通过 Frame3D cs_zscore 确保一致性）
    factor_cols = [c for c in df.columns if c.startswith('factor_cnt_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    nan_counts = {col: result.df[col].isna().sum() for col in factor_cols}
    logging.debug(f"[{name}] Counting NaN counts: {nan_counts}")

    return Frame3D(result.df[factor_cols].copy())
