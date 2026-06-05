"""
质量因子（高级）— 4 个因子：收益率偏度×2 / up-down / 复合。
自相关/尾部风险已拆分至 quality_autocorr 节点。
skew 用 ts_rolling，up_down 用 sliding_window_view 向量化。
"""
import numpy as np
import logging
from numpy.lib.stride_tricks import sliding_window_view
from qpipe.frame3d import Frame3D


def compute_quality_advanced_factors(name: str, f3d: Frame3D, context) -> Frame3D:
    """计算 4 个质量高级因子。"""
    result = f3d.copy()
    ret = f3d.ts_pct_change('close', 1).df['close']
    result = result.add_column('_ret', ret)
    grp = f3d.df.index.get_level_values('name')

    # ---- 1-2: 收益率偏度 ----
    skew60 = f3d.copy().add_column('_ret', ret).ts_rolling('_ret', 60, 'skew').df['_ret']
    result = result.add_column('factor_qa_skew_60d', skew60)
    skew120 = f3d.copy().add_column('_ret', ret).ts_rolling('_ret', 120, 'skew').df['_ret']
    result = result.add_column('factor_qa_skew_120d', skew120)

    # ---- 3: Up/Down capture ratio（向量化） ----
    def _up_down_ratio_vec(series, window):
        arr = series.values
        n = len(arr)
        if n < max(2, window // 2):
            return np.full(n, np.nan)
        win = sliding_window_view(arr, window)
        pos_mask = win > 0
        neg_mask = win < 0
        pos_cnt = pos_mask.sum(axis=1)
        neg_cnt = neg_mask.sum(axis=1)
        pos_mean = np.where(pos_cnt > 0,
                            (win * pos_mask).sum(axis=1) / np.maximum(pos_cnt, 1), 0.0)
        neg_mean = np.where(neg_cnt > 0,
                            np.abs((win * neg_mask).sum(axis=1)) / np.maximum(neg_cnt, 1), 1e-6)
        neg_mean[neg_mean == 0] = 1e-6
        ratio = pos_mean / neg_mean
        result_arr = np.full(n, np.nan)
        result_arr[window - 1:] = ratio
        return result_arr

    up_down = ret.groupby(grp).transform(lambda x: _up_down_ratio_vec(x, 60))
    result = result.add_column('factor_qa_up_down_60d', up_down)

    # ---- 4: 复合（仅用本节点因子） ----
    result = result.add_column('factor_qa_composite',
                                (result.df['factor_qa_skew_60d'] +
                                 result.df['factor_qa_up_down_60d']) / 2)

    # 截面标准化
    factor_cols = [c for c in result.df.columns if c.startswith('factor_qa_')]
    result = result.cs_zscore_batch(factor_cols, cp=False)

    logging.debug(f"[{name}] QA-Adv NaN: "
                  f"{ {c: result.df[c].isna().sum() for c in factor_cols} }")
    return Frame3D(result.df[factor_cols].copy())
