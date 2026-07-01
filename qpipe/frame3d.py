"""
Frame3D — 三维数据容器（时间 × 股票 × 属性）。
索引：pd.MultiIndex，层级为 [key, name]（key=时间, name=股票）。
列是属性维度（如 open, close, volume 等）。

本模块提供时序（沿 key 层）和截面（沿 name 层）的计算 API。
所有方法返回新的 Frame3D，不修改原始数据。
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.width', 1024)


class Frame3D:
    """三维数据容器：时间 × 股票 × 属性。"""

    _df: pd.DataFrame

    def __init__(self, df: pd.DataFrame) -> None:
        assert isinstance(df.index, pd.MultiIndex), 'Index must be MultiIndex (key, code)'
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def copy(self) -> Frame3D:
        return Frame3D(self._df.copy(deep=True))

    def __repr__(self) -> str:
        df = self._df
        n_times = df.index.get_level_values('key').nunique()
        n_stocks = df.index.get_level_values('code').nunique()
        n_cols = len(df.columns)
        total_cells = len(df) * n_cols
        valid_cells = df.notna().sum().sum() if total_cells > 0 else 0
        valid_pct = valid_cells / total_cells * 100 if total_cells > 0 else 0.0
        return (
            f'Frame3D[{len(df)} rows, {n_times}t x {n_stocks}s x {n_cols}c, '
            f'valid_data={valid_pct:.1f}%](\n{self._df!r}\n)'
        )

    def reset_index(self) -> Frame3D:
        if isinstance(self._df.index, pd.MultiIndex) and 'key' in self._df.index.names:
            arrays = []
            for name in self._df.index.names:
                if name == 'key':
                    # 对 key 层进行精度统一和截断
                    arr = self._df.index.get_level_values(name).normalize().astype('datetime64[ns]')
                else:
                    # 保持其他层（如 code）原样
                    arr = self._df.index.get_level_values(name)
                arrays.append(arr)
            # 重新构造 MultiIndex
            self._df.index = pd.MultiIndex.from_arrays(arrays, names=self._df.index.names)
        return self

    def last_key(self) -> int:
        return self._df.index.get_level_values(0).max()

    def first_key(self) -> int:
        return self._df.index.get_level_values(0).min()

    def last_frame(self) -> Frame3D:
        return Frame3D(self._df[self._df.index.get_level_values(0) == self.last_key()].copy())

    def first_frame(self) -> Frame3D:
        return Frame3D(self._df[self._df.index.get_level_values(0) == self.first_key()].copy())

    def to(self, dtype: type = np.float32) -> Frame3D:
        # 筛选出所有浮点数列
        float_cols = self._df.select_dtypes(include=['floating']).columns
        # 批量转换
        if len(float_cols) > 0:
            # 修复：如果 _df 是视图（例如从外部切片传入），先 copy 避免警告
            if getattr(self._df, '_is_view', False):
                self._df = self._df.copy()
            self._df.loc[:, float_cols] = self._df.loc[:, float_cols].astype(dtype)
            # self._df[float_cols] = self._df[float_cols].astype(dtype, copy=False)
        return self

    # ========================================================================
    # 时序 API — 沿 key（时间）层计算，每个 stock 独立
    # ========================================================================

    def ts_delay(self, col: str, periods: int, cp: bool = True) -> Frame3D:
        """时序滞后：将指定列在每个 stock 内部向下平移 periods 个时间单位。
        NaN 填充缺失值。cp=False 时原地操作，避免深拷贝。"""
        df = self._df.copy() if cp else self._df
        df[col] = df.groupby('code')[col].shift(periods)
        return Frame3D(df)

    def ts_delta(self, col: str, periods: int, cp: bool = True) -> Frame3D:
        """时序差分：col(t) - col(t-periods)，每个 stock 独立。
        cp=False 时原地操作。"""
        delayed = self.ts_delay(col, periods)
        df = self._df.copy() if cp else self._df
        df[col] = df[col] - delayed.df[col]
        return Frame3D(df)

    def ts_pct_change(self, col: str, periods: int, cp: bool = True) -> Frame3D:
        """时序百分比变化：(col(t) - col(t-periods)) / col(t-periods)。
        cp=False 时原地操作。"""
        delayed = self.ts_delay(col, periods)
        df = self._df.copy() if cp else self._df
        denom = delayed.df[col]
        with np.errstate(divide='ignore', invalid='ignore'):
            df[col] = (df[col] - denom) / denom.replace(0, np.nan)
        return Frame3D(df)

    def ts_pct_change_multi(
        self, col: str, periods: list[int], prefix: str = '', cp: bool = True
    ) -> Frame3D:
        """批量时序百分比变化：一次 GroupBy 完成多周期计算。

        对每个 period in periods，生成列 '{prefix}_{period}d'（若 prefix 非空）
        或 '{col}_pct_{period}d'（默认列名）。

        相比多次调用 ts_pct_change，大幅减少深拷贝和 GroupBy 开销。
        """
        df = self._df.copy() if cp else self._df
        grp = df.groupby('code')[col]
        for p in periods:
            col_name = f'{prefix}_{p}d' if prefix else f'{col}_pct_{p}d'
            shifted = grp.shift(p)
            with np.errstate(divide='ignore', invalid='ignore'):
                df[col_name] = (df[col] - shifted) / shifted.replace(0, np.nan)
        return Frame3D(df)

    def ts_rolling(self, col: str, window: int, agg_fn: str, cp: bool = True) -> Frame3D:
        """时序滚动聚合，每个 stock 独立计算。

        agg_fn 支持: 'mean', 'std', 'min', 'max', 'sum', 'skew', 'kurt'。
        min_periods = max(1, window // 2)。
        cp=False 时原地操作。
        """
        min_periods = max(1, window // 2)
        df = self._df.copy() if cp else self._df
        df[col] = df.groupby('code')[col].transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).agg(agg_fn)
        )
        return Frame3D(df)

    def ts_rolling_multi(
        self, col: str, windows: list[int], agg_fn: str, prefix: str = '', cp: bool = True
    ) -> Frame3D:
        """批量时序滚动聚合：在一次 GroupBy 循环中完成多个窗口计算。

        对每个 w in windows，生成列 '{prefix}_{w}d'（若 prefix 非空）
        或 '{col}_{agg_fn}_{w}d'（默认列名）。

        相比多次调用 ts_rolling，减少深拷贝和 GroupBy 重复扫描。
        """
        df = self._df.copy() if cp else self._df
        for w in windows:
            col_name = f'{prefix}_{w}d' if prefix else f'{col}_{agg_fn}_{w}d'
            min_periods = max(1, w // 2)
            df[col_name] = df.groupby('code')[col].transform(
                lambda x, ww=w, mp=min_periods: x.rolling(window=ww, min_periods=mp).agg(agg_fn)
            )
        return Frame3D(df)

    def ts_zscore(self, col: str, window: int, cp: bool = True) -> Frame3D:
        """时序标准化：(x - rolling_mean) / rolling_std，每个 stock 独立。
        min_periods = max(1, window // 2)。cp=False 时原地操作。"""
        min_periods = max(1, window // 2)
        df = self._df.copy() if cp else self._df
        grp = df.groupby('code')[col]
        roll_mean = grp.transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).mean()
        )
        roll_std = grp.transform(lambda x: x.rolling(window=window, min_periods=min_periods).std())
        with np.errstate(divide='ignore', invalid='ignore'):
            df[col] = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
        return Frame3D(df)

    def ts_rank(self, col: str, window: int, cp: bool = True) -> Frame3D:
        """时序排名：滚动窗口内当前值的百分位排名（0~1），每个 stock 独立。
        cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df
        df[col] = df.groupby('code')[col].transform(
            lambda x: x.rolling(window=window, min_periods=2).apply(
                lambda w: (w.rank().iloc[-1] - 1) / (len(w) - 1) if len(w) > 1 else np.nan,
                raw=False,
            )
        )
        return Frame3D(df)

    # ========================================================================
    # 截面 API — 沿 name（股票）层计算，每个 time 独立
    # ========================================================================

    def cs_zscore(self, col: str, cp: bool = True) -> Frame3D:
        """截面标准化：(x - cross_sectional_mean) / cross_sectional_std。
        对每个 time 独立计算。std=0 时返回 0。cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df
        grp = df.groupby('key')[col]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cs_mean = grp.transform('mean')
            cs_std = grp.transform('std')
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (df[col] - cs_mean) / cs_std.replace(0, np.nan)
            df[col] = z.fillna(0.0)
        return Frame3D(df)

    def cs_zscore_batch(self, cols: list[str], cp: bool = True) -> Frame3D:
        """批量截面标准化：对多个列一次性做 (x-mean)/std。
        避免逐列调用的重复深拷贝和 groupby 开销。
        std=0 时返回 0。cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df
        grp = df.groupby('key')[cols]
        # pandas transform('std'/'mean') 内部调 np.nanstd/nanmean，
        # 在全 NaN 切片上触发 RuntimeWarning。errstate 无法穿透 pandas。
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cs_mean = grp.transform('mean')
            cs_std = grp.transform('std')
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (df[cols] - cs_mean) / cs_std.replace(0, np.nan)
            df[cols] = z.fillna(0.0)
        return Frame3D(df)

    def cs_zscore_batch_trimmed(self, cols: list[str], cp: bool = True,
                        trim: float = 0.01) -> Frame3D:
        """批量截面截尾标准化：
        1) 按 key 分组，剔除每组 top/bottom `trim` 分位后计算 mean/std；
        2) 用该 mean/std 对原始（未截尾）数据做 (x-mean)/std；
        3) std=0 或组内非 NaN 样本数 ≤ 2 时返回 0。
        4) 增加校验：若组内非 NaN 样本数不足以支撑 trim 比例（如 trim=0.01 需至少 100 个样本），
           则该组不进行截尾，直接使用原始数据计算统计量。
        cp=False 时原地操作。trim=0.01 即剔除上下各 1%。
        """
        df = self._df.copy() if cp else self._df
        key = df.index.get_level_values('key')  # MultiIndex: 'key' 是索引层级名
        grp = df.groupby(key, sort=False)[cols]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # 1. 统计各组每列的非 NaN 样本数，并广播到每一行
            counts = grp.transform('count')
            # 2. 判定是否满足截尾所需的最低样本量
            # 例如 trim=0.01 表示剔除 1%，至少需要 100 个样本才能稳定剔除 1 个点
            min_required = (1.0 / trim) if trim > 0 else 0
            do_trim = counts >= min_required
            # 3. 计算各组上下分位数（q_lo, q_hi）
            q_lo = grp.transform(lambda s: np.nanquantile(s, trim))
            q_hi = grp.transform(lambda s: np.nanquantile(s, 1.0 - trim))
            # 4. 构造有效截尾掩码
            # 条件1：数据本身非 NaN
            # 条件2：要么满足截尾样本量且处于分位区间内，要么不满足样本量（保留全部非 NaN）
            is_in_range = (df[cols] >= q_lo) & (df[cols] <= q_hi)
            effective_mask = df[cols].notna() & (is_in_range | ~do_trim)
            # 5. 将不在区间内或为 NaN 的值置为 NaN，用于计算截尾 mean/std
            masked = df[cols].where(effective_mask)
            # 6. 截尾后的 mean / std（按 key 分组）
            mgrp = masked.groupby(key, sort=False)
            cs_mean = mgrp.transform('mean')
            cs_std = mgrp.transform('std')

        # 7. 用截尾 mean/std 对原始数据做 zscore
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (df[cols] - cs_mean) / cs_std.replace(0, np.nan)
            df[cols] = z.fillna(0.0)

        return Frame3D(df)

    def cs_rank(self, col: str, cp: bool = True) -> Frame3D:
        """截面排名百分位（0~1），对每个 time 独立计算。
        使用 (rank-1)/(N-1) 公式，使得 min=0, max=1。
        单股票截面返回 0.5。cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df

        def _rank_pct(x: pd.Series) -> pd.Series:
            n = len(x)
            if n <= 1:
                return pd.Series(0.5, index=x.index)
            return (x.rank() - 1) / (n - 1)

        df[col] = df.groupby('key')[col].transform(_rank_pct)
        return Frame3D(df)

    def cs_rank_batch(self, cols: list[str], cp: bool = True) -> Frame3D:
        """批量截面排名百分位：一次 GroupBy 完成多列截面排名。
        cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df

        def _rank_pct(x: pd.Series) -> pd.Series:
            n = len(x)
            if n <= 1:
                return pd.Series(0.5, index=x.index)
            return (x.rank() - 1) / (n - 1)

        for c in cols:
            df[c] = df.groupby('key')[c].transform(_rank_pct)
        return Frame3D(df)

    def cs_demean(self, col: str, cp: bool = True) -> Frame3D:
        """截面去均值：x - cross_sectional_mean。cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df
        cs_mean = df.groupby('key')[col].transform('mean')
        df[col] = df[col] - cs_mean
        return Frame3D(df)

    def cs_neutralize(self, col: str, by: list[str], cp: bool = True) -> Frame3D:
        """截面中性化：对 col 按 by 中的列做截面回归，取残差。
        回归前自动对 by 做 cs_zscore。

        对每个 time 独立做 OLS 回归，返回残差。
        如果某一天的数据不足以做回归（如 by 全 NaN 或样本过少），
        则返回 cs_demean 结果作为降级处理。
        cp=False 时原地操作。
        """
        df = self._df.copy() if cp else self._df
        f3d_tmp = Frame3D(df)
        for b in by:
            f3d_tmp = f3d_tmp.cs_zscore(b, cp=False)

        def _ols_residual(grp: pd.DataFrame) -> pd.Series:
            """对单个时间截面做 OLS，返回残差。"""
            y = grp[col].values.astype(float)
            X = grp[by].values.astype(float)
            mask = ~np.isnan(y)
            if X.shape[1] > 0:
                mask = mask & (~np.any(np.isnan(X), axis=1))
            if mask.sum() < max(3, len(by) + 2):
                with np.errstate(all='ignore'):
                    m = float(np.nanmean(y)) if np.any(~np.isnan(y)) else 0.0
                return pd.Series(y - m, index=grp.index)  # type: ignore[reportOperatorIssue]
            y_clean = y[mask]
            X_clean = X[mask]
            try:
                beta = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]
                y_pred = X @ beta
                residual = y - y_pred  # type: ignore[reportOperatorIssue]
            except np.linalg.LinAlgError:
                residual = y - np.nanmean(y)  # type: ignore[reportOperatorIssue]
            return pd.Series(residual, index=grp.index)

        result = df.groupby('key', group_keys=False).apply(_ols_residual)
        df[col] = result
        return Frame3D(df)

    # ========================================================================
    # 工具 API
    # ========================================================================

    def get_cs_series(self, col: str, time_key: Any) -> pd.Series:
        """获取指定时间截面的 Series，index 为 stock name。"""
        mask = self._df.index.get_level_values('key') == time_key
        return self._df.loc[mask, col].droplevel('key')

    def get_ts_series(self, stock: str, col: str) -> pd.Series:
        """获取指定股票的时序 Series，index 为 time key。"""
        mask = self._df.index.get_level_values('code') == stock
        return self._df.loc[mask, col].droplevel('code')

    def add_column(self, name: str, values: pd.Series | np.ndarray, cp: bool = True) -> Frame3D:
        """安全添加列，自动对齐索引。

        - 若 values 为 Series，会与内部 df 按 index 对齐。
        - 若 values 为 np.ndarray，要求长度等于 df 行数。
        cp=False 时原地操作。
        """
        df = self._df.copy() if cp else self._df
        if isinstance(values, pd.Series):
            df[name] = values.reindex(df.index)
        else:
            assert len(values) == len(df), (
                f'Length mismatch: values({len(values)}) vs df({len(df)})'
            )
            df[name] = values
        return Frame3D(df)

    def filter_stocks(self, mask: pd.Series, cp: bool = True) -> Frame3D:
        """按布尔 mask 过滤股票（截面维度）。

        mask: index 为 stock name，值为 bool 的 Series。
        返回只保留 mask 中为 True 的 stock 的新 Frame3D。
        cp=False 时原地操作。
        """
        valid_stocks = mask[mask].index
        # 修复：先切片，再根据 cp 决定是否拷贝，提高性能并避免视图问题
        sub_df = self._df[self._df.index.get_level_values('code').isin(valid_stocks)]
        return Frame3D(sub_df.copy() if cp else sub_df)
