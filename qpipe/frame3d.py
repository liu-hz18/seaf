"""
Frame3D — 三维数据容器（时间 × 股票 × 属性）。
索引：pd.MultiIndex，层级为 [key, name]（key=时间, name=股票）。
列是属性维度（如 open, close, volume 等）。

本模块提供时序（沿 key 层）和截面（沿 name 层）的计算 API。
所有方法返回新的 Frame3D，不修改原始数据。
"""
import pandas as pd
import numpy as np
from typing import List, Union


class Frame3D:
    def __init__(self, df: pd.DataFrame):
        assert isinstance(df.index, pd.MultiIndex), "Index must be MultiIndex (key, name)"
        self._df = df

    @property
    def df(self):
        return self._df

    def copy(self):
        return Frame3D(self._df.copy(deep=True))

    def __repr__(self):
        df = self._df
        n_times = df.index.get_level_values('key').nunique()
        n_stocks = df.index.get_level_values('name').nunique()
        n_cols = len(df.columns)
        # 有效数据占比（非 NaN 单元格比例）
        total_cells = len(df) * n_cols
        valid_cells = df.notna().sum().sum() if total_cells > 0 else 0
        valid_pct = valid_cells / total_cells * 100 if total_cells > 0 else 0.0
        return (
            f"Frame3D[{len(df)} rows, {n_times}t x {n_stocks}s x {n_cols}c, "
            f"valid_data={valid_pct:.1f}%](\n{repr(self._df)}\n)"
        )

    # ========================================================================
    # 时序 API — 沿 key（时间）层计算，每个 stock 独立
    # ========================================================================

    def ts_delay(self, col: str, periods: int, cp: bool = True) -> 'Frame3D':
        """时序滞后：将指定列在每个 stock 内部向下平移 periods 个时间单位。
        NaN 填充缺失值。cp=False 时原地操作，避免深拷贝。"""
        df = self._df.copy() if cp else self._df
        df[col] = df.groupby('name')[col].shift(periods)
        return Frame3D(df)

    def ts_delta(self, col: str, periods: int, cp: bool = True) -> 'Frame3D':
        """时序差分：col(t) - col(t-periods)，每个 stock 独立。
        cp=False 时原地操作。"""
        delayed = self.ts_delay(col, periods)
        df = self._df.copy() if cp else self._df
        df[col] = df[col] - delayed.df[col]
        return Frame3D(df)

    def ts_pct_change(self, col: str, periods: int, cp: bool = True) -> 'Frame3D':
        """时序百分比变化：(col(t) - col(t-periods)) / col(t-periods)。
        cp=False 时原地操作。"""
        delayed = self.ts_delay(col, periods)
        df = self._df.copy() if cp else self._df
        denom = delayed.df[col]
        with np.errstate(divide='ignore', invalid='ignore'):
            df[col] = (df[col] - denom) / denom.replace(0, np.nan)
        return Frame3D(df)

    def ts_pct_change_multi(self, col: str, periods: List[int],
                            prefix: str = '', cp: bool = True) -> 'Frame3D':
        """批量时序百分比变化：一次 GroupBy 完成多周期计算。

        对每个 period in periods，生成列 '{prefix}_{period}d'（若 prefix 非空）
        或 '{col}_pct_{period}d'（默认列名）。

        相比多次调用 ts_pct_change，大幅减少深拷贝和 GroupBy 开销。
        """
        df = self._df.copy() if cp else self._df
        grp = df.groupby('name')[col]
        for p in periods:
            col_name = f'{prefix}_{p}d' if prefix else f'{col}_pct_{p}d'
            shifted = grp.shift(p)
            with np.errstate(divide='ignore', invalid='ignore'):
                df[col_name] = (df[col] - shifted) / shifted.replace(0, np.nan)
        return Frame3D(df)

    def ts_rolling(self, col: str, window: int, agg_fn: str, cp: bool = True) -> 'Frame3D':
        """时序滚动聚合，每个 stock 独立计算。

        agg_fn 支持: 'mean', 'std', 'min', 'max', 'sum', 'skew', 'kurt'。
        min_periods = max(1, window // 2)。
        cp=False 时原地操作。
        """
        min_periods = max(1, window // 2)
        df = self._df.copy() if cp else self._df
        df[col] = df.groupby('name')[col].transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).agg(agg_fn)
        )
        return Frame3D(df)

    def ts_rolling_multi(self, col: str, windows: List[int], agg_fn: str,
                         prefix: str = '', cp: bool = True) -> 'Frame3D':
        """批量时序滚动聚合：在一次 GroupBy 循环中完成多个窗口计算。

        对每个 w in windows，生成列 '{prefix}_{w}d'（若 prefix 非空）
        或 '{col}_{agg_fn}_{w}d'（默认列名）。

        相比多次调用 ts_rolling，减少深拷贝和 GroupBy 重复扫描。
        """
        df = self._df.copy() if cp else self._df
        for w in windows:
            col_name = f'{prefix}_{w}d' if prefix else f'{col}_{agg_fn}_{w}d'
            min_periods = max(1, w // 2)
            df[col_name] = df.groupby('name')[col].transform(
                lambda x, ww=w, mp=min_periods:
                    x.rolling(window=ww, min_periods=mp).agg(agg_fn)
            )
        return Frame3D(df)

    def ts_zscore(self, col: str, window: int, cp: bool = True) -> 'Frame3D':
        """时序标准化：(x - rolling_mean) / rolling_std，每个 stock 独立。
        min_periods = max(1, window // 2)。cp=False 时原地操作。"""
        min_periods = max(1, window // 2)
        df = self._df.copy() if cp else self._df
        grp = df.groupby('name')[col]
        roll_mean = grp.transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).mean()
        )
        roll_std = grp.transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).std()
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            df[col] = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
        return Frame3D(df)

    def ts_rank(self, col: str, window: int, cp: bool = True) -> 'Frame3D':
        """时序排名：滚动窗口内当前值的百分位排名（0~1），每个 stock 独立。
        cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df
        df[col] = df.groupby('name')[col].transform(
            lambda x: x.rolling(window=window, min_periods=2).apply(
                lambda w: (w.rank().iloc[-1] - 1) / (len(w) - 1) if len(w) > 1 else np.nan,
                raw=False
            )
        )
        return Frame3D(df)

    # ========================================================================
    # 截面 API — 沿 name（股票）层计算，每个 time 独立
    # ========================================================================

    def cs_zscore(self, col: str, cp: bool = True) -> 'Frame3D':
        """截面标准化：(x - cross_sectional_mean) / cross_sectional_std。
        对每个 time 独立计算。std=0 时返回 0。cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df
        grp = df.groupby('key')[col]
        cs_mean = grp.transform('mean')
        cs_std = grp.transform('std')
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (df[col] - cs_mean) / cs_std.replace(0, np.nan)
            df[col] = z.fillna(0.0)
        return Frame3D(df)

    def cs_zscore_batch(self, cols: list, cp: bool = True) -> 'Frame3D':
        """批量截面标准化：对多个列一次性做 (x-mean)/std。
        避免逐列调用的重复深拷贝和 groupby 开销。
        std=0 时返回 0。cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df
        grp = df.groupby('key')[cols]
        cs_mean = grp.transform('mean')
        cs_std = grp.transform('std')
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (df[cols] - cs_mean) / cs_std.replace(0, np.nan)
            df[cols] = z.fillna(0.0)
        return Frame3D(df)

    def cs_rank(self, col: str, cp: bool = True) -> 'Frame3D':
        """截面排名百分位（0~1），对每个 time 独立计算。
        使用 (rank-1)/(N-1) 公式，使得 min=0, max=1。
        单股票截面返回 0.5。cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df
        def _rank_pct(x):
            n = len(x)
            if n <= 1:
                return pd.Series(0.5, index=x.index)
            return (x.rank() - 1) / (n - 1)
        df[col] = df.groupby('key')[col].transform(_rank_pct)
        return Frame3D(df)

    def cs_rank_batch(self, cols: list, cp: bool = True) -> 'Frame3D':
        """批量截面排名百分位：一次 GroupBy 完成多列截面排名。
        cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df
        def _rank_pct(x):
            n = len(x)
            if n <= 1:
                return pd.Series(0.5, index=x.index)
            return (x.rank() - 1) / (n - 1)
        # 对每列分别做 rank（pandas 不支持多列 rank）
        for c in cols:
            df[c] = df.groupby('key')[c].transform(_rank_pct)
        return Frame3D(df)

    def cs_demean(self, col: str, cp: bool = True) -> 'Frame3D':
        """截面去均值：x - cross_sectional_mean。cp=False 时原地操作。"""
        df = self._df.copy() if cp else self._df
        cs_mean = df.groupby('key')[col].transform('mean')
        df[col] = df[col] - cs_mean
        return Frame3D(df)

    def cs_neutralize(self, col: str, by: List[str], cp: bool = True) -> 'Frame3D':
        """截面中性化：对 col 按 by 中的列做截面回归，取残差。
        回归前自动对 by 做 cs_zscore。

        对每个 time 独立做 OLS 回归，返回残差。
        如果某一天的数据不足以做回归（如 by 全 NaN 或样本过少），
        则返回 cs_demean 结果作为降级处理。
        cp=False 时原地操作。
        """
        df = self._df.copy() if cp else self._df
        # 先对中性化变量做截面标准化（原地）
        f3d_tmp = Frame3D(df)
        for b in by:
            f3d_tmp = f3d_tmp.cs_zscore(b, cp=False)
        # df 已被 f3d_tmp 修改（共享 self._df）

        # 对每个 time 做回归取残差
        def _ols_residual(grp):
            """对单个时间截面做 OLS，返回残差。"""
            y = grp[col].values.astype(float)
            X = grp[by].values.astype(float)
            # 移除包含 NaN 的行
            mask = ~np.isnan(y)
            if X.shape[1] > 0:
                mask = mask & (~np.any(np.isnan(X), axis=1))
            if mask.sum() < max(3, len(by) + 2):
                # 样本不足 → 返回 demean；若全 NaN 则返回 NaN
                with np.errstate(all='ignore'):
                    m = np.nanmean(y) if np.any(~np.isnan(y)) else 0.0
                return pd.Series(y - m, index=grp.index)
            y_clean = y[mask]
            X_clean = X[mask]
            # OLS: β = (X'X)^-1 X'y
            try:
                beta = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]
                y_pred = X @ beta
                residual = y - y_pred
            except np.linalg.LinAlgError:
                residual = y - np.nanmean(y)
            return pd.Series(residual, index=grp.index)

        result = df.groupby('key', group_keys=False).apply(_ols_residual)
        df[col] = result
        return Frame3D(df)

    # ========================================================================
    # 工具 API
    # ========================================================================

    def get_cs_series(self, col: str, time_key) -> pd.Series:
        """获取指定时间截面的 Series，index 为 stock name。"""
        mask = self._df.index.get_level_values('key') == time_key
        return self._df.loc[mask, col].droplevel('key')

    def get_ts_series(self, stock: str, col: str) -> pd.Series:
        """获取指定股票的时序 Series，index 为 time key。"""
        mask = self._df.index.get_level_values('name') == stock
        return self._df.loc[mask, col].droplevel('name')

    def add_column(self, name: str, values: Union[pd.Series, np.ndarray],
                   cp: bool = True) -> 'Frame3D':
        """安全添加列，自动对齐索引。

        - 若 values 为 Series，会与内部 df 按 index 对齐。
        - 若 values 为 np.ndarray，要求长度等于 df 行数。
        cp=False 时原地操作。
        """
        df = self._df.copy() if cp else self._df
        if isinstance(values, pd.Series):
            df[name] = values.reindex(df.index)
        else:
            assert len(values) == len(df), \
                f"Length mismatch: values({len(values)}) vs df({len(df)})"
            df[name] = values
        return Frame3D(df)

    def filter_stocks(self, mask: pd.Series, cp: bool = True) -> 'Frame3D':
        """按布尔 mask 过滤股票（截面维度）。

        mask: index 为 stock name，值为 bool 的 Series。
        返回只保留 mask 中为 True 的 stock 的新 Frame3D。
        cp=False 时原地操作。
        """
        df = self._df.copy() if cp else self._df
        valid_stocks = mask[mask].index
        df = df[df.index.get_level_values('name').isin(valid_stocks)]
        return Frame3D(df)
