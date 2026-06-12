"""
合成数据生成器 — 生成具有可预测结构的模拟量价数据（含 IPO/退市机制）。

数据生成逻辑：
1. 每只股票有独立的 log_price 随机游走路径，波动率异质。
2. 5 个"隐藏因子" h1..h5 是 AR(1) 过程（ρ=0.95），驱动未来收益。
3. 真实 20 日前瞻收益 = Σ(w_i * h_i) + noise，其中 noise_ratio 控制信噪比。
4. OHLC 从 log_price 推导；成交量与日内波动正相关；市值服从慢变对数正态分布。
5. 所有随机数使用 np.random.default_rng(seed)，确保可复现。
6. 若提供 start_date，time index 使用真实交易日（跳过周末）；否则使用整数索引。
7. IPO/退市：股价首次低于 0.005 的股票在次日退市，同时激活一只新股（随机初始价格）。

V3: 动态股票池模式 — 预生成 pool_size=n_stocks*2 的参数，逐日管理活跃股票集合。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from qpipe.frame3d import Frame3D

if TYPE_CHECKING:
    from collections.abc import Iterator

# 退市阈值：股价（close）低于此值即触发退市
DELIST_PRICE_THRESHOLD: float = 0.005
# 股票池倍数：预生成的股票总数为 n_stocks * POOL_MULTIPLIER
POOL_MULTIPLIER: int = 2


def _generate_hidden_factors(n_times: int, rng: np.random.Generator) -> np.ndarray:
    """生成 5 个隐藏因子，每个是 AR(1) 过程，衰减系数 0.95。

    Returns: (n_times, 5) 的 ndarray。
    """
    n_factors = 5
    rho = 0.95
    factors = np.zeros((n_times, n_factors))
    factors[0] = rng.normal(0, 1, n_factors)
    for t in range(1, n_times):
        factors[t] = rho * factors[t - 1] + np.sqrt(1 - rho**2) * rng.normal(0, 1, n_factors)
    return factors


def _generate_stock_params(n_stocks: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """生成每只股票的参数：波动率 σ 和隐藏因子权重 w_i。"""
    sigma = rng.lognormal(mean=-3.5, sigma=1.0, size=n_stocks)
    weights = rng.normal(0, 1, (n_stocks, 5))
    return {'sigma': sigma, 'weights': weights}


def _make_time_keys(n_times: int, start_date: str | None) -> list:
    """构造时间索引列表。

    若 start_date 为 None，返回 [0, 1, ..., n_times-1]（向后兼容）。
    否则从 start_date 开始，跳过周末，取 n_times 个交易日。
    """
    if start_date is None:
        return list(range(n_times))
    dates = pd.bdate_range(start=start_date, periods=n_times)
    return list(dates)


def generate_synthetic_data(
    n_times: int = 1000,
    n_stocks: int = 500,
    noise_ratio: float = 0.3,
    seed: int = 42,
    start_date: str | None = None,
) -> Iterator[Frame3D]:
    """主生成器：逐日 yield 包含 OHLC + turnover + volume + market_cap 的 Frame3D。

    noise_ratio=0 时完全可预测，noise_ratio=1 时信号淹没在噪声中。
    若提供 start_date（YYYY-MM-DD），time index 使用真实交易日。
    支持 IPO/退市：股价 < DELIST_PRICE_THRESHOLD 次日退市，同时上市新股。
    """
    rng = np.random.default_rng(seed)

    pool_size = n_stocks * POOL_MULTIPLIER
    time_keys = _make_time_keys(n_times, start_date)

    # ---- 预生成隐藏因子 ----（+20 用于前瞻收益）
    hidden_factors = _generate_hidden_factors(n_times + 20, rng)
    hidden_factors_trimmed = hidden_factors[:n_times, :]

    # ---- 预生成 pool_size 只股票的参数 ----
    params = _generate_stock_params(pool_size, rng)
    sigma = params['sigma']        # (pool_size,)
    weights = params['weights']    # (pool_size, 5)

    # ---- 预计算信号（所有股票的面板前瞻收益）----
    signal = hidden_factors_trimmed @ weights.T          # (n_times, pool_size)
    noise_raw = rng.normal(0, 1, (n_times, pool_size))
    true_fwd_ret = signal + noise_ratio * noise_raw      # (n_times, pool_size)

    # ---- 每个股票的运行时状态 ----
    # stock_state[si]: dict with keys:
    #   log_price, mcap, active, active_since, name
    stock_state: list[dict] = []

    def _init_stock(si: int, t: int, name_prefix: str) -> dict:
        """创建并返回一个股票的初始状态。"""
        return {
            'log_price': rng.normal(0, 0.8),
            'mcap': rng.lognormal(np.log(100), 1.2),
            'active': True,
            'active_since': t,
            'name': name_prefix + str(si).zfill(4),
        }

    # 初始化前 n_stocks 只股票
    for si in range(n_stocks):
        stock_state.append(_init_stock(si, 0, 'S'))

    # 预初始化剩余 reserve 股票（暂未激活）
    for si in range(n_stocks, pool_size):
        stock_state.append({
            'log_price': 0.0,
            'mcap': 0.0,
            'active': False,
            'active_since': -1,
            'name': 'R' + str(si - n_stocks).zfill(4),
        })

    # ---- 退市调度表：{stock_idx: delist_at_time} ----
    delist_schedule: dict[int, int] = {}
    next_reserve: int = n_stocks     # 下一个待激活的 reserve 索引

    # ---- 前一日 log_price 记录（用于生成当天 open price）----
    prev_log_prices: dict[int, float | None] = dict.fromkeys(range(n_stocks))

    for t in range(n_times):
        tk = time_keys[t]

        # === 处理当日到期的退市 ===
        to_delist = [si for si, dt in delist_schedule.items() if dt == t]
        for si in to_delist:
            stock_state[si]['active'] = False
            del delist_schedule[si]
            # 激活下一只新股
            if next_reserve < pool_size:
                new_si = next_reserve
                next_reserve += 1
                # 用原 reserve 的参数重新初始化
                stock_state[new_si] = _init_stock(new_si, t, 'N')
                prev_log_prices[new_si] = None   # 新股无前日价格
                logging.info(
                    f'[DataGen][t={t}] Stock {stock_state[si]["name"]} delisted, '
                    f'new stock {stock_state[new_si]["name"]} listed.'
                )
            else:
                logging.warning(
                    f'[DataGen][t={t}] Reserve pool exhausted. '
                    f'Stock {stock_state[si]["name"]} delisted, no replacement.'
                )

        # === 收集当日活跃股票 ===
        active_indices = [si for si, st in enumerate(stock_state) if st['active']]
        n_active = len(active_indices)
        active_names = [stock_state[si]['name'] for si in active_indices]

        # === 当日 log_price ===
        log_prices_t = np.array([stock_state[si]['log_price'] for si in active_indices])
        close_t = np.exp(log_prices_t)

        # === 生成当日 OHLC ===
        eps_open_std = 0.002
        eps_range_std = 0.003
        open_t = np.zeros(n_active)
        for i, si in enumerate(active_indices):
            prev_lp = prev_log_prices.get(si)
            if prev_lp is not None:
                open_t[i] = np.exp(prev_lp) * np.exp(rng.normal(0, eps_open_std))
            else:
                # 上市首日，open 在 close 附近随机
                open_t[i] = close_t[i] * np.exp(rng.normal(0, eps_open_std))
        eps_high = np.abs(rng.normal(0, eps_range_std, n_active))
        eps_low = np.abs(rng.normal(0, eps_range_std, n_active))
        high_t = np.maximum(open_t, close_t) * (1 + eps_high)
        low_t = np.minimum(open_t, close_t) * (1 - eps_low)

        # === 当日市值、成交量、换手率 ===
        mcap_t = np.array([stock_state[si]['mcap'] for si in active_indices])
        intraday_range = np.abs(close_t - open_t)
        volume_t = intraday_range * mcap_t * 0.01 * np.exp(rng.normal(0, 0.5, n_active))
        # 体积地板（含每股票微小噪声，避免截面退化）。
        # 注意：price 回落后 1e-4 的 floor 会反客为主（0.01 vs 实际 volume≈0.006），
        # 故降低至 1e-7，确保 floor 仅在极端情况下生效。
        floor_noise = np.exp(rng.normal(0, 0.0001, n_active))
        floor = mcap_t * 1e-7 * floor_noise
        volume_t = np.maximum(volume_t, floor)
        turnover_t = volume_t / mcap_t

        # === 构建当日 Frame3D ===
        arrays = [[tk] * n_active, active_names]
        mi = pd.MultiIndex.from_arrays(arrays, names=['key', 'name'])

        # close_uq：不复权收盘价（模拟除权除息）。
        # 除权/分红只会让不复权价 ≤ 后复权价，故用 -|ε| 确保 close_uq ≤ close。
        close_uq_t = close_t * np.exp(-np.abs(rng.normal(0, 0.0005, n_active)))

        # OHLC 价格精度对齐真实股市：统一 2 位小数
        open_t = np.round(open_t, 2)
        high_t = np.round(high_t, 2)
        low_t = np.round(low_t, 2)
        close_t = np.round(close_t, 2)
        close_uq_t = np.round(np.minimum(close_uq_t, close_t), 2)

        df = pd.DataFrame(
            {
                'stock_name': active_names,        # 股票名（便于 CSV 可读，主键仍是 index level 'name'）
                'open': open_t,
                'high': high_t,
                'low': low_t,
                'close': close_t,
                'close_uq': close_uq_t,
                'turnover': turnover_t,
                'volume': volume_t,
                'market_cap': mcap_t,
            },
            index=mi,
        )
        if t % 100 == 0:
            logging.info(
                f'[DataGen] Day {t}/{n_times}, noise_ratio={noise_ratio}, '
                f'active={n_active}, delisted={len(delist_schedule)}'
            )
        yield Frame3D(df)

        # === 检查退市条件：close < DELIST_PRICE_THRESHOLD ===
        for i, si in enumerate(active_indices):
            if close_t[i] < DELIST_PRICE_THRESHOLD and si not in delist_schedule:
                delist_schedule[si] = t + 1  # 次日退市

        # === 更新状态供下一日使用 ===
        # 记录当前 log_price 为下日 open 计算的前日价
        for i, si in enumerate(active_indices):
            prev_log_prices[si] = log_prices_t[i]
            # 市值慢变漂移
            stock_state[si]['mcap'] = mcap_t[i] * np.exp(rng.normal(0, 0.001))

        # 计算下一日 log_price（仅活跃股票，当前日 t < n_times-1）
        if t < n_times - 1:
            drift = 0.0002
            innovation = rng.normal(0, 0.01, n_active)
            for i, si in enumerate(active_indices):
                # true_fwd_ret 是 20 日级信号（std≈7.2），直接用作每日漂移会导致
                # 1000 天后价格爆炸（e^80+），使 liquidity/value 等因子退化。
                # 除以 sqrt(20)≈4.5 转换为日度尺度，再乘以 0.05 控制日波动率。
                daily_signal = true_fwd_ret[t, si] * 0.05 / 4.5
                stock_state[si]['log_price'] = (
                    log_prices_t[i] + drift
                    + sigma[si] * (daily_signal + innovation[i])
                )

        time.sleep(0.2)


class DataSourceCallable:
    """模块级可调用类，用于 pickle 安全的 SourceNode gen_func。"""

    def __init__(
        self,
        n_times: int,
        n_stocks: int,
        noise_ratio: float,
        seed: int,
        start_date: str | None = None,
    ) -> None:
        self.n_times = n_times
        self.n_stocks = n_stocks
        self.noise_ratio = noise_ratio
        self.seed = seed
        self.start_date: str | None = start_date

    def __call__(self):
        return generate_synthetic_data(
            n_times=self.n_times,
            n_stocks=self.n_stocks,
            noise_ratio=self.noise_ratio,
            seed=self.seed,
            start_date=self.start_date,
        )
