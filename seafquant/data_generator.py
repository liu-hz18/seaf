"""
合成数据生成器 — 生成具有可预测结构的模拟量价数据。

数据生成逻辑：
1. 每只股票有独立的 log_price 随机游走路径，波动率异质。
2. 5 个"隐藏因子" h1..h5 是 AR(1) 过程（ρ=0.95），驱动未来收益。
3. 真实 20 日前瞻收益 = Σ(w_i * h_i) + noise，其中 noise_ratio 控制信噪比。
4. OHLC 从 log_price 推导；成交量与日内波动正相关；市值服从慢变对数正态分布。
5. 所有随机数使用 np.random.default_rng(seed)，确保可复现。
6. 若提供 start_date，time index 使用真实交易日（跳过周末）；否则使用整数索引。
"""
import time
import numpy as np
import pandas as pd
import logging
from typing import Iterator, Optional
from qpipe.frame3d import Frame3D


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


def _generate_stock_params(n_stocks: int, rng: np.random.Generator) -> dict:
    """生成每只股票的参数：波动率 σ 和隐藏因子权重 w_i。"""
    sigma = rng.lognormal(mean=-3.5, sigma=0.5, size=n_stocks)  # mean ≈ 0.025/day
    weights = rng.normal(0, 1, (n_stocks, 5))  # (n_stocks, 5)
    return {'sigma': sigma, 'weights': weights}


def _generate_ohlc(log_prices: np.ndarray, rng: np.random.Generator) -> dict:
    """从 log_price 推导 OHLC 价格。"""
    n_times, n_stocks = log_prices.shape
    close = np.exp(log_prices)
    open_price = np.zeros_like(close)
    high = np.zeros_like(close)
    low = np.zeros_like(close)
    
    eps_open_std = 0.002
    eps_range_std = 0.003
    
    for t in range(n_times):
        if t == 0:
            open_price[t] = close[t] * np.exp(rng.normal(0, eps_open_std, n_stocks))
        else:
            open_price[t] = close[t - 1] * np.exp(rng.normal(0, eps_open_std, n_stocks))
        eps_high = np.abs(rng.normal(0, eps_range_std, n_stocks))
        eps_low = np.abs(rng.normal(0, eps_range_std, n_stocks))
        high[t] = np.maximum(open_price[t], close[t]) * (1 + eps_high)
        low[t] = np.minimum(open_price[t], close[t]) * (1 - eps_low)
    
    return {'open': open_price, 'high': high, 'low': low, 'close': close}


def _generate_market_cap(n_times: int, n_stocks: int, rng: np.random.Generator) -> np.ndarray:
    """慢变随机游走的市值序列，截面服从对数正态分布。"""
    base = rng.lognormal(mean=np.log(100), sigma=1.2, size=n_stocks)
    mcap = np.zeros((n_times, n_stocks))
    mcap[0] = base
    for t in range(1, n_times):
        mcap[t] = mcap[t - 1] * np.exp(rng.normal(0, 0.001, n_stocks))
    return mcap


def _make_time_keys(n_times: int, start_date: Optional[str]) -> list:
    """构造时间索引列表。
    
    若 start_date 为 None，返回 [0, 1, ..., n_times-1]（向后兼容）。
    否则从 start_date 开始，跳过周末，取 n_times 个交易日。
    """
    if start_date is None:
        return list(range(n_times))
    # pd.bdate_range: 跳过周末（周六/周日）的交易日序列
    dates = pd.bdate_range(start=start_date, periods=n_times)
    return list(dates)


def generate_synthetic_data(
    n_times: int = 1000,
    n_stocks: int = 500,
    noise_ratio: float = 0.3,
    seed: int = 42,
    start_date: Optional[str] = None,
) -> Iterator[Frame3D]:
    """主生成器：逐日 yield 包含 OHLC + turnover + volume + market_cap 的 Frame3D。
    
    noise_ratio=0 时完全可预测，noise_ratio=1 时信号淹没在噪声中。
    若提供 start_date（YYYY-MM-DD），time index 使用真实交易日。
    """
    rng = np.random.default_rng(seed)
    stock_names = [f'S{str(i).zfill(4)}' for i in range(n_stocks)]
    
    # 时间索引
    time_keys = _make_time_keys(n_times, start_date)
    
    # 1. 隐藏因子（+20 用于前瞻收益）
    hidden_factors = _generate_hidden_factors(n_times + 20, rng)
    hidden_factors_trimmed = hidden_factors[:n_times, :]
    
    # 2. 股票参数
    params = _generate_stock_params(n_stocks, rng)
    sigma = params['sigma']
    weights = params['weights']
    
    # 3. log_price 路径
    log_prices = np.zeros((n_times, n_stocks))
    log_prices[0] = rng.normal(0, 0.3, n_stocks)
    
    signal = hidden_factors_trimmed @ weights.T
    noise = rng.normal(0, 1, (n_times, n_stocks))
    true_fwd_ret = signal + noise_ratio * noise
    
    for t in range(1, n_times):
        drift = 0.0002
        log_prices[t] = log_prices[t - 1] + drift + sigma * (true_fwd_ret[t - 1] + rng.normal(0, 0.5, n_stocks))
    
    # 4. OHLC
    ohlc = _generate_ohlc(log_prices, rng)
    
    # 5. market_cap
    mcap = _generate_market_cap(n_times, n_stocks, rng)
    
    # 6. volume 和 turnover
    intraday_range = np.abs(ohlc['close'] - ohlc['open'])
    volume = intraday_range * mcap * 0.01 * np.exp(rng.normal(0, 0.5, (n_times, n_stocks)))
    volume = np.maximum(volume, mcap * 0.0001)
    turnover = volume / mcap
    
    # 7. 逐日生成 Frame3D
    for t in range(n_times):
        tk = time_keys[t]
        arrays = [
            [tk] * n_stocks,
            stock_names
        ]
        mi = pd.MultiIndex.from_arrays(arrays, names=['key', 'name'])
        df = pd.DataFrame({
            'open': ohlc['open'][t],
            'high': ohlc['high'][t],
            'low': ohlc['low'][t],
            'close': ohlc['close'][t],
            'turnover': turnover[t],
            'volume': volume[t],
            'market_cap': mcap[t],
        }, index=mi)
        if t % 100 == 0:
            logging.info(f"[DataGen] Generated day {t}/{n_times}, noise_ratio={noise_ratio}")
        yield Frame3D(df)
        time.sleep(0.2)
