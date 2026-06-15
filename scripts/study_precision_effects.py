"""
精度损失影响研究 — 独立实验脚本。

探究价格舍入精度 (precision) 对 lgbm / mlp 模型预测能力的影响，
并验证多种缓解方案的有效性。

实验设计：
1. 生成含已知信号结构的合成数据 (n_stocks × n_times, 含隐藏因子)
2. 在高精度 (precision=6) 和低精度 (precision=2) 下做特征工程
3. 分别用 lgbm 和 mlp 训练预测
4. 对比测试集 IC / MSE，验证精度效应
5. 引入缓解方案 (有效数字因子 / VWAP / 梯度特征 / 回报率特征) 并对比提升

用法:
  python scripts/study_precision_effects.py [--n-stocks 50] [--n-times 300] [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s][%(asctime)s] %(message)s',
    stream=sys.stdout,
)
_log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 实验配置
# ═══════════════════════════════════════════════════════════════════════════════

EPS: float = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Precision effects study')
    p.add_argument('--n-stocks', type=int, default=60)
    p.add_argument('--n-times', type=int, default=400)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model-window', type=int, default=250)
    p.add_argument('--fwd', type=int, default=20)
    p.add_argument('--retrain-every', type=int, default=20)
    p.add_argument('--cv-folds', type=int, default=3)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 合成数据生成 — 含隐藏因子结构
# ═══════════════════════════════════════════════════════════════════════════════


def generate_data(
    n_stocks: int,
    n_times: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    生成 shape=(n_times, n_stocks, n_features) 的合成面板数据。

    信号结构：
      - 5 个隐藏因子 (H)   → 驱动未来收益
      - 5 个纯噪声因子 (N)  → 无预测力
      - OHLC price 数据

    Returns: {prices, features, label}
    """
    t = np.arange(n_times, dtype=np.float64).reshape(-1, 1, 1)
    s = np.arange(n_stocks, dtype=np.float64).reshape(1, -1, 1)

    # ---- 隐藏因子（决定未来收益）----
    H = rng.normal(0, 1, (n_times, n_stocks, 5))
    # 缓慢漂移的截面结构
    H += 0.3 * np.sin(t / 50 + s / 10)
    H += 0.2 * np.cos(t / 80 - s / 15)

    # ---- 价格生成 (高精度) ----
    log_price = np.cumsum(
        rng.normal(0.0003, 0.015, (n_times, n_stocks)), axis=0
    ) + np.log(100.0)
    log_price += 0.02 * H[:, :, 0]  # 因子0 驱动价格趋势
    price = np.exp(log_price)  # 高精度基准价格

    # ---- OHLC (高精度) ----
    high_price = price * np.exp(np.abs(rng.normal(0, 0.005, price.shape)))
    low_price = price * np.exp(-np.abs(rng.normal(0, 0.005, price.shape)))
    open_price = price * np.exp(rng.normal(0, 0.003, price.shape))
    volume = np.exp(rng.normal(15, 0.8, price.shape))
    turnover = volume / (1e6 + price * 1e3)

    # ---- 未来收益 (label) ----
    fwd_ret = np.full_like(price, np.nan)
    fwd_ret[:-20] = (price[20:] - price[:-20]) / (price[:-20] + EPS)
    # 标签主要由 H[:,:,0] H[:,:,1] 驱动，加截面噪音
    signal = 0.3 * H[:-20, :, 0] + 0.2 * H[:-20, :, 1]
    cs_noise = rng.normal(0, 0.5, signal.shape)
    fwd_ret[:-20] = signal + cs_noise

    return {
        'price': price,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'volume': volume,
        'turnover': turnover,
        'H': H,
        'label': fwd_ret,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 特征工程 — 不同精度下的因子计算 + 缓解方案
# ═══════════════════════════════════════════════════════════════════════════════


def _cs_zscore(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """截面 z-score（沿 stock 维度）。"""
    mu = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True) + EPS
    return (x - mu) / std


def compute_factors(
    data: dict[str, np.ndarray],
    precision: int,
    *,
    mitigation: str = 'none',
) -> np.ndarray:
    """
    计算因子矩阵 shape=(n_times-fwd, n_stocks, n_features)。

    mitigation 选项:
      'none'          — 标准因子
      'sig_digits'    — 增加有效数字因子
      'vwap'          — 增加 VWAP 因子 (高精度)
      'ret_based'     — 回报率替代价格
      'gradient'      — 价格梯度特征
      'all'           — 以上所有
    """
    price_raw = data['price']
    o, h, lo = data['open'], data['high'], data['low']
    vol, to = data['volume'], data['turnover']
    n_times_full, n_stocks = price_raw.shape

    # 舍入价格
    price = np.round(price_raw, precision)
    _ = np.round(o, precision)
    h_r = np.round(h, precision)
    l_r = np.round(lo, precision)

    fwd = 20
    n_times = n_times_full - fwd
    features: list[np.ndarray] = []

    # === 基础因子（回报率/波动率/量比）===
    for p in [5, 20]:
        ret = (price[p:] - price[:-p]) / (price[:-p] + EPS)
        features.append(ret[:n_times])
        features.append(_cs_zscore(ret[:n_times]))

    # 波动率（逐日 rolling std）
    for w in [20, 60]:
        log_ret = np.log(price[1:] / (price[:-1] + EPS) + 1)
        vol_feat = np.zeros((n_times, n_stocks))
        for i in range(w, n_times + 1):
            vol_feat[i - 1] = np.nanstd(log_ret[i - w:i], axis=0)
        features.append(vol_feat)

    # 成交量特征（当日量 / w日均量）
    for w in [20, 60]:
        vol_chg = np.zeros((n_times, n_stocks))
        for i in range(w, n_times + 1):
            ma = np.nanmean(vol[i - w:i], axis=0)
            vol_chg[i - 1] = vol[i - 1] / (ma + EPS)
        features.append(vol_chg)

    # 换手率特征
    features.append(to[:n_times])
    features.append(_cs_zscore(to[:n_times]))

    # 价格范围
    features.append((h_r[:n_times] - l_r[:n_times]) / (price[:n_times] + EPS))

    # === 缓解方案 ===

    if mitigation in ('sig_digits', 'all'):
        # —— 有效数字因子：量化舍入对每个因子值的影响 ——
        # 对每个返回特征，计算 raw_ret 和 rounded_ret 的差值
        for p in [5, 20]:
            ret_raw = (price_raw[p:] - price_raw[:-p]) / (price_raw[:-p] + EPS)
            ret_rnd = (price[p:] - price[:-p]) / (price[:-p] + EPS)
            sig_dig = np.abs(ret_raw - ret_rnd)  # 精度损失量
            features.append(sig_dig[:n_times])

    if mitigation in ('vwap', 'all'):
        # —— VWAP 因子：用高精度成交量加权均价 ——
        vwap = (h_r + l_r + price) / 3  # 简化 VWAP
        # VWAP 比单个 OHLC 更稳定
        for p in [20]:
            vwap_ret = (vwap[p:] - vwap[:-p]) / (vwap[:-p] + EPS)
            features.append(vwap_ret[:n_times])
        # VWAP 偏离度
        features.append((price[:n_times] - vwap[:n_times]) / (vwap[:n_times] + EPS))

    if mitigation in ('ret_based', 'all'):
        # —— 回报率替代价格 ——
        # 用高精度回报率（计算后再舍入）
        ret_1d = (price_raw[1:] - price_raw[:-1]) / (price_raw[:-1] + EPS)
        for w in [5, 20]:
            cum_ret = np.zeros((n_times, n_stocks))
            for i in range(w, n_times + 1):
                cum_ret[i - w:i] = np.sum(ret_1d[i - w:i], axis=0)
            features.append(cum_ret[:n_times])

    if mitigation in ('gradient', 'all'):
        # —— 价格梯度 + 惯性特征 ——
        # 一阶梯度（变化方向）
        grad_1d = np.zeros((n_times, n_stocks))
        for i in range(1, n_times + 1):
            grad_1d[i - 1] = price[i] - price[i - 1]
        features.append(grad_1d)
        # 二阶梯度（加速度）
        grad_2d = np.zeros((n_times, n_stocks))
        for i in range(2, n_times + 1):
            grad_2d[i - 1] = price[i - 1] - 2 * price[i - 2] + price[i - 3]
        features.append(grad_2d)

    # === 堆叠 + 标准化 ===
    X = np.stack(features, axis=-1)  # (n_times, n_stocks, n_feats)
    # 截面标准化
    X = _cs_zscore(X, axis=1)
    # 填充 NaN
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 模型训练与评估
# ═══════════════════════════════════════════════════════════════════════════════


def _lgbm_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
) -> np.ndarray:
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, num_leaves=31,
        learning_rate=0.05, verbose=-1, random_state=42,
        force_col_wise=True,
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _mlp_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
) -> np.ndarray:
    import torch
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_feats = x_train.shape[1]

    model = torch.nn.Sequential(
        torch.nn.Linear(n_feats, 128),
        torch.nn.LayerNorm(128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 64),
        torch.nn.LayerNorm(64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 32),
        torch.nn.LayerNorm(32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    ).to(device)

    X_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(200):
        optim.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optim.step()

    model.eval()
    X_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(X_test_t).cpu().numpy().ravel()


@dataclass
class Result:
    label: str
    model_type: str
    precision: int
    mitigation: str
    ic_pearson: float = np.nan
    ic_spearman: float = np.nan
    mse: float = np.nan
    train_time_s: float = np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# 实验主流程
# ═══════════════════════════════════════════════════════════════════════════════


def run_experiment(args: argparse.Namespace) -> list[Result]:
    rng = np.random.default_rng(args.seed)
    n_stocks = args.n_stocks
    n_times_full = args.n_times
    fwd = args.fwd
    model_window = args.model_window
    retrain_every = args.retrain_every

    _log.info(f'Generating data: {n_times_full}t × {n_stocks}s ...')
    t0 = time.perf_counter()
    data = generate_data(n_stocks, n_times_full, rng)
    _log.info(f'  done in {time.perf_counter() - t0:.1f}s')

    label = data['label']  # (n_times, n_stocks)
    n_times = n_times_full - fwd

    # 滚动训练测试
    test_start = model_window
    n_rolls = (n_times - test_start) // retrain_every

    _log.info(f'Rolling test: {n_rolls} roll(s), '
              f'window={model_window}, retrain_every={retrain_every}')

    results: list[Result] = []

    configs: list[tuple[int, str, str]] = [
        # (precision, model_type, mitigation)
        (2, 'lgbm', 'none'),
        (2, 'mlp', 'none'),
        (6, 'lgbm', 'none'),
        (6, 'mlp', 'none'),
        # 缓解方案 (仅在 precision=2 时测试)
        (2, 'lgbm', 'sig_digits'),
        (2, 'mlp', 'sig_digits'),
        (2, 'lgbm', 'vwap'),
        (2, 'mlp', 'vwap'),
        (2, 'lgbm', 'ret_based'),
        (2, 'mlp', 'ret_based'),
        (2, 'lgbm', 'gradient'),
        (2, 'mlp', 'gradient'),
        (2, 'lgbm', 'all'),
        (2, 'mlp', 'all'),
    ]

    for precision, model_type, mitigation in configs:
        _log.info(f'  [{model_type}] precision={precision}, mitigation={mitigation}')

        t_start = time.perf_counter()
        X = compute_factors(data, precision, mitigation=mitigation)

        preds_all: list[np.ndarray] = []
        ys_all: list[np.ndarray] = []

        for start in range(test_start, n_times, retrain_every):
            if start + model_window > n_times:
                continue
            if start + model_window + retrain_every > n_times:
                test_end = n_times
            else:
                test_end = start + retrain_every

            X_train = X[start:start + model_window].reshape(-1, X.shape[-1])
            y_train = label[start:start + model_window].reshape(-1)
            X_test = X[start + model_window:test_end].reshape(-1, X.shape[-1])
            y_test = label[start + model_window:test_end].reshape(-1)

            valid = ~np.isnan(y_train) & ~np.isnan(y_test).all(
                axis=0 if y_test.ndim > 1 else None
            )
            if not valid.any():
                continue

            if model_type == 'lgbm':
                pred = _lgbm_fit_predict(X_train, y_train, X_test)
            else:
                pred = _mlp_fit_predict(X_train, y_train, X_test)

            preds_all.append(pred)
            ys_all.append(y_test)

        if not preds_all:
            results.append(Result(
                label='', model_type=model_type,
                precision=precision, mitigation=mitigation,
            ))
            continue

        pred_flat = np.concatenate(preds_all)
        y_flat = np.concatenate(ys_all)
        valid = ~np.isnan(pred_flat) & ~np.isnan(y_flat)
        pred_flat, y_flat = pred_flat[valid], y_flat[valid]

        if len(pred_flat) < 30:
            results.append(Result(
                label='', model_type=model_type,
                precision=precision, mitigation=mitigation,
            ))
            continue

        ic_p, _ = pearsonr(pred_flat, y_flat)
        ic_s = spearmanr(pred_flat, y_flat).correlation
        mse = float(np.mean((pred_flat - y_flat) ** 2))

        elapsed = time.perf_counter() - t_start
        label_str = f'{model_type}_p{precision}_{mitigation}'
        results.append(Result(
            label=label_str, model_type=model_type,
            precision=precision, mitigation=mitigation,
            ic_pearson=ic_p, ic_spearman=ic_s, mse=mse,
            train_time_s=elapsed,
        ))
        _log.info(f'    IC={ic_p:.4f} (Pearson), {ic_s:.4f} (Spearman), '
                  f'MSE={mse:.4f}, time={elapsed:.1f}s')

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════════════════════════════════════════


def print_report(results: list[Result]) -> None:
    baseline = {r.label: r for r in results if r.mitigation == 'none'}
    mitigated = [r for r in results if r.mitigation != 'none']

    print('\n' + '=' * 95)
    print('  精度影响研究报告')
    print('=' * 95)
    print(f'{"配置":<32s} {"IC(Pearson)":>12s} {"IC(Spearman)":>13s} {"MSE":>10s} {"时间":>8s}')
    print('-' * 95)

    for r in results:
        label = f'{r.model_type}(p={r.precision}, {r.mitigation})'
        print(f'{label:<32s} {r.ic_pearson:>+12.4f} {r.ic_spearman:>+13.4f} '
              f'{r.mse:>10.4f} {r.train_time_s:>7.1f}s')

    print('-' * 95)
    print('\n> 精度效应:')
    for mt in ['lgbm', 'mlp']:
        base_6 = baseline.get(f'{mt}_p6_none')
        base_2 = baseline.get(f'{mt}_p2_none')
        if base_6 and base_2:
            delta = base_6.ic_pearson - base_2.ic_pearson
            print(f'  {mt}: precision=6→2, ΔIC={delta:+.4f} '
                  f'({base_6.ic_pearson:.4f} → {base_2.ic_pearson:.4f})')

    print('\n> 缓解方案提升 (vs precision=2 baseline):')
    for mt in ['lgbm', 'mlp']:
        base_2 = baseline.get(f'{mt}_p2_none')
        if base_2 is None:
            continue
        for r in mitigated:
            if r.model_type == mt:
                gain = r.ic_pearson - base_2.ic_pearson
                marker = '+' if gain > 0 else '-'
                print(f'  {mt}+{r.mitigation:<12s}: ΔIC={gain:+7.4f} {marker} '
                      f'(MSE={r.mse:.4f})')

    print('\n> 最优组合:')
    best = max(results, key=lambda r: r.ic_pearson)
    print(f'  {best.model_type}(p={best.precision}, {best.mitigation}) '
          f'IC={best.ic_pearson:.4f}')

    print('=' * 95)


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == '__main__':
    args = parse_args()
    results = run_experiment(args)
    print_report(results)
