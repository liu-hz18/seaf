"""
因子计算主入口 — 聚合 8 大类共 128 个因子。
每类因子在独立模块中实现，可在独立的进程节点中并行计算。
"""
from seafquant.factors_momentum import compute_momentum_factors
from seafquant.factors_reversal import compute_reversal_factors
from seafquant.factors_volatility import compute_volatility_factors
from seafquant.factors_liquidity import compute_liquidity_factors
from seafquant.factors_value import compute_value_factors
from seafquant.factors_quality import compute_quality_factors
from seafquant.factors_trend import compute_trend_factors
from seafquant.factors_size import compute_size_factors

# 因子函数注册表：名称 → 函数
FACTOR_REGISTRY = {
    'momentum': compute_momentum_factors,
    'reversal': compute_reversal_factors,
    'volatility': compute_volatility_factors,
    'liquidity': compute_liquidity_factors,
    'value': compute_value_factors,
    'quality': compute_quality_factors,
    'trend': compute_trend_factors,
    'size': compute_size_factors,
}

# 每个大类对应的列名前缀
FACTOR_PREFIXES = {
    'momentum': 'factor_mom',
    'reversal': 'factor_rev',
    'volatility': 'factor_vol',
    'liquidity': 'factor_liq',
    'value': 'factor_val',
    'quality': 'factor_qual',
    'trend': 'factor_trend',
    'size': 'factor_size',
}

# 导出的因子函数列表
FACTOR_FUNCTIONS = list(FACTOR_REGISTRY.values())


def all_factor_columns() -> list:
    """返回所有 128 个因子列名列表。"""
    cols = []
    for category, prefix in FACTOR_PREFIXES.items():
        for i in range(1, 17):
            cols.append(f"{prefix}_{i:02d}")
    return cols


__all__ = [
    'compute_momentum_factors', 'compute_reversal_factors',
    'compute_volatility_factors', 'compute_liquidity_factors',
    'compute_value_factors', 'compute_quality_factors',
    'compute_trend_factors', 'compute_size_factors',
    'FACTOR_REGISTRY', 'FACTOR_PREFIXES', 'FACTOR_FUNCTIONS',
    'all_factor_columns',
]
