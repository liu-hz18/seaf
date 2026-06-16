"""
因子计算主入口 — 聚合所有因子类别（合并后 10 个节点）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from seafquant.factor.counting import compute_counting_factors
from seafquant.factor.cross_section import compute_cross_section_factors
from seafquant.factor.interaction import compute_interaction_factors
from seafquant.factor.liquidity import compute_liquidity_factors
from seafquant.factor.momentum import compute_momentum_factors
from seafquant.factor.precision import compute_precision_factors
from seafquant.factor.quality_autocorr import compute_quality_autocorr_factors
from seafquant.factor.quality_merged import compute_quality_merged_factors
from seafquant.factor.quality_pattern import compute_quality_pattern_factors
from seafquant.factor.trend import compute_trend_factors
from seafquant.factor.tspct import compute_tspct_factors
from seafquant.factor.value import compute_value_factors
from seafquant.factor.volatility import compute_volatility_factors

if TYPE_CHECKING:
    from collections.abc import Callable

    from qpipe.frame3d import Frame3D

FACTOR_REGISTRY: dict[str, Callable[[str, Frame3D, Any], Frame3D]] = {
    'momentum': compute_momentum_factors,
    'volatility': compute_volatility_factors,
    'liquidity': compute_liquidity_factors,
    'value': compute_value_factors,
    'quality_merged': compute_quality_merged_factors,
    'quality_autocorr': compute_quality_autocorr_factors,
    'quality_pattern': compute_quality_pattern_factors,
    'trend': compute_trend_factors,
    'counting': compute_counting_factors,
    'interaction': compute_interaction_factors,
    'cross_section': compute_cross_section_factors,
    'precision': compute_precision_factors,
    'tspct': compute_tspct_factors,
}

# 各因子节点的个性化窗口配置。
# window: 滑动窗口大小 = max_rolling_window + 10
# min_periods: 最小数据量 = max_rolling_window
# 模型/IC 节点的窗口基于 max_window 动态计算。
FACTOR_WINDOWS: dict[str, dict[str, int]] = {
    # max=120 模块 — 需要长窗口（130/120）
    'momentum':          {'window': 130, 'min_periods': 120},
    'volatility':        {'window': 130, 'min_periods': 120},
    'value':             {'window': 130, 'min_periods': 120},
    'quality_merged':    {'window': 130, 'min_periods': 120},
    'quality_autocorr':  {'window': 130, 'min_periods': 120},
    'quality_pattern':   {'window': 130, 'min_periods': 120},
    'trend':             {'window': 130, 'min_periods': 120},
    # max=60 模块 — 短窗口（70/60）
    'liquidity':         {'window': 70, 'min_periods': 60},
    'counting':          {'window': 70, 'min_periods': 60},
    'interaction':       {'window': 70, 'min_periods': 60},
    'cross_section':     {'window': 70, 'min_periods': 60},
    # max=60 模块（VWAP 最大滚动窗口 = 60）
    'precision':         {'window': 70, 'min_periods': 60},
    # max=60（ts_rank 最大窗口 = 60）
    'tspct':             {'window': 70, 'min_periods': 60},
}

# 全局最大窗口（模型/IC 节点基于此计算）
GLOBAL_MAX_FACTOR_WINDOW = max(c['window'] for c in FACTOR_WINDOWS.values())

# 因子节点输入列过滤 — 每个模块只接收实际用到的 OHLCV 列。
# source 发送 7 列到所有因子队列，通过 input_columns 过滤后
# time_order_buffer 仅保留必要列，窗口缓冲内存节省 30-85%。
FACTOR_INPUT_COLUMNS: dict[str, list[str]] = {
    'momentum':         ['close', 'open', 'high', 'low', 'volume', 'turnover'],
    'volatility':       ['close', 'open', 'high', 'low', 'volume'],
    'liquidity':        ['close', 'volume', 'turnover', 'market_cap'],
    'value':            ['close', 'market_cap', 'turnover'],
    'quality_merged':   ['close', 'high', 'low', 'market_cap', 'volume'],
    'quality_autocorr': ['close'],
    'quality_pattern':  ['close', 'high', 'low'],
    'trend':            ['close', 'volume'],
    'counting':         ['close', 'turnover'],
    'interaction':      ['close', 'high', 'low', 'volume', 'turnover', 'market_cap'],
    'cross_section':    ['close', 'volume'],
    'precision':        ['close', 'high', 'low'],
    'tspct':            ['open', 'high', 'low', 'close', 'volume', 'turnover'],
}

FACTOR_PREFIXES: dict[str, str] = {
    'momentum': 'factor_mom',
    'volatility': 'factor_vol',
    'liquidity': 'factor_liq',
    'value': 'factor_val',
    'quality_merged': 'factor_qb',
    'quality_autocorr': 'factor_qa',
    'quality_pattern': 'factor_qa',
    'trend': 'factor_trend',
    'counting': 'factor_cnt',
    'interaction': 'factor_inter',
    'cross_section': 'factor_cs',
    'precision': 'factor_prec',
    'tspct': 'factor_tspct',
}


__all__ = [
    'FACTOR_PREFIXES',
    'FACTOR_REGISTRY',
    'FACTOR_WINDOWS',
    'GLOBAL_MAX_FACTOR_WINDOW',
    'compute_counting_factors',
    'compute_cross_section_factors',
    'compute_interaction_factors',
    'compute_liquidity_factors',
    'compute_momentum_factors',
    'compute_quality_autocorr_factors',
    'compute_quality_merged_factors',
    'compute_quality_pattern_factors',
    'compute_trend_factors',
    'compute_value_factors',
    'compute_volatility_factors',
]
