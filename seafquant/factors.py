"""因子计算主入口 — 12 节点（含估值因子，baostock 专用）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from seafquant.factor.counting import compute_counting_factors
from seafquant.factor.interaction import compute_interaction_factors
from seafquant.factor.liquidity import compute_liquidity_factors
from seafquant.factor.momentum import compute_momentum_factors
from seafquant.factor.precision import compute_precision_factors
from seafquant.factor.quality_merged import compute_quality_merged_factors
from seafquant.factor.quality_pa import compute_quality_pa_factors
from seafquant.factor.trend_cs import compute_trend_cs_factors
from seafquant.factor.tspct import compute_tspct_factors
from seafquant.factor.valuation import compute_valuation_factors
from seafquant.factor.value import compute_value_factors
from seafquant.factor.volatility import compute_volatility_factors

if TYPE_CHECKING:
    from collections.abc import Callable

    from qpipe.frame3d import Frame3D


FACTOR_REGISTRY: dict[str, Callable[[str, int, Frame3D, Any], Frame3D]] = {
    'momentum':         compute_momentum_factors,
    'volatility':       compute_volatility_factors,
    'liquidity':        compute_liquidity_factors,
    'value':            compute_value_factors,
    'quality_merged':   compute_quality_merged_factors,
    'quality_pa':       compute_quality_pa_factors,
    'trend_cs':         compute_trend_cs_factors,
    'counting':         compute_counting_factors,
    'interaction':      compute_interaction_factors,
    'precision':        compute_precision_factors,
    'tspct':            compute_tspct_factors,
    'valuation':        compute_valuation_factors,
}

FACTOR_WINDOWS: dict[str, dict[str, int]] = {
    'momentum':         {'window': 130, 'min_periods': 120},
    'volatility':       {'window': 130, 'min_periods': 120},
    'value':            {'window': 130, 'min_periods': 120},
    'quality_merged':   {'window': 130, 'min_periods': 120},
    'quality_pa':       {'window': 130, 'min_periods': 120},
    'trend_cs':         {'window': 130, 'min_periods': 120},
    'liquidity':        {'window': 70, 'min_periods': 60},
    'counting':         {'window': 70, 'min_periods': 60},
    'interaction':      {'window': 70, 'min_periods': 60},
    'precision':        {'window': 70, 'min_periods': 60},
    'tspct':            {'window': 70, 'min_periods': 60},
    'valuation':        {'window': 130, 'min_periods': 120},  # max=120 (MA120, Z-score120)
}

GLOBAL_MAX_FACTOR_WINDOW = max(c['window'] for c in FACTOR_WINDOWS.values())

FACTOR_INPUT_COLUMNS: dict[str, list[str]] = {
    'momentum':         ['close', 'open', 'high', 'low', 'volume', 'turnover'],
    'volatility':       ['close', 'open', 'high', 'low', 'volume'],
    'liquidity':        ['close', 'volume', 'turnover', 'market_cap'],
    'value':            ['close', 'market_cap', 'turnover'],
    'quality_merged':   ['close', 'high', 'low', 'market_cap', 'volume'],
    'quality_pa':       ['close', 'high', 'low'],
    'trend_cs':         ['close', 'volume'],
    'counting':         ['close', 'turnover'],
    'interaction':      ['close', 'high', 'low', 'volume', 'turnover', 'market_cap'],
    'precision':        ['close', 'vwap'],
    'tspct':            ['open', 'high', 'low', 'close', 'volume', 'turnover'],
    'valuation':        ['close', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM'],
}

FACTOR_PREFIXES: dict[str, str] = {
    'momentum':         'factor_mom',
    'volatility':       'factor_vol',
    'liquidity':        'factor_liq',
    'value':            'factor_val',
    'quality_merged':   'factor_qb',
    'quality_pa':       'factor_qa',
    'trend_cs':         'factor_trend',
    'counting':         'factor_cnt',
    'interaction':      'factor_inter',
    'precision':        'factor_prec',
    'tspct':            'factor_tspct',
    'valuation':        'factor_est',
}

__all__ = ['FACTOR_PREFIXES','FACTOR_REGISTRY','FACTOR_WINDOWS','GLOBAL_MAX_FACTOR_WINDOW']
