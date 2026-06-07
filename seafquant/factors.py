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
from seafquant.factor.quality_autocorr import compute_quality_autocorr_factors
from seafquant.factor.quality_merged import compute_quality_merged_factors
from seafquant.factor.quality_pattern import compute_quality_pattern_factors
from seafquant.factor.trend import compute_trend_factors
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
}

__all__ = [
    'FACTOR_PREFIXES',
    'FACTOR_REGISTRY',
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
