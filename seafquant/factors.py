"""
因子计算主入口 — 聚合所有因子类别。
每类因子在独立模块中实现，可在独立的进程节点中并行计算。
"""
from seafquant.factor.factors_momentum import compute_momentum_factors
from seafquant.factor.factors_reversal import compute_reversal_factors
from seafquant.factor.factors_volatility import compute_volatility_factors
from seafquant.factor.factors_liquidity import compute_liquidity_factors
from seafquant.factor.factors_value import compute_value_factors
from seafquant.factor.factors_quality_basic import compute_quality_basic_factors
from seafquant.factor.factors_quality_advanced import compute_quality_advanced_factors
from seafquant.factor.factors_quality_autocorr import compute_quality_autocorr_factors
from seafquant.factor.factors_quality_pattern import compute_quality_pattern_factors
from seafquant.factor.factors_quality_sign import compute_quality_sign_factors
from seafquant.factor.factors_trend import compute_trend_factors
from seafquant.factor.factors_trend_macd import compute_trend_macd_factors
from seafquant.factor.factors_size import compute_size_factors
from seafquant.factor.factors_counting import compute_counting_factors
from seafquant.factor.factors_counting_streak import compute_counting_streak_factors
from seafquant.factor.factors_counting_nh import compute_counting_nh_factors
from seafquant.factor.factors_intraday import compute_intraday_factors
from seafquant.factor.factors_interaction import compute_interaction_factors
from seafquant.factor.factors_cross_section import compute_cross_section_factors
from seafquant.factor.factors_cross_section_neut import compute_cross_section_neut_factors

FACTOR_REGISTRY = {
    'momentum': compute_momentum_factors,
    'reversal': compute_reversal_factors,
    'volatility': compute_volatility_factors,
    'liquidity': compute_liquidity_factors,
    'value': compute_value_factors,    'quality_basic': compute_quality_basic_factors,
    'quality_advanced': compute_quality_advanced_factors,
    'quality_autocorr': compute_quality_autocorr_factors,
    'quality_pattern': compute_quality_pattern_factors,
    'quality_sign': compute_quality_sign_factors,
    'trend': compute_trend_factors,
    'trend_macd': compute_trend_macd_factors,
    'size': compute_size_factors,
    'counting': compute_counting_factors,
    'counting_streak': compute_counting_streak_factors,
    'counting_nh': compute_counting_nh_factors,
    'intraday': compute_intraday_factors,
    'interaction': compute_interaction_factors,
    'cross_section': compute_cross_section_factors,
    'cross_section_neut': compute_cross_section_neut_factors,
}

FACTOR_PREFIXES = {
    'momentum': 'factor_mom', 'reversal': 'factor_rev',
    'volatility': 'factor_vol', 'liquidity': 'factor_liq',
    'value': 'factor_val',    'quality_basic': 'factor_qb', 'quality_advanced': 'factor_qa',
    'quality_autocorr': 'factor_qa', 'quality_pattern': 'factor_qa',
    'quality_sign': 'factor_qa',
    'trend': 'factor_trend', 'trend_macd': 'factor_trend',
    'size': 'factor_size',
    'counting': 'factor_cnt', 'counting_streak': 'factor_cnt',
    'counting_nh': 'factor_cnt',
    'intraday': 'factor_intra', 'interaction': 'factor_inter',
    'cross_section': 'factor_cs', 'cross_section_neut': 'factor_cs',
}

__all__ = [
    'compute_momentum_factors', 'compute_reversal_factors',
    'compute_volatility_factors', 'compute_liquidity_factors',
    'compute_value_factors',    'compute_quality_basic_factors', 'compute_quality_advanced_factors',
    'compute_quality_autocorr_factors',
    'compute_quality_pattern_factors', 'compute_quality_sign_factors',
    'compute_trend_factors', 'compute_trend_macd_factors',
    'compute_size_factors',
    'compute_counting_factors', 'compute_counting_streak_factors',
    'compute_counting_nh_factors',
    'compute_intraday_factors',
    'compute_interaction_factors', 'compute_cross_section_factors',
    'compute_cross_section_neut_factors',
    'FACTOR_REGISTRY', 'FACTOR_PREFIXES',
]
