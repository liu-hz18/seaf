"""
SEAF 量化回测框架主入口 — Pipeline 组装与执行。

拓扑：1 source → 10 factor nodes → model → ic_analysis

运行：python pipeline.py --noise-ratio 0.3 --n-times 1000 --n-stocks 500 --start-date 2020-01-02 --fwd 20
"""
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from qpipe.flow import Flow
from seafquant.data_generator import generate_synthetic_data
from seafquant.factor.trend import compute_trend_factors
from seafquant.factor.momentum import compute_momentum_factors
from seafquant.factor.volatility import compute_volatility_factors
from seafquant.factor.liquidity import compute_liquidity_factors
from seafquant.factor.value import compute_value_factors
from seafquant.factor.quality_merged import compute_quality_merged_factors
from seafquant.factor.quality_pattern import compute_quality_pattern_factors
from seafquant.factor.quality_autocorr import compute_quality_autocorr_factors
from seafquant.factor.counting import compute_counting_factors
from seafquant.factor.interaction import compute_interaction_factors
from seafquant.factor.cross_section import compute_cross_section_factors
from seafquant.model_node import model_train_predict
from seafquant.ic_analysis import ic_analysis_fn, ic_epilogue


class DataSourceCallable:
    """模块级可调用类，用于 pickle 安全的 SourceNode gen_func。"""
    def __init__(self, n_times: int, n_stocks: int, noise_ratio: float, seed: int,
                 start_date: str | None = None):
        self.n_times = n_times
        self.n_stocks = n_stocks
        self.noise_ratio = noise_ratio
        self.seed = seed
        self.start_date = start_date

    def __call__(self):
        return generate_synthetic_data(
            n_times=self.n_times, n_stocks=self.n_stocks,
            noise_ratio=self.noise_ratio, seed=self.seed,
            start_date=self.start_date,
        )


def main():
    parser = argparse.ArgumentParser(description='SEAF Quantitative Backtest Framework')
    parser.add_argument('--noise-ratio', type=float, default=0.3)
    parser.add_argument('--n-times', type=int, default=1000)
    parser.add_argument('--n-stocks', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start-date', type=str, default='2020-01-02')
    parser.add_argument('--model-type', type=str, default='lgbm', choices=['lgbm', 'ridge'])
    parser.add_argument('--fwd', type=int, default=20,
                        help='Forward prediction horizon in days (controls model/IC windows)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format='%(message)s', stream=sys.stdout)

    # ===== 因子节点注册（10 个并行节点，已合并耗时匹配） =====
    factor_nodes = [
        ('factor_trend', compute_trend_factors),             # 趋势 (MA+MACD) 16cols≈1.07s
        ('factor_momentum', compute_momentum_factors),       # 动量+反转 32cols≈0.96s
        ('factor_volatility', compute_volatility_factors),   # 波动+日内 33cols≈0.99s
        ('factor_liquidity', compute_liquidity_factors),     # 流动+规模 32cols≈1.12s
        ('factor_value', compute_value_factors),             # 价值 16cols≈0.76s
        ('factor_quality_merged', compute_quality_merged_factors), # 质量合并(basic+cs_neut) 25cols≈1.02s
        ('factor_quality_pattern', compute_quality_pattern_factors), # 质量形态+高级 9cols≈1.07s
        ('factor_quality_autocorr', compute_quality_autocorr_factors), # 自相关 4cols≈1.18s
        ('factor_counting', compute_counting_factors),       # 计数 16cols≈0.87s
        ('factor_cross_section', compute_cross_section_factors), # 截面排名 10cols≈1.03s
        ('factor_interaction', compute_interaction_factors), # 交互 16cols≈0.80s
    ]

    # 窗口参数（基于 fwd 动态计算）
    fwd = args.fwd
    FACTOR_WINDOW = 130
    FACTOR_MIN_PERIODS = 2
    MODEL_WINDOW = FACTOR_WINDOW + fwd
    MODEL_MIN_PERIODS = MODEL_WINDOW
    IC_WINDOW = fwd + 1
    IC_MIN_PERIODS = IC_WINDOW

    flow = Flow()

    # ===== 1. 数据源节点 =====
    gen_callable = DataSourceCallable(args.n_times, args.n_stocks, args.noise_ratio, args.seed, args.start_date)
    src_output_queues = [f'q_ohlc_to_{name}' for name, _ in factor_nodes]
    src_output_queues.extend(['q_close_to_model', 'q_close_to_ic'])
    flow.add_source('src_data', gen_callable, src_output_queues)

    # ===== 2. 因子计算节点 =====
    factor_output_queues = []
    for fname, ffunc in factor_nodes:
        q_out = f'q_{fname}_out'
        factor_output_queues.append(q_out)
        flow.add_node(name=fname, func=ffunc,
                      input_from=f'q_ohlc_to_{fname}',
                      output_to=[q_out],
                      window=FACTOR_WINDOW, min_periods=FACTOR_MIN_PERIODS)

    # ===== 3. 模型训练预测节点 =====
    model_context = {
        'model_type': args.model_type,
        'fwd': fwd,
        'model_window': MODEL_WINDOW,
    }
    flow.add_node(name='model', func=model_train_predict,
                  input_from=factor_output_queues + ['q_close_to_model'],
                  output_to=['q_signal'],
                  window=MODEL_WINDOW, min_periods=MODEL_MIN_PERIODS,
                  context=model_context)

    # ===== 4. IC 分析节点 =====
    ic_context = {'fwd': fwd}
    flow.add_node(name='ic_analysis', func=ic_analysis_fn,
                  input_from=['q_signal', 'q_close_to_ic'],
                  output_to=[],
                  window=IC_WINDOW, min_periods=IC_MIN_PERIODS,
                  epilogue_fn=ic_epilogue,
                  context=ic_context)

    # ===== 启动 =====
    logging.info("=" * 50)
    logging.info(f"SEAF Pipeline: n_times={args.n_times}, n_stocks={args.n_stocks}, "
                 f"noise_ratio={args.noise_ratio}, seed={args.seed}, start_date={args.start_date}, "
                 f"fwd={fwd}, model_type={args.model_type}")
    logging.info(f"Topology: 1 source -> {len(factor_nodes)} factor nodes -> model -> ic_analysis")
    logging.info(f"Windows: factor={FACTOR_WINDOW}, model={MODEL_WINDOW}, ic={IC_WINDOW}")
    logging.info("=" * 50)

    flow.start()
    flow.join()
    logging.info("Pipeline completed.")


if __name__ == '__main__':
    main()