"""
SEAF 量化回测框架主入口 — Pipeline 组装与执行。

拓扑：
  src_data ──→ 8 个因子节点 (momentum, reversal, volatility, liquidity,
               value, quality, trend, size) → 各输出 16 个因子列
  src_data ──→ model (close 数据) ──→ pred_signal ──→ ic_analysis
  src_data ──→ ic_analysis (close 数据)

运行：python pipeline.py --noise-ratio 0.3 --n-times 1000 --n-stocks 500
"""
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from qpipe.flow import Flow
from seafquant.data_generator import generate_synthetic_data
from seafquant.factors_momentum import compute_momentum_factors
from seafquant.factors_reversal import compute_reversal_factors
from seafquant.factors_volatility import compute_volatility_factors
from seafquant.factors_liquidity import compute_liquidity_factors
from seafquant.factors_value import compute_value_factors
from seafquant.factors_quality import compute_quality_factors
from seafquant.factors_trend import compute_trend_factors
from seafquant.factors_size import compute_size_factors
from seafquant.model_node import model_train_predict
from seafquant.ic_analysis import ic_analysis_fn, ic_epilogue


class DataSourceCallable:
    """模块级可调用类，用于 pickle 安全的 SourceNode gen_func。"""
    def __init__(self, n_times: int, n_stocks: int, noise_ratio: float, seed: int):
        self.n_times = n_times
        self.n_stocks = n_stocks
        self.noise_ratio = noise_ratio
        self.seed = seed

    def __call__(self):
        return generate_synthetic_data(
            n_times=self.n_times,
            n_stocks=self.n_stocks,
            noise_ratio=self.noise_ratio,
            seed=self.seed,
        )


def main():
    parser = argparse.ArgumentParser(description='SEAF Quantitative Backtest Framework')
    parser.add_argument('--noise-ratio', type=float, default=0.3,
                        help='Noise ratio for synthetic data (0=clean, 1=pure noise)')
    parser.add_argument('--n-times', type=int, default=1000,
                        help='Number of time steps (trading days)')
    parser.add_argument('--n-stocks', type=int, default=500,
                        help='Number of stocks')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(message)s',
        stream=sys.stdout,
    )

    # ===== 因子节点注册 =====
    factor_nodes = [
        ('factor_momentum', compute_momentum_factors),
        ('factor_reversal', compute_reversal_factors),
        ('factor_volatility', compute_volatility_factors),
        ('factor_liquidity', compute_liquidity_factors),
        ('factor_value', compute_value_factors),
        ('factor_quality', compute_quality_factors),
        ('factor_trend', compute_trend_factors),
        ('factor_size', compute_size_factors),
    ]

    # 因子窗口：最长周期 120，加缓冲 = 130
    FACTOR_WINDOW = 130
    FACTOR_MIN_PERIODS = 2

    # 模型窗口：200 天因子 + 20 天前瞻 close
    MODEL_WINDOW = 220
    MODEL_MIN_PERIODS = 220

    # IC 窗口：需要 21 天数据（20 日前瞻）
    IC_WINDOW = 21
    IC_MIN_PERIODS = 21

    flow = Flow()

    # ===== 1. 数据源节点 =====
    gen_callable = DataSourceCallable(args.n_times, args.n_stocks, args.noise_ratio, args.seed)

    # Source 输出到 10 个队列：8 因子 + model_close + ic_close
    src_output_queues = [f'q_ohlc_to_{name}' for name, _ in factor_nodes]
    src_output_queues.append('q_close_to_model')
    src_output_queues.append('q_close_to_ic')

    flow.add_source('src_data', gen_callable, src_output_queues)

    # ===== 2. 因子计算节点 =====
    factor_output_queues = []
    for fname, ffunc in factor_nodes:
        q_out = f'q_{fname}_out'
        factor_output_queues.append(q_out)
        flow.add_node(
            name=fname,
            func=ffunc,
            input_from=f'q_ohlc_to_{fname}',
            output_to=[q_out],
            window=FACTOR_WINDOW,
            min_periods=FACTOR_MIN_PERIODS,
        )

    # ===== 3. 模型训练预测节点 =====
    model_input_queues = factor_output_queues + ['q_close_to_model']
    flow.add_node(
        name='model',
        func=model_train_predict,
        input_from=model_input_queues,
        output_to=['q_signal'],
        window=MODEL_WINDOW,
        min_periods=MODEL_MIN_PERIODS,
    )

    # ===== 4. IC 分析节点 =====
    flow.add_node(
        name='ic_analysis',
        func=ic_analysis_fn,
        input_from=['q_signal', 'q_close_to_ic'],
        output_to=[],  # 终端节点
        window=IC_WINDOW,
        min_periods=IC_MIN_PERIODS,
        epilogue_fn=ic_epilogue,
    )

    # ===== 启动 =====
    logging.info("=" * 50)
    logging.info(f"SEAF Pipeline: n_times={args.n_times}, n_stocks={args.n_stocks}, "
                 f"noise_ratio={args.noise_ratio}, seed={args.seed}")
    logging.info(f"Topology: 1 source → 8 factor nodes → model → ic_analysis")
    logging.info("=" * 50)

    flow.start()
    flow.join()

    logging.info("Pipeline completed.")


if __name__ == '__main__':
    main()
