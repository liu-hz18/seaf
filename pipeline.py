"""
SEAF 量化回测框架主入口 — Pipeline 组装与执行。

拓扑：1 source → 10 factor nodes → model → ic_analysis

运行：python pipeline.py --noise-ratio 0.3 --n-times 1000 --n-stocks 500 --start-date 2020-01-02 --fwd 20
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

from qpipe.flow import Flow
from seafquant.data_generator import DataSourceCallable
from seafquant.factors import (
    FACTOR_INPUT_COLUMNS,
    FACTOR_REGISTRY,
    FACTOR_WINDOWS,
    GLOBAL_MAX_FACTOR_WINDOW,
)
from seafquant.ic_analysis import ic_analysis_fn, ic_epilogue
from seafquant.model_node import model_train_predict
from seafquant.strategy import strategy_epilogue, strategy_fn


def main() -> None:
    parser = argparse.ArgumentParser(description='SEAF Quantitative Backtest Framework')
    parser.add_argument('--noise-ratio', type=float, default=0.3)
    parser.add_argument('--n-times', type=int, default=1000)
    parser.add_argument('--n-stocks', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start-date', type=str, default='2020-01-02')
    parser.add_argument('--model-type', type=str, default='lgbm', choices=['lgbm', 'ridge', 'mlp'])
    parser.add_argument(
        '--fwd',
        type=int,
        default=20,
        help='Forward prediction horizon in days (controls IC window)',
    )
    parser.add_argument(
        '--model-window',
        type=int,
        default=200,
        help='Model training window in days (factor history for training)',
    )
    parser.add_argument(
        '--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR']
    )
    parser.add_argument('--no-mlflow', action='store_true', default=False, help='Disable MLflow tracking')
    parser.add_argument(
        '--snapshot-interval', type=int, default=100,
        help='Snapshot input/output every N calls (0=disabled)',
    )
    parser.add_argument(
        '--precision', type=int, default=2,
        help='Price/market-cap rounding precision (decimal places)',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"args: {args}")
    # ===== 因子节点注册（由 FACTOR_REGISTRY 派生，10 个并行节点） =====
    factor_nodes = [(f'factor_{name}', func) for name, func in FACTOR_REGISTRY.items()]

    # 窗口参数
    # Model 节点独立于 Factor 节点：它接收的是因子输出（截面），需要独立的历史窗口来训练。
    fwd = args.fwd
    MODEL_WINDOW = args.model_window + args.fwd
    MODEL_MIN_PERIODS = MODEL_WINDOW
    IC_WINDOW = fwd + 1
    IC_MIN_PERIODS = IC_WINDOW

    # ===== MLflow 初始化 =====
    experiment_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not args.no_mlflow:
        import mlflow  # 延迟导入 — 仅在需要时加载

        mlflow.set_tracking_uri('sqlite:///mlruns.db')
        mlflow.set_experiment(experiment_name)
        run_name = f'{args.model_type}-w{args.model_window}-f{args.fwd}-{args.start_date}'
        mlflow_run = mlflow.start_run(run_name=run_name)
        mlflow_run_id: str = mlflow_run.info.run_id
    else:
        mlflow_run_id = ''

    # ---- 日志文件：主进程写入 logs/{run_id}.txt ----
    _log = logging.getLogger(__name__)
    if mlflow_run_id:
        os.makedirs('logs', exist_ok=True)
        _log.addHandler(logging.FileHandler(f'logs/{mlflow_run_id}.txt', encoding='utf-8'))
    else:
        os.makedirs('logs', exist_ok=True)
        _log.addHandler(logging.FileHandler('logs/no_mlflow.txt', encoding='utf-8'))

    flow = Flow(queue_maxsize=GLOBAL_MAX_FACTOR_WINDOW)

    # ===== 1. 数据源节点 =====
    gen_callable = DataSourceCallable(
        args.n_times, args.n_stocks, args.noise_ratio, args.seed, args.start_date,
        precision=args.precision,
    )
    src_output_queues = [f'q_ohlc_to_{name}' for name, _ in factor_nodes]
    src_output_queues.extend(['q_close_to_model', 'q_close_to_ic', 'q_close_to_strategy'])
    flow.add_source(
        'src_data',
        gen_callable,
        src_output_queues,
        context={'mlflow_run_id': mlflow_run_id, 'start_date': args.start_date, 'precision': args.precision},
        snapshot_interval=args.snapshot_interval,
        log_level=args.log_level,
    )

    # ===== 2. 因子计算节点 =====
    factor_output_queues = []
    for fname, ffunc in factor_nodes:
        # 提取模块短名（"factor_counting" → "counting"）
        module_key = fname.removeprefix('factor_')
        fw = FACTOR_WINDOWS.get(module_key, {'window': 130, 'min_periods': 60})
        input_cols = FACTOR_INPUT_COLUMNS.get(module_key)
        q_out = f'q_{fname}_out'
        factor_output_queues.append(q_out)
        flow.add_node(
            name=fname,
            func=ffunc,
            input_from=f'q_ohlc_to_{fname}',
            input_columns=input_cols,
            output_to=[q_out],
            window=fw['window'],
            min_periods=fw['min_periods'],
            context={'mlflow_run_id': mlflow_run_id, 'start_date': args.start_date, 'precision': args.precision},
            snapshot_interval=args.snapshot_interval,
            log_level=args.log_level,
        )

    # ===== 3. 模型训练预测节点 =====
    # Label: cs_zscore(ln(close[t+fwd]) - ln(close[t+1])) — (fwd-1)日截面对数超额收益
    # t+1买入、t+fwd卖出，对齐实盘交易执行和IC指标
    # model_context 显式列出所有可配置参数（与 model_node.setdefault 默认值对齐）
    model_context = {
        'model_type': args.model_type,
        'fwd': fwd,
        'model_window': MODEL_WINDOW,
        'mlflow_run_id': mlflow_run_id,
        'start_date': args.start_date,
        'precision': args.precision,
        # 以下参数使用 model_node 默认值，此处仅为文档可读性显式列出
        # 'retrain_every': 20,
    }
    flow.add_node(
        name='model',
        func=model_train_predict,
        input_from=[*factor_output_queues, 'q_close_to_model'],
        output_to=['q_signal', 'q_signal_to_strategy'],
        window=MODEL_WINDOW,
        min_periods=MODEL_MIN_PERIODS,
        # 排除原始 OHLCV 列：基建层在入队前删除，防止泄露到特征空间。
        # close 保留用于 label 计算，model_node 内部会排除出特征列。
        exclude_input_columns=['open', 'high', 'low', 'close_uq',
                               'turnover', 'volume', 'market_cap'],
        context=model_context,
        snapshot_interval=args.snapshot_interval,
        log_level=args.log_level,
    )

    # ===== 4. IC 分析节点 =====
    # IC：(fwd-1)日截面对数超额收益的 Spearman rank 相关系数
    # 与 model 节点的 label 定义对齐（ln(close[t+fwd]) - ln(close[t+1])）
    ic_context = {
        'fwd': fwd, 'num_groups': 10,
        'mlflow_run_id': mlflow_run_id, 'start_date': args.start_date,
        'precision': args.precision,
    }
    flow.add_node(
        name='ic_analysis',
        func=ic_analysis_fn,
        input_from=['q_signal', 'q_close_to_ic'],
        output_to=[],
        window=IC_WINDOW,
        min_periods=IC_MIN_PERIODS,
        input_columns=['pred_signal', 'close'],
        epilogue_fn=ic_epilogue,
        context=ic_context,
        snapshot_interval=args.snapshot_interval,
        log_level=args.log_level,
    )

    # ===== 5. 策略绩效节点 =====
    strategy_context = {
        'fwd': fwd,
        'num_groups': 10,
        'initial_cash': 10_000_000,
        'mlflow_run_id': mlflow_run_id,
        'start_date': args.start_date,
        'precision': args.precision,
    }
    flow.add_node(
        name='strategy',
        func=strategy_fn,
        input_from=['q_signal_to_strategy', 'q_close_to_strategy'],
        output_to=[],
        window=2,
        min_periods=2,
        input_columns=['pred_signal', 'close', 'close_uq', 'stock_name'],
        epilogue_fn=strategy_epilogue,
        context=strategy_context,
        snapshot_interval=args.snapshot_interval,
        log_level=args.log_level,
    )

    # ===== 记录启动参数与 git 版本 =====
    if not args.no_mlflow:
        for key, val in vars(args).items():
            mlflow.log_param(key, val)
        try:
            git_hash = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                check=False, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if git_hash:
                mlflow.log_param('git_commit', git_hash)
        except Exception:
            pass

    # ===== 启动 =====
    logging.info('=' * 50)
    if not args.no_mlflow:
        logging.info(
            f'MLflow experiment="{experiment_name}" run_id={mlflow_run_id}'
            f' (tracking_uri=sqlite:///mlruns.db)'
        )
    logging.info(
        f'SEAF Pipeline: n_times={args.n_times}, n_stocks={args.n_stocks}, '
        f'noise_ratio={args.noise_ratio}, seed={args.seed}, start_date={args.start_date}, '
        f'fwd={fwd}, model_type={args.model_type}'
    )
    logging.info(f'Topology: 1 source -> {len(factor_nodes)} factor nodes -> model -> ic_analysis + strategy')
    logging.info(f'Model window={MODEL_WINDOW}, IC window={IC_WINDOW}')
    factor_window_summary = ', '.join(
        f'{k}={v["window"]}' for k, v in sorted(FACTOR_WINDOWS.items())
    )
    logging.info(f'Factor windows: {factor_window_summary}')
    logging.info('=' * 50)

    flow.start()
    flow.join()
    if not args.no_mlflow:
        mlflow.end_run()
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri('')  # reset to avoid stale URI in parent process
    logging.info('Pipeline completed.')


if __name__ == '__main__':
    main()
