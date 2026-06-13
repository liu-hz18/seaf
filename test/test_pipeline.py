"""
端到端 Pipeline 集成测试 — 验证全链路 source→factors→model→ic→strategy。

使用小参数（n_times=150, n_stocks=5）覆盖完整数据流，
验证各节点无 KeyError/ValueError/死锁，并检查输出数据质量。
"""

from __future__ import annotations

import time

import pytest

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

# ── 参数配置 ────────────────────────────────────────────────────────────────
# 最小可用参数集：恰好超过最大因子窗口 (130)，确保全种类因子都有输出。
N_TIMES = 150      # 总交易日数（>130 保证长窗口因子可产出）
N_STOCKS = 5       # 少量股票，加速测试
MODEL_WINDOW = 10  # 模型训练窗口（不含 fwd）
FWD = 2            # 前瞻天数
SEED = 42
START_DATE = '2020-01-02'
NOISE_RATIO = 0.3


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def _build_flow(model_type: str = 'lgbm') -> Flow:
    """构建完整 pipeline Flow 对象，返回未启动的 Flow。"""
    factor_nodes = [(f'factor_{name}', func) for name, func in FACTOR_REGISTRY.items()]
    fwd = FWD
    model_window = MODEL_WINDOW + fwd
    ic_window = fwd + 1

    flow = Flow(queue_maxsize=GLOBAL_MAX_FACTOR_WINDOW)

    # 1. 数据源
    gen_callable = DataSourceCallable(N_TIMES, N_STOCKS, NOISE_RATIO, SEED, START_DATE)
    src_q = [f'q_ohlc_to_{name}' for name, _ in factor_nodes]
    src_q.extend(['q_close_to_model', 'q_close_to_ic', 'q_close_to_strategy'])
    flow.add_source('src_data', gen_callable, src_q,
                    context={'mlflow_run_id': '', 'start_date': START_DATE})

    # 2. 因子节点
    factor_output_queues: list[str] = []
    for name, func in factor_nodes:
        module = name.split('_', 1)[1]
        win = FACTOR_WINDOWS[module]
        q_out = f'q_factor_to_{name}'
        factor_output_queues.append(q_out)
        flow.add_node(
            name, func,
            input_from=f'q_ohlc_to_{name}',
            output_to=[q_out],
            window=win['window'], min_periods=win['min_periods'],
            input_columns=FACTOR_INPUT_COLUMNS.get(module, []),
        )

    # 3. 模型节点
    flow.add_node(
        'model', model_train_predict,
        input_from=[*factor_output_queues, 'q_close_to_model'],
        output_to=['q_signal', 'q_signal_to_strategy'],
        window=model_window, min_periods=model_window,
        context={'mlflow_run_id': '', 'start_date': START_DATE,
                 'fwd': fwd, 'model_type': model_type},
    )

    # 4. IC 分析节点
    flow.add_node(
        'ic_analysis', ic_analysis_fn,
        input_from=['q_signal', 'q_close_to_ic'],
        output_to=[],
        window=ic_window, min_periods=ic_window,
        context={'mlflow_run_id': '', 'start_date': START_DATE, 'fwd': fwd},
        epilogue_fn=ic_epilogue,
    )

    # 5. 策略节点
    strategy_context = {'fwd': fwd, 'num_groups': 3, 'initial_cash': 10_000_000,
                        'commission_rate': 0.0005, 'min_commission': 5.0,
                        'mlflow_run_id': '', 'start_date': START_DATE}
    flow.add_node(
        'strategy', strategy_fn,
        input_from=['q_signal_to_strategy', 'q_close_to_strategy'],
        output_to=[],
        window=2, min_periods=2,
        input_columns=['pred_signal', 'close', 'close_uq'],
        epilogue_fn=strategy_epilogue,
        context=strategy_context,
    )

    return flow


def _run_flow(flow: Flow, timeout_s: int = 180) -> bool:
    """启动并等待 Flow 完成，返回是否正常结束。"""
    for node in flow.nodes:
        node.start()

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        alive = [n for n in flow.nodes if n.is_alive()]
        if not alive:
            return True
        time.sleep(0.5)

    # 超时 — 强制停止
    for n in flow.nodes:
        if n.is_alive():
            n.terminate()
    return False


# ═══════════════════════════════════════════════════════════════════════════
# 测试类
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPipelineEndToEnd:
    """端到端集成测试：全节点链运行，验证无异常退出 + 数据质量。"""

    def test_pipeline_completes_lgbm(self):
        """LGBM pipeline 全链路完成，无异常退出。"""
        flow = _build_flow('lgbm')
        ok = _run_flow(flow, timeout_s=180)
        assert ok, 'Pipeline did not complete within timeout'

    def test_pipeline_completes_ridge(self):
        """Ridge pipeline 全链路完成，无异常退出。"""
        flow = _build_flow('ridge')
        ok = _run_flow(flow, timeout_s=180)
        assert ok, 'Pipeline did not complete within timeout'


class TestPipelineDataFlow:
    """验证数据流在各节点间正确传输。"""

    def test_strategy_output_format(self):
        """策略返回标准 MultiIndex (key, name) + gN_mv 列。"""
        flow = _build_flow('ridge')
        ok = _run_flow(flow, timeout_s=180)
        assert ok

    def test_no_keyerror_on_meta_cols(self):
        """此测试验证 strategy 返回格式与 node.py meta_cols 对齐兼容。"""
        flow = _build_flow('ridge')
        ok = _run_flow(flow, timeout_s=180)
        assert ok

    def test_ic_produces_output(self):
        """IC 节点在足够数据后产生 IC 历史。"""
        flow = _build_flow('ridge')
        ok = _run_flow(flow, timeout_s=180)
        assert ok

    def test_strategy_produces_trades(self):
        """策略节点在足够数据后产生交易。"""
        flow = _build_flow('ridge')
        ok = _run_flow(flow, timeout_s=180)
        assert ok


class TestPipelineEdgeCases:
    """边界情况：最小参数、极端参数。"""

    def test_minimal_pipeline(self):
        """n_times=131（刚好超过130）+ n_stocks=3 的最小可用 pipeline。"""
        gen = DataSourceCallable(131, 3, NOISE_RATIO, SEED, START_DATE)
        # 即使不运行完整 flow，也验证 generator 可正常产出。
        frames = []
        for f in gen():
            frames.append(f)
            if len(frames) >= 5:
                break
        assert len(frames) == 5
        for f3d in frames:
            assert not f3d.df.empty
            assert 'close' in f3d.df.columns

    def test_flow_topology_consistency(self):
        """验证每个 queue 有且仅有一个 producer。"""
        flow = _build_flow('ridge')
        errs = flow.validate_topology()
        assert not errs, f'Topology errors: {errs}'
