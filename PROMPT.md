────────────────────────────────────────────────────────────

量化回测框架（SEAF）完整开发指令
────────────────────────────────────────

0. 角色与前置约束

你是一位拥有 15 年经验的量化系统架构师。你的工作方式是先阅读再动手，先验证再推进。在本项目的任何代码编写之前，你必须完成 Phase 0 的全部阅读任务，并在你的推理中输出一份结构化的代码理解摘要。跳过阅读就直接编码是本指令明确禁止的行为。

关键约束（硬性）：
- 单文件不超过 300 行。超过即拆分。
- 任何数值计算函数必须先写单元测试（test/ 目录），测试通过后再集成。
- 每个 Phase 完成后必须运行 pipeline.py 验证端到端正确性。不允许多个 Phase 连续实现后再验证。
- 所有 git commit 信息必须中英文双语，格式：[Phase X] <简短描述> | <English desc>。
- 每次改动须同步更新 CHANGELOG.md，记录：变更内容、变更原因、遇到的问题、解决方案、下一步计划。

────────────────────────────────────────────────────────────

1. 项目架构理解

1.1 目录结构
seaf/
├── qpipe/           # 基建层（已实现，需扩展）
│   ├── __init__.py
│   ├── frame3d.py   # Frame3D 类（需扩展 API）
│   ├── node.py      # MultiInputNode, SourceNode（需增加 context/epilogue_fn）
│   ├── flow.py      # Flow 编排器（已完备）
│   └── example.py   # 示例代码
├── seafquant/       # 业务层（待实现）
│   └── __init__.py
├── test/            # 单元测试
│   └── __init__.py
├── pipeline.py      # 程序入口（待实现）
└── requirements.txt

1.2 基建层核心机制（必须理解后再动手）

Frame3D：
- 内部 df 是一个 pd.DataFrame，索引为 pd.MultiIndex，层级为 [time, stock]，即 df.index.names = ['key', 'name']。
- time 层是时序维度，stock 层是截面维度。列是属性维度。
- 当前 Frame3D 没有任何时序或截面的计算 API。你需要扩展它，但必须遵循以下设计原则：
- 所有方法返回新的 Frame3D 或 pd.Series/DataFrame，不原地修改。
- 时序操作沿 key 层计算，截面操作沿 name 层计算。
- 方法命名：时序用 ts_* 前缀，截面用 cs_* 前缀。

MultiInputNode 数据合并语义：
- 来自多个上游队列的同一 time 的 Frame3D，通过 pd.concat(df_list, axis=1) 按列合并 。这意味着如果不同上游输出了同名列，会出现列重复。你的 pipeline 设计中必须用 input_columns 精确指定需要的列，避免歧义和有明显含义的列名重名。
- 合并后的数据进入 time_order_buffer（deque），按 window 和 min_periods 控制滑动窗口。
- func(name, f3d) 接收的 f3d 包含窗口内全部时间的数据。函数返回后，node 自动提取最新时间的行输出到下游队列。
- 这意味着一件事：如果你在 func 中做时序计算（如 rolling mean），你实际上只需要计算最新那一天的 rolling 结果，因为只有最新一天会被输出。但你可以利用窗口内的历史数据来完成这个计算。

SourceNode：
- gen_func() 返回 Iterator[Frame3D]。每次 yield 一个 Frame3D，node 自动取其最新 time 行广播到输出队列。
- 生成器函数的调用发生在子进程中，因此它必须是 pickle 可序列化的。这意味着：
- ❌ 闭包（在函数内部定义的函数）
- ❌ lambda 表达式
- ✅ 模块级定义的函数
- ✅ 实现了 __call__ 的类实例（类本身必须在模块顶层定义）

Flow 拓扑约束（由 validate_topology 强制执行）：
1. 节点名全局唯一
2. 每条 queue 只能有一个生产者（不能多写一读）
3. 每条 queue 必须有消费者
4. 无孤立节点
5. 有向无环（DAG）

────────────────────────────────────────────────────────────

2. Phase 0：深度代码阅读（必须先完成）

在编写任何代码之前，逐文件阅读并输出理解摘要：

2.1 阅读清单
- [ ] qpipe/frame3d.py — 输出：Frame3D 的索引结构、现有方法清单、缺失的 API 列表
- [ ] qpipe/node.py — 输出：MultiInputNode 的完整生命周期（启动→收数→合并→滑窗→计算→输出→退出）、SourceNode 的完整生命周期、哪些地方需要改造
- [ ] qpipe/flow.py — 输出：Flow 的拓扑验证规则、节点注册流程
- [ ] qpipe/example.py — 输出：示例中的数据流拓扑图（ASCII 图 + 文字描述）

2.2 阅读完成后
在 CHANGELOG.md 中记录你对基建层的理解摘要，包括：
- 框架的优势（流式、多进程并行、滑窗机制）
- 框架的劣势或限制（无状态 node、Frame3D API 贫乏、pickle 约束）

────────────────────────────────────────────────────────────

3. Phase 1：基建层扩展

3.1 Frame3D 扩展（`qpipe/frame3d.py`）

你需要新增以下 API。每个方法必须先有单元测试（test/test_frame3d.py）。

时序 API（沿 `key` 层 / 时间轴计算）

def ts_delay(self, col: str, periods: int) -> 'Frame3D':
    """时序滞后：将指定列向下平移 periods 个时间单位。NaN 填充。
    注意：这会在每个 stock 内部独立平移，不是整体平移。"""

def ts_delta(self, col: str, periods: int) -> 'Frame3D':
    """时序差分：col(t) - col(t-periods)。"""

def ts_pct_change(self, col: str, periods: int) -> 'Frame3D':
    """时序百分比变化：(col(t) - col(t-periods)) / col(t-periods)。"""

def ts_rolling(self, col: str, window: int, agg_fn: str) -> 'Frame3D':
    """时序滚动聚合。agg_fn 支持: 'mean','std','min','max','sum','skew','kurt'。
    每个 stock 独立计算。min_periods = max(1, window//2)。"""

def ts_zscore(self, col: str, window: int) -> 'Frame3D':
    """时序标准化：(x - rolling_mean) / rolling_std，每个 stock 独立。"""

def ts_rank(self, col: str, window: int) -> 'Frame3D':
    """时序排名：在滚动窗口内，当前值的历史百分位排名（0~1）。"""

截面 API（沿 `name` 层 / 股票轴计算）

def cs_zscore(self, col: str) -> 'Frame3D':
    """截面标准化：(x - cross_sectional_mean) / cross_sectional_std。
    对每个 time 独立计算。处理 std=0 的情况：返回 0。"""

def cs_rank(self, col: str) -> 'Frame3D':
    """截面排名百分位（0~1），对每个 time 独立计算。"""

def cs_demean(self, col: str) -> 'Frame3D':
    """截面去均值：x - cross_sectional_mean。"""

def cs_neutralize(self, col: str, by: List[str]) -> 'Frame3D':
    """截面中性化：对 col 按 by 中的列做截面回归，取残差。
    回归前自动对 by 做 cs_zscore。"""

通用工具 API

def get_cs_series(self, col: str, time_key) -> pd.Series:
    """获取指定时间截面的 Series，index 为 stock name。"""

def get_ts_series(self, stock: str, col: str) -> pd.Series:
    """获取指定股票的时序 Series，index 为 time key。"""

def add_column(self, name: str, values: Union[pd.Series, np.ndarray]) -> 'Frame3D':
    """安全添加列，自动对齐索引。"""

def filter_stocks(self, mask: pd.Series) -> 'Frame3D':
    """按布尔 mask 过滤股票（截面维度）。"""

实现规范：
- 操作过程中出现的 NaN（如因窗口不足导致的），保持 NaN，不填充。下游由各自的处理逻辑决定如何处理。
- 所有方法内部使用 groupby 沿 name 层做时序操作，用 groupby 沿 key 层做截面操作。
- 返回值始终是新的 Frame3D，不修改原始数据。
- 时序算子虽然使用了 rolling, 但最后框架还是会选择当前计算结果 Frame3D 中最新的一天的截面数据作为输出，这也正是流式框架的优势所在，不再需要关心时序计算的问题，在节点内编程的安全性高了很多。

测试要求（test/test_frame3d.py）：
- 构造一个小型 Frame3D（3 time × 3 stock × 5 cols）。
- 逐一测试每个 API。
- 特别测试边界：单 stock、单 time、std=0、全 NaN 列。

3.2 MultiInputNode 扩展（`qpipe/node.py`）

在 MultiInputNode.__init__ 中增加两个可选参数：

def __init__(self, ...,
    context: Optional[Any] = None,           # 有状态上下文
    epilogue_fn: Optional[Callable[[str, Any], None]] = None,  # 退出前回调
):

`context`：
- 在 func(name, f3d, context) 被调用时，作为第三个参数传入。
- 每次调用返回时，如果 func 返回 Tuple[Frame3D, Any]，则用返回值的第二部分更新 context。如果只返回 Frame3D，则 context 保持不变。
- 初始值由 add_node 时的 context 参数指定。

`epilogue_fn`：
- 签名：epilogue_fn(name: str, context: Any) -> None。
- 在进程退出前（run() 方法的 finally 块中）调用。
- 用于汇总分析（如计算全时段 mean IC、ICIR）。
- 必须是 pickle 可序列化的（模块级函数或 __call__ 类实例）。

`func` 签名变更：
- 旧：func(name: str, f3d: Frame3D) -> Frame3D
- 新：func(name: str, f3d: Frame3D, context: Any) -> Union[Frame3D, Tuple[Frame3D, Any]]
- 向后兼容：如果 func 只接受 2 个参数，自动包装（不传 context）。

Flow.add_node 同步更新：增加 context 和 epilogue_fn 参数。

测试要求（test/test_node.py）：
- 测试 context 传递和更新。
- 测试 epilogue_fn 在退出时被调用。
- 测试向后兼容（不传 context 时正常运行）。

3.3 SourceNode 扩展

在 SourceNode.__init__ 中增加：

def __init__(self, ...,
    context: Optional[Any] = None,
    epilogue_fn: Optional[Callable[[str, Any], None]] = None,
):

用于支持 source 节点的有状态初始化（如预加载数据）和退出清理。

────────────────────────────────────────────────────────────

4. Phase 2：数据生成节点

4.1 数据生成函数（`seafquant/data_generator.py`）

模块级函数（非闭包），签名：

def generate_synthetic_data(
    n_times: int = 1000,
    n_stocks: int = 500,
    noise_ratio: float = 0.3,
    seed: int = 42,
) -> Iterator[Frame3D]:

数据结构：每行一个 (time, stock)，列包括：
- open, high, low, close：OHLC 价格序列
- turnover：换手率
- volume：成交量
- market_cap：市值

数据生成逻辑（必须在代码注释中清晰说明）：

1. 基准价格路径：对每只股票，生成一个随机游走的基础价格路径：
- log_price[t] = log_price[t-1] + ε_t，其中 ε_t ~ N(0, σ²)
- 不同股票的波动率 σ 从对数正态分布中采样，模拟真实市场中的异质波动。

2. 可预测信号结构（核心）：
- 生成 5 个"隐藏因子" h1..h5，每个是独立的 AR(1) 过程，衰减系数 0.95。
- 真正的未来收益由这些隐藏因子的线性组合决定：
true_fwd_ret[t] = Σ(w_i * h_i[t]) + η_t
- 其中 w_i 是固定的权重向量（每个 stock 不同），η_t ~ N(0, noise_ratio * σ_ret)
- noise_ratio 控制可预测信号与噪声的比例。noise_ratio=0 时完全可预测，noise_ratio=1 时信号完全淹没在噪声中。

3. OHLC 生成：
- 从 log_price 推导出 close 价格。
- open = close[t-1] * exp(ε_open)
- high = max(open, close) * (1 + |ε_high|)
- low = min(open, close) * (1 - |ε_low|)
- ε 参数控制日内波动幅度。

4. 辅助数据：
- volume：与 |close - open| 正相关 + 噪声。
- turnover：volume / market_cap。
- market_cap：慢变随机游走，截面服从对数正态分布。

代码结构约束：
- 生成器函数不超过 100 行。辅助函数拆分到同文件的模块级函数。
- 所有随机数使用 np.random.default_rng(seed)，确保可复现。
- 在每个 time 的生成末尾打日志：logging.info(f"[DataGen] Generated day {t}/{n_times}, noise_ratio={noise_ratio}")。

测试要求（test/test_data_generator.py）：
- 测试数据 shape、列名完整性。
- 测试 no noise 时 IC 显著 > 0，high noise 时 IC 趋近于 0。
- 测试可复现性（相同 seed 生成相同数据）。

────────────────────────────────────────────────────────────

5. Phase 3：因子计算节点

5.1 因子分类与设计原则

构造 128 个因子，分为 8 大类，每类 16 个。每大类一个独立的计算节点（factor_momentum, factor_reversal, factor_volatility, factor_liquidity, factor_value, factor_quality, factor_trend, factor_size）。

设计原则：
1. 复用计算结构：同一类内的因子共享底层变换（如都基于收益率序列），减少重复 groupby。
2. 流式友好：在 MultiInputNode 的 func 中，你拥有 window 天的历史数据。时序因子直接基于窗口内数据计算最新一天的值。你不需要额外维护历史状态（除非因子本身需要长记忆）。
3. 截面标准化：每个因子的最终输出必须经过 cs_zscore()（截面标准化），确保量纲一致。
4. 平稳性：优先使用收益率（pct_change）而非价格水平，使用排名而非绝对值，使用 ratio 而非 difference。
5. NaN 处理：不在因子计算中随意填 0。记录每个因子产生 NaN 的原因（窗口不足？除零？空数据？），在 CHANGELOG.md 中汇总。

5.2 每类因子的具体规格

5.2.1 动量类 (factor_momentum, 16 factors)
基于 close 价格的多周期收益率。参数：周期 [1, 3, 5, 10, 20, 40, 60, 120]，每个周期做原始收益率 + 波动率调整收益率（收益/标准差）。共 8×2=16。

5.2.2 反转类 (factor_reversal, 16 factors)
短期反转效应。基于 close 的 1/3/5 日收益率的多种变体：原始反转、隔夜反转（close→open）、日内反转（open→close）、结合成交量的反转。用 ts_zscore 检测极端偏离。

5.2.3 波动率类 (factor_volatility, 16 factors)
基于日收益率的波动性度量。参数：周期 [5, 10, 20, 60] 。变体：已实现波动率、下行波动率、波动率的波动率、高低价范围/close（Parkinson 估计）。

5.2.4 流动性类 (factor_liquidity, 16 factors)
基于 turnover 和 volume。变体：日均换手率、换手率变化、成交量变化、Amihud 非流动性指标（|ret|/volume）。

5.2.5 价值类 (factor_value, 16 factors)
由于我们没有基本面数据（如 EPS、BV），用代理变量：价格/市值比的各种变体、价格相对历史均值的偏离、市值对数等。

5.2.6 质量类 (factor_quality, 16 factors)
代理：收益稳定性（收益的 std 的倒数）、价格趋势的 Sharpe-like ratio、高低价差的稳定性。

5.2.7 趋势类 (factor_trend, 16 factors)
价格与移动均线的关系。MA 周期 [5, 10, 20, 60, 120]。变体：价格/MA、MA 交叉、MACD 类信号。

5.2.8 规模类 (factor_size, 16 factors)
market_cap 的各种变换：log、排名、市值变化率、市值波动。

5.3 因子计算节点实现（`seafquant/factors.py`）

每个因子大类的函数签名：

def compute_momentum_factors(name: str, f3d: Frame3D, context: Any) -> Frame3D:
    # 输入 f3d 包含 window 天的 close, turnover, volume, market_cap 数据
    # 输出 f3d 只包含 16 个动量因子列，每列已做截面标准化

必须遵守的规则：
- 每个大类函数在一个独立的模块文件中定义，如 seafquant/factors_momentum.py。文件不超过 300 行。
- 所有因子函数必须是模块级函数（pickle 兼容）。
- 输出列命名：factor_mom_ret_1d, factor_mom_ret_3d 等，前缀标识大类。
- 对每个因子输出列做 cs_zscore()。
- 在函数内部记录 NaN 统计：nan_counts = {col: df[col].isna().sum() for col in factor_cols}，通过 logging.debug 输出。

单元测试（test/test_factors.py）：
- 用 Phase 2 的数据生成器生成小规模数据（50 time × 20 stock）。
- 逐一测试每个因子类的函数，验证输出 shape、列数、无全 NaN 列。

────────────────────────────────────────────────────────────

6. Phase 4：模型训练与预测节点

6.1 模型节点逻辑（`seafquant/model_node.py`）

模块级函数签名：

def model_train_predict(name: str, f3d: Frame3D, context: Any) -> Tuple[Frame3D, Any]:

输入：
- 来自因子节点的 pipeline：window=200 天的因子数据（200 time × 500 stock × 128 factors）
- 来自 data source 的 pipeline：window=220 天的 close 数据（用于计算 fwd_ret_xd）

为什么是 220？ 因为需要 20 天前瞻窗口，且训练需要 200 天因子数据扩展到 0-199 的 label，我们需要 factor window 为 200，close window 为 220（200 + 20 前瞻）。

context 设计：
context = {
    'trained_model': None,       # 当前训练好的模型
    'is_trained': False,         # 是否已完成一次训练
    'retrain_every': 20,         # 每 20 个交易日重新训练一次
    'days_since_train': 0,       # 距上次训练的天数
    'feature_cols': [...],       # 128 个因子列名
}

训练流程：
1. 当 days_since_train >= retrain_every 或首次运行时，触发训练。
2. 从 f3d 中提取前 200 天的因子数据（time 0~199）。
3. 从 close 数据计算 20 日前瞻截面超额收益：
- fwd_ret = close[t+20] / close[t] - 1（对每个 stock）
- fwd_ret_xd = cs_zscore(fwd_ret)（截面标准化）
- 时间穿越防护：label 的时间索引必须严格大于 feature 的时间索引。对于 time=t 的 feature，label 使用 time=t 的 fwd_ret_xd（该 fwd_ret 基于 t 到 t+20 的实际收益，这是一个前瞻量。在训练时，这是合法的因为我们用历史数据训练）。但注意：对于 time=180~199 的 feature，无法计算 fwd_ret_xd（因为需要 t+20 超出窗口）。所以只有 time 0~179 可用于训练。
4. 使用 LightGBM 训练回归模型：
- LGBMRegressor(n_estimators=100, max_depth=6, num_leaves=31, reg_alpha=0.1, reg_lambda=0.1, ...)
- 显式正则化：reg_alpha(L1) 和 reg_lambda(L2) 处理共线性。
- 早停：early_stopping_rounds=10。
- 评估：5 折交叉验证的 R² 和 IC。
5. 训练完成后，对所有 500 只股票的第 199 天因子进行预测，输出预测信号。
6. 对预测信号做截面标准化（cs_zscore），输出列名 pred_signal。

预测流程（非训练日）：
- 直接用 context['trained_model'] 对最新一天的因子做 predict。
- 输出 pred_signal。

输出：
- 返回 (Frame3D(df_with_pred_signal), updated_context)。
- 输出列：pred_signal（每股一个预测值）。

日志：
- 每次训练时输出：logging.info(f"[Model] Training: window_size=200, train_samples={n_train}, features={n_features}, cv_ic={cv_ic_mean:.4f}, cv_r2={cv_r2_mean:.4f}")
- 每次预测时输出：logging.info(f"[Model] Predict day {t}, signal_mean={signal_mean:.4f}, signal_std={signal_std:.4f}")

6.2 模型节点的数据拓扑

src_data --[close_raw]--> model     (window=220, 输入 close)
src_data --[close_raw]--> ic_analysis (window=1, 输入 close)
factors --[factors_128]--> model    (window=200, 输入 128 个因子)
model --[pred_signal]--> ic_analysis (window=1, 输出预测信号)

注意：model 节点有两个输入队列，需要 input_columns 分别指定 close_raw 和 128 个因子列; ic_analysis 也有两个输入队列

────────────────────────────────────────────────────────────

7. Phase 5：信号 IC 分析节点

7.1 IC 分析逻辑（`seafquant/ic_analysis.py`）

模块级函数签名：

def ic_analysis_fn(name: str, f3d: Frame3D, context: Any) -> Tuple[Frame3D, Any]:

输入：
- 来自 model 的 pipeline：pred_signal（每日截面向量）
- 来自 data source 的 pipeline：close 数据，window=21 天（需计算 20 日 fwd_ret）

context 设计：
context = {
    'ic_history': [],       # List[float]，每日 IC
    'cumsum_ic': 0.0,      # IC 累计值
    'day_count': 0,        # 交易日计数
}

每日计算流程：
1. 提取当天的 pred_signal 向量（截面）。
2. 计算当天的 20 日 fwd_ret_xd：
- fwd_ret = close[t+20] / close[t] - 1
- fwd_ret_xd = cs_zscore(fwd_ret)
- 时间穿越防护：这里的 fwd_ret 是基于未来实际收益的，这在 IC 分析中是合法的（我们在事后评估信号质量）。
3. 计算截面 Spearman rank IC：
- ic = spearmanr(pred_signal, fwd_ret_xd).correlation
- 处理 NaN：如果 pred_signal 或 fwd_ret_xd 全 NaN，则该日 IC = NaN，不纳入统计。
4. 更新 context：
- ic_history.append(ic)
- cumsum_ic += ic
- day_count += 1

日志：
- 每日输出：logging.info(f"[IC] day={day_count}, ic={ic:.4f}, cumsum_ic={cumsum_ic:.4f}")

退出汇总（epilogue_fn）：

def ic_epilogue(name: str, context: dict) -> None:
    ics = [x for x in context['ic_history'] if not np.isnan(x)]
    if len(ics) < 10:
        logging.warning("[IC Epilogue] Insufficient IC data for summary.")
        return
    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    winrate = sum(1 for x in ics if x > 0) / len(ics)
    
    logging.info(f"[IC Epilogue] ========== IC Summary ==========")
    logging.info(f"[IC Epilogue]   N={len(ics)}, Mean IC={mean_ic:.4f}, ICIR={icir:.4f}")
    logging.info(f"[IC Epilogue]   WinRate={winrate:.2%}, CumSum IC={context['cumsum_ic']:.4f}")
    logging.info(f"[IC Epilogue]   IC Std={std_ic:.4f}, IC Skew={pd.Series(ics).skew():.4f}")
    logging.info(f"[IC Epilogue] ======================================")

────────────────────────────────────────────────────────────

8. Phase 6：Pipeline 组装与验证

8.1 pipeline.py（程序入口）

文件路径：pipeline.py（项目根目录）

import argparse
import logging
from qpipe.flow import Flow
# ... 导入所有节点函数

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-ratio', type=float, default=0.3,
                        help='Noise ratio for synthetic data (0=clean, 1=pure noise)')
    parser.add_argument('--n-times', type=int, default=1000)
    parser.add_argument('--n-stocks', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    flow = Flow()
    
    # === 定义节点拓扑 ===
    # (此处写拓扑定义，预期约 40-60 行)
    
    flow.start()
    flow.join()

if __name__ == '__main__':
    main()

拓扑约束：
- Source: src_data → 输出到 q_close_to_model, q_close_to_ic, q_ohlc_to_factors
- Nodes: factor_momentum, factor_reversal, ..., factor_size（8 个因子节点）→ 输出到 q_factors_to_model
- Node: model → 输入 q_factors_to_model + q_close_to_model → 输出到 q_signal_to_ic
- Node: ic_analysis → 输入 q_signal_to_ic + q_close_to_ic

注意：
- 8 个因子节点是并行的（每个是独立进程），它们的输出汇集到一个 queue q_factors_to_model。这意味着 model 节点的 input_from 可以是一个包含多个上游的列表，MultiInputNode 会自动按 time 合并。
- 但 Flow 的拓扑验证要求每个 queue 只有一个 producer。所以 8 个因子节点必须输出到不同的 queue（q_factors_0 到 q_factors_7 ），然后 model 节点从这 8 个 queue 读取。这是当前 Flow 的限制。

8.2 验证方案

编写 scripts/validate_noise_sweep.py：

"""遍历 20 种噪声强度，验证 ICIR 随噪声单调递减。"""
import subprocess
import numpy as np

noise_ratios = np.linspace(0.0, 0.95, 20)
results = []

for nr in noise_ratios:
    # 运行 pipeline，捕获 ICIR
    # pipeline 需支持将 ICIR 输出到文件或 stdout
    ...
    
# 验证：ICIR 与 noise_ratio 的 Spearman 相关系数应 < -0.8
corr = spearmanr(noise_ratios, [r['icir'] for r in results]).correlation
assert corr < -0.8, f"ICIR-noise correlation {corr:.4f} > -0.8, framework may be broken"
print(f"Validation PASSED: ICIR-noise Spearman corr = {corr:.4f}")

────────────────────────────────────────────────────────────

9. 时间穿越防护清单（每次编码时必须自我审查）

在编写任何涉及"未来信息"的计算之前，按以下清单逐项确认：

- [ ] Label 计算：用于模型训练的 y（fwd_ret_xd）是否严格基于 feature 时间之后的数据？即 label_time = feature_time + horizon ，且 label_time 的数据在训练时不泄露到 feature 的构建中。
- [ ] 因子计算：当天的因子值是否只使用了当天及之前的数据？如果使用了 ts_rolling，rolling 窗口是否完全落在历史区间内？
- [ ] 截面标准化：cs_zscore 在同一天内计算均值和标准差时，是否混入了未来的股票数据？答案：不会，因为它是按 time 分组的。
- [ ] 模型训练：训练集和验证集的划分是否严格按时间顺序（如 time-ordered split），而非随机 shuffle？答案：使用 TimeSeriesSplit 或手动按时间切分。
- [ ] IC 计算：虽然在事后计算（这是合法的），但确保 pred_signal[t] 只与 fwd_ret_xd[t]（基于 t→t+20 的收益）配对，没有错位。

────────────────────────────────────────────────────────────

10. NaN/Inf 诊断协议

每次遇到 NaN/Inf 时，按以下步骤排查（不可跳过直接填 0）：

1. 定位：nan_cols = df.columns[df.isna().any()].tolist()，确定哪些列、哪些行。
2. 分类：
- 除零：分母为 0 或接近 0（如 cs_zscore 中 std=0）。
- 窗口不足：ts_rolling 的 min_periods 不满足。
- 空输入：上游节点未输出该列。
- 对数/开方：输入负数或 0。
- 合并不对齐：pd.concat 时索引不完全匹配导致 NaN。
3. 解决（按优先级）：
- 修复计算逻辑（如加 epsilon 防止除零）。
- 调整窗口参数。
- 在合理位置做前向填充（仅在与逻辑一致时）。
- 如果 NaN 是不可避免的（如因缺少历史数据），保持 NaN，但在下游做显式处理。
4. 记录：在 CHANGELOG.md 中记录每次 NaN 事件的根因和解决方案。

────────────────────────────────────────────────────────────

11. Pickle 安全编码规范

多进程环境中，mp.Process 通过 pickle 序列化传递对象。以下模式必须遵守：

✅ 允许的模式：
# 1. 模块级函数
def my_func(name, f3d, ctx):
    ...

# 2. 模块级可调用类
class MyCallable:
    def __init__(self, param):
        self.param = param
    def __call__(self, name, f3d, ctx):
        ...

# 3. 使用 dill 替代 pickle（在 node.py 中 import dill 并在需要时用 dill.dumps/loads）
# 注意：requirements.txt 已包含 dill>=0.3.8

❌ 禁止的模式：
# 1. 闭包
def outer():
    x = 1
    def inner(name, f3d, ctx):  # ❌ 闭包无法 pickle
        return x + ...

# 2. lambda
func = lambda name, f3d, ctx: ...  # ❌ lambda 无法 pickle

# 3. 实例方法绑定
obj.method  # ❌ 绑定方法可能无法 pickle（取决于 class 定义位置）

────────────────────────────────────────────────────────────

12. Git 提交与文档规范

12.1 Commit Message 格式
[Phase X] <中文简短描述> | <English short description>

<详细中文描述>
<Detailed English description>

12.2 CHANGELOG.md 格式
## [Phase X] 2026-06-03

### 变更内容
- 新增 xxx 文件
- 修改 yyy 方法

### 变更原因


### 遇到的问题

### 解决方案

### 下一步计划

────────────────────────────────────────────────────────────

13. 执行顺序与检查点

┌────────────────────────────────────────┬────────────────────────────────────────┬────────────────────────────────────────┐
Phase                                  │ 内容                                   │ 检查点                                 │
├────────────────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
0                                      │ 深度阅读                               │ 输出 CHANGELOG.md 中的理解摘要         │
├────────────────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
1                                      │ 基建扩展（Frame3D API +                │ test_frame3d.py 全部通过 +             │
                                       │ context/epilogue_fn）                  │ test_node.py 全部通过                  │
├────────────────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
2                                      │ 数据生成节点                           │ test_data_generator.py 通过 +          │
                                       │                                        │ pipeline.py 单节点运行验证             │
├────────────────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
3                                      │ 因子计算节点                           │ test_factors.py 通过 + 5 天小数据      │
                                       │                                        │ pipeline 验证                          │
├────────────────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
4                                      │ 模型节点                               │ 手动验证：有噪声时 IC > 0，无噪声时 IC │
                                       │                                        │ 接近 1                                 │
├────────────────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
5                                      │ IC 分析节点                            │ epilogue_fn 输出完整统计               │
├────────────────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
6                                      │ 全流程 + 噪声扫描验证                  │ ICIR 随 noise 单调递减                 │
└────────────────────────────────────────┴────────────────────────────────────────┴────────────────────────────────────────┘

每个 Phase 完成后：
1. 运行对应测试，确认全部通过。
2. 运行 python pipeline.py --n-times 50 --n-stocks 20 做小规模端到端验证。
3. git commit。
4. 更新 CHANGELOG.md。

────────────────────────────────────────────────────────────

14. 最终交付清单

- [ ] pipeline.py：程序入口，通过 argparse 配置参数。
- [ ] qpipe/frame3d.py：扩展后的 Frame3D（含时序+截面 API）。
- [ ] qpipe/node.py：扩展后的 MultiInputNode 和 SourceNode（含 context + epilogue_fn）。
- [ ] qpipe/flow.py：扩展后的 Flow（含 context/epilogue_fn 参数传递）。
- [ ] seafquant/data_generator.py：合成数据生成器。
- [ ] seafquant/factors.py：因子主入口（调度 8 个子模块）。
- [ ] seafquant/factors_momentum.py 等 8 个因子模块。
- [ ] seafquant/model_node.py：LightGBM 训练+预测节点。
- [ ] seafquant/ic_analysis.py：IC 分析节点 + epilogue_fn。
- [ ] test/test_frame3d.py：Frame3D API 单元测试。
- [ ] test/test_node.py：node 扩展单元测试。
- [ ] test/test_data_generator.py：数据生成器测试。
- [ ] test/test_factors.py：因子计算测试。
- [ ] scripts/validate_noise_sweep.py：噪声扫描验证脚本。
- [ ] CHANGELOG.md：完整开发日志。
