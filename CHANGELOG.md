# SEAF 开发变更日志

## [Phase 0] 2026-06-03 代码阅读理解与架构分析 | Code Reading & Architecture Analysis

### 基建层理解摘要

#### 1. Frame3D (`qpipe/frame3d.py`)

**索引结构**：内部 `df` 为 `pd.DataFrame`，索引是 `pd.MultiIndex`，层级为 `[key, name]`（key=时间维度, name=股票维度）。列是属性维度（如 open, close 等）。

**现有方法清单**：
- `__init__(df)` — 初始化，断言 index 必须是 MultiIndex
- `df` (property) — 返回内部 DataFrame
- `copy()` — 深拷贝
- `__repr__()` — 字符串表示

**缺失 API**（需要扩展）：
- 时序 API：ts_delay, ts_delta, ts_pct_change, ts_rolling, ts_zscore, ts_rank
- 截面 API：cs_zscore, cs_rank, cs_demean, cs_neutralize
- 工具 API：get_cs_series, get_ts_series, add_column, filter_stocks

#### 2. MultiInputNode (`qpipe/node.py`)

**完整生命周期**：
1. `__init__` — 设置参数（name, func, input_queues, output_queues, window, min_periods, input_columns, output_columns）
2. `start()` → 子进程入口 `run()`
3. 为每个 input_queue 启动 daemon 线程运行 `receive_worker`
4. `receive_worker`：阻塞等待 queue.get() → 解析 time → 存到 `buffers[queue_idx][time]` → 设置 ready_event
5. 主循环：
   - 等待所有活跃 worker 的 ready_event（心跳超时检测死 worker）
   - 计算 shared_times（所有输入队列共有时间的交集）
   - 对每个 shared_time，合并多路输入：`pd.concat(df_list, axis=1)` → 按列合并
   - 应用 input_columns 过滤
   - 添加到 `time_order_buffer`（deque）
   - 当 buffer 长度 >= min_periods：滑动窗口计算
     - window_length 控制在 [min_periods, window] 之间
     - 取 buffer 前 window_length 个 Frame3D concat
     - 调用 `func(name, run_input_f3d)`
     - 取输出最新 time 行 → 推送到 output_queues
     - window_tail_index += 1
   - 所有 worker 死亡 → 退出
6. `finally`：设置 global_exit → 等待线程退出 → 每个 output_queue 发送 stop_signal

**关键机制**：
- 窗口滑动：每次计算后 window_tail_index+1，当窗口>window 时 popleft 丢失旧数据
- 输出截取：func 返回完整窗口数据，node 自动只取最新 time 行输出
- 合并语义：`pd.concat(axis=1)` 按列合并，需用 input_columns 避免列冲突

**需要改造的地方**：
- func 签名目前是 `(name, f3d) → Frame3D`，需要改为 `(name, f3d, context) → Frame3D | (Frame3D, context)`
- 需要在 `run()` 的 finally 中调用 epilogue_fn
- 需要在 func 调用处支持 context 的传入和更新

#### 3. SourceNode (`qpipe/node.py`)

**生命周期**：
1. `run()` 中迭代 `gen_func()`
2. 对每个 yield 的 Frame3D，取最新 time 行 → 推送到所有 output_queues
3. 迭代完毕 → 每个 output_queue 发送 stop_signal

**需要改造**：增加 context/epilogue_fn 支持（用于有状态初始化/退出清理）

#### 4. Flow (`qpipe/flow.py`)

**拓扑验证规则**：
1. 节点名全局唯一
2. 每个 queue 只能有一个 producer（多写一读禁止）
3. 每个 queue 必须有 consumer
4. 无孤立节点
5. 有向无环（DAG，DFS 染色法检测环）

**节点注册流程**：
- `create_queue(name)` → mp.Queue (幂等)
- `add_source(name, gen_func, output_to)` → 创建 SourceNode，记录拓扑
- `add_node(name, func, input_from, output_to, window, min_periods, input_columns, output_columns)` → 创建 MultiInputNode，记录拓扑
- `start()` → 先 validate_topology()，再逐个 node.start()
- `join()` → 逐个 node.join()

**需要改造**：add_node 和 add_source 需要支持 context 和 epilogue_fn 参数

#### 5. Example (`qpipe/example.py`)

**数据流拓扑**：
```
src1 ──→ sumprod:0 ──→ sumprod ──→ printer ──→ printer-final
  └──→ printer:0 ──→ printer-src
                          
src2 ──→ sumprod:1 ──→ (sumprod)
  └──→ printer:1 ──→ (printer-src)
```

关键模式：source 输出到多个独立 queue（如 `sumprod:0`, `sumprod:1`），下游 node 从多个 queue 合并读取。queue 命名约定为 `nodename:N` 来区分不同上游。

### 框架优势
1. **流式处理**：逐日计算，天然适合时序滚动策略回测
2. **多进程并行**：每个 node 独立进程，因子计算可并行
3. **滑窗机制**：自动管理历史窗口，func 内只需关注计算逻辑
4. **拓扑验证**：编译期检查数据流合法性
5. **自动数据合并**：多上游自动按时间对齐合并

### 框架限制与风险
1. **无状态 node**：当前 func 无 context，无法做累积统计（需改造）
2. **Frame3D API 贫乏**：无时序/截面计算能力（需扩展）
3. **单 producer 约束**：一个 queue 不能多写，factor 节点需各自独立 queue
4. **pickle 序列化**：多进程通信要求所有传递对象可 pickle（闭包/lambda 禁止）
5. **exit 信号机制**：stop_signal 是字符串哨兵值，可能与真实数据混淆
6. **合并语义**：pd.concat(axis=1) 按列合并，同名列会重复，必须用 input_columns 指定

### 下一步计划
进入 Phase 1：基建层扩展（Frame3D API + Node context/epilogue_fn）

---

## [Phase 1] 2026-06-03 基建层扩展 | Infrastructure Extension

### 变更内容
- **qpipe/frame3d.py**：新增 14 个计算 API（6 时序 + 4 截面 + 4 工具）
  - 时序：ts_delay, ts_delta, ts_pct_change, ts_rolling, ts_zscore, ts_rank
  - 截面：cs_zscore, cs_rank, cs_demean, cs_neutralize
  - 工具：get_cs_series, get_ts_series, add_column, filter_stocks
- **qpipe/node.py**：MultiInputNode 和 SourceNode 新增 context 和 epilogue_fn 支持
  - func 签名兼容新式 `(name, f3d, context)` 和旧式 `(name, f3d)`
  - context 通过 `(Frame3D, context)` tuple 返回值更新
  - epilogue_fn 在 finally 块中调用，用于汇总分析
  - pickle 兼容：使用运行时 inspect.signature 动态检测，不创建闭包
- **qpipe/flow.py**：add_node / add_source 增加 context 和 epilogue_fn 参数透传
- **test/test_frame3d.py**：23 个单元测试（含边界：单 stock、std=0、不可变性）
- **test/test_node.py**：6 个单元测试（context 传递、tuple 更新、epilogue、向后兼容）

### 变更原因
按项目 spec Phase 1 要求，为上层业务代码提供完备的计算 API 和有状态节点能力。

### 遇到的问题
1. **cs_rank 默认用 pandas rank(pct=True) 得到 rank/N 而非 (rank-1)/(N-1)**：自己实现 _rank_pct 函数，单股票返回 0.5。
2. **_wrap_func 创建闭包导致 pickle 失败**：改为在 run() 中使用 _call_func 运行时检查签名。
3. **epilogue 跨进程测试用 mp.Queue 在 Windows spawn 下序列化失败**：改为文件写入方式（EpilogueFileWriter 模块级可调用类）。

### 解决方案
所有 API 使用 groupby 沿 key（截面）或 name（时序）计算，返回新 Frame3D 不修改原始数据。cs_neutralize 使用 np.linalg.lstsq 做 OLS 回归取残差。

### 下一步计划
进入 Phase 2：数据生成节点

---

## [Optimization] 2026-06-05 因子模块性能优化与节点拆分 | Factor Performance Optimization & Node Split

### 变更内容

**quality_advanced 拆分 + 向量化**：
- `factors_quality_advanced.py`（16→8 因子）：保留 skew / up_down / autocorr / tail_risk / composite，移除 dd_duration / hl_stability / kurt / 符号变化
- `factors_quality_pattern.py`（新建，8 因子）：包含 dd_duration / hl_stability / kurt / consec_sign_change / max_consec_pos
- pipeline.py 拓扑从 13→14 个因子节点

**向量化消除 rolling().apply() 瓶颈**：
- `quality_advanced`: tail_risk 用 `rolling().quantile(0.05)` 替代 `apply(_tail_risk)`；autocorr 用 `rolling().corr(shift(1))` 替代 `apply(_autocorr_lag1)`；up_down 用 `sliding_window_view` 向量化
- `quality_pattern`: dd_duration / sign_change / max_consec_pos 全部用 `sliding_window_view` 向量化
- `factors_counting.py`: streak / new_high_low / run_pct 全部用 `sliding_window_view` 向量化

### 性能提升（80×100 数据）

| 模块 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| quality_advanced | 6.98s | 0.32s | **22×** |
| quality_pattern | 7.30s | 0.20s | **36×** |
| counting | ~5.0s | 0.55s | **9×** |

并行瓶颈从 ~7s 降至 ~0.55s，各节点延迟接近，充分利用多进程流式框架的并行优势。

### 变更原因
`rolling().apply(python_fn)` 是 pandas 中最慢的模式——每个 (stock, window) 组合都调用一次 Python 函数。改为 `sliding_window_view` 批量计算 + `rolling().quantile()`/`rolling().corr()` 内置向量化方法。

### 解决方案
- 所有自定义 apply 函数替换为 numpy `sliding_window_view` 批量矩阵操作
- 利用 `np.argmax(axis=1)` / `np.diff(axis=1)` / `np.sum(axis=1)` 等向量化聚合
- 辅助函数在 `groupby().transform()` 内 per-stock 调用（500 次），而非 per-window 调用（500K 次）

### 影响范围
- `seafquant/factors_quality_advanced.py` - 缩减至 8 因子，全向量化
- `seafquant/factors_quality_pattern.py` - 新建，8 因子
- `seafquant/factors_counting.py` - 全向量化
- `seafquant/factors.py` - 注册 quality_pattern
- `pipeline.py` - 14 节点拓扑
- `test/test_factors.py` - quality_pattern 测试

---

## [Optimization v2] 2026-06-05 全模块性能均衡优化 | All-Module Latency Equalization

### 变更内容

**节点拆分（2→4 节点）**：
- `factors_counting.py`（16→9）→ `factors_counting_streak.py`（7）— streak/新高新低分离
- `factors_trend.py`（16→8）→ `factors_trend_macd.py`（8）— MACD/动量/复合分离

**直接 pandas groupby rolling（10 模块）**：
消除 `f3d.copy().add_column().ts_rolling()` 链式深拷贝瓶颈。
- `factors_quality_basic.py` / `volatility` / `intraday` / `value` / `cross_section`
- `liquidity` / `momentum` / `reversal` / `counting` / `trend`

### 性能提升（200t×300s 数据）

| 模块 | 优化前 | 优化后 | 改善 | 手段 |
|------|--------|--------|------|------|
| counting | 4.87s | 0.32s | **15×** | 拆分+直接rolling |
| trend | 3.85s | 0.17s | **23×** | 拆分+直接rolling |
| quality_basic | 3.38s | 0.48s | **7.0×** | 直接rolling |
| volatility | 3.56s | 0.53s | **6.7×** | 直接rolling |
| intraday | 3.31s | 0.47s | **7.0×** | 直接rolling |
| liquidity | 2.92s | 0.45s | **6.5×** | 直接rolling |
| momentum | 2.31s | 0.51s | **4.5×** | 直接rolling |
| reversal | 2.35s | 0.88s | **2.7×** | 直接rolling |
| value | 3.20s | 0.86s | **3.7×** | 直接rolling |
| cross_section | 3.26s | 1.25s | **2.6×** | 直接rolling+cs_neutralize |

**并行瓶颈**：4.87s → **1.25s**（3.9× 整体加速）
**因子节点**：13 → 16 个（+counting_streak, +trend_macd, +quality_pattern）
**测试**：全部 17 个通过

### 优化方法总结

1. **消除 `rolling().apply()`** — 用 `sliding_window_view` / `rolling.quantile()` / `rolling.corr()` 替换 Python 回调
2. **直接 pandas groupby rolling** — 绕过 Frame3D 不可变 copy，通过 `.values` 赋值避免索引对齐问题
3. **节点拆分** — 将 16 因子模块拆为 7-9 因子的并行子节点
4. **复合因子重构** — 拆分后 composite 仅依赖同节点因子

### 影响范围
- 新建：`factors_counting_streak.py`, `factors_trend_macd.py`
- 重写：11 个 factors_*.py 模块（直接 rolling 优化）
- 更新：`factors.py`, `pipeline.py`（16 节点拓扑）, `test/test_factors.py`（17 测试）

## [Phase 5] 2026-06-05 计算效率优化 | Performance Optimization

### 目标
因子节点计算耗时优化，降低并行流水线瓶颈。

### 方案 A: Frame3D 副本消除 + 批量 API
- **`cp` 参数**：所有 `ts_*` / `cs_*` 方法增加 `cp: bool = True`，`cp=False` 时原地操作跳过深拷贝
- **`ts_pct_change_multi`**：一次 GroupBy 完成多周期 pct_change，替代逐次调用
- **`ts_rolling_multi`**：一次 GroupBy 循环完成多窗口滚动聚合
- **`cs_rank_batch`**：一次 GroupBy 完成多列截面排名
- **全量模块改造**：20 个 factor 模块的 `cs_zscore_batch(factor_cols)` → `cs_zscore_batch(factor_cols, cp=False)`，消除尾部冗余深拷贝

### 方案 B: Numpy 向量化替代 groupby-transform
- **counting_streak**：Pivot (times×stocks) → numpy → flatten，消除 500 次 groupby-transform 调用开销
  - `_streaks_2d`: 批量化连续涨跌计数
  - `_run_pct_2d`: `sliding_window_view` 沿时间轴批量计算同向占比
- **momentum**：`ts_pct_change_multi` 批量替代 8 次独立 `df.groupby('name')['close'].pct_change(p)`
- **quality_autocorr**：尝试 numpy sliding_window_view 但 pivot+stack 开销超过收益；回滚保持 pandas `rolling.corr`（C 实现已最优）

### 方案 C: Node 窗口缓存
- 分析结论：瓶颈在因子计算内部，`pd.concat` 操作 <10ms，缓存收益极微，跳过。

### 性能数据（130t × 500s, 5-run avg）

| 指标 | 优化前 | 优化后 | 改善 |
|---|---|---|---|
| 总串行耗时 | 17.28s | **9.19s** | **-46.8%** |
| 并行瓶颈 | 1.505s (quality_autocorr) | **1.225s** (quality_autocorr) | **-18.6%** |
| counting_streak | 1.339s | **0.150s** | **-88.8%** |
| intraday | 1.026s | **0.328s** | **-68.0%** |
| counting | 0.830s | **0.266s** | **-68.0%** |
| trend | 0.425s | **0.136s** | **-68.0%** |
| liquidity | 0.794s | **0.290s** | **-63.5%** |

### 影响范围
- 修改：`qpipe/frame3d.py`（+cp 参数, +3 batch API）
- 修改：全部 20 个 `seafquant/factor/factors_*.py`（cs_zscore_batch cp=False）
- 重写：`factors_momentum.py`（批量 pct_change）、`factors_counting_streak.py`（pivot→numpy）
- 新增测试：`test/test_frame3d.py`（+9 测试，总计 32）
- 新增脚本：`scripts/_bench_after.py`, `scripts/_validate_mom.py`, `scripts/_validate_opt.py`

## [Maintenance] 2026-06-05 因子文件重命名 + 基准脚本修复 | Factor Rename & Bench Fix

### 概述
清理代码库命名规范、修复基准测试脚本的路由错误。

### 因子文件重命名（12 个文件）

`seafquant/factor/factors_*.py` → `seafquant/factor/*.py`，去掉冗余 `factors_` 前缀：

| 旧名 | 新名 |
|------|------|
| `factors_counting.py` | `counting.py` |
| `factors_cross_section.py` | `cross_section.py` |
| `factors_cross_section_neut.py` | `cross_section_neut.py` |
| `factors_interaction.py` | `interaction.py` |
| `factors_liquidity.py` | `liquidity.py` |
| `factors_momentum.py` | `momentum.py` |
| `factors_quality_autocorr.py` | `quality_autocorr.py` |
| `factors_quality_basic.py` | `quality_basic.py` |
| `factors_quality_pattern.py` | `quality_pattern.py` |
| `factors_trend.py` | `trend.py` |
| `factors_value.py` | `value.py` |
| `factors_volatility.py` | `volatility.py` |

### Import 路径更新
- `pipeline.py`：12 行 import 路径同步更新
- `seafquant/factors.py`：FACTOR_REGISTRY import 路径同步 + 注释修正（11→12 节点）

### bench_all_factors.py 修复
- **Bug**: `ACTIVE_MODULES` 硬编码了因子合并前的 20 个模块名（`reversal`、`quality_advanced`、`trend_macd` 等），因子合并重构后 FACTOR_REGISTRY 仅保留 12 个 key，导致 `KeyError: 'reversal'`
- **修复**: `ACTIVE_MODULES = list(FACTOR_REGISTRY.keys())`，与注册表自动对齐，彻底消除漂移

### 测试加速
- `test_data_generator.py`：`n_times` 150 → 80，两个可预测性测试从 65s 降至 35s

### 辅助工具
- 新增 `scripts/_rename_imports.py`：批量更新所有 `.py` 文件中的 `factors_*` import 引用

### 验证
- 全部 56 个测试通过（pytest，149.79s）
- `bench_all_factors.py` 12 模块全部跑通（瓶颈 quality_autocorr 1.26s，最快 cross_section_neut 0.33s）

## [Optimization] 2026-06-05 节点合并: quality_basic + cross_section_neut → quality_merged

### 动机
因子节点并行效率不均——`cross_section_neut`（6cols / 0.33s）和 `quality_basic`（19cols / 0.69s）
耗时远低于瓶颈模块（quality_autocorr 1.26s），浪费进程资源且不贡献并行加速。

### 变更
- **合并模块**: `quality_basic.py` + `cross_section_neut.py` → `quality_merged.py`（25 列因子）
  - Part A: 质量基础+符号（19 cols, prefix `factor_qb_` / `factor_qa_`）
  - Part B: 截面中性化（6 cols, prefix `factor_cs_`）
  - 统一尾部 `cs_zscore_batch` 批处理
- **节点数**: 12 → 10（-17% 进程资源）
- **FACTOR_REGISTRY**: `quality_basic` + `cross_section_neut` → `quality_merged`

### 效率提升

合并前后 bench_all_factors 对比（200t×500s, 10-run avg）：

| 模块 | 合并前 | 合并后 | 列数 |
|------|--------|--------|------|
| quality_basic | 0.69s | — | 19 |
| cross_section_neut | 0.33s | — | 6 |
| **quality_merged** | — | **1.23s** | 25 |

并行瓶颈比变化：

| 指标 | 合并前 | 合并后 | 改善 |
|------|--------|--------|------|
| 瓶颈 | quality_autocorr 1.26s | quality_merged 1.23s | -2.4% |
| 最快 | cross_section_neut 0.33s | value 0.75s | — |
| **瓶颈比** | **3.8x** | **1.6x** | **-57.9%** |

### 受影响文件
- 新增：`seafquant/factor/quality_merged.py`
- 修改：`seafquant/factors.py`（import + registry + prefixes + docstring）
- 修改：`pipeline.py`（12→10 nodes + import 替换 + 注释更新）
- 修改：`test/test_factors.py`（2 tests→1 test, 25 cols, 总计 55 测试）
- 修改：`seafquant/factor/cross_section.py`（注释同步）

### 验证
- 全部 55 个测试通过（pytest，149s）
- `bench_all_factors.py` 10 模块全部跑通（瓶颈 quality_merged 1.23s）

## [Refactor] 2026-06-06 pipeline.py 去冗余 — 消除 FACTOR_REGISTRY 双维护

### 问题
`pipeline.py` 中存在两处与 `seafquant/factors.py` 的 `FACTOR_REGISTRY` 重复：
1. 10 行独立的 `from seafquant.factor.xxx import compute_xxx_factors` — 与 `factors.py` 中的相同 import 重复
2. `factor_nodes` 列表 — 与 `FACTOR_REGISTRY` 键值对完全等价（仅前缀不同）

### 变更
- **import 精简**：10 行 → 1 行 (`from seafquant.factors import FACTOR_REGISTRY`)
- **factor_nodes 派生**：`[(f'factor_{name}', func) for name, func in FACTOR_REGISTRY.items()]`
  — 因子增删改只需维护 `FACTOR_REGISTRY` 一处，pipeline 自动同步
- **pipeline.py**：144 行 → 122 行（-15%）

## [Tooling] 2026-06-07 全局类型标注 + Lint 工具链配置

### 动机
项目缺少类型标注和自动化 lint 流程，代码可维护性不足。

### 全局类型标注

**基建层 (qpipe/)**：
- `frame3d.py` — `from __future__ import annotations`，所有方法签名添加完整参数/返回类型，
  `_df: pd.DataFrame` 实例属性声明，`Union`→`|`，`List`→`list`，新增 `Frame3D` 类文档
- `node.py` — 类型别名 `FactorFunc` / `EpilogueFunc` / `GenFunc`，
  `MultiInputNode.__init__` 完整参数类型，`receive_worker` 参数类型，
  内部局部变量类型标注（`dead_workers`, `time_order_buffer`, `timings` 等）
- `flow.py` — `Flow` 类实例属性 inline 类型标注，`add_source`/`add_node` 参数类型，
  `validate_topology` 内部 DFS 变量类型，`create_queue` 返回 `mp.Queue[Any]`

**业务层 (seafquant/)**：
- `data_generator.py` — `_generate_stock_params` 返回类型 `dict[str, np.ndarray]`，
  生成器参数 `start_date: str | None`
- `factors.py` — `FACTOR_REGISTRY: dict[str, Callable[...]]`，`FACTOR_PREFIXES: dict[str, str]`
- `model_node.py` — `from __future__ import annotations`，`Tuple`→`tuple`，`_build_model`→`Any`
- `ic_analysis.py` — `from __future__ import annotations`，`ic_epilogue` 参数 `dict[str, Any] | None`
- **10 个 factor 模块** — 统一添加 `from __future__ import annotations`，清理旧 typing 导入

**入口 & 脚本**：
- `pipeline.py` — `DataSourceCallable.__init__` 返回 `None`，`main() -> None`，
  `start_date: str | None` 属性声明
- `bench_all_factors.py` — `results` 字典类型标注

### Lint 工具链配置

**pyproject.toml**：
- Ruff 19 条规则集（E/W/F/I/N/UP/B/C4/SIM/RUF/PT/RET/PIE/PL/PERF/FURB/RSE/TCH）
- Mypy 类型检查配置，第三方库忽略（lightgbm/sklearn/scipy/mlflow）
- Pytest 配置（timeout=300s）
- CJK 全角标点豁免（RUF002/RUF003），测试/脚本目录宽松规则

**scripts/pre_commit.py**：
- 自动化提交前验证链：ruff check → ruff format → pytest → bench → git diff → commit
- 支持交互式确认：`python scripts/pre_commit.py "msg"`

### 验证
- Ruff: 0 errors（全部自动修复 150 + auto-fix 41 条）
- Pytest: 55 passed
- 格式统一：ruff format 全量文件

## [Model] 2026-06-08 Model 节点重构 — Label 对齐实盘 + IC 导向训练 + 充分日志

### 动机
原有 model_node.py 存在三个核心问题：
1. **Label 计算含时间穿越**：`close[t+fwd] / close[t] - 1` 包含了 t→t+1 的隔夜收益，
   但实盘交易在 t+1 买入、t+fwd 卖出，不应包含 t→t+1 这一段。
2. **Label 全局标准化而非截面标准化**：`(y - mean(y)) / std(y)` 对所有时间混合计算，
   引入了未来时间的均值/方差信息。
3. **日志不充分**：缺少训练样本统计、CV 每折 IC、特征重要性等关键调试信息。

### Label 修正

**之前**：
```python
close_fwd = close[t+fwd]; close_t = close[t]
fwd_ret = close_fwd / close_t - 1                    # 含 t→t+1 隔夜
y_xd = (y - mean(y)) / std(y)                        # 全局标准化
```

**之后**：
```python
close_buy  = close[t+1]; close_sell = close[t+fwd]
fwd_ret = close_sell / close_buy - 1                  # t+1→t+fwd 纯持有期
label_xd = cs_zscore(fwd_ret)                         # 逐截面独立标准化
```

### 训练对齐实盘指标

- **损失函数**：MSE on cs_zscore labels ≈ 最大化截面 Pearson IC（数学等价）
  - 对于标准化向量：`||pred - y||² ∝ -cov(pred, y)`，MSE 最小化 ⇒ 协方差最大化 ⇒ IC 最大化
- **验证指标**：CV 使用 Spearman rank IC（非 MSE），与 IC 分析节点统一
- **预测输出**：`cs_zscore(model.predict(X_latest))`，截面标准化保证量纲一致

### 日志增强

| 阶段 | 日志内容 |
|------|---------|
| 训练开始 | 模型类型、fwd、窗口尺寸、因子数 |
| 样本构建 | 截面数、总样本数、label mean/std/min/max、截面 ret_mean/ret_std |
| NaN 清理 | 移除样本数、剩余样本数 |
| CV | 每折 IC、训练/验证样本数 |
| 特征重要性 | LGBM top-10 feature importance（仅 lgbm）|
| 预测 | 股票数、signal mean/std/min/max、NaN 特征警告 |

### IC 分析节点同步修正
- `t_past` 从 `times[-(fwd+1)]` 改为 `times[-(fwd)]`
  — 与 model label 对齐：close[t+fwd] / close[t+1] - 1

## [Refactor] 2026-06-08 Node Context 语义简化 + Pipeline 重构 + Model/IC 节点完善

### 动机
经过多个迭代轮次的问题沉淀，本轮对框架的三个核心模块进行系统性重构：
1. **node.py**: context 参数语义澄清——从可能为 None 的可选参数改为固定空字典 `{}`
2. **pipeline.py**: 窗口参数解耦——model 窗口与 factor 窗口独立配置
3. **model_node.py**: Label 对齐实盘、IC 导向训练、充分日志

### Node Context 简化

**之前的问题**：
- `MultiInputNode` 和 `SourceNode` 的 `self.context` 初始化可能为 `None`
- `_call_func` 调用后需要判断 `isinstance(result, tuple)` 来提取 context
- 业务函数需要 `return Frame3D(df), context` 才能更新 context

**修改后**：
- `self.context` 统一初始化为 `{}`（空字典）
- context 为可变对象（dict），func 内直接修改 `context['key'] = value`
- 业务函数不再返回 context：`return Frame3D(df)` 即可
- `node.py` 去掉了 `isinstance(result, tuple)` 的分支处理
- 计时操作直接写入 `current_context['elapsed_ms']`

**旧式兼容**：
- `_call_func` 仅捕获 `TypeError`（签名不匹配时回退到旧式 `(name, f3d)` 调用）
- 不再捕获 `ValueError`，避免将因子运行时错误误判为"缺少 context 参数"

### Pipeline 窗口解耦

- 新增 `--model-window` 参数（default=200），控制模型训练窗口大小
- `FACTOR_WINDOWS` 注册表：每个因子模块独立配置 `window` 和 `min_periods`
- `MODEL_WINDOW = model_window + fwd`（模型窗口 = 因子历史 + 前瞻期）
- 因子窗口由 `factors.py` 中的 `FACTOR_WINDOWS` 集中管理

### Model 节点 Label 修正

**核心变更**：
- **Label 买入点**：`close[t+1]`（次日收盘买入，不含 t→t+1 隔夜）
- **Label 卖出点**：`close[t+fwd]`
- **标准化**：`cs_zscore(close_sell / close_buy - 1)`（逐截面独立，无未来信息泄漏）
- **训练目标**：MSE on cs_zscore labels ≈ 最大化截面 Pearson IC
- **CV 验证**：使用 Spearman rank IC

**日志增强**：训练开始/样本构建/NaN 清理/CV 每折 IC/特征重要性/预测信号

### IC 分析节点修正
- `t_past` 从 `times[-(fwd+1)]` 改为 `times[-(fwd)]`，与 model label 对齐
- 函数返回值移除 context（遵循新的 context 语义）

### 影响范围
- `qpipe/node.py` — MultiInputNode/SourceNode context 默认 {}、简化 tuple 分支
- `qpipe/frame3d.py` — `_call_func` 仅捕获 TypeError
- `pipeline.py` — `--model-window`、FACTOR_WINDOWS、窗口解耦
- `seafquant/factors.py` — 新增 FACTOR_WINDOWS 注册表
- `seafquant/model_node.py` — Label 修正、IC 导向、日志增强、移除 context 返回
- `seafquant/ic_analysis.py` — t_past 修正、移除 context 返回
- `test/test_node.py` — 适配新 context 语义
- `TODO.md` — 记录 mlflow 集成等后续计划

## [Infra] 2026-06-08 MLflow 全流程集成

### 动机
之前 pipeline 无实验追踪能力，无法系统化比较不同参数配置下的 IC/ICIR 等指标。

### 变更内容

**MLflow 基础设施**：
- `qpipe/utils.py`（新建）：子进程安全的 MLflow 工具函数
  - `mlflow_init()` / `mlflow_end()` — 实验生命周期管理
  - `mlflow_log_metric()` / `mlflow_log_metrics()` — 单值/批量指标记录
  - 实验名 = 启动时间戳（`2026-06-08_23-43-15` 格式）
  - SQLite 后端：`sqlite:///mlruns.db`
- `pipeline.py`：`--no-mlflow` 参数、git hash 自动记录为 param

**Model 节点指标**：
- 训练阶段：`n_samples`、`n_features`、`label_mean/std/min/max`
- 预测阶段：`signal_mean/std/min/max`、`n_predictions`
- 每次重训练后写入 MLflow step

**IC 分析节点**：
- 指标日志化：`ic_mean`、`ic_std`、`ic_ir`、`ic_winrate`、`cumsum_ic`（epilogue 阶段）
- epilogue 函数通过 `context` 读取完整 IC 序列并计算汇总指标

**工具脚本**：
- `scripts/_check_mlflow.py` — 快速查看最新实验的 IC 指标
- `scripts/_verify_steps.py` — 验证最新 mlflow 实验是否存在 IC 指标

### 后续扩展
- `MultiInputNode` 新增 `time_order_buffer_len` 指标（de6f649），暴露窗口队列积压长度，用于诊断流水线背压。

## [Model] 2026-06-09 特征重要性 + ModelWrapper 协议解耦

### 概述
两轮紧密相关提交（5bf48e9 + c418748）的整体效果：
1. 模型训练后输出特征重要性（LGBM built-in、Ridge coefficient magnitude）
2. 模型实现从 model_node.py 中抽离为独立 `model_wrappers.py`

### 特征重要性
- **LGBM**：`feature_importances_`（split-based），导出 top-10 日志 + 完整 JSON artifact
- **Ridge**：`abs(coef_)` 排序，日志 top-10 + 完整 JSON
- artifact 路径：`feature_importance_{model_type}.json`

### ModelWrapper 协议解耦 (c418748) — 211 行新增
- `seafquant/model_wrappers.py`（新建）：
  - `ModelWrapper` 抽象基类：`train(X, y)` / `predict(X)` / `get_feature_importance()`
  - `LGBMWrapper` / `RidgeWrapper` / `MLPWrapper` 三个实现
  - `MLPWrapper` 将 PyTorch MLP（2 隐藏层 + BatchNorm + Dropout）封装为统一接口
- `model_node.py` 退化为编排层：数据准备 → `wrapper.train()` → `wrapper.predict()` → 输出信号
- 模型新增由 FACTORY 注册表 + CLI `--model-type` 参数控制（`lgbm` / `ridge` / `mlp`）

## [Bugfix] 2026-06-09 系统性 rolling .values 索引错位 + model_node 测试套件

### 问题
全部 8 个因子模块中使用 `groupby('name').rolling(window).apply(func)` 模式，
在 `apply` 内部取 `.values` 时，rolling 窗口内 DataFrame 的索引是原始全局索引切片，
`values` 取值与实际窗口数据可能出现索引错位——在退市/股票池变化时尤为致命。

### 修复
- 10 处 `arr.value` / `series.values` → 先 `.reset_index(drop=True).values` 再取值
- 模块：momentum/volatility/liquidity/counting/counting_streak/quality_autocorr/quality_merged/value
- 影响面：price_rsi / garman_klass / amihud / streak / autocorr 等约 40 个因子

### 新增测试
- `test/test_model_node.py`（623 行，47 tests）：
  - LGBM / Ridge 训练-预测-信号形状全流程
  - NaN 处理、feature importance、小样本守卫、CV IC 指标
  - 重训练间隔逻辑、retrain_every 边界
- `TestRollingAlignment`：回归测试验证修复后 `values` 取值与慢速基准一致

## [Test] 2026-06-09 因子对拍验证测试套件 (511 tests)

### 概述
新增全部 11 个因子模块的系统性对拍验证（cross-validation）。每个因子的实时输出
与独立基准函数计算结果逐值比对，最大绝对误差 < 1e-10。

### 新增文件
- `test/crossval_helpers.py`：通用基准计算工具（`_cs_rank_pct`、`_parkinson_bench`、滚动聚合等）
- `test/test_factors_crossval_{module}.py`（11 个文件）：
  - counting / cross_section / interaction / liquidity / momentum
  - quality_autocorr / quality_merged / quality_pattern / regression
  - trend / value / volatility
- 总计 511 tests，全部通过（pytest, ~230s）

### 设计原则
- 基准使用 pandas groupby 慢速但正确的计算（不依赖 Frame3D API）
- 避免测试代码与被测代码共享计算路径（防止"两个相同 bug 互相抵消"）
- 每个测试在 130 时间 × 20 股票微型数据集上运行

## [Chore] 2026-06-09 Source→IC 属性精简

- `pipeline.py`：source 节点到 ic 分析节点的 pipeline 只传递 `close` 列（而非全部 7 列）
- 减少跨进程序列化数据量 ~80%，IC 节点本就不需要 OHLC/volume/market_cap

## [Infra + Model] 2026-06-09 IPO/退市机制 + NaN 处理系统性加固

### IPO/退市机制

**数据生成器 V3** (`data_generator.py`)：
- 预生成 `pool_size = n_stocks * 2` 的股票参数（sigma、weights）
- 逐日管理活跃股票集合，动态股票池
- 退市规则：`close < 0.005` → 次日退市，同时激活一只新股（名称前缀 N）
- OHLC / 市值 / 成交量从预生成改为逐日生成（支持动态股票池）
- 新股上市首日 open 在 close 附近随机（无前日价格）
- 验证脚本：`scripts/_check_delist.py`

**窗口对齐** (`node.py`)：
- `MultiInputNode.run` 中 window_frames 拼接后，以最新时间片的股票集合为权威集合
- 前序时间片 `reindex(latest_stocks, fill_value=NaN)`：
  - 退市股票自动删除
  - 新上市股票补 NaN（非 0.0，避免 `log(0)` 等下游运算异常）

### Model NaN 处理改进 (`model_node.py`)

**旧逻辑**：标签 y 含 NaN 或特征 X 任意列含 NaN → 丢弃该样本。在退市场景下，新上市股票前 60 天大量 NaN 导致全部样本被丢弃，模型无法训练。

**新逻辑**：
- 标签 y 为 NaN → 丢弃
- 特征 X 超过半数列为 NaN → 丢弃该样本
- 特征 X ≤ 半数列为 NaN → NaN 置 0.0，保留样本

### 因子模块 NaN 防护（5 个文件）

| 模块 | 变更 | 原因 |
|------|------|------|
| `liquidity.py` | `np.log(dollar_vol/mcap/ratio)` 前加 `> 0` 守卫 | IPO 窗口内 mcap 为 NaN 导致 `log(0) = -inf` |
| `value.py` | `np.log(mcap)` 守卫 | 同上 |
| `volatility.py` | `np.log(high/low)`, `np.log(close/open)` 守卫 | high/low 可能为 0 |
| `counting.py` | `_new_high_count` 守卫 `n < window`（而非 `n < max(2, window//2)`） | 此前遗漏的一处守卫条件不一致 |

### 测试参数调整
- `test_data_generator.py`：`n_times` 80→200, `n_stocks` 20→30（增强统计显著性）

### Cross-section 对拍测试修复
- `test_factors_crossval_cross_section.py`：`rank_delta` 基准计算从「先 shift 后 rank」改为「先 rank 后 shift」
  - 在线框架中，`compute_cross_section_factors` 收到的是窗口内所有时间片的完整数据，先对每个时间片独立 rank，再沿时序 shift
  - 旧基准方案在 IPO/退市场景下「先 shift 后 rank」会因股票集合不一致而产生不同结果

## [Perf] 2026-06-09 内存系统性优化 — 队列背压 + input_columns + float32 输出层 + gc 调度

### 背景
n_stocks=50、n_times=1000 运行动辄被系统 kill。根本原因：
1. `mp.Queue` 无 `maxsize`，上游（source 0.2s/item）远超下游消费速度（factor ~1s/item），队列无限膨胀直至 OOM
2. 14 个子进程各独立加载 numpy/pandas（84MB/进程），1.2GB 纯 import 开销
3. 因子节点窗口缓冲无列过滤，全量 7 列 OHLCV 串行化/反串行化
4. Python 堆碎片多轮迭代后从 96MB 漂移至 194MB（无 gc.collect）

### 四项优化

**A. 队列背压（flow.py）**：
- `Flow.__init__` 新增 `queue_maxsize` 参数，`create_queue` 默认使用
- `pipeline.py` 设为 `GLOBAL_MAX_FACTOR_WINDOW` (=130)
- 每个 mp.Queue 上限 130 项 → 队列内存从"无限"变为有界

**B. input_columns 列过滤（pipeline.py）**：
- 新增 `FACTOR_INPUT_COLUMNS` 注册表，精确指定每个因子模块需要的 OHLCV 列
- 11 个模块平均只需 3.5 列（vs 全量 7 列），`time_order_buffer` 列数减半
- 省最多的是 `quality_autocorr`（7→1，省 86%）、`trend`（7→2，省 71%）

**C. float32 输出层（node.py）**：
- 新增 `_to_float32()` — 所有节点输出前将 float64→float32
- SourceNode / MultiInputNode 调用点：`latest_f3d = _to_float32(latest_f3d)` 
- 队列序列化数据、下游所有 `time_order_buffer`、窗口 concat 全为 fp32 → 精确 50% 内存节省
- 仅转换 float64 列，保留索引/int/object 列不变

**D. gc 调度 + RSS 日志（node.py）**：
- 每次迭代后 `gc.collect()` 回收因子计算产生的临时 DataFrame
- 每 10 次调用打 RSS 日志（需 `psutil`，可选依赖）
- 堆碎片从 ~100MB（无 gc）降至 ~5MB（有 gc 50 轮后）

### 新增工具
- `scripts/profile_memory.py` — 独立内存监测，5s 间隔采样进程 RSS 输出 CSV
- `scripts/_mem_breakdown.py` — 单因子节点逐阶段 RSS 分析脚本

### 实测数据（n_stocks=50, n_times=150, 14 进程）

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 队列上限 | 无限 (OOM) | 130 项/queue |
| 进程 RSS (单因子) | ~194 MB | ~100 MB |
| 堆碎片漂移 | ~100 MB | ~5 MB |
| 总 RSS | ~2.9 GB | ~1.4 GB |

### 生产预估（6000 stocks, 11 因子 + model + IC）

| 组件 | 优化前 | 优化后 |
|------|--------|--------|
| 单因子节点（重） | ~1500 MB | ~800 MB (input_cols + fp32) |
| 单因子节点（轻） | ~500 MB | ~200 MB |
| 队列（主进程） | 无限增长 | ~780 MB (bounded) |
| Import 开销 (×14) | 1.2 GB | 1.2 GB (不变) |
| **总计** | **OOM** | **~6-8 GB** |

## [Bugfix] 2026-06-08 sliding_window_view 守卫条件修复 + 框架异常传播强化

### 问题
运行 `pipeline.py --n-times 500 --n-stocks 50` 时，`factor_counting` 和
`factor_quality_pattern` 在窗口初期（数据不足 20 天）崩溃：
`ValueError: window shape cannot be larger than input array shape`。

### 根因
`sliding_window_view` 的防卫条件是 `max(2, window//2) > T`，当 `window=20, T=12`
时 `max(2,10)=10` 不满足 `>12`，守卫被绕过，导致窗口尺寸大于数组长度。

### 修复
- `counting.py: _run_pct_2d`, `_new_high_count`, `_new_low_count`
  — 守卫条件 `max(2, window//2) > n` → `window > n`
- `quality_pattern.py: _up_down_ratio_vec`
  — 同上

### 框架异常传播强化
- `node.py: _call_func` 只捕获 `TypeError`（签名不匹配的回退），
  不再捕获 `ValueError`，避免将因子计算运行时错误误判为"缺少 context 参数"
- 框架既有的 `try-except-finally` 结构能正确终止崩溃节点并发送 stop_signal 到下游：
  异常 → `except`（日志 + traceback）→ `finally`（`global_exit.set()` +
  `stop_signal` 推送上/下游 queue）→ 下游通过心跳超时检测并依次退出