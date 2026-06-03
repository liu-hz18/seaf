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
