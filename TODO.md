需要人来做的事情：
1. 开展回测
2. 回测 label_clip 的效果
3. node.py 中增加 mlflow 对 buffer size 和 两个游标 的记录
4. 排查策略节点 position_value 先上升后下降最后又接近0的问题 [是资金不足或者分组不够细致的原因，如果分10组，截面上是5000支股票，每只持仓10000元以上，分散持仓20天，则每组至少需要 20*10000*5000/10 = 1_0000_0000 = 1亿元的资金]

作为资深的机器学习量化研究员，我们的模型训练的 label 为 cs_zscore(close_{t+N} / close_{t+1} - 1), 这代表未来 N 日个股截面超额收益，也说明我们会在模型给出预测后的 t+1 收盘时进行买入，t+N 日收盘时进行卖出。并且，我们的交易是逐日进行的，尽管模型的预测周期是N, 但是我们在将资金分摊到了 N 天来进行预测和交易。这样既降低了滑点成本，也使得交易样本数量增加，从而与模型训练阶段的信号绩效更加对齐。至于损失函数，我们的模型训练应该最大化 截面IC, 也就是截面上 预测收益率向量 和 实际收益率向量的 pearson 相关系数。可以看到，对于一个支持做多和做空的市场，预测信号的截面IC就是该日风险调整后的相对于大盘收益均值的超额收益，截面IC越大，则预测越准确，策略将在单位波动率下获得更大的超额收益。这使得模型的训练、预测和实盘交易的指标更加对齐。此外，为了方便调试，你还需要在 model 节点的各个关键功能处给出充分的日志记录，用于前期团队对该节点的正确性和完整性进行测试和评估。你需要审视 model_node.py 查看是否有与上述要求不一致的地方，结合框架全局和业务特点，给出系统性的model节点搭建方案。

1. node.py 中 MultiInputNode 和 SourceNode 的 self.context，当初始化参数 context 为空时，应该置为空字典，而不是None
2. 基于1的修改，call_func所调用的函数返回 context 是不必要的。因为 context 是个字典，在 node.py 中的 result = self._call_func(self.name, run_input_f3d, current_context) 这行代码已经传入其中，函数可以对 current_context 进行修改，而不需要在函数返回值中返回，并交由 node.py 中的
```
if isinstance(result, tuple):
    output_f3d, current_context = result
else:
    output_f3d = result
```
这样的代码处理。node.py 中的节点计时操作，也是直接向 current_context 这个变量赋值即可，不需要处理该变量不存在的情况。
3. model_node.py 和 ic_analysis.py 中的节点逻辑函数返回值需要相应修改

对 model 节点 和 IC analysis 节点引入 mlflow 用于记录每次程序的实时运行情况。
1. 记录本次主程序的各项启动参数、代码库git版本号，以方便实验溯源
2. model 节点需要记录训练集上的训练损失、样本条数、nan比例、预测超额收益率的最大值、最小值、skew
3. IC analysis 节点逐日记录当日的截面pearson IC, 截面 rank IC, raw（未经过 cs_zscore的）截面收益率的 std, 当日累计 IC (cumsum pearson IC, cumsum rank IC)
4. 在 node.py 中记录每个节点执行运算的实际运行时间（即原来日志通过输出的计时部分，也要放到 mlflow 中来记录）
5. 在 node.py 中记录每个节点的 output queue 的队列实时大小，按照 call_func 调用的频率来记录以和每个交易日对齐
我们的框架为多进程流式数据框架，mlflow 的 run_id 注意在多进程之间需要保持一致，这样才会记录到同一个实验中，实验的名字为启动主程序的时间。后端存储采用 sqlite 后端而不是文件后端。

重要迭代：在 qpipe\node.py 中引入股票的上市和退市机制，导致涉及到时间序列的节点，需要根据最新日期的标的集合，调整窗口期内每个时间片的标的集合。对于退市的，最新的时间片必然不包含该标的，那么前续的每个时间片的该标的都应该删除；对于上市的股票，最新的时间片包含该标的，而前续的若干个时间片可能不包含该标的，这时应该将前续的时间片都补上该标的，同时列属性默认设置为0.0。主要涉及到的代码是 MultiInputNode.run 中的 window_frames 的合并逻辑；此外，我们的数据生成器 seafquant\data_generator.py 中也需要模拟真实的退市上市机制，暂时设定为股价首次低于 0.005 元的股票在次日退市，同时补充一个新的股票作为新上市的股票，初始价格随机，其他的OHLC等属性参照目前的逻辑生成；


小迭代：对于模型训练节点 seafquant\model_node.py 中的 nan 样本处理，只有当 某条样本 x 的特征列有超过一半的nan时，才删除此样本，否则将该条样本的nan数值置为 0.0。目前的逻辑是样本中有一个nan就会丢掉该条样本，需要进一步完善。当前的代码：
# 3. NaN 处理
valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
nan_count = sum(~valid)
nan_total = nan_count + len(y)
nan_ratio = nan_count / nan_total if nan_total > 0 else 0.0
X, y = X[valid], y[valid]
logging.info(f'[{name}] NaN removal: {nan_count} removed, {len(y)} remain')


内存优化迭代：
1. 对节点的输入设置 input_columns 来减少时间片占用的内存大小；
2. 我们先将每个节点在输出时的数据设置为 fp32 精度，这样队列中传输的数据以及缓冲就都是 fp32 了，这样改动范围较小，取得的内存收益还可以。
结合框架特点和基建特点，设计系统性的优化方案，并完成本次迭代。


我们的业务框架是基于多进程流式数据框架进行的，在 qpipe\node.py 的 MultiInputNode.run 和 SourceNode.run 中，每个节点的计算逻辑函数 如 _call_func 和 gen_func 都会根据输入的一定时间窗的数据来进行计算或记录，然后输出最新时间片的数据（也就是计算结果）。现在，我需要你增加一个功能，就是当这些函数调用一定频次时（如100次），就采样一次这个节点的输入数据(Frame3D)和输出数据(Frame3D)，作为快照保存在 mlflow artifact 中，文件命名为 {node_name}_in|out_{time}.csv, 其中 time 就是frame3d的最新时间片的日期(key)。请阅读和分析框架代码与基建代码，理清项目全局，设计合适的方案，完成本次迭代。注意迭代应该保持低耦合度、低代码复杂度和可扩展性，以方便合作开发者以及后续我们自己的继续迭代。

【重要迭代、较复杂迭代】增加截面选股策略与绩效计算记录模块 strategy。
1. 首先在源节点的数据生成函数中增加 close_uq 列，表示不复权的收盘价。此前已经有的 open, close, high, low 等等都是后复权价格，是用来计算截面选股的信号（已经由model节点给出）和策略净值的。
2. pipeline 中增加 strategy 节点。它的上游节点为 source 和 model, 即接收 source 节点的 close(后复权), close_uq(不复权)的截面数据（单日），以及 model 节点的截面信号数据（单日）。经由基建框架合并之后，进行分组选股、交易和净值的计算。 
3. strategy 节点内部，会缓存 T日的 signal 数据，并在 T+1 时间片进行选股和交易，这样是为了防止时间穿越，因为我们的信号是基于前一天的收盘价给出的，到次日才有机会交易，我们设置次日的交易时间为收盘时，这与前续模型节点的训练标签是一致的。
4. strategy 节点是截面选股策略，分为 group 组进行交易和持仓。首先对截面上的 signal 排序并分为 group 组，group 0 始终买入 signal分位数0-10% 的股票、group 1 始终买入 signal分位数10-20% 的股票、... 以此类推, 共 group 组。
5. 因为实际的股票有分红送股等行为，所有我们提供了 不复权 close_uq 和后复权 close 的收盘价。不复权数据主要用于交易，计算真实股数；复权数据主要用于计算净值NAV。此外复权数据还用于在因子和model节点用来计算最终的信号。
6. 我们是 fwd 日滚动交易，以提高统计数量，对齐信号的生成和策略的交易，也就是把资金平均分配到 fwd 日，每日都根据前一日的信号，在当日临近收盘时进行交易，持仓周期是 fwd 日，初始资金为 initial_cash.
7. 买入或卖出的手续费都是0.0005，最低5元。
8. 每个 group 的每日净值、持仓、交易、回撤等数据都保存到 mlflow 中。
9. 每个函数要配置对应的测试代码，以验证每个模块或单元的逻辑正确性和计算正确性。
对于每个 group 内的交易和净值计算，我初步设计了一套计算流程，伪代码（变量名不一定和上面的叙述保持一致，只是逻辑大概行得通）如下：
┌─────────────────────────────────────────────────────────────┐
│  FUNCTION init_context(initial_cash, fwd=20)                │
│    context.cash = initial_cash                              │
│    context.positions = {}        # (sid, batch_dc) → record │
│    context.pending_signal = None # T-1日信号，待T日执行       │
│    context.day_counter = 0                                  │
│    context.trade_log = []                                   │
│    context.position_log = []                                │
│    context.nav_log = []                                     │
│    RETURN context                                           │
│                                                             │
│  FUNCTION on_bar(context, date, signal_T, close_uq, hfq)   │
│    day_counter += 1                                         │
│                                                             │
│    ── Step 1: 计算今日复权因子 F_T ──                        │
│    FOR each stock: F_T[sid] = hfq[sid] / uq[sid]           │
│                                                             │
│    ── Step 2: 执行昨日待执行信号 ──                          │
│    IF pending_signal exists:                                │
│      找到今日到期的持仓 (mature_dc == day_counter)            │
│      total_equity = 现金 + 所有持仓市值                      │
│      slice_capital = total_equity / fwd                     │
│                                                             │
│      FOR sid IN signal ∩ maturing:                          │
│        旧仓实际股数 = Σ N_initial_i × (F_T / F_buy_i)      │
│        目标股数 = floor(slice_capital × weight / uq / 100)×100│
│        delta = 目标股数 - 旧仓实际股数                       │
│        IF delta > 0:  补仓(扣现金+手续费)                    │
│        IF delta < 0:  减仓(加现金-手续费)                    │
│        删除旧批次，创建新批次(N_initial=实际股数, F_buy=F_T) │
│                                                             │
│      FOR sid IN signal - maturing:                          │
│        新开仓：买入目标股数(扣现金+手续费)                    │
│        创建新批次(N_initial=买入股数, F_buy=F_T)            │
│                                                             │
│      FOR sid IN maturing - signal:                          │
│        全部平仓(加现金-手续费)                               │
│        删除旧批次                                           │
│                                                             │
│    ── Step 3: 存储今日信号 ──                               │
│    pending_signal = signal_T                                │
│                                                             │
│    ── Step 4: 计算并记录净值 ──                              │
│    total_equity = 现金 + Σ(N_initial × hfq / F_buy)        │
│    nav_log.append(date, cash, total_equity)                 │
│                                                             │
│    ── Step 5: 记录持仓快照 ──                               │
│    FOR each position:                                       │
│      实际股数 = N_initial × (F_T / F_buy)                   │
│      市值 = N_initial × (hfq / F_buy)                      │
│      position_log.append(...)                               │
│                                                             │
│    RETURN context                                           │
└─────────────────────────────────────────────────────────────┘
上述针对每个 group 进行交易和计算净值的伪代码，转化为的python代码示例如下（仅供参考和分析使用）：
```
import math
import pandas as pd
import numpy as np
from collections import defaultdict

# ════════════════════════════════════════════════════════════════
#  一、初始化
# ════════════════════════════════════════════════════════════════
def init_context(initial_cash=1_000_000, fwd=20,
                 commission_rate=0.0005, min_commission=5.0):
    """
    初始化回测上下文

    Parameters
    ----------
    initial_cash   : float  初始资金
    fwd            : int    持仓周期（交易日）= 资金均分份数
    commission_rate: float  买卖手续费率（万五 = 0.0005）
    min_commission : float  最低手续费
    """
    return {
        # ── 配置 ──
        'initial_cash'    : initial_cash,
        'fwd'             : fwd,
        'commission_rate' : commission_rate,
        'min_commission'  : min_commission,

        # ── 核心状态 ──
        'cash'            : initial_cash,
        # 持仓字典 key=(stock_id, batch_dc)  value=dict
        #   stock_id   : 股票代码
        #   batch_dc   : 建仓时的 day_counter
        #   N_initial  : 锚定不复权股数（黄金公式的核心变量）
        #   F_buy      : 建仓日的复权因子
        #   mature_dc  : 到期 day_counter = batch_dc + fwd
        #   entry_date : 建仓日期（用于输出展示）
        'positions'       : {},

        # T-1日信号，待T日收盘执行
        'pending_signal'  : None,
        'day_counter'     : 0,

        # ── 输出日志 ──
        'trade_log'       : [],   # 逐笔交易
        'position_log'    : [],   # 每日持仓快照
        'nav_log'         : [],   # 每日净值
    }


# ════════════════════════════════════════════════════════════════
#  二、内部辅助函数
# ════════════════════════════════════════════════════════════════

def _calc_commission(trade_value, rate, min_comm):
    """计算手续费：万五，最低5元"""
    return max(abs(trade_value) * rate, min_comm)


def _get_actual_shares(pos, F_today):
    """
    由锚定股数 + 复权因子推算当前实际股数
    ─────────────────────────────────────
    如果中间发生了10送5，F 从 F_buy 变为 1.5×F_buy
    actual_shares = N_initial × (1.5 × F_buy) / F_buy = 1.5 × N_initial
    ─────────────────────────────────────
    """
    ft = F_today.get(pos['stock_id'])
    if ft is None or pos['F_buy'] <= 0:
        return 0.0
    return pos['N_initial'] * (ft / pos['F_buy'])


def _get_position_value(pos, close_hfq):
    """
    ★ 黄金公式 ★  用后复权价计算持仓真实人民币市值
    ─────────────────────────────────────
    市值 = N_initial × (P_hfq / F_buy)

    证明：
      今日市值 = 实际股数 × 不复权价
               = [N_initial × (F_today/F_buy)] × P_uq
               = N_initial × (P_uq × F_today) / F_buy
               = N_initial × P_hfq / F_buy  ✓
    ─────────────────────────────────────
    """
    p_hfq = close_hfq.get(pos['stock_id'], 0)
    if p_hfq <= 0 or pos['F_buy'] <= 0:
        return 0.0
    return pos['N_initial'] * (p_hfq / pos['F_buy'])


def _compute_total_equity(context, close_uq, close_hfq):
    """总资产 = 现金 + 所有持仓市值（黄金公式）"""
    total = context['cash']
    for pos in context['positions'].values():
        total += _get_position_value(pos, close_hfq)
    return total


def _log_trade(context, date, stock_id, action, shares, price, value, commission):
    """记录一笔交易"""
    context['trade_log'].append({
        'date': date, 'stock_id': stock_id, 'action': action,
        'shares': shares, 'price': price, 'value': value, 'commission': commission,
    })


def _create_position(context, stock_id, dc, n_initial, f_buy, date):
    """创建新持仓批次"""
    context['positions'][(stock_id, dc)] = {
        'stock_id'  : stock_id,
        'batch_dc'  : dc,
        'N_initial' : n_initial,
        'F_buy'     : f_buy,
        'mature_dc' : dc + context['fwd'],
        'entry_date': date,
    }


# ════════════════════════════════════════════════════════════════
#  三、三种交易处理逻辑
# ════════════════════════════════════════════════════════════════

def _process_delta_trade(context, date, dc, sid, weight, slice_capital,
                         maturing_keys, close_uq, close_hfq, F_today):
    """
    差额交易：到期持仓 + 新信号继续持有
    → 只补仓/减仓，节省手续费，锚点重置为新批次
    """
    p_uq = close_uq.get(sid, 0)
    if p_uq <= 0:  # 停牌无法交易，延期一天到期
        for key in maturing_keys:
            context['positions'][key]['mature_dc'] = dc + 1
        return

    rate     = context['commission_rate']
    min_comm = context['min_commission']

    # ── 旧仓实际股数（含送股产生的零碎股）──
    old_shares = sum(
        _get_actual_shares(context['positions'][k], F_today) for k in maturing_keys
    )

    # ── 目标股数（100的整数倍）──
    target_value  = slice_capital * weight
    target_shares = math.floor(target_value / p_uq / 100) * 100

    # ── 差额交易 ──
    delta = target_shares - old_shares

    if delta > 0:                          # 补仓
        trade_value = delta * p_uq
        commission  = _calc_commission(trade_value, rate, min_comm)
        if context['cash'] >= trade_value + commission:
            context['cash'] -= (trade_value + commission)
            _log_trade(context, date, sid, 'buy', delta, p_uq, trade_value, commission)
        else:                              # 资金不足，尽力买入
            max_aff   = max(0, context['cash'] - min_comm)
            buy_shares = math.floor(max_aff / p_uq / 100) * 100
            if buy_shares > 0:
                trade_value = buy_shares * p_uq
                commission  = _calc_commission(trade_value, rate, min_comm)
                if context['cash'] >= trade_value + commission:
                    context['cash'] -= (trade_value + commission)
                    _log_trade(context, date, sid, 'buy', buy_shares,
                               p_uq, trade_value, commission)
                    target_shares = old_shares + buy_shares
                else:
                    target_shares = old_shares
            else:
                target_shares = old_shares

    elif delta < 0:                        # 减仓
        sell_shares = min(abs(delta), old_shares)
        trade_value = sell_shares * p_uq
        commission  = _calc_commission(trade_value, rate, min_comm)
        context['cash'] += (trade_value - commission)
        _log_trade(context, date, sid, 'sell', sell_shares,
                   p_uq, trade_value, commission)
        target_shares = old_shares - sell_shares

    # delta == 0 时无需交易，target_shares = old_shares

    # ── 删除旧批次，创建新批次（锚点重置）──
    for key in maturing_keys:
        del context['positions'][key]

    if target_shares > 0 and sid in F_today:
        _create_position(context, sid, dc, target_shares, F_today[sid], date)


def _process_new_trade(context, date, dc, sid, weight, slice_capital,
                       close_uq, close_hfq, F_today):
    """新开仓：信号中的股票，当前无到期持仓"""
    p_uq = close_uq.get(sid, 0)
    if p_uq <= 0 or sid not in F_today:
        return

    rate     = context['commission_rate']
    min_comm = context['min_commission']

    target_value  = slice_capital * weight
    target_shares = math.floor(target_value / p_uq / 100) * 100
    if target_shares <= 0:
        return

    trade_value = target_shares * p_uq
    commission  = _calc_commission(trade_value, rate, min_comm)

    if context['cash'] >= trade_value + commission:
        context['cash'] -= (trade_value + commission)
        _log_trade(context, date, sid, 'buy', target_shares,
                   p_uq, trade_value, commission)
        _create_position(context, sid, dc, target_shares, F_today[sid], date)
    else:                                  # 资金不足，尽力买入
        max_aff    = max(0, context['cash'] - min_comm)
        buy_shares = math.floor(max_aff / p_uq / 100) * 100
        if buy_shares > 0:
            trade_value = buy_shares * p_uq
            commission  = _calc_commission(trade_value, rate, min_comm)
            if context['cash'] >= trade_value + commission:
                context['cash'] -= (trade_value + commission)
                _log_trade(context, date, sid, 'buy', buy_shares,
                           p_uq, trade_value, commission)
                _create_position(context, sid, dc, buy_shares, F_today[sid], date)


def _process_close_trade(context, date, dc, sid, maturing_keys,
                         close_uq, close_hfq, F_today):
    """全部平仓：到期持仓不在新信号中"""
    p_uq = close_uq.get(sid, 0)
    if p_uq <= 0:                          # 停牌无法卖出，延期
        for key in maturing_keys:
            context['positions'][key]['mature_dc'] = dc + 1
        return

    rate     = context['commission_rate']
    min_comm = context['min_commission']

    for key in maturing_keys:
        pos = context['positions'][key]
        actual_shares = _get_actual_shares(pos, F_today)
        if actual_shares > 0:
            trade_value = actual_shares * p_uq
            commission  = _calc_commission(trade_value, rate, min_comm)
            context['cash'] += (trade_value - commission)
            _log_trade(context, date, sid, 'sell', actual_shares,
                       p_uq, trade_value, commission)
        del context['positions'][key]


# ════════════════════════════════════════════════════════════════
#  四、主函数：逐日调用
# ════════════════════════════════════════════════════════════════

def on_bar(context, date, signal, close_uq, close_hfq):
    """
    逐日调用 — 对齐信号、交易与净值

    Timeline
    --------
    T日收盘:  生成 signal_T
    T+1收盘:  执行 signal_T（用 T+1 的不复权价撮合）

    所以: on_bar 在 T日 被调用时，
      - 执行 context['pending_signal']（即 signal_{T-1}）→ 用 T日不复权价撮合
      - 存储 signal_T → 供 T+1 日执行

    Parameters
    ----------
    context   : dict   回测上下文
    date      :        当前交易日 T
    signal    : dict   T日收盘信号 {stock_id: weight}，T+1收盘执行
    close_uq  : dict   T日不复权收盘价 {stock_id: price}
    close_hfq : dict   T日后复权收盘价 {stock_id: price}

    Returns
    -------
    context   : dict   更新后的上下文
    """
    context['day_counter'] += 1
    dc = context['day_counter']

    # ═══════════════════════════════════════
    #  Step 1: 计算今日隐含复权因子
    # ═══════════════════════════════════════
    F_today = {}
    for sid in close_uq:
        puq, phfq = close_uq[sid], close_hfq.get(sid, 0)
        if puq > 0 and phfq > 0:
            F_today[sid] = phfq / puq

    # ═══════════════════════════════════════
    #  Step 2: 执行昨日待执行信号
    # ═══════════════════════════════════════
    if context['pending_signal'] is not None:
        sig = context['pending_signal']

        # 2.1 找出今日到期的持仓批次
        maturing = defaultdict(list)      # stock_id -> [position_keys]
        for key, pos in list(context['positions'].items()):
            if pos['mature_dc'] == dc:
                maturing[pos['stock_id']].append(key)

        # 2.2 计算当前总资产 → 确定新切片资金量
        total_equity = _compute_total_equity(context, close_uq, close_hfq)
        slice_capital = total_equity / context['fwd']

        sig_sids = set(sig.keys())
        mat_sids = set(maturing.keys())

        # 2.3 信号 ∩ 到期 → 差额交易 + 锚点重置（省手续费）
        for sid in sig_sids & mat_sids:
            _process_delta_trade(
                context, date, dc, sid, sig[sid], slice_capital,
                maturing[sid], close_uq, close_hfq, F_today
            )

        # 2.4 信号 - 到期 → 新开仓
        for sid in sig_sids - mat_sids:
            _process_new_trade(
                context, date, dc, sid, sig[sid], slice_capital,
                close_uq, close_hfq, F_today
            )

        # 2.5 到期 - 信号 → 全部平仓
        for sid in mat_sids - sig_sids:
            _process_close_trade(
                context, date, dc, sid, maturing[sid],
                close_uq, close_hfq, F_today
            )

    # ═══════════════════════════════════════
    #  Step 3: 存储今日信号，供明日执行
    # ═══════════════════════════════════════
    context['pending_signal'] = signal

    # ═══════════════════════════════════════
    #  Step 4: 计算并记录净值
    # ═══════════════════════════════════════
    total_equity = _compute_total_equity(context, close_uq, close_hfq)
    context['nav_log'].append({
        'date'          : date,
        'day_counter'   : dc,
        'cash'          : context['cash'],
        'total_equity'  : total_equity,
        'position_value': total_equity - context['cash'],
    })

    # ═══════════════════════════════════════
    #  Step 5: 记录持仓快照
    # ═══════════════════════════════════════
    for key, pos in context['positions'].items():
        sid = pos['stock_id']
        actual_shares = _get_actual_shares(pos, F_today) if sid in F_today else 0
        market_value  = _get_position_value(pos, close_hfq)
        context['position_log'].append({
            'date'         : date,
            'day_counter'  : dc,
            'stock_id'     : sid,
            'batch_dc'     : pos['batch_dc'],
            'N_initial'    : pos['N_initial'],
            'F_buy'        : pos['F_buy'],
            'F_today'      : F_today.get(sid),
            'actual_shares': actual_shares,      # 当前真实股数（含送股零头）
            'market_value' : market_value,        # 人民币市值
            'p_uq'         : close_uq.get(sid, 0),
            'p_hfq'        : close_hfq.get(sid, 0),
            'mature_dc'    : pos['mature_dc'],
            'entry_date'   : pos['entry_date'],
        })

    return context


# ════════════════════════════════════════════════════════════════
#  五、结果提取
# ════════════════════════════════════════════════════════════════

def get_nav_df(context):
    return pd.DataFrame(context['nav_log'])

def get_trade_df(context):
    return pd.DataFrame(context['trade_log'])

def get_position_df(context):
    return pd.DataFrame(context['position_log'])


if __name__ == '__main__':
    # ── 初始化 ──
    context = init_context(initial_cash=1_000_000, fwd=20)

    # ── 逐日调用 ──
    for i, dt in enumerate(trading_dates):
        # signal:   T日收盘信号，dict {stock_id: weight}
        # close_uq: T日不复权收盘价，dict {stock_id: price}
        # close_hfq:T日后复权收盘价，dict {stock_id: price}
        on_bar(context, dt, signal[i], close_uq[i], close_hfq[i])

    # ── 取结果 ──
    nav_df      = get_nav_df(context)       # 每日净值
    trade_df    = get_trade_df(context)     # 逐笔交易记录
    position_df = get_position_df(context)  # 每日持仓快照
```
请你根据上述要求，完成 strategy 节点函数的设计和编写，保持低耦合度和可扩展性，方便合作开发者和我们后续的进一步迭代和修正。保持耐心，细粒度拆解任务，结合框架特点和基建特点，以及真实量化交易的实际场景，设计合适的细粒度方案，然后一步步完成。

在 seafquant\ic_analysis.py 中，你需要给出我们截面选股策略 top多头组 - bottom 多头组的理论对数净值差(log(top nav) - log(bottom nav))。如果下游的选股策略是将截面N等分组，并且组内对不同股票是等权重持仓的，那么其近似的关系为：
$$
\ln(NAV_{top, T}) - \ln(NAV_{bot, T}) \approx 2N \cdot \phi\left(\Phi^{-1}\left(\frac{1}{N}\right)\right) \cdot \overline{\sigma_r} \cdot \sum_{t=1}^T \rho_t
$$
符号说明：
$NAV_{top, T}, NAV_{bot, T}$：第 $T$ 期 Top 组和 Bottom 组的净值
$N$：截面等分的组数
$\phi(\cdot)$：标准正态分布的概率密度函数 (PDF)
$\Phi^{-1}(\cdot)$：标准正态分布的逆累积分布函数 (分位数函数)
$\overline{\sigma_r}$：时间序列上截面等权对数收益率标准差的均值
$\rho_t$ 或 $IC_t$：第 $t$ 期的截面 Pearson IC

所以你需要做一些修正：
1. 在模型训练节点 seafquant\model_node.py，截面超额收益率(训练标签)定义为对数收益率：ln(close_{t+fwd}) - ln(close_{t+1}), 而不是简单收益率。注意收益率依然需要做截面的标准化。
2. 在IC计算节点 seafquant\ic_analysis.py，根据上述公式，给出理论上近似的 top-bottom 对数净值差，并将其记录在 mlflow metric 中。可以在 ic_epilogue 中实现，这样我们就可以计算 平均的 截面对数收益率的标准差。
3. strategy 节点中已经有了 top-bottom 对数净值差 的实际计算结果，已经记录在 mlflow 的 metric 中了，方便我们后续对比两者是否一致。

seafquant\strategy.py 策略文件中，因为我们针对 T日收盘后由前续节点计算出的信号，T+1日收盘才会进行交易。所以实际上 T日收盘后应该给出次日的交易计划，即次日交易哪些股票，每只股票交易的市值是多少、占总资产的比例（但此时还不知道次日实际的股价，因此算不出股数，只能有计划交易的市值）。请将每组每日的次日交易计划记录在 mlflow artifact 中，每组一个文件夹，每组的文件夹内每日是一个csv文件。你还需要注意交易的先后顺序，应该是先卖后买的。这个计划应该在调用函数的当日就可以给出了，而不需要等到次日的调用。请理解上述需求，结合框架结构和基本逻辑，设计方案并实现

1. strategy/strategy_g*/strategy_g*_trades.csv|strategy_g*_positions.csv 中的价格并没有保留两位小数，理论上讲，这个价格是由上游传过来的，上游的价格都是保留了两位小数，但是这两个 mlflow artifact 却没有，请排查问题
2. strategy/strategy_g*/strategy_g*_trades.csv|strategy_g*_positions.csv，strategy/strategy_g*/daily_plans/strategy_g*_daily_plan_xxxx-xx-xx.csv 中没有 stock_name 列，只有 stock_id 列，请排查问题。并且其他节点的数据中 stock_id 都是称为 name 列，strategy 节点注意保持和其他节点列属性含义的一致。
3. 将框架中所有的代表股票代码的 name 列替换为 code 列（更改列名，即原来 (key, name) 的 multiindex 修改为 (key, code)）
4. seafquant\data_generator.py 中 code 列的生成方式保持不变，stock_name 列采用随机常用汉字（3或4个）来生成（即随机汉字名）。注意到，该文件中
```
def _init_stock(si: int, t: int, name_prefix: str) -> dict:
    """创建并返回一个股票的初始状态。"""
    return {
        'log_price': rng.normal(0, 0.8),
        'mcap': rng.lognormal(np.log(100), 1.2),
        'active': True,
        'active_since': t,
        'name': name_prefix + str(si).zfill(4),
    }

# 初始化前 n_stocks 只股票
for si in range(n_stocks):
    stock_state.append(_init_stock(si, 0, 'S'))

# 预初始化剩余 reserve 股票（暂未激活）
for si in range(n_stocks, pool_size):
    stock_state.append({
        'log_price': 0.0,
        'mcap': 0.0,
        'active': False,
        'active_since': -1,
        'name': 'R' + str(si - n_stocks).zfill(4),
    })
```
这部分有 'name' 属性，但是这里的name实际上是我们期望的 code，我希望你将列名修正，与我们框架的设定保持一致，否则容易混淆。此外，你还需要生成一个 stock_name 列, 由常见汉字组成。
5. 整个代码库中的代码的价格属性/市值属性都用了 round(..., 2) 的舍入，以对齐真实股市和交易场景的价格/价值情况。我们要在 pipeline.py 中引入一个 precision 的 args 参数，用来代表舍入的精度。默认为2，但也可以由用户自行设置。所以代码中的 round 就变成了 round(..., precision). 节点代码中的 precision 可以通过 context 参数传入。请充分理解本多进程流式框架的结构和逻辑关系，完成本次迭代。

seafquant\model_wrappers.py中，扩展或自定义 lgbm 模型或 ridge 模型，三种模型要支持 pearson ic 损失函数, 也就是对于单批次的样本给出的预测值 (Batch size, feature) -> (Batch size, ) 。计算其与 真实值 (Batch size,) 的 -Pearson ic（取负值表示最大化）。pearson ic 是一个序列损失函数而不是MSE一样的单点损失函数。基于torch的神经网络因为自动求导机制，这个功能将很好实现。但是对于 lgbm 和 ridge 机器学习器，可能需要结合api自定义一些模块或工具。损失函数的选择需要配置在 pipeline.py 的 argparse 中，目前期望的可选项是 ["mse", "ic"], 请仔细研究框架代码和逻辑架构，调研实现方案，设计合理的抽象和设计模式，保持低耦合度和可扩展性，方便后续扩展损失函数族。请完成上述需求。

你是一个资深的量化机器学习研究员，你需要进行模型 IC 性能下降的原因探究: 经过遍历 precision 参数，发现是价格精度下降导致的，如何补偿这部分精度损失 ?
    观察到的现象：价格精度降低后，模型节点的 train mse 上升了，代表训练表现变差；IC节点计算结果显示 pearson ic 也下降了，代表样本外的测试表现也变差了。
    mlp 出现了类似的性能下降，不过 mlp 的 signal 要比 lgbm 的更加标准正态，偏度和极端值更小，变化也更平稳。对于IC节点给出的测试集（实际上是滚动训练与测试）效果，整体上的样本外表现：
mlp(precision=6) > lgbm(precision=6) > mlp(precision=2) > lgbm(precision=2)
如何解决？价格节点里增加一个 有效数字 的因子？引入 vwap 数据作为价格特征？将价格的舍入机制对齐到训练环节？
你可以编写一个类似的脚本，通过大量模拟的数据和加入适当噪音的标签，进行 lgbm 和 mlp 的大规模训练，验证数据集数值精度对测试集效果的影响，并结合多种可能的解决方案缓解这种现象。你的研究和解决方案应该先在独立的脚本中进行充分完备的验证，验证有效后再我们的多进程流式数据框架中落实。

pipeline 的数据流中增加 ensemble (bagging) 功能，由 pipeline.py 的 argparse 提供一个 list type 的参数，list 中的每个成员是 ['lgbm', 'ridge', 'mlp'] 中的一种。根据这个参数列表的设置，会有 并行的 len(list) 种model 节点，他们的输入是一致的，输出连接到每个model特定的 ic_analysis 节点，用于记录每个模型节点输出信号的IC. 多个 model 的输出还会汇集到 bagging 节点，由 bagging 对各个模型的截面信号进行等权融合后输出最终信号；bagging节点的输出接入到 strategy 节点用于选股策略，还会接入到一个 ic_analysis 节点进行融合信号的IC分析。所以，实际上你需要完成的任务主要有如下：1. 引入新的argparse参数，扩展数据流到多个并行的 model 节点，以及配置每个 model 节点下游的 ic 节点；2. 将多个 model 节点输出的信号都输出到 bagging 节点，bagging 节点内部对多源模型的信号输入进行等权融合，作为最终信号；3. bagging 节点的输出信号接入到目前的 strategy 节点，作为选股策略的驱动信号；4. 从单一模块和全流程的角度，编写完备的单元测试测试上述改动的逻辑正确性和计算正确性、功能正确性。请掌握我们多进程流式数据框架的大局，理清目前节点之间的拓扑关系和基本职责，形成迭代方案并进行迭代。要求迭代保持可扩展性、代码整洁性和低耦合度，方便团队和后续继续深入迭代。

重要迭代：接下来我们要接入 baostock 的真实历史日k线数据，不再使用 seafquant\data_generator.py 给出的模拟数据. 我们增加一个新的源节点 baostock_data. 请求 baostock 交易日信息、A股股票信息、股票的量价与基本面数据 见 api_test.py，首先阅读该示例文件了解 api 的调用方法、返回格式与注意事项。注意：每日baostock API请求不能超过5万次，超过后进入黑名单控制；关于数据库（关乎数据下载后的保存形式与频率）的建立方案，见 DATABASE.md；baostock_data 节点的功能是下载数据【到现实时间的“今天”为止，起始时间由 argparse 决定，对于缺失的数据应该完整补充】、将数据入库并通过迭代器的形式 yield 每日的截面股票数据（OHLCV, turnover, market_cap, pe, pb, pcf, ps）等。
1. 现实时间中，如果在 T 日调用，只能获得截至 T-1 日的数据；
2. 因为调用次数的限制，我们在获取数据时，每次调用需要获取多日的数据（如单只股票一年的数据）；
3. 约定: 数据的下载日期从 2007-01-01 到今日（今日没有数据，最新到 T-1 日），每个股票需要编排获取的日期；
4. 开发时，需要先按小数据区间，如 1年，并开展测试，包括数据下载、入库、生成器迭代过程、接入 pipeline.py 等
5. 调用api时偶尔有卡住或请求失败的现象，需要注意研究并实现防api卡死的解决方案。
请阅读相关文件和代码，充分理解需求，系统性研究解决方案，细化阶段性任务，保持耐心细致严谨，并一步步实施方案。


首尾成功、中间失败的情况是不允许出现的。你是一个资深的软件工程师，针对我们此前遇到的大量问题，以及数据节点的历史遗留问题，我提出了较为完整的解决方案：
1. 尝试从库中读取 args.start_date 开始截至今天（如果现实时间 > 18:00(UTC+8)则为今天，否则为 T-1日，这个变量称为 day_now）的全部交易日列表，如果该列表不存在，或者存储的交易日数据最后一天<day_now，则从 baostock api “bs.query_trade_dates(start_date=start_date, end_date=end_date)” 获取最新的交易日信息，将最新的截至 day_now 的交易日信息入库。这样我们就得到了交易日列表数据。
2. 从交易日列表数据中最初的交易日开始，每隔 STOCK_LIST_INTERVAL 个交易日，就尝试读取库中该交易日的全部A股股票列表（包括 date, code, name 信息），如果库中没有，则从 baostock api "bs.query_all_stock(day=day_str)" 获取该日的全部股票列表（这个每次调用大概耗时1分钟左右），并入库。这样，我们就得到了每隔STOCK_LIST_INTERVAL的股票列表数据，以及股票code 对应的 name 信息
3. 对于 2 中所执行的 每隔STOCK_LIST_INTERVAL 个交易日的循环，获得股票列表数据之后，便开始尝试获取从被遍历的交易日当日(trade_day)到 day_now 的股票日频数据，数据来自 bs.query_history_k_data_plus(code=..., fields=..., start_date=..., end_date=..., frequency='d', adjustflag=...) ，后复权数据的 adjustflag='1', 不复权数据的 adjustflag='3'，需要获取 后复权数据的BAOSTOCK_FIELDS和不复权数据的CLOSE_UQ_FIELDS。股票日频数据的具体获取方法如下：
a. 首先对该交易日的全部股票列表 (stock_list_on_trade_day) ，从其交易日当日到day_now, 划分chunk，划分的依据是按年度第一个交易日为间断点进行划分，即首年到首年底、次年全年、...、最后一年年初到day_now (比如 stock_list_on_trade_day=2007-05-01, day_now=2026-06-17, 则 chunk=[[2007-05-01, 2007-12-31], [2008-01-01, 2008-12-31], [2009-01-01, 2009-12-31], ..., [2026-01-01, 2026-06-17]]), 然后遍历每个股票的每隔chunk，检查数据库中是否有该股票该chunk的全部交易日的数据（如果出现chunk内 >0 个交易日的缺失，则标记为重新获取数据）。汇总得到所有应该获取的 task=(code, chunk) 列表后，开始从 baostock api 获取该task 的数据，获取数的过程由子进程 mp.Process 进行，执行结束后，返回数据。注意网络请求、baostock 登录等都应有重试机制，出现网络错误后应logout 再login。多个子进程并行完成所有的 task, 并将数据返回数据节点的主进程。主进程将每个 task process 获取的数据入库。注意，每个 api 访问前后，都应给出日志，包括调用api的信息汇总、调用耗时、总调用次数统计、返回信息的汇总；因为网络失败或执行错误，导致数据缺失或不全的，不设置缺省值，不进行入库，这样下次运行的时候，框架就会检测到缺失，根据上面的机制主动补全数据；不复权数据所获取的close_uq 列、股票列表中存储的 name 列，都应该并入 后复权数据的数据表，并保存到数据库中。
通过上述流程，我们就完成了全部数据的获取，并且每次执行 pipeline.py 时，都会及时获取或补齐目前缺失的全部数据，没有遗漏。
获取完数据之后，就开始沿着数据库存储的日期进行逐日迭代，以迭代器的形式返回日度的截面股票数据，供多进程流式框架的下游使用。
此前的实现过于粗糙，而以上我提出的是一个比较完整的数据加载、获取、迭代的逻辑，能最大程度防止网络失败导致的数据缺失，并且具有一定的鲁棒性，逻辑也比较清晰，便于维护。请结合框架代码和基建代码，充分理解上述设计，形成完整详实的实施方案，保持工程师的严谨细致，一步步完成方案，迭代本次大修改。迭代完成后，你还需要针对每个模块给出完备的单元测试，来验证其逻辑正确性和鲁棒性。

重要迭代：修改框架之间的数据流传递内容以统一各个节点之间的 day_idx (或称idx), 便于进行日志和 mlflow 记录。
目前，根据 qpipe\node.py 中的设置，SourceNode.run 负责数据生成器的迭代、将数据发送给下游，MultiInputNode.run 负责读取上游数据、执行计算逻辑、将数据发送给下游。数据以日度截面的形式传递，由 frame3d 数据结构进行维护。因为是日度数据，所以现在需要新增一个 idx 变量，表示当前时间片（截面）的数据在源节点遍历的idx是多少。这个 idx 由源节点给出，后续节点都采用这个 idx 作为时间片的唯一标识。
在代码层面，需要：
1. 数据源节点函数（迭代器）返回 (idx, frame3d) 的元组。针对 seafquant\baostock_data.py 和 seafquant\data_generator.py 这两类源数据迭代器都应修改其 yield 的返回值
2. SourceNode.run 将 (idx, frame3d) 压入下游队列 outq.put(...)
3. MultiInputNode.receive_worker 的 input_queue.get(timeout=0.5) 从上游队列获取 (idx, frame3d) 数据元组，将 idx 作为 time_value 保存到该上游队列对应的数据缓冲区：self.buffers[queue_idx]
4. MultiInputNode.run 根据上游队列 buffers 的 time_value 进行时间片合并成 Frame3D(merged_df)（这一步应该和之前是一致的，保留原逻辑即可）然后 time_order_buffer 保存的数据 也是 (idx, Frame3D(merged_df)) 元组，这里的 idx 是上游传递过来的 idx 数据，不再是原来根据 input_queue 中 get 的 frame3d 确定的日期数据。执行完预处理逻辑之后，MultiInputNode.run调用 self._call_func 计算得到 output_f3d，后处理得到 latest_f3d 之后，将 (idx, latest_f3d) 压入队列，这样 idx 就可以被下游节点收到并执行类似的逻辑。
5. 各个节点对应的 mlflow 记录，其 step 参数不再采用 step = trading_step(current_context.get('start_date', ''), max_key) 格式，而是采用 idx 作为 step 进行记录；各节点的 logging 模块，输出的日志格式应该补充带有 [{idx}] 信息。
仔细阅读框架代码，理清模块之间的关系和模块内部的职责，形成完整严谨的迭代方案，以工程师的严谨和耐心，一步步完成上述迭代。

seafquant\factor\cross_section.py 和 seafquant\factor\trend.py 是因子计算节点中耗时比较长的节点，尤其是截面股票数量上升之后，这两个节点的计算时间是其他节点的4倍左右。这些节点的逻辑都是输入一个[时间，截面，数据列]的三维datafrrame，计算最后一个时间片(T=-1)的截面因子向量，输出的形状为[截面，因子列]。你可以阅读 seafquant\factor 的因子计算逻辑，理解框架结构，通过向量化计算、just-in-time计算、拆分节点计算逻辑到多个节点等方式，优化因子部分的  性能瓶颈（尤其是截面股票数量较大时，如5000），并编写单元测试，测试性能优化的结果，对所有节点在(时间=200，截面=5000，数据列=和框架一致)的随机数据上的性能表现进行测试，并为进一步优化各个节点的计算时延做准备。

==========================================================================================
Module                          100s       500s      1000s      2000s      5000s
--------------------------------------------------------------------------------
counting                       0.337      0.812      3.770      6.479     16.085
interaction                    0.592      0.908      3.758      6.223     15.091
momentum                       0.418      0.786      3.287      6.045     13.956
quality_pattern                0.170      0.686      2.512      5.191     12.485
value                          0.866      0.934      3.294      5.571     12.049
quality_merged                 0.880      0.991      3.049      4.968     10.927
liquidity                      0.651      0.640      1.972      3.439      6.983
volatility                     0.074      0.281      1.012      2.397      5.856
tspct                          0.117      0.300      1.127      2.135      5.338
precision                      0.099      0.244      0.999      1.917      4.986
quality_autocorr               0.038      0.196      0.549      1.525      3.949
cross_section                  0.453      0.360      1.222      1.855      3.668
trend                          0.034      0.133      0.494      0.980      2.304
--------------------------------------------------------------------------------
TOTAL                          4.729      7.270     27.045     48.724    113.678
==========================================================================================
性能测试的结果如上，需要把快的节点合并一下、慢的节点拆分成两个，节省内存。整体压在 8s 左右.

在 pipeline.py 开启 --data-source baostock 模式时，框架会从数据库 duckdb 
中加载历年每日的真实股票k线数据，这些数据中，相比于 synthetic 模式，多了若干列属性，主要是估值属性，包括：
"peTTM"       DOUBLE,
"pbMRQ"       DOUBLE,
"psTTM"       DOUBLE,
"pcfNcfTTM"   DOUBLE,
我们希望在 seafquant\factor 文件夹下增加一个新的因子节点：估值因子节点。这个节点根据 close 属性与上述4个估值属性（如果你结合实际的量化研究场景，认为需要引入其他的列属性也可以），在一定时间窗口内，计算每只股票的若干估值或基本面相关的因子。你可以参考同目录下其他因子节点的实现方式，并基于自己对框架的深度理解以及量化因子研究的基础或前沿知识，设计20个用到估值指标的因子，并将其引入我们的数据流水线中。注意在 synthetic 模式下，没有这些属性，所以你在 pipeline.py 的拓扑定义上可能需要做一个区分。
充分理解上述需求和框架与业务代码的结构与逻辑，保持耐心细致，完成上述需求。


数据下载提速工程：
你已经是一个资深的架构工程师，现有的 baostock 数据下载节点采用子进程的方式下载数据，每个子进程下载一个task(一个股票一个时间区间内)的数据，这就需要在每个task 都要进行一次 socket 建立和断开连接的过程（baostock 依赖库是通过 socket 建立连接的）
然而，每次建立和中断连接操作是耗时的，并且由于 baostock 平台的限制，我们无法通过多进程的方式访问 baostock api.
因此，考虑到 baostock api 存在一定的连接稳定性，我们可以将 一个 batch 的 task 集中送入子进程，子进程进行一次登录（登录bs.login就是socket 建立连接的过程）（如果task执行过程中遇到获取数据或连接失败的情况，当然需要尝试logout 并 重新login），就可以串行获取一个 batch 的 task, 那么我们的预期数据下载速度将得到大幅提升。
我们在 seafquant\baostock_data.py 中引入 BATCH_SIZE 这一全局变量，用于设置每个子进程负责的 task 数量。原来的派发任务代码在 seafquant\baostock_data.py:525-584, 你需要改造该流程；
原来的子进程单任务执行代码在 seafquant\baostock_worker.py 中，每个任务涉及到login,logout 以及 2次api调用，并由 _MAX_RETRIES 全局变量控制每个api的重试次数。每次失败时，都要重新执行 logout 和 login 来尝试重建连接。你需要将该部分代码改造成适应 batch tasks 的形式。
此外为了实现数据下载过程中全流程的监控，设置了很多logging日志，你在batch化的代码中也要保留并适配这些日志。
请认真梳理 baostock data 节点的框架结构、逻辑关系和关键变量，仔细规划方案和步骤，耐心完成任务。

【关键迭代】策略每日数据逐日dump到mlflow artifact: 
strategy 节点 (seafquant\strategy.py) 在 strategy_epilogue 函数中会保存若干 artifact 文件，包括 nav, trade_log, position_log, daily_plans. 其中 trade_log 是每日实际交易时记录的，记录实际交易内容，在 seafquant\strategy_core.py 文件中实现；position_log 是每日交易结束时记录的，记录了当日的持仓快照，在 seafquant\strategy_daily.py 中实现；daily_plans 是 strategy 节点接收到当日信号之后生成的，在 seafquant\strategy.py 和 seafquant\strategy_daily.py 分别有调用和实现。
目前的代码是在节点退出时，统一 dump 这三类日志。并且 trade_log, position_log 是将所有数据合并成了一个大的 dataframe 然后保存到 artifact 文件的；daily_plans 是每日有一个 artifact 文件。
现在，你需要对该部分逻辑进行改造：收集 trade_log, position_log, daily_plans 当日记录的数据，并在 strategy_fn 中（也就是数据产生当日），记录到 mlflow 的 artifact 中，每日保存一个当日的文件。保存的文件格式分别为：f'strategy/{name}_g{gid}/trade/trade_{date_str}', f'strategy/{name}_g{gid}/position/position_{date_str}', f'strategy/{name}_g{gid}/daily_plans/daily_plan_{date_str}'. 其中 daily_plans 的保存方式与现在是一致的，trade_log和position_log你需要适当的改造。
请充分认真理解 strategy 节点的框架结构、数据流和子模块逻辑关系，仔细设计合理的方案，保持耐心，完成上述任务。

seafquant\ic_analysis.py 的 IC 节点中，ic_analysis_fn 返回值是 f3d，几乎就是这个节点的输入值，没有体现该节点的任何信息。现在我们希望 IC 节点的返回值能更有意义。要求返回：t+fwd 卖出日当天 的 code, stock_name, close, close_uq, fwd_ret(buy_t -> sell_t), cs_excess_fwd_ret(buy_t -> sell_t), pred_signal(at pred_t) 这几个属性。请充分理解 IC 节点的逻辑和架构，完成该任务。

seafquant\strategy.py 中 strategy_fn 最后会记录逐股逐组持仓市值 Frame3D。但是我运行 python pipeline.py --data-source baostock --start-date 2007-01-01 --fwd 20 --model-window 200 --ensemble mlp 发现实际记录的 持仓市值存在为负的情况，并且都是-0.0x 的小数字，而且不同的股票在多个 group 中的mv是完全一样的，如下：
keycodestock_nameg0_mvg1_mvg2_mvg3_mvg4_mvg5_mvg6_mvg7_mvg8_mvg9_mv
6/23/2009sh.600000浦发银行0-0.026659558-0.026659558-0.040116259-0.026659558-0.026659558-0.037696345-0.026659558-0.042988241-0.046923322
6/23/2009sh.600001邯郸钢铁0-0.026659558-0.026659558-0.040116259-0.026659558-0.026659558-0.037696345-0.026659558-0.042988241-0.046923322
6/23/2009sh.600004白云机场0-0.026659558-0.026659558-0.040116259-0.026659558-0.026659558-0.037696345-0.026659558-0.042988241-0.046923322
6/23/2009sh.600005武钢股份0-0.026659558-0.026659558-0.040116259-0.026659558-0.026659558-0.037696345-0.026659558-0.042988241-0.046923322
请仔细检查 strategy 节点的逻辑，排查出现该现象的原因。