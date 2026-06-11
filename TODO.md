需要人来做的事情：
1. IC skew 的负向问题研究

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
