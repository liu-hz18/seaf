需要人来做的事情：
1. 解决多进程基建下，占用内存过多导致被 kill 的问题。排查各个进程占用的内存大小。
2. 增加数据采样，每隔固定日期，采样节点 input 和 output ，写入到 mlflow artifact 中

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
