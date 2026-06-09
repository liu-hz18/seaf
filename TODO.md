需要人来做的事情：
1. 不同节点的正确性验证，以及差异化的 period 配置
2. 不同模型要支持特征重要性的计算和记录，记录为 mlflow artifacts

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

