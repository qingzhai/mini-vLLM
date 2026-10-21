# DeepSpec 参考实现探索记录

最新：[三轮正式测量与消融分析](formal-20261019/README.md)。12 条输入、256 输出、三轮全部完成；相对 AR 的完整序列一致性未通过。DSpark No Markov/Markov Only 之间 36 组输出相同，内部吞吐比约 1.89×；置信阈值没有观察到收益。完整数据和失败边界随目录保留。

已归档 [2026-10-19 Smoke](smoke-20261019/README.md)。单 A100 PCIe 40GB，Qwen3-8B BF16 SDPA，batch=1，各数据集一条、32 输出、单轮；存在一个 Greedy 分叉，正在精度回归。

[同日诊断](diagnostic-20261019/README.md) 稳定复现该分叉，并隔离 Cache 历史、Query 形状与局部 FP32 Head 结果；尚待整段精度回归，不是新增性能 Benchmark。

**不能作为正式三轮消融结果、任务质量评测或 Mini-vLLM Serving 性能。** 原始失败样本、逐 Token 输出、接受轨迹均保留。正式长测在诊断和协议确认后另建结果目录。
