# 精选实验结果

此目录保留最终配置的指标、正确性摘要、原始记录文件名和 SHA256。完整日志、输入 token、输出及调度轨迹保存在原始 records ZIP，本仓库不重复分发全部历史调试现场。

- [DFlash 与 CUDA Graph](dflash_graph/README.md)：四项对照，三轮中位数。
- [单卡 TP2 DP2 和 1P1D](distributed/README.md)：相同输入、输出及驻留约束，单次正式测量。
- [ShareGPT 调度](sharegpt/README.md)：固定 100 请求，完整 Prefill / Chunk / Chunk＋Prefix。
- [AWQ W4A16](awq/README.md)：官方量化 checkpoint、FP16 对照及存储开销。
- [EAGLE3 / DFlash / DSpark](speculative_compare/legacy-target-formal-20261021/README.md)：原 Target 适配、十二条三轮、全部输出一致；Markov/Confidence 消融及历史失败记录分别保留。

各表精度与执行路径不同；加速比只适用于表内指定基线，不能跨表相乘。
