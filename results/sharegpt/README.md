# ShareGPT 调度对照：精选结果

固定版本真实 ShareGPT 派生输入：80 个不同 prompt 加 20 个受控完整 prompt 重放，共 100 请求，50% short（256..512 tokens）和 50% long（1024..1536 tokens）；每请求 greedy 生成 64 tokens，忽略 EOS，同时入队，驻留上限 8。

单 A100 40GB，Qwen3-8B BF16 主干 + FP32 LM Head；固定算术和全阶段 per-query PagedAttention，纯 Decode 使用 Graph。三种模式输出均逐 token 一致，正式计时排除预热和诊断，每模式测一次。

同一 Chunk 配置启用 Prefix：吞吐 **43.674→52.510 tokens/s（+20.23%）**；平均 TTFT **72.091→68.250 s**；复用 13,568 输入 tokens（17.20%）。

相对完整 Prefill 的 48.339 tokens/s，Chunk+Prefix 吞吐为 **1.086×**，但平均 TTFT **64.675→68.250 s**、平均 E2E **69.913→75.134 s**。Chunk 单独未在本轮加速。

TTFT/E2E 包含排队，不含 HTTP、网络、分词和模型加载。20% 为实验构造的重复请求比例；原 Flash Prefill 未用于本轮参考配置。完整原始记录本地保存，`SOURCES.json` 提供 SHA256。
