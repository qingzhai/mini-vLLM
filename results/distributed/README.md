# Single / TP2 / DP2 / 1P1D：精选结果

单机双 A100-PCIE-40GB（PHB），Qwen3-8B FP16 主干 + FP32 LM Head，固定计算形状和 per-query PagedAttention。统一驻留上限 4，请求各输出 64 tokens。ShareGPT 派生 burst 32 请求、stream 16 请求；每模式每负载正式测量一次。

- DP2 相对单卡总吞吐分别提高 **26.20% / 19.49%**；两卡总吞吐提升不能等同每卡效率提升。
- 1P1D 相对 DP2 吞吐近似持平；最大 token 间隔分别由 **2157.99→207.92 ms / 1792.52→200.24 ms**。P99 并未全面改善。
- DP2 与 1P1D 两负载各 48 请求全部与单卡逐 token 一致；TP2 为 47/48，吞吐下降，每卡峰值显存约 20.08→10.10 GiB。
- 1P1D 完成 36 层 K/V 的 NCCL GPU 迁移与物理页重映射；D 卡不执行 Prefill，P 卡不执行 Decode。handoff 为同步协议。

`summary.csv` 的吞吐是整组请求的总吞吐；`output_tokens_per_gpu_s` 单列每卡效率。Graph、DFlash、AWQ、Prefix 和 Chunk 在本轮关闭。完整原始记录本地保存，`SOURCES.json` 提供 SHA256。
