# Mini-vLLM 推理引擎扩展与投机解码优化 | 2026.10

**项目概述：** 基于开源 MinivLLM，面向 Qwen3-8B 构建可复现的推理优化与评测框架，扩展 KV Cache、请求调度及多 GPU 推理，集成 DFlash 与 Target CUDA Graph，并接入官方 EAGLE-3、DFlash、DSpark 完成统一 Target 对照与消融。技术栈：Python、PyTorch、Triton、CUDA Graph、NCCL。

**KV Cache 与请求调度：** 扩展链式块哈希前缀复用、引用计数/LRU 及 Chunked Prefill 混合调度，支持共享 KV、部分 Prefill 不采样与动态补位。100 请求 ShareGPT 四组、各三轮对照中，Prefix-only 输出吞吐中位数较 Full Prefill 提升 **22.53%**；同一 Chunk 配置启用 Prefix 的配对吞吐提升 **16.61%**。

**DFlash 与 CUDA Graph：** 接入官方 DFlash Draft，实现多请求 Target 批量验证、连续前缀接受及 KV 回滚；通过静态缓冲与 Batch/Query 形状分桶捕获 Target 前向图。单 A100 40GB、每请求 256 输出 Token，单请求及三请求总吞吐分别达到 AR Graph 基线的 **2.99× / 1.52×**，三轮输出与接受轨迹一致，正式验证前向无 Graph Fallback。

**投机解码对照与消融：** 将官方 EAGLE-3、DFlash、DSpark 接入统一 Qwen3-8B Target，开展固定候选预算的方法对照与 DSpark 推理消融。单 A100、Batch=1、无 Graph 下，三种方法吞吐分别达到配对 AR Eager 的 **2.93× / 3.48× / 3.90×**（DSpark Markov Only），七种投机配置共 **252/252** 组完整输出一致；启用 Markov 的配对吞吐提升 **84.68%**，Confidence 截断未改善整体吞吐。

**AWQ 量化推理：** 集成官方 AWQ W4A16 权重加载与量化 Linear 参考验证，实现 Triton 量化 GEMM，权重及量化元数据存储较 FP16 减少 **62.77%**；所测峰值 allocated 显存由 **19.87～20.03 GiB 降至 10.31～10.47 GiB**。所测量化推理未获得吞吐提升。

**多 GPU 推理：** 完善 TP2 分片加载与推理，实现 DP2 双副本路由及单机 1P1D Prefill/Decode 分离，通过 NCCL 完成全部 36 层 KV 迁移与物理页表重映射。双 A100 两组受控负载下，DP2 相较单卡总吞吐提升 **19%～26%**；1P1D 在与 DP2 吞吐接近时，最大 ITL 降低约 **89%～90%**。
