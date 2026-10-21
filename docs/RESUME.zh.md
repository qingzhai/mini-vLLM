# Mini-vLLM 推理引擎扩展与投机解码优化 | 2026.10

**项目概述：** 基于开源 MinivLLM，面向 Qwen3-8B 扩展 KV Cache 与推理调度，集成 DFlash、CUDA Graph 及多 GPU 推理，并接入官方 EAGLE-3、DFlash、DSpark，构建统一 Target 验证与消融实验。技术栈：Python、PyTorch、Triton、CUDA Graph、NCCL。

**KV Cache 与请求调度：** 基于分页 KV 管理实现链式块哈希前缀复用、引用计数/LRU 及 Chunked Prefill 混合调度，支持共享 KV、部分 Prefill 不采样与请求动态补位。100 请求 ShareGPT 受控负载下，同一 Chunk 配置启用 Prefix 后，输出吞吐提升 **20.23%**。

**DFlash 与 CUDA Graph：** 接入官方 DFlash Draft，实现多请求 Target 批量验证、连续前缀接受及 KV 回滚；通过静态缓冲与 Batch/Query 形状分桶捕获 Target 前向图。在单 A100 40GB、256-Token 实验中，相较 AR Graph 基线，单请求及三请求总吞吐分别提升 **2.99× / 1.52×**，三轮输出和接受轨迹一致，验证前向无 Graph Fallback。

**投机解码对照与消融：** 将官方 EAGLE-3、DFlash、DSpark 接入统一 Qwen3-8B Target，开展固定候选预算下的方法对照与 DSpark 推理消融。单 A100、Batch=1、无 Graph 实验中，相对配对 AR Eager，三种方法吞吐分别提升 **2.93× / 3.48× / 3.90×**（DSpark Markov Only）；同一 DSpark Checkpoint 启用 Markov 后吞吐提升 **84.68%**，发现 Confidence 截断虽提高接受率，却未改善整体吞吐。

**AWQ 量化推理：** 集成官方 AWQ W4A16 权重加载与量化 Linear 参考验证，实现 Triton 量化 GEMM，权重及量化元数据存储较 FP16 减少 **62.77%**；所测峰值 allocated 显存由 **19.87～20.03 GiB 降至 10.31～10.47 GiB**。所测量化推理未获得吞吐提升。

**多 GPU 推理：** 实现 TP2 张量分片、DP2 双副本路由及单机 1P1D Prefill/Decode 分离，通过 NCCL 完成全部 36 层 KV 迁移与物理页表重映射。双 A100 两组受控负载下，DP2 相较单卡总吞吐提升 **19%～26%**；1P1D 在与 DP2 吞吐接近时，最大 ITL 降低约 **89%～90%**。
