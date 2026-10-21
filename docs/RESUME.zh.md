# 2026.10　轻量级大模型推理引擎复现与优化

**项目概述：** 基于开源 MinivLLM，围绕 Qwen3-8B 扩展 KV Cache 与请求调度，集成 DFlash 批量投机解码、Target CUDA Graph，并完成 EAGLE3/DFlash/DSpark 方法对照、Markov/Confidence 消融及双卡部署实验。技术栈：Python、PyTorch、Triton、CUDA Graph、NCCL、AWQ。

**缓存与请求调度：** 扩展分页 KV、链式块哈希前缀匹配与 LRU，支持 Chunked Prefill 和 P/D 混合连续批处理，验证部分 Prefill 不采样、请求补位与页回收。固定版本 ShareGPT 派生 100 请求中，三种调度输出逐 Token 一致；同一 Chunk 配置开启 Prefix，单次吞吐 **43.67→52.51 tokens/s（+20.2%）**。

**DFlash 与 CUDA Graph：** 接入官方 Draft，实现批量 Target 验证、连续前缀接受与 KV 回滚，按 batch/query 分桶捕获 Target 前向图。单 A100 40GB、固定每请求 256 输出，三轮中位数相对 **AR Graph**：单请求 **5.513→1.846 s**，驻留≤2 的三请求总耗时 **11.653→7.651 s**，总吞吐 **2.99×/1.52×**；输出和接受轨迹一致，验证前向无 fallback。

**投机方法对照与消融：** 将官方 EAGLE3/DFlash/DSpark 接入同一稳定 Target。单 A100 40GB、12 条固定输入、每条 256 输出、三轮测量，相对配对 **AR Eager** 的吞吐分别为 **2.93×/3.48×/3.90×**（DSpark Markov Only），**252/252** 组投机输出逐 Token 一致；同一 DSpark checkpoint 启用 Markov 后配对吞吐 **+84.68%**，Confidence 截断未获得总体吞吐收益。

**双卡 DP 与 P/D 分离：** 实现双副本独立 KV、按驻留数路由及单机 1P1D 的全部 36 层 K/V NCCL 迁移与页表重映射。两组 ShareGPT 派生负载单次测量中，DP2 较单卡总吞吐 **+19%～26%**；1P1D 在接近 DP2 的吞吐下，将最大 Token 间隔由 **1.79～2.16 s 降至 0.20～0.21 s**；两模式各 48 条请求与单卡输出一致。

**量化与张量并行：** 加载官方 AWQ W4A16 并验证量化 Linear，权重及量化元数据存储较 FP16 **−62.77%**；完成 NCCL TP2 分片加载与推理，每卡峰值显存约 **20.08→10.10 GiB**。所测 AWQ、TP2 均未获得吞吐提升。

---

以上为可并入个人简历的项目经历。DFlash＋Graph 使用 z-lab b16 Draft，相对 AR Graph；三方法对照使用 DeepSpec 官方七候选 checkpoint，相对各自同进程 AR Eager，计时包括 Prefill 与完整生成，Batch=1、无 Graph，252 组为 12 条输入×3 轮×7 种投机模式。三方法不是匹配训练条件的结构消融；Markov/Confidence 是同一 checkpoint 的推理策略消融。Prefix 的 20.2% 相对同一 Chunk 配置；量化数字仅为权重及元数据存储。没有宣称自行训练 Draft、生产 vLLM 优势或全部功能联合启用。证据见 [项目总结](PROJECT_SUMMARY.zh.md) 与 [三方法正式结果](../results/speculative_compare/legacy-target-formal-20261021/README.md)。
