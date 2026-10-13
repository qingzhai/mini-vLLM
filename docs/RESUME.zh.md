# 2026.10　轻量级大模型推理引擎复现与优化

**项目概述：** 基于开源 MinivLLM 扩展 Qwen3-0.6B/8B 推理流程，围绕 Prefill/Decode、分页 KV Cache 和请求调度实现与验证优化；集成 Prefix Caching、Chunked Prefill、DFlash 批量投机解码与 Target CUDA Graph，并完成单机双卡 TP/DP/1P1D 对照。技术栈：Python、PyTorch、Triton、CUDA Graph、NCCL、AWQ。

**推理与缓存调度：** 完成模型加载、分页 KV 管理、链式块哈希前缀匹配与 LRU 驱逐，支持 Chunked Prefill 与 P/D 混合连续批处理；验证部分 Prefill 不采样、共享前缀复用、请求补位与 KV 页回收。建立固定版本 ShareGPT 派生 100 请求评测，三种调度输出逐 token 一致；同一 Chunk 配置开启 Prefix 后，单次吞吐由 **43.67 提升至 52.51 tokens/s（+20.2%）**，复用 **13,568** 个输入 tokens。

**投机解码与 CUDA Graph：** 接入官方 DFlash Draft，完成批量 Target 验证、接受判断及 KV 回滚；通过静态输入缓冲和 batch/query 形状分桶捕获 Target 前向图。在单 A100 40GB、Qwen3-8B 固定 256-token 负载下，三轮中位数相对已开启 Graph 的自回归基线，单请求耗时 **5.513→1.846 s**，batch=2 的三请求总耗时 **11.653→7.651 s**，总吞吐分别达到 **2.99×/1.52×**；三轮输出与投机接受轨迹一致，验证前向无 Graph fallback。

**双卡请求并行：** 实现双副本独立 KV 管理及按驻留请求数路由，构建单卡、TP2、DP2 和 1P1D 的统一对照。在双 A100 40GB、两组 ShareGPT 派生负载单次测量中，DP2 相对单卡总吞吐提高 **19%～26%**；48 条请求均与单卡输出逐 token 一致，并记录每卡吞吐、显存和排队时延。

**Prefill/Decode 分离：** 实现单机 1P1D 推理，完成全部 36 层分页 K/V 的 NCCL 迁移、独立物理页表重映射及页所有权回收；P 卡负责 Prefill、D 卡负责 Decode。在与 DP2 近似吞吐下，两组负载最大 token 间隔由 **1.79～2.16 s 降至 0.20～0.21 s**，48 条请求输出逐 token 一致。

**量化与张量并行：** 集成官方 AWQ W4A16 权重加载与量化 Linear 参考验证，权重及量化元数据存储较 FP16 减少 **62.77%**；完成 NCCL TP2 分片加载与推理，在所测负载下将每卡峰值显存由约 **20.08 降至 10.10 GiB**。

---

以上是项目经历，可直接并入个人简历；项目名和表述明确保留开源基础与官方模型来源。实验精度、负载、比较对象、测量次数及未验证项见 [项目总结](PROJECT_SUMMARY.zh.md)。DFlash＋Graph 的倍数是相对 AR Graph；Prefix 的 20.2% 是相对同一 Chunk 配置；量化数字为权重存储统计。各实验配置独立，未宣称全部功能同时启用。
