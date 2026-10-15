# Qwen3 推理引擎复现与优化：项目总结

项目时间：2026.10。基于 [Wenyueh/MinivLLM](https://github.com/Wenyueh/MinivLLM) 扩展和整理，使用官方 Qwen3、AWQ checkpoint 与 DFlash Draft；本项目聚焦推理执行、调度及实验验证。所有性能数字来自已完成的 GPU 运行记录，本次整理没有重新执行 GPU 实验。

## 1. 完成范围

| 模块 | 已完成和验证的内容 |
|---|---|
| Qwen 推理 | Qwen3-8B 为主要优化与评测对象；Qwen3-0.6B 用于早期基础推理及缓存/分块调度功能检查 |
| 分页与前缀缓存 | 分页 KV、链式块哈希匹配、完整页前缀复用与 LRU；早期 16 页压力场景发生 14 次驱逐，最后页全部回收 |
| 分块与连续批处理 | Chunked Prefill、Prefill/Decode 混合步骤、部分 Prefill 不采样、驻留上限和请求补位 |
| Attention | 使用 Triton PagedAttention；原 Flash Prefill 有独立小规模算子参考检查。正式跨调度正确性使用统一的 per-query PagedAttention 参考配置 |
| 投机解码 | 官方 BF16 DFlash Draft、批量 Target 验证、接受/拒绝、KV 回滚和请求补位 |
| CUDA Graph | 普通 Decode 与 DFlash Target 验证图；静态缓冲、动态元数据更新、batch 1/2 × query 1/16 四种形状 |
| 多卡 | NCCL TP2 分片、DP2 双副本路由、单机 1P1D 全层 KV 迁移与物理页重映射 |
| W4A16 | 官方 AWQ checkpoint 消费检查、量化 Linear 参考验证、推理与显存统计 |
| 评测 | 记录模型元数据及固定数据版本、逐 token 输出与时间、调度轨迹、源代码摘要、正确性和性能分别验收 |

这些模块有独立实验入口和配置；没有把全部功能同时启用后验收。正式 ShareGPT 调度、DFlash＋Graph、AWQ 和 TP/DP/1P1D 性能结果均针对 Qwen3-8B；0.6B 未完成同等范围的评测。发布仓库保留精简代码与精选结果，完整历史原始记录另行本地保存。

## 2. 硬件、精度与指标

主要环境为 A100-PCIE-40GB、Python 3.11.14、PyTorch 2.9.1+cu128、CUDA Runtime 12.8、Triton 3.5.1、Transformers 4.57.3。双卡互联拓扑为 PHB。不同实验精度和执行方式如下，不能跨表直接比较吞吐或相乘加速比。

| 实验 | 精度与执行路径 | 正式次数 |
|---|---|---:|
| DFlash＋Graph | FP16 Target 主干＋FP32 LM Head，BF16 Draft，固定计算形状，全阶段 per-query PagedAttention | 每 case/模式 3 次，中位数 |
| ShareGPT 调度 | BF16 主干＋FP32 LM Head，固定计算形状，全阶段 per-query PagedAttention，纯 Decode 使用 Graph | 每模式 1 次 |
| Single/TP2/DP2/1P1D | FP16 主干＋FP32 LM Head，固定计算形状，全阶段 per-query PagedAttention；Graph/DFlash/AWQ/Prefix/Chunk 关闭 | 每 case/模式 1 次 |
| AWQ | FP16 激活＋FP32 LM Head，官方 AWQ W4A16 GEMM group128；Graph/Prefix/Chunk 关闭 | 每 case/模式 3 次，各指标中位数 |

TTFT 是从入队至首 token，E2E 是入队至完成，均包含引擎排队；TPOT 是首 token 后的平均生成间隔。ITL 是相邻输出 token 间隔，最大 ITL 和 P99 含义不同。总吞吐为整组输出 tokens/整组耗时；多请求总吞吐不是单个请求的生成速度。计时不包含模型加载、分词、预热、诊断或 Graph 捕获，不包含 HTTP 或网络服务开销。投机解码批量返回 token，其平均 TPOT 不能代替逐 token 可见间隔分布。

## 3. DFlash 与 CUDA Graph 四项对照

单 A100。single-code 为 1 请求，batch2-requests3 为 3 请求且驻留上限 2；每请求 greedy 生成 256 tokens、忽略 EOS。使用固定提示词，不是 ShareGPT。各模式三轮，顺序旋转；所有模式驻留相同 Draft 和四张图，也统一采集 Target 的五层特征。

| 模式 | 单请求总耗时 s | 单请求 tokens/s | 三请求总耗时 s | 三请求总 tokens/s |
|---|---:|---:|---:|---:|
| AR eager | 25.416 | 10.07 | 51.343 | 14.96 |
| AR Graph | 5.513 | 46.43 | 11.653 | 65.91 |
| DFlash eager | 5.372 | 47.65 | 18.763 | 40.93 |
| DFlash＋Target Graph | **1.846** | **138.70** | **7.651** | **100.38** |

相对 AR Graph，联合模式单请求总吞吐 **2.987×**、总耗时降低 **66.52%**；三请求总吞吐 **1.523×**、总耗时降低 **34.34%**。相对 DFlash eager 为 2.911×/2.452×。单独 DFlash eager 对 AR Graph 为 1.026×/0.621×，说明只比较 AR eager 会夸大相对已有 Graph 基线的收益。

| 诊断与执行证据 | 结果 |
|---|---|
| 正式完整输出 | 24 次运行、48 次请求执行、12,288 个 token 全部匹配参考 |
| 同历史前向 | 8 组覆盖 batch 1/2、query 1/16、256/512 页边界；logits/hidden/features/picks 精确一致，max_abs=0 |
| 实际 Graph 执行 | 所有验证前向实际 replay，fallback=0；初始 Prefill 图外 |
| 投机状态 | eager/Graph 接受、拒绝、推进轨迹一致；三请求场景请求补位 1 次、回滚 9 页；结束全部页归还 |
| Target 前向 | 单请求 256→46 次；三请求 512→144 次 |
| Draft 接受率 | 单请求 211/675=31.26%；三请求 536/3465=15.47% |
| 捕获开销 | 四张图共 3.533 s，捕获时 allocated 增量 77.39 MiB |
| 三轮总耗时范围 | 联合模式单请求 1.837～1.846 s，三请求 7.634～7.753 s |

单请求平均 TTFT 基本相同（AR Graph 约 0.101 s、联合模式 0.100 s），主要收益来自生成阶段；平均 TPOT 为 21.223→6.843 ms。三请求平均 TTFT 为 2.185→1.176 s，E2E 为 7.695→5.302 s，含第三个请求等待空位的时间。

Graph 仅覆盖 Target 前向，Draft、初始 Prefill、接受判断、回滚在图外。峰值 allocated 约 21.98/22.27 GiB，所有模式都驻留 Draft 和图资源，不能据此比较独立 AR/DFlash 部署的最低显存。本结果是本项目固定算术执行器的受控对照，不是相对 vLLM 等生产引擎的加速比。

精选指标与来源摘要：[results/dflash_graph](../results/dflash_graph/README.md)。

## 4. ShareGPT 调度对照

固定数据集 `heka-ai/sharegpt-english-10k-vllm-serving-benchmark`，revision `854a6eb3c310f07f9e8b269a735bf6fe547e543d`。80 个不同 prompt 加 20 个受控完整 prompt 重放，形成 100 请求；50 条短输入 256..512 tokens、50 条长输入 1024..1536 tokens。每请求 greedy 输出 64 tokens、忽略 EOS，100 请求同时入队，最多驻留 8；330 个 KV 页，每页 256 tokens。

| 配置 | 总耗时 s | 平均 TTFT s | 平均 TPOT ms | 平均 E2E s | 输出 tokens/s | 输出序列一致 |
|---|---:|---:|---:|---:|---:|---:|
| 完整 Prefill | 132.397 | 64.675 | 83.134 | 69.913 | 48.339 | 基线 |
| Chunked | 146.539 | 72.091 | 131.181 | 80.356 | 43.674 | 100/100 |
| Chunked＋Prefix | 121.881 | 68.250 | 109.267 | 75.134 | 52.510 | 100/100 |

同一 Chunk 配置启用 Prefix：吞吐 +20.23%，平均 TTFT 72.091→68.250 s，平均 E2E 80.356→75.134 s；复用 13,568 个输入 tokens，占 78,889 输入 tokens 的 17.20%。Chunk/Chunk+Prefix 分别发生 267/214 个部分 Prefill 块、188/158 个混合步骤，最后页全部归还。

相对完整 Prefill，Chunk+Prefix 吞吐 +8.63%，但平均 TTFT +5.53%、E2E +7.47%；不能写为同时改善三项指标。20% 是人为构造的重复请求比例，不是天然共享前缀率。每模式只正式测一次，未评估运行间波动。原 Flash Prefill 没有用于本轮参考配置，本轮通过不能代表原 Flash 路径已修复。

精选指标与来源摘要：[results/sharegpt](../results/sharegpt/README.md)。

## 5. Single、TP2、DP2 与单机 1P1D

双 A100 PHB；统一总驻留上限 4，KV 数据缓冲区合计 2.25 GiB。复用固定版本 ShareGPT 派生输入，burst 为 32 请求同时到达，stream 为 16 请求按固定时间到达；每请求输出 64 tokens。每模式每负载正式测一次，计时包括排队、调度/RPC、采样与 KV 迁移。

| 负载 | 模式 | 总耗时 s | 输出 tokens/s | tokens/(GPU·s) | 平均 TTFT s | 平均 E2E s | P99 ITL ms | 最大 ITL ms | 序列一致 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| burst | Single | 85.290 | 24.012 | 24.012 | 39.959 | 48.081 | 1864.866 | 4400.130 | 基线 |
| burst | TP2 | 94.241 | 21.732 | 10.866 | 44.411 | 53.360 | 2056.610 | 4914.372 | 31/32 |
| burst | DP2 | 67.585 | 30.303 | 15.151 | 30.887 | 37.864 | 115.328 | 2157.990 | 32/32 |
| burst | 1P1D | 67.457 | 30.360 | 15.180 | 30.291 | 36.518 | 142.570 | 207.916 | 32/32 |
| stream | Single | 41.077 | 24.929 | 24.929 | 15.510 | 22.921 | 619.347 | 3077.076 | 基线 |
| stream | TP2 | 46.930 | 21.820 | 10.910 | 18.015 | 26.452 | 705.923 | 3531.997 | 16/16 |
| stream | DP2 | 34.378 | 29.787 | 14.893 | 11.890 | 18.441 | 150.294 | 1792.517 | 16/16 |
| stream | 1P1D | 34.421 | 29.749 | 14.875 | 11.915 | 18.221 | 143.253 | 200.235 | 16/16 |

DP2 是两个完整模型副本、独立 KV 管理，按最少驻留请求路由。相对单卡总吞吐 +26.20%/+19.49%，TTFT −22.70%/−23.34%；但两卡未达到线性扩展，每卡吞吐下降。

1P1D 在 P 卡打包全部 36 层 prompt K/V，通过 NCCL GPU send/recv 传至 D 卡，再按独立物理页表重映射。D 卡 Prefill 次数为 0、P 卡 Decode 次数为 0；两组迁移共 3,713,236,992/1,769,766,912 bytes，平均 handoff 13.03/12.42 ms，包含打包、等待、传输和写入，不能换算纯链路带宽。

1P1D 对 DP2 吞吐 +0.19%/−0.13%，视为基本持平；最大 ITL −90.37%/−88.83%。P99 没有全面改善：burst 全体 P99 从 115.33 增至 142.57 ms。handoff 本身为同步协议，未实现异步传输和计算重叠，也未执行 CUDA kernel profiling。全部逻辑页所有权归还，预分配显存池仍驻留。

TP2 的主要结果是分摊每卡内存，峰值约 20.08→10.10 GiB；本轮吞吐下降。同历史 8 行 logits max_abs=0.08344、relative_RMSE=0.001803，低于预设 0.35/0.01 阈值；完整生成 47/48 一致，数值容差检查不等于自由生成全部一致。尚未定位该条生成分叉原因。

精选指标与来源摘要：[results/distributed](../results/distributed/README.md)。

## 6. AWQ W4A16

官方 `Qwen/Qwen3-8B-AWQ` revision `4da05a8edb55c6046cce958586c33b61da07bb79`；GEMM group128。903 个 checkpoint 张量全部消费、144 个合并量化模块，6 项量化 Linear 对解包参考检查通过，另验证有限值生成与页回收。没有执行自主 AWQ 校准或任务质量评测。

| 存储口径 | FP16 bytes | AWQ bytes | 降低 |
|---|---:|---:|---:|
| 权重及量化元数据 | 16,381,470,720 | 6,098,479,104 | 62.772% |
| 加上额外 FP32 LM Head | 18,870,790,144 | 8,587,798,528 | 54.492% |

每请求生成 32 tokens；short 35 输入 tokens、long 530 输入 tokens。每 case 各三轮，各指标为对应中位数。

| Case | FP16 总耗时 s | AWQ 总耗时 s | FP16 tokens/s | AWQ tokens/s | FP16→AWQ 峰值 allocated GiB |
|---|---:|---:|---:|---:|---|
| short-b1 | 2.642 | 9.332 | 12.111 | 3.429 | 19.873→10.314 |
| short-b4 | 2.723 | 10.545 | 47.001 | 12.139 | 19.884→10.325 |
| long-b1 | 2.554 | 17.785 | 12.531 | 1.799 | 19.922→10.363 |
| long-b4 | 2.792 | 35.897 | 45.840 | 3.566 | 20.028→10.469 |

当前实现更慢，结论限于权重存储与显存下降；62.772% 不是整机显存峰值的下降比例，也没有量化后生成完全相同或任务质量无损结论。

精选指标与来源摘要：[results/awq](../results/awq/README.md)。

## 7. 证据、归档与结项边界

`results/` 发布的是精选摘要，含条件、指标及源记录 SHA256。每个目录的 `SOURCES.json` 记录原始 ZIP 文件名、ZIP SHA256 和所引用记录的 SHA256。完整 records ZIP 含 token、trace、配置与日志，由项目持有者本地保留；公开仓库不发布长 prompt、逐 token 原始记录、机器路径或连接信息。模型权重、虚拟环境和缓存不属于代码归档。

旧 DFlash 的诊断单次结果与早期 TP 实验不再作为主要性能结论：本总结优先使用四项 Graph 对照与统一双卡对照。不同实验的精度、负载和测量次数有区别，不把其数值横向合并。

未宣称完成其他模型适配、PP、Draft Graph、异步 KV handoff、生产 HTTP 服务、所有功能的联合配置、原 Flash Prefill 完整模型正确性修复，或相对生产推理引擎的性能优势。当前完成的是可复现的推理系统复现、扩展及受控实验。
