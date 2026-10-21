# Qwen3 推理引擎复现与优化：项目总结

项目时间：2026.10。基于 [Wenyueh/MinivLLM](https://github.com/Wenyueh/MinivLLM) 扩展和整理，使用官方 Qwen3、AWQ checkpoint，以及 EAGLE-3、DFlash、DSpark Draft；Draft 均非本项目训练。除原 DFlash＋Graph 自定义运行时外，新增 `LegacyTarget` 将 Mini-vLLM Qwen3-8B 接入官方 DeepSpec 循环，在统一 Target Eager 下完成方法对照与 DSpark 推理消融。所有性能数字来自已完成的 GPU 运行记录，本次整理没有重新执行 GPU 实验。

贡献边界：上游已有模型、算子、Paged KV Cache 与基础 Continuous Batching；本项目在此基础上扩展完整页 Prefix/LRU、Chunk 混合调度、自定义 DFlash 接受/回滚与 Target Graph，并实现多卡实验控制器和 AWQ 适配。新三方法对照保留官方 Draft 提议、验证和接受函数体，由本项目适配 Target 前向、特征、分页 KV 接口及实验协议，不是独立研发或训练新的投机算法。具体调用边界见 [LegacyTarget](../experiments/speculative_compare/legacy_target.py)、[benchmark.worker()/_worker()](../experiments/speculative_compare/benchmark.py) 与[来源声明](../THIRD_PARTY_NOTICES.md)。

## 1. 完成范围

| 模块 | 已完成和验证的内容 |
|---|---|
| Qwen 推理 | Qwen3-8B 为主要优化与评测对象；Qwen3-0.6B 用于早期基础推理及缓存/分块调度功能检查 |
| 分页与前缀缓存 | 分页 KV、链式块哈希匹配、完整页前缀复用与 LRU；早期 16 页压力场景发生 14 次驱逐，最后页全部回收 |
| 分块与连续批处理 | Chunked Prefill、Prefill/Decode 混合步骤、部分 Prefill 不采样、驻留上限和请求补位 |
| Attention | 使用 Triton PagedAttention；原 Flash Prefill 有独立小规模算子参考检查。正式跨调度正确性使用统一的 per-query PagedAttention 参考配置 |
| 投机解码 | 官方 BF16 DFlash Draft、批量 Target 验证、接受/拒绝、KV 回滚和请求补位 |
| 三种 Draft 对照 | 官方 DeepSpec EAGLE3/DFlash/DSpark 接原 Target 的串行适配、Markov/Confidence 推理消融，十二输入三轮均通过 Greedy 检查 |
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
| 三种 Draft 对照 | 原 Target FP16＋FP32 Head、BF16 Draft、固定算术与 per-query PagedAttention；Draft SDPA MATH，无 Graph | 十二条输入，每模式 3 轮，配对比值中位数 |

三种 Draft 对照使用独立 Python 3.11.12 / Transformers 5.10.2 环境。它只记录串行完整生成时间和接受轨迹，不提供 Serving TTFT/ITL，也不与旧 Graph 路径混算加速比。

TTFT 是从入队至首 token，E2E 是入队至完成，均包含引擎排队；TPOT 是首 token 后的平均生成间隔。ITL 是相邻输出 token 间隔，最大 ITL 和 P99 含义不同。总吞吐为整组输出 tokens/整组耗时；多请求总吞吐不是单个请求的生成速度。计时不包含模型加载、分词、预热、诊断或 Graph 捕获，不包含 HTTP 或网络服务开销。投机解码批量返回 token，其平均 TPOT 不能代替逐 token 可见间隔分布。

## 3. DFlash 与 CUDA Graph 四项对照

单 A100。single-code 为 1 请求，batch2-requests3 为 3 请求且驻留上限 2；每请求 greedy 生成 256 tokens、忽略 EOS。使用官方 z-lab b16 Draft，15 个候选＋Pending，验证 Q=16，支持多请求 Target 合批。使用固定提示词，不是 ShareGPT。各模式三轮，顺序旋转；所有模式驻留相同 Draft 和四张图，也统一采集 Target 的五层特征。

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

## 7. EAGLE3 / DFlash / DSpark 方法对照与消融

单 A100 40GB，`LegacyTarget` 将原 Mini-vLLM Qwen3-8B 适配到官方 DeepSpec 的提议、验证和接受循环；官方 EAGLE-3/DFlash/DSpark 每轮最多 7 个候选＋Pending，与第 3 节的 z-lab b16 路径不同。GSM8K / HumanEval / MT-Bench 各四条固定输入，共十二条，每条实际生成 256 Token、忽略 EOS、三轮。平均输入 120.33 Token，Batch=1，Target Eager，无 Graph。相同 Draft 在其 AR 基线中保持驻留；固定形状补行和逐 Query Attention 有计算成本，此 AR 是受控参考配置，不是生产最优 AR。

| 方法 | 完整生成 tokens/s 中位数 | 相对配对 AR Eager | 逐 Token 一致 |
|---|---:|---:|---:|
| EAGLE3 | 47.260 | 2.934× | 36/36 |
| DFlash | 56.407 | 3.479× | 36/36 |
| DSpark No Markov | 34.283 | 2.111× | 36/36 |
| DSpark Markov Only | 63.180 | 3.898× | 36/36 |
| DSpark Fixed | 63.225 | 3.894× | 36/36 |
| DSpark Confidence 0.5 | 61.839 | 3.823× | 36/36 |
| DSpark Confidence 0.8 | 51.741 | 3.193× | 36/36 |

Fixed 吞吐中位数 63.225 tokens/s 略高于 Markov Only 的 63.180，Markov Only 的配对加速比中位数 3.898× 则略高于 Fixed 的 3.894×。吞吐取各模式三轮中位数，加速比取同轮配对比值中位数，不能互换排序或把微小差异当成稳定优劣。

速度包含 Prefill 与完整生成，不是纯 Decode 或 ShareGPT 并发吞吐。不同 Draft 训练和容量不匹配，不宣称严格结构消融。

同一 DSpark checkpoint 推理时开启 Markov 的三轮配对吞吐比为 **1.846838×（+84.68%）**。Markov Only 与 Fixed 的输出、提议、接受轨迹完全相同；Fixed 额外计算 confidence 但不截断，本负载速度没有明显差别。Confidence 0.5/0.8 相对 Fixed 的配对吞吐分别 **−1.97% / −18.16%**：0.8 的已送验候选接受率升到 83.21%，实际输出/Verification 却从 4.826 降到 3.873，更多验证轮抵消了截短窗口的收益。

Confidence Head 独立预测分数，经 sigmoid 和固定阈值选择连续送验前缀，不是 Token Softmax 概率。当前只测 Batch=1 的固定阈值；没有队列负载、动态预算或吞吐成本模型，不能据此否定高并发下的负载感知 Confidence Scheduling。

### Target 数值一致性与 Cache 适配证据

旧 HF 路径中，AR 与多 Token 验证曾在相同输出历史下分叉。`humaneval:19` 的一个现场中，BF16 LM Head 对 Token 5980/429 均输出 31.625；相同 Hidden 的真实 FP32 Head 分数为 31.637157/31.575722。按原提议、验证、crop 历史重放，36 层 K/V 可逐元素重建现场；AR 单步 Cache 则不同。这支持该次分叉涉及分块数值历史与 Head 舍入，不能证明所有缓存或算子都没有问题。[原始控制实验](../results/speculative_compare/diagnostic-20261019/README.md)

只替换 FP32 Head 仍分叉；HF FP16 短回归也只通过 20/21。最终组合使用 FP16 Target、真实 FP32 Head、BF16 Draft，固定关键 Linear/RMS/激活计算行数、统一逐 Query PagedAttention，并控制归约和 Draft MATH 后端。`LegacyCache` 将官方 Target Cache 的长度与裁剪映射到 `PagePool`，每 Query 因果长度屏蔽无效尾部；Draft DynamicCache 独立，五层 Target 特征在 `draft.fc` 入口转 BF16，请求结束 finally 归还 Target 页。实现见 [install_arithmetic()](../experiments/common/target.py)、[LegacyCache.crop()/LegacyTarget.forward()](../experiments/speculative_compare/legacy_target.py)、[install_draft_feature_bridge()](../experiments/speculative_compare/precision.py)。

该组合先通过三请求 21/21 短回归，再完成十二输入三轮正式测量：七种投机配置各 **36/36**，总计 **252/252**；连同配对 AR 共 **360** 次请求、**92,160** 输出 Token，无运行错误、Target 页全部回收。它解决了受测负载中的一致性问题，**尚未通过单变量实验确定历史所有分叉的唯一根因**，也不是对任意输入与旧 HF AR 等价的证明。旧失败与诊断记录不改写。

没有 EAGLE3 树验证、普通 Serving 接入、随机采样分布检验或 GSM8K/HumanEval 任务评分。完整条件、三轮范围、接受率、分任务数据及来源见 [正式结果](../results/speculative_compare/legacy-target-formal-20261021/README.md)。

## 8. 证据、归档与结项边界

`results/` 发布条件、指标及源记录 SHA256。每个目录的 `SOURCES.json` 记录原始 ZIP 文件名、ZIP SHA256 和所引用记录的 SHA256。ShareGPT 等历史系统实验以精选摘要为主；新增三方法对照保留公开任务的输入 Token ID、逐请求输出与接受轨迹 JSONL，便于离线复算首次分叉和消融指标。这些记录包含当时的通用工作目录路径，不含连接凭据。完整 ZIP、终端长日志和硬件清单由项目持有者本地保留；模型权重、虚拟环境和缓存不提交到代码仓库。

旧 DFlash 的诊断单次结果与早期 TP 实验不再作为主要性能结论：本总结优先使用四项 Graph 对照与统一双卡对照。不同实验的精度、负载和测量次数有区别，不把其数值横向合并。

未宣称完成其他模型适配、PP、Draft Graph、异步 KV handoff、生产 HTTP 服务、所有功能的联合配置、原 Flash Prefill 完整模型正确性修复，或相对生产推理引擎的性能优势。当前完成的是可复现的推理系统复现、扩展及受控实验。
