# EAGLE3 / DFlash / DSpark 三轮正式测量

**测量完整完成，尚未通过相对 AR 的 Greedy 一致性验收。** 下表的相对 AR 倍数是观测吞吐比，不是已验证的无损加速。原始分叉样本没有删除，各模式没有提前跳过。

## 配置与统计口径

- 单 A100 PCIe 40GB，Python 3.11、PyTorch 2.9.1+cu128、Transformers 5.10.2，DeepSpec commit `005e03b81cec38b7da6399833d609ee89a2587f2`。
- Qwen3-8B Target：BF16 主干＋FP32 Head；三个 Draft 保持 BF16。TF32 关闭，BF16 reduced-precision reduction 开启。原 checkpoint 不修改，所有 AR/投机模式共用同一精度控制。
- GSM8K、HumanEval、MT-Bench 各 4 条，共 12 条；输入 35～341 Token，平均 120.33。不是 ShareGPT，也不代表长上下文分布。
- batch=1，逐请求执行；每条实际输出 256 Token，temperature=0，忽略 EOS，不使用 PAD 补齐。各模式三轮，轮内配对顺序轮换；无 CUDA Graph / compile。
- 每轮每模式 12×256=3072 输出 Token；共 360 次请求执行，92,160 个输出 Token。生成计时合计 2271.28 s，约 37.85 分钟，不含模型加载、准备、分词和预热。
- 时间与 tokens/s 为三轮中位数；倍数为每轮与各自驻留 Draft 的 AR 配对吞吐比的中位数。它不一定等于两个中位数直接相除。
- `output_tokens_per_s` 包含 Prefill、Draft 初始化和完整生成，**不是纯 Decode Tokens/s**，也不是并发 Serving 吞吐。没有测 TTFT、ITL 或任务准确率。

原始 [report.json](report.json)、[summary.csv](summary.csv)、[逐请求 DSpark 记录](dspark-records.jsonl)、[分析指标](analysis.json) 与 [归档指纹](SOURCES.json) 随目录保存。完整 ZIP 在本地另行保留，含实际源码和官方源码快照。

## 总体对照

| 方法 | AR 耗时 s | 实验耗时 s | 实验 tokens/s | 相对配对 AR 观测比 | 与 AR 全序列相同 | 峰值 allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| EAGLE3 | 139.38 | 54.81 | 56.05 | 2.545× | 15/36 | 19.53 |
| DFlash | 139.33 | 44.49 | 69.05 | 3.129× | 15/36 | 20.94 |
| DSpark No Markov | 137.23 | 75.18 | 40.86 | 1.830× | 15/36 | 21.08 |
| DSpark Markov Only | 137.23 | 39.85 | 77.09 | 3.448× | 15/36 | 21.09 |
| DSpark Fixed | 137.23 | 39.84 | 77.12 | 3.445× | 15/36 | 21.09 |
| DSpark Confidence 0.5 | 137.23 | 40.02 | 76.75 | 3.433× | 15/36 | 21.09 |
| DSpark Confidence 0.8 | 137.23 | 47.97 | 64.04 | 2.864× | 18/36 | 21.07 |

时间是每轮十二条请求的完整生成耗时之和，不是单请求耗时。36 次是十二条输入×三轮；不是 36 条不同 Prompt。AR 吞吐分别为 EAGLE3 进程 22.04、DFlash 进程 22.05、DSpark 进程 22.39 tokens/s。峰值是各模式三轮中的最大 allocated 值，不是纯权重大小。

`completed=true`，各模式完整，`errors=[]`、`precision_errors=[]`、`ar_divergences=[]`；但 `qualified_greedy_comparison=false`。三个进程的 AR 以及每个模式自身的输出均跨轮稳定。

## DSpark 内部消融：哪些可以独立解释

| 对照（右者相对左者） | 三轮配对吞吐比中位数 | 配对完整输出一致 | 配对提议/接受轨迹一致 |
|---|---:|---:|---:|
| No Markov → Markov Only | **1.886×（+88.58%）** | **36/36** | 0/36，候选质量与轮数改变 |
| Markov Only → Fixed | 1.00019×，近乎相同 | **36/36** | **36/36** |
| Fixed → Confidence 0.5 | 0.99532×（−0.47%） | 30/36 | 6/36 |
| Fixed → Confidence 0.8 | 0.83138×（−16.86%） | 24/36 | 0/36 |

**Markov 的同输出消融有独立证据。** 同一个 DSpark checkpoint、相同常驻权重、输入、输出预算及 Target，推理时打开 Markov，全部 36 组输出与 No Markov 完全一致；时延合计的中位数从 75.18 s 降为 39.85 s。这个结论是本次内部对照，不代表它已与 AR 完全一致，也不能解释为重新训练两种结构的严格消融。

**Fixed 没有改变候选策略。** 它只比 Markov Only 多计算 confidence head，阈值 0 不截断。输出和全部提议/接受轨迹都相同。三轮差异有正有负，没有证据将如此小的时间差解释为显著加速或明确开销。

**两个置信阈值没有带来总吞吐收益。** 0.5 略慢，0.8 明显慢，且它们与 Fixed 也不是所有输出都相同。不能将这些速度差全部视为同输出条件下的纯调度因果效应，更不能因为某阈值有更多请求碰巧与 AR 对齐，就推断任务准确率更高。

## 接受率高，为什么仍可能更慢

以下接受率、长度按三轮全部 Verification 汇总；候选长度指实际送入 Target 的数量，不是 Draft 完整计算的 7 位置。有效推进排除 Prefill 首 Token，并按最终 256 Token 预算截断。

| 方法 | 平均验证候选数 | 平均接受 Draft 数 | 接受率 | 有效输出 Token/Verification | Verification 总数 |
|---|---:|---:|---:|---:|---:|
| EAGLE3 | 7.00 | 2.96 | 42.24% | 3.89 | 2361 |
| DFlash | 7.00 | 3.45 | 49.24% | 4.37 | 2100 |
| DSpark No Markov | 7.00 | 1.58 | 22.50% | 2.55 | 3600 |
| DSpark Markov Only / Fixed | 7.00 | 3.99 | 56.94% | 4.87 | 1884 |
| DSpark Confidence 0.5 | 5.74 | 3.91 | 68.21% | 4.83 | 1902 |
| DSpark Confidence 0.8 | 3.52 | 2.94 | 83.50% | 3.89 | 2358 |

0.8 丢掉较难接受的候选后，接受率达到 83.50%，但每次实际推进从 4.87 降为 3.89，需要更多轮验证；Draft backbone 也仍按完整 block7 执行。因此“接受率更高”不等于“每秒输出更多”。本轮没有分离 Draft/Target Kernel 时间，不能进一步声称某项 CPU 或通信开销是唯一瓶颈。单卡参考路径也没有跨 GPU 模型通信。

作为粗略关系，生成时间由 Prefill 加每轮 Draft、Target 验证、接受/crop 等开销构成；候选截短同时改变每轮成本和总轮数，必须看完整生成时间，不能只看接受比例。

## 方法间比较与机制

EAGLE3 是官方单链自回归 Draft；DFlash 使用 Masked 并行候选；DSpark 在并行 backbone 上加轻量顺序 Markov 修正及 confidence 预测。它们都由同一个 Target 验证，候选数量不等于接受数量。当前 confidence 是固定阈值截断，没有实现论文的完整在线负载感知调度器。

EAGLE3、DFlash、DSpark No Markov、Markov Only、Fixed 在本次 **36 组配对的完整输出彼此完全相同**。DFlash 对 EAGLE3 的三轮配对吞吐比中位数为 1.226×；Markov Only 对 DFlash 为 1.120×。这些是本轮匹配输出的参考路径方法对照，三个 checkpoint 的容量、训练不同，不能当成严格结构消融，也不能推广到 Mini-vLLM 并发服务。

## 一致性失败的范围

上述五种不截短模式相对 AR 的七个首次分叉，在三轮中重复出现：

| 输入 | 首次不同的输出 Token 索引（从 0 开始） |
|---|---:|
| `humaneval:19` | 15 |
| `gsm8k:1017` | 34 |
| `humaneval:79` | 44 |
| `humaneval:127` | 82 |
| `mt-bench:9` | 154 |
| `mt-bench:11` | 158 |
| `humaneval:4` | 220 |

Confidence 0.5 将 `mt-bench:9` 的首次分叉提前到 106；0.8 在 `mt-bench:9` 也是 106，并在 `mt-bench:47` 的 194 分叉，其余失败范围详见原始 report。

独立方法出现相同输出及首次分叉、各自跨轮稳定，使共同 Target 计算路径成为优先研究对象；这不是对所有分叉底层原因的证明。前一轮 BF16 Head 索引 26 的局部诊断，不能直接解释这轮 FP32 Head 下的七个新分叉。没有逐点排除状态/掩码问题，也不能把完整序列不一致简单等同于任务质量下降，因为本轮没有任务答案评分。

## 如何用于项目和简历

可以陈述：完成 DeepSpec 三种投机方法及 DSpark 推理策略的可复现对照，统一 Target/输入/预算/精度，保存三轮原始输出与失败轨迹；同 checkpoint 的 No Markov/Markov Only 在 36 组相互一致的输出下，观测到约 1.89× 内部吞吐提升；confidence 截断在该负载未获收益。

不能陈述：已实现相对 AR 无损 3.45× 加速、复现论文全部调度收益、提升 HumanEval 准确率，或把 batch=1 参考结果当成 Mini-vLLM Serving 吞吐。原项目已有、另行验收通过的 DFlash＋CUDA Graph 结果不受本轮失败影响，也不能与这些倍数相乘。

本轮按用户要求完整测量并保留失败，不再自动追加 GPU 排查或重测。若后续修数值/计算路径，需要新的配置、同输出验收和新的测量，不能改写本次归档为通过。
