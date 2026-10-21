# EAGLE3 / DFlash / DSpark：原 Mini-vLLM Target 正式对照

**本轮正式测量与完整 Greedy 一致性均通过。** 共 360 次请求执行、92,160 个实际输出 Token；七种投机模式各 36/36 与同轮同进程 AR 完整一致，跨进程与跨轮 AR 也一致。所有请求均记录 Target 页回收；运行、精度及 AR 分叉错误列表均为空。数据是十二条唯一请求、各三次重复，不能称为 36 条独立 Prompt。

原包 `speculative-legacy-target-formal-records.zip` 的 SHA256 为 `ccf68dcc6969494e759f5d41c8bfc15fde6ff4ea529b083d753f09681c8c935b`。原始文件哈希见 [SOURCES.json](SOURCES.json)，[report.json](report.json)、[summary.csv](summary.csv) 及三个 JSONL 均保持返回内容。另有只读离线生成的 [消融汇总](ablation/ablation-analysis.json)、[分任务表](ablation/by_task.csv)、[接受轨迹表](ablation/trajectories.csv) 和 [逐请求分叉检查](divergence/divergence-analysis.json)。本次没有重跑 GPU。

## 条件和口径

- GPU 为单张 A100 PCIe 40GB；Python 3.11.12、PyTorch 2.9.1+cu128、CUDA 12.8、Triton 3.5.1、Transformers 5.10.2，完整环境随原记录保存。
- Target 为原 Mini-vLLM Qwen3-8B：FP16 主干、实际 FP32 Head、固定形状算术、全阶段逐 Query PagedAttention。关闭 TF32 及 FP16/BF16 reduced-precision reduction；无 CUDA Graph。
- 三个 Draft 是官方 BF16 checkpoint，使用 SDPA MATH、Target 第 1/9/17/25/33 层特征，每轮最多七个候选。来源与 revision 见 [lock.json](lock.json)；Target 历史 revision 未知，使用文件 SHA256 固定内容。官方 Draft 没有由本项目训练。
- 原 GSM8K / HumanEval / MT-Bench 各四条输入，分词后共十二条；平均 120.33、最短 35、最长 341 个输入 Token。每条 temperature=0，实际生成 256 Token、忽略 EOS；预热每模式 32 Token，测量三轮，配对模式轮换顺序。
- Batch=1，串行执行。不同 Draft 分进程加载，各自有重新运行的 AR，AR 时同一 Draft 仍驻留。三种 Draft 的训练和容量不同，这是方法对照；DSpark 内部开关才是同一 checkpoint 的推理策略消融。
- 时间包括 Prefill、Draft/Target、采样、接受与 Cache 更新、必要的主机同步；不含加载、分词、预热或磁盘写入。本轮关闭 Top-2 采集，完整 Logits 仍用于验证。**不是纯 Decode tokens/s，也不是 ShareGPT 并发 Serving 吞吐。**

## 主要性能

每轮每模式输出 `12 × 256 = 3072` Token，吞吐为 3072 / 十二请求生成耗时之和，再取三轮中位数。加速比是三轮同进程配对比值的中位数，可能与两个中位数直接相除略有差异。

| 模式 | 配对 AR tokens/s | 本模式 tokens/s | 十二请求累计耗时 s | 配对加速比 | 完整一致 |
|---|---:|---:|---:|---:|---:|
| EAGLE3 | 16.172 | 47.260 | 65.002 | 2.934× | 36/36 |
| DFlash | 16.168 | 56.407 | 54.461 | 3.479× | 36/36 |
| DSpark No Markov | 16.235 | 34.283 | 89.608 | 2.111× | 36/36 |
| DSpark Markov Only | 16.235 | 63.180 | 48.623 | 3.898× | 36/36 |
| DSpark Fixed | 16.235 | 63.225 | 48.588 | 3.894× | 36/36 |
| DSpark Confidence 0.5 | 16.235 | 61.839 | 49.677 | 3.823× | 36/36 |
| DSpark Confidence 0.8 | 16.235 | 51.741 | 59.372 | 3.193× | 36/36 |

三个进程的 AR 十二请求累计耗时分别为 189.961、190.006、189.220 秒。三轮吞吐范围：EAGLE3 47.231～47.629、DFlash 56.248～56.646、Markov Only 62.989～63.504 tokens/s。这些是同一小样本集合的运行波动，不是总体泛化的置信区间。

本轮全部计时累计 **2953.297 秒，约 49.22 分钟**，不含加载、预热和核验。

## DSpark 消融说明

| 变化 | 三轮配对吞吐比中位数 | 吞吐变化 | 结论 |
|---|---:|---:|---|
| No Markov → Markov Only | 1.846838× | +84.68% | 本负载下启用已训练 Markov 修正收益明确 |
| Markov Only → Fixed | 1.000643× | +0.06% | 未观察到明显速度差别；不能据微小差值宣称收益 |
| Fixed → Confidence 0.5 | 0.980311× | −1.97% | 总体无收益 |
| Fixed → Confidence 0.8 | 0.818370× | −18.16% | 过早截断降低整体推进效率 |

Markov Only 关闭 confidence head，Fixed 保留 confidence head 计算但阈值为零、不截断候选；两者**不是相同计算实现**。实际 36 组输出、提议长度、接受长度和验证次数完全相同，只有额外 confidence 计算与计时的差别。所有模式保留相同权重驻留，不通过移除权重降低显存。

No Markov 与 Markov Only 使用相同 DSpark checkpoint，在推理时切换 Markov；关闭 Markov 时也关闭依赖它的 confidence head。它不能被称为“重新训练的 DFlash”，也不能推断移除 Markov 后重新训练的结果。

### 接受率更高，为什么反而慢

| 模式 | 平均送验候选数 | 平均接受 Draft 数 | 已送验候选接受率 | 实际输出/Verification | 三轮验证总次数 |
|---|---:|---:|---:|---:|---:|
| EAGLE3 | 7.000 | 2.931 | 41.87% | 3.859 | 2379 |
| DFlash | 7.000 | 3.320 | 47.43% | 4.256 | 2157 |
| DSpark No Markov | 7.000 | 1.560 | 22.29% | 2.539 | 3615 |
| DSpark Markov Only / Fixed | 7.000 | 3.913 | 55.90% | 4.826 | 1902 |
| DSpark Confidence 0.5 | 5.711 | 3.807 | 66.66% | 4.722 | 1944 |
| DSpark Confidence 0.8 | 3.520 | 2.929 | 83.21% | 3.873 | 2370 |

Confidence 0.8 把送验候选从平均 7 个降到 3.52 个，筛掉低置信候选使已送验集合的接受率上升，但平均每次验证实际交付从 4.826 降为 3.873 Token，完成相同输出需要更多轮。源码先执行完整 Draft Block、采样和 confidence 预测，再截断送验前缀，不能把候选长度变短等同于少执行同样比例的 Draft 计算。

源码依据：官方 `deepspec/eval/dspark/evaluator.py::_propose()`、`draft_ops.py::forward_dspark_draft_block()` / `build_dspark_proposal()` / `_confident_prefix_length()`；Markov 通过 `deepspec/modeling/dspark/qwen3/modeling.py::sample_draft_tokens()` 进入轻量顺序修正。原源码在 ZIP 的 `deepspec-source/` 下。没有采集 Draft/Target 分段 CUDA 时间，因而不能为其开销占比或唯一瓶颈给出数字。

`平均接受 Draft 数+1` 是计算推进长度；表中的实际输出/Verification 则用实际交付的输出减去初始首 Token，再除验证次数。最后一轮可能超出剩余预算并被截断，两者不能混为同一指标。

## 数据集差异

| 方法 | GSM8K tokens/s / 配对加速 | HumanEval tokens/s / 配对加速 | MT-Bench tokens/s / 配对加速 |
|---|---|---|---|
| EAGLE3 | 66.929 / 4.148× | 54.466 / 3.402× | 33.293 / 2.049× |
| DFlash | 77.096 / 4.778× | 62.344 / 3.863× | 41.299 / 2.531× |
| DSpark Markov Only | 80.877 / 4.968× | 71.442 / 4.424× | 47.729 / 2.908× |

每个子表仅四条唯一 Prompt，不代表完整任务集能力；没有进行 GSM8K 答案评分或 HumanEval 单元测试。整体配置下 confidence 截断无收益，也不能推导所有任务、阈值和并发负载下一律无收益。

## 显存、数值与项目边界

- EAGLE3 / DFlash / DSpark 投机峰值 allocated 分别约 **23.180 / 24.570 / 24.715 GiB**，三个配对 AR 也保留相应 Draft。不能据此得出投机相对独立纯 AR 几乎没有额外显存。
- Target 原 FP16 参数约 15.256 GiB，另有 FP32 Head 副本 **2,489,319,424 bytes**，KV、激活和 Draft 另外占用；这是模型参数与运行峰值的不同口径。没有 AWQ 量化或多卡介入本轮。
- 历史 HF BF16 / FP32 Head 与 HF FP16 的失败结果保留。本轮是复用原 Mini-vLLM Target 的新适配路径，改动包含 Attention、固定形状、归约和 Draft SDPA 后端；不能单独宣布“BF16 是所有分叉根因”。没有与原 HF AR 对齐输出作为本轮标准，也没有完成失败位置的全部 KV/数值因果对照。
- 这里是普通 AR **Eager** 基线；不能把 3.48× 写成相对 AR Graph、vLLM、原项目全部负载的普遍加速，也不能与旧 DFlash＋Graph 的加速比相乘。
- 当前三个 Draft 接的是串行 Target 适配器，尚未接入普通引擎并发 Serving 或 CUDA Graph 路径。没有验证长上下文、更多到达模式、随机采样分布、EAGLE3 树验证或完整线上调度器。

## 可复述的项目结论

“基于官方 DeepSpec Draft，在 Mini-vLLM 的稳定 Target 执行路径上完成 EAGLE3、DFlash、DSpark 方法对照及 Markov/Confidence 推理消融。单 A100 40GB、十二条固定输入、三轮测量中，相对配对 AR Eager 的吞吐提升约 2.93×、3.48×、3.90×，全部 252 组投机输出逐 Token 一致；Markov 修正带来约 84.68% 吞吐收益，而 confidence 截断虽提高接受率，整体吞吐反而下降。”

适合描述受控推理实验和适配工作，不应包装为自行训练算法、生产级 Serving 加速或论文全任务复现。

## 离线复算

```bash
python experiments/speculative_compare/analyze_ablation.py \
  --records-dir results/speculative_compare/legacy-target-formal-20261021 \
  --output-dir /tmp/minivllm-formal-ablation-new
```

输出目录中的目标文件必须尚不存在，避免覆盖原派生结果。此脚本不加载模型，不访问网络，不执行 GPU。
