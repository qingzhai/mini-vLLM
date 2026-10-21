# 正式分叉聚合与旧版 DFlash 对照

本分析只读已有 JSONL 和源码，没有启动 GPU。原性能记录、失败样本和阈值均保留。

## 十二条唯一请求，三轮分别核对

表内数字为首次不同的输出 Token 索引，从 0 开始；也是分叉前相同的 Token 数。`通过` 表示完整 256 Token 一致。每个单元格的三轮结果相同，不是只取第一轮。

五种方法指 EAGLE3、DFlash、DSpark No Markov、Markov Only、Fixed。它们不只首次分叉相同，三轮共 36 组的**完整输出也彼此完全相同**。

| 唯一请求 | 上述五种方法各自 | Confidence 0.5 | Confidence 0.8 |
|---|---:|---:|---:|
| gsm8k:456 | 通过 | 通过 | 通过 |
| gsm8k:816 | 通过 | 通过 | 通过 |
| gsm8k:466 | 通过 | 通过 | 通过 |
| gsm8k:1017 | 34 | 34 | 34 |
| humaneval:19 | 15 | 15 | 15 |
| humaneval:127 | 82 | 82 | 82 |
| humaneval:4 | 220 | 220 | 通过 |
| humaneval:79 | 44 | 44 | 44 |
| mt-bench:11 | 158 | 158 | 通过 |
| mt-bench:47 | 通过 | 通过 | 194 |
| mt-bench:71 | 通过 | 通过 | 通过 |
| mt-bench:9 | 154 | 106 | 106 |

| 方法 | 三轮均通过的唯一请求 | 通过执行次数 | 平均公共前缀 Token | 中位公共前缀 Token | 仅失败请求平均公共前缀 |
|---|---:|---:|---:|---:|---:|
| EAGLE3 / DFlash / No Markov / Markov Only / Fixed（各自） | 5/12 | 15/36 | 165.58 | 189 | 101.00 |
| Confidence 0.5 | 5/12 | 15/36 | 161.58 | 189 | 94.14 |
| Confidence 0.8 | 6/12 | 18/36 | 167.58 | 225 | 79.17 |

平均/中位公共前缀把通过请求记为 256；不能把它解释为逐 Token 准确率。各模式自身三轮输出稳定，三个 worker 的 AR 输出也完全相同。这使共同 Target 计算路径成为优先排查对象；不构成对数值误差或 KV 正确性的证明。

完整的每模式、每请求、每轮结果和 AR/投机 Token ID 见 [divergence-analysis.json](divergence-analysis.json)，按唯一请求聚合见 [by-unique-request.csv](by-unique-request.csv)。JSON 记录原始三份 JSONL 的 SHA256；没有修改原始正式报告。

## 为什么不能现在断言“全是 BF16 并列”

正式 JSONL **没有保存 Target Logits、Top-2 Margin、Hidden States 或 KV 张量**。从输出 Token ID 无法倒推出 Margin，分析中统一记为 `null / not_recorded_in_formal_jsonl`。

已证实的 BF16 Head 并列只属于 [旧 Smoke 的局部诊断](../diagnostic-20261019/README.md)：`humaneval:19` 的索引 26，5980/429 均为 31.625，真实 argmax 为 429；相同 Hidden State 的 FP32 Head 可以区分它们。但本轮已经使用 FP32 Head，首次分叉改为索引 15，AR=11、投机=3070，不能用索引 26 的证据替代新位置的 Logits。

由接受轨迹可以定位应观察的预测行。例如 DFlash 的 `humaneval:19`：第 3 轮，Pending 输出索引 11，验证前有效 Cache 长 125，Q=8；索引 15 对应 **Target 输出的第 3 行（从 0 开始）**，是该轮修正/额外 Token。不是每个分叉都发生在第一行，诊断若始终只观察第 0 行，会看错位置。

`verification_site` 是按已记录接受长度推导的索引，不是对真实 Cache 张量的测量。后续控制实验必须在相同已提交历史下比较 AR 与 Verification，并核对各层 KV 长度、位置、掩码和数值；不能把正常的分块计算差异直接认定为错误 KV，也不能跳过状态检查。

## 以前的 DFlash 确实通过，但实际实现不同

已验收的 [DFlash + Graph 汇总](../../dflash_graph/summary.json) 记录 24 次模式运行、48 次请求执行、12,288 个输出 Token 全部一致，另有 8 次前向精确检查。**AR Eager、AR Graph、DFlash Eager、DFlash Graph 都通过**，因此不能简单把通过归因于 CUDA Graph。

| 项目 | 旧 Mini-vLLM 实验 | 当前 DeepSpec 正式实验 |
|---|---|---|
| Target 主干 | FP16 | BF16 |
| Target Head | 真正 FP32 GEMM，固定 32 行 | 真正 FP32 GEMM，行数随输入变化 |
| Linear | 固定 32 行分块并补齐计算 | HF 原始可变形状 |
| RMSNorm / 激活 | 固定行数，RMS 32/1024，激活 32 | 没有这些固定形状适配 |
| Target Attention | 每个 Query 使用同一 PagedAttention；FP32 score/softmax/value reduction | HF Qwen3 SDPA，逐 Token AR 与多 Token 验证的形状不同 |
| Reduced-precision GEMM reduction | FP16/BF16 两个开关都关闭 | BF16 开启 |
| Draft | z-lab block16，15 个候选 | deepseek-ai block7，7 个候选 |
| 请求 | 固定 code/explain 提示，单请求和最多两个驻留请求 | 十二条 GSM8K/HumanEval/MT-Bench，batch=1 |

源码依据：

- [common/target.py](../../../experiments/common/target.py)：`install_arithmetic()` 固定 Linear、RMS、激活形状，替换 Target Attention，使用 FP32 固定行 Head；`Target.__init__()` 创建 FP16 参数和 KV。
- [fixed_shape.py](../../../experiments/common/fixed_shape.py)：`fixed_rows_forward()`、`fixed_linear()`，Padding 只参与算子计算，不进入逻辑 Token/KV。
- [dflash_graph/benchmark.py](../../../experiments/dflash_graph/benchmark.py)：`main()` 关闭 reduced reduction，四模式同驻留、同算术；入口还将 Draft SDPA 限定为 MATH。Target 自身使用上表的自定义 Attention。
- [新 benchmark.py](../../../experiments/speculative_compare/benchmark.py)：`worker()` 调用官方 BF16 `build_models()`，随后设置 FP32 Head；旧正式运行没有固定行或 per-query Target 替换。
- [precision.py](../../../experiments/speculative_compare/precision.py)：`configure_target_head()` 只提高 Head 精度，不能消除之前各层 Hidden States / KV 的差异。
- 官方 DeepSpec pin `005e03b81cec38b7da6399833d609ee89a2587f2` 的 `deepspec/eval/base_evaluator.py::verify_draft_tokens()`、`generate_decoding_sample()`：一次验证 Pending 加候选，连续前缀接受，`start += accepted+1` 后 `crop(start)`。实际源码随原完整 ZIP 归档；初步阅读未发现明显长度错一，但尚未对本轮七处分叉逐点验证状态。

旧运行时 `state.py::accept_prefix()` / `commit()` 与官方实现使用同一种 Pending / 连续前缀语义；最后输出预算截断的处理有所不同，但不能仅凭实现方式不同就归因于 Cache Bug。

## 目前最合理的判断与下一步

共同 Target 的逐 Token / 分块计算差异比“三种 Draft 同时独立损坏”更值得优先研究。旧版不止改了 Head，还对主干形状、Attention 和精度做过稳定性控制；新参考实现没有沿用它们。**这是一组可核实的差别，不是已经完成的因果隔离。**

2026-10-19 用户决定先切换 FP16 Target。新配置为 FP16 主干参数、FP32 Head、BF16 Draft；Target RoPE 等浮点 Buffer 保留原精度，进入 Draft 的特征在 `draft.fc` 前转 BF16，FP16 reduced reduction 关闭，BF16 reduction 保持原设置。它不是旧算术的完全移植，也不是只改变一个独立变量，因此即使通过，也不能声称已证明 BF16 是唯一根因。

先取 `humaneval:19`、`humaneval:4`、`mt-bench:9`，每条完整 256 输出、一轮，保留所有方法与两个置信阈值。覆盖早期、后期及模式相关分叉。不自动重跑三轮正式 Benchmark，不使用原 BF16 速度作为 FP16 速度。执行方式见 [FP16 操作说明](../../../experiments/speculative_compare/FP16.zh.md)。

旧 DFlash + Graph 的验收结论仍在其配置/负载内成立。本轮 AR 相对吞吐比仍是未通过一致性的观测值；No Markov → Markov Only 的内部同输出对照保留，但仍不能转述成相对 AR 的无损收益。

## 复算方式

在仓库根目录运行；目标目录须为新目录，避免覆盖已有分析：

```bash
python3 experiments/speculative_compare/analyze_divergence.py \
  --records-dir results/speculative_compare/formal-20261019 \
  --output-dir /tmp/minivllm-divergence-recomputed
```
