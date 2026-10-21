# 2026-10-19 Smoke：已执行，有未解决分叉

来源 `deepspec-audit-20261019-080548Z.zip`，内容指纹见 [SOURCES.json](SOURCES.json)。官方 DeepSpec 的 92 个源码文件与 lock 一致，commit `005e03b81cec38b7da6399833d609ee89a2587f2`，服务器 Git 工作树干净。归档中的 `benchmark.py`、`protocol.py` 与本项目当时版本一致。

GSM8K `gsm8k:456`、HumanEval `humaneval:19`、MT-Bench `mt-bench:11` 的输入分别为 100、114、44 Token，平均 **86 Token**，不能代表历史 ShareGPT 的输入长度分布。每条输出 32 Token，忽略 EOS，单轮。以下是三个输入合计 96 个输出 Token / 总生成时间，包含 Prefill。

| 模式 | 生成耗时合计 s | 完整生成 tokens/s | 相对同进程 AR | 三条输出与 AR 一致 |
|---|---:|---:|---:|---|
| EAGLE3 | 2.548 | 37.68 | 1.603× | 是 |
| DFlash | 1.747 | 54.94 | 2.409× | 是 |
| DSpark No Markov | 3.330 | 28.83 | 1.246× | 是 |
| DSpark Markov Only | 1.651 | 58.14 | 2.514× | 是 |
| DSpark Fixed | 1.652 | 58.11 | 2.512× | 是 |
| DSpark Confidence 0.5 | 1.687 | 56.92 | 2.461× | 是 |
| DSpark Confidence 0.8 | 2.013 | 47.70 | 2.062×（未合格） | **否** |

三个独立 AR 的耗时合计分别为 4.083、4.210、4.150 s；不是把三种驻留 Draft 的基线合并为一个。原始数值见 [summary.csv](summary.csv)。

当前只能观察到：这三个短输入中 Markov Only 比 No Markov 约快 2.017×；Fixed 与 Markov Only 的三条输出及提议/接受轨迹完全一致，额外耗时约 0.001 s，单次测量不能证明如此小的差异显著。Confidence 0.5 对 Fixed 的吞吐约低 2.05%，不能包装为已观察到调度正收益。所有这些均为 Smoke 观察，不能推广到正式负载。

Confidence 0.8 的 `humaneval:19` 在输出索引 26 分叉：AR 为 5980，实验为 429。`report.json` 的 `qualified_greedy_comparison=false`，`errors=[]`、`ar_divergences=[]`。其加速比保留为失败现场，不能作为正确性通过的性能结论。

旧 [diagnostic-console.txt](diagnostic-console.txt) 第 5 轮记录两个 Logits 同为 31.625，却写 `argmax:5980`；实际输出是 429。归档未含生成该字段的临时诊断脚本。Top-k 在并列时的首项不能替代真正 argmax，该日志不足以单独证明真实采样选择或底层数值差异的来源。新增诊断将分别记录真实 argmax、并列集合、官方 one-hot 概率 argmax，并隔离 Cache 历史与 Query 形状。

模型不可变 revision：EAGLE3 `f6485ba8d21e11942958617dbe7e71b467f38f38`，DFlash `9e44dbbb6cb68b0c943abf9c5fc3c17c00897cdf`，DSpark `03326e5043815da1f81b109078b2889737c26017`。Target 是本地文件，没有历史 revision，内容 SHA256 在 `lock.json`；不能自行补一个 Target commit。

这是 DeepSpec 参考路径，尚未接入 Mini-vLLM；也没有使用旧 DFlash＋Graph 的 b16 Draft、FP16 Target 或 CUDA Graph。不能与旧实验的加速比相乘。
