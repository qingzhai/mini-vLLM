# 原 Mini-vLLM Target 适配回归：21/21 组投机输出一致

原始包 `speculative-legacy-target-records.zip`，SHA256：`6ab675e1b650d836ccac5d120d5bfbe14da3f27685456c8cdf84d34ed4a7c9c8`。提取文件的逐项哈希见 [SOURCES.json](SOURCES.json)，原 JSON/JSONL、环境、日志均保留原内容；[分叉汇总](divergence/divergence-analysis.json) 为离线派生分析。

单张 A100 PCIe 40GB，Qwen3-8B 原 Mini-vLLM Target：FP16 主干、实际 FP32 Head、固定形状算术、全阶段逐 Query PagedAttention；关闭 TF32 及 FP16/BF16 reduced-precision reduction。三个官方 DeepSpec Draft 保持 BF16，SDPA 使用 MATH。无 CUDA Graph，并发 Batch=1。

| 投机模式 | 请求数 | 完整输出与配对 AR 一致 | 公共输出前缀 |
|---|---:|---:|---:|
| EAGLE3 | 3 | 3/3 | 每条 256 Token |
| DFlash | 3 | 3/3 | 每条 256 Token |
| DSpark No Markov | 3 | 3/3 | 每条 256 Token |
| DSpark Markov Only | 3 | 3/3 | 每条 256 Token |
| DSpark Fixed | 3 | 3/3 | 每条 256 Token |
| DSpark Confidence 0.5 | 3 | 3/3 | 每条 256 Token |
| DSpark Confidence 0.8 | 3 | 3/3 | 每条 256 Token |

输入为原正式工作负载中的 `humaneval:19`（114 Token）、`humaneval:4`（145 Token）、`mt-bench:9`（53 Token），每条实际生成 256 Token，temperature=0、忽略 EOS、一轮。共完成 30 次请求执行：21 次投机、9 次配对 AR。跨进程 AR 输出一致，所有请求均记录 Target 页回收；`errors=[]`、`precision_errors=[]`、`ar_divergences=[]`，`qualified_greedy_comparison=true`。

已逐字节比较返回包与本地的 `benchmark.py`、`protocol.py`、`precision.py`、`legacy_target.py`、`legacy-source-manifest.json`，均相同。三个 worker 的原 Target 清单指纹一致：`b5606a50aa4524d2a1e7438322272e138af076c2ee4eccb3758a6b958b1b68d2`，锁定 31 个文件；算子模块数为 Linear 144、RMS 145、Activation 36、Attention 36。

## 这个结果支持什么

前一轮 HF FP16 路径仅 Confidence 0.8 的 `mt-bench:9` 在索引 24 分叉，本轮三个请求全部一致，支持这套原 Target 适配计算配置在短回归上有效。

这里同时改变了 Target Attention/固定形状算术及归约、Draft SDPA 后端，不能单独归因于 BF16 或某个算子；原 HF FP16 分叉没有保存 Top-2 或 KV 内容，因而未完成该位置数值与 Cache 差异的因果定位。也没有证明与历史 HF AR 的输出相同；验收基线是本轮相同 Target 配置下重新运行的 AR。

## 性能和下一步

本轮开启 Target Top-2 采集和 CPU 导出，三请求累计的所有十种模式生成计时约 **269.42 s**，不含加载、预热与核验。原始速度仍保留在 [summary.csv](summary.csv)，但不把诊断配置的速度作为正式结论，也不与原 Mini-vLLM Graph 结果混算。

后续沿用本轮计算源码、Target/Draft 权重、精度和接受算法，关闭 Top-2 收集，恢复原 GSM8K / HumanEval / MT-Bench 各四条输入，共十二条、三轮。[正式结果](../legacy-target-formal-20261021/README.md) 已返回，七种模式均 36/36 一致。两轮证据分别保存；均不是 ShareGPT Serving 压测。
