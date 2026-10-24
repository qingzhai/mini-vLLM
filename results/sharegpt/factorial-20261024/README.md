# ShareGPT：Chunk × Prefix 四组正式消融（2026-10-24）

本轮完成四组各三轮 GPU 测量，并从原始 ZIP 离线复算输入身份、逐请求时间、吞吐、执行配置和完整输出一致性。原始 ZIP 保持原样；[AUDIT.json](AUDIT.json) 是本地审查结果，不是重新跑 GPU。

## 配置与比较对象

单 A100-PCIE-40GB；Qwen3-8B BF16 主干＋FP32 LM Head；Torch 2.9.1+cu128、Transformers 4.57.3、Triton 3.5.1、Python 3.11.12。所有阶段使用固定算术与 per-query PagedAttention，纯 Decode 启用 CUDA Graph，没有使用原 Flash Prefill。

固定 ShareGPT revision `854a6eb3c310f07f9e8b269a735bf6fe547e543d`；100 条请求、80 个不同 Prompt＋20 次完整 Prompt 重放、50 短/50 长，平均输入 788.89 Token。请求同时入队，驻留上限 8、每步 Token Budget 2048、330 页×256 Token，每请求 greedy 实际生成 64 Token，忽略 EOS。

四组统一 Decode-first，仅改变 Chunk/Prefix 两个开关。关闭 Chunk 时，未缓存 Prompt 后缀必须一次算完，预算不足就推迟，不隐式切块；因此 Full/Prefix-only 仍可和 Decode 混合，同样有 49 个 mixed steps。每次 trial 重建冷前缀缓存，四组各用 16 条真实输入预热；三轮旋转模式顺序，但未实现四个位置完全平衡。协议和入口见 [操作说明](../../../docs/SHAREGPT_FACTORIAL.zh.md)。

## 三轮指标中位数

各列单独取中位数，不代表同一次运行；TTFT/E2E 含排队，不含 HTTP、网络、分词、加载、预热和 Graph 诊断。每 trial 输出 6400 Token。

| 模式 | 总耗时 s | 平均 TTFT s | 平均 TPOT ms | 平均 E2E s | 输出 Token/s | 吞吐范围 Token/s |
|---|---:|---:|---:|---:|---:|---|
| Full Prefill | 146.063 | 72.652 | 137.668 | 81.358 | 43.817 | 43.730～48.810 |
| Chunk-only | 146.871 | 72.339 | 130.844 | 80.582 | 43.576 | 39.465～44.046 |
| Prefix-only | 119.207 | 66.258 | 112.804 | 73.364 | 53.688 | 50.819～56.369 |
| Chunk+Prefix | 137.117 | 76.874 | 121.739 | 84.544 | 46.675 | 46.360～51.363 |

Prefix 两组均命中 13,568 输入 Token，占 78,889 的 17.20%；20% 是重放请求比例。Chunk-only/Chunk+Prefix 每 trial 分别有 267/214 个部分 Prefill 块、188/158 个混合步骤。峰值 allocated 约 29.56 GiB。

## 同轮配对吞吐比

先算每轮候选吞吐÷同轮基线吞吐，再对三个比值取中位数。这与上表“吞吐中位数之比”不同；例如 Prefix-only/Full 的后者是 1.2253，不能写成配对收益 22.53%。轮内各模式也先后执行，配对并不能消除所有时间漂移。

| 对照 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 配对比值中位数 | 吞吐变化 |
|---|---:|---:|---:|---:|---:|
| Full Prefill → Prefix-only | 1.1549 | 1.2253 | 1.1621 | 1.1621 | +16.21% |
| Chunk-only → Chunk+Prefix | 1.1661 | 1.0639 | 1.1827 | 1.1661 | +16.61% |
| Full Prefill → Chunk-only | 0.9024 | 0.9945 | 0.9025 | 0.9025 | -9.75% |
| Prefix-only → Chunk+Prefix | 0.9112 | 0.8635 | 0.9185 | 0.9112 | -8.88% |

当前受控负载中，Prefix 在有/无 Chunk 时三轮均提升吞吐；Chunk 的两组配对结果三轮均为负。Prefix-only 每轮吞吐最高。没有证明 Chunk 对其他到达模式或生产内核无益，也没有 Profiling 能把下降归于唯一瓶颈。

## 各轮总耗时（秒）

| 轮次 | Full | Chunk-only | Prefix-only | Chunk+Prefix |
|---|---:|---:|---:|---:|
| 1 | 131.121 | 145.304 | 113.538 | 124.603 |
| 2 | 146.063 | 146.871 | 119.207 | 138.051 |
| 3 | 146.353 | 162.168 | 125.938 | 137.117 |

三轮存在约 10% 量级的运行耗时波动，尚未采集 GPU 频率、功耗或 kernel profiling 来解释原因。只有三轮，也没有完整位置平衡或多机器重复，不给稳定置信区间，不把很小的差异当成一般规律。

## 正确性、Tokenizer 与历史边界

- 12 次正式测量、1200 次请求执行、76,800 个实际输出 Token；所有完整输出与历史参考相同，跨模式/跨轮均一致，重放请求一致，有限值与 KV 回收通过。
- 正式计时内没有新增 Graph 诊断。离线审查重新从 Token 时间戳计算逐请求 TTFT/TPOT/E2E，并复算 trial 吞吐、三轮指标中位数和配对比值。
- Tokenizer 清单唯一差异是额外的空文件 `tokenizer.json.download.curl.log`，历史四个真实 Tokenizer 文件及模型 config 哈希均相同，全部请求 Token 哈希也相同。准备代码保留实际文件清单和差异，没有伪造历史元数据或关闭输出比较。
- [2026-10-11 旧三组](../README.md) 每组测一次，Full 与 Chunk 还改变调度优先级。旧 +20.23% 保留为历史 Chunk→Chunk+Prefix 结果；本轮统一调度、各三轮，分别是 +16.21%/+16.61%，不能拼表、平均或相乘。
- 这不是 DFlash/三种 Draft、AWQ 或多卡实验，不与其加速比混用；不是相对生产 vLLM 的性能结论，也不是任务质量评测。

## 记录与复算

保留服务器原始 [report.json](report.json)、[factorial-effects.json](factorial-effects.json)、[依赖清单](installed-packages.json)；[summary.csv](summary.csv) 等 CSV 仅统一为 LF 换行，数值未改。[trials.json](trials.json) 保留 12 个完整 summary 与逐请求输出哈希；[SOURCES.json](SOURCES.json) 记录原始 ZIP、读入记录和服务器源码快照的 SHA256，并注明 CSV 换行转换。原始 Prompt、输入/输出数组、步骤 trace 和完整 ZIP 在本地保存，不提交到 Git。

持有原 ZIP 时，无需 GPU 或 Torch 即可审查：

```bash
python3 scripts/import_sharegpt_factorial.py /实际路径/sharegpt-factorial-20261024-024058Z-s283aej0-records.zip
```
