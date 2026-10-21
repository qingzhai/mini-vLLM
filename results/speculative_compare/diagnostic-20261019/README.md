# HumanEval 分叉的离线控制实验

本轮为诊断，不计性能收益。来源 `dspark-greedy-diagnostic-records.zip`，指纹见 [SOURCES.json](SOURCES.json)；返回的三份 Python 源码与发出的已审查诊断包一致，lock、workload 与旧 Smoke 完全相同。

`diagnostic_complete=true`、`reproduced_historical_failure=true`、`trace_matches_untraced=true`、`focus_history_matches_ar=true`。原分叉稳定复现，诊断 wrapper 没有改变本轮输出和提议/接受轨迹。模型仍为原 BF16 主干与 BF16 Head，BF16 reduced-precision reduction 开启，TF32 关闭。

## 直接观察

第 5 轮 `start=139`，输入 `[6337,429]`，position `[139,140]`。全部 36 层 Target KV 的有效长度为 139，Draft 在提议前缓存 133 个位置，新 Target features 为 6，提议后裁回 139；进入验证时的 Token 历史与 AR 的公共历史一致。

| Cache 与 Query | BF16 Logit 5980 | BF16 Logit 429 | 真正 argmax | 相同 Hidden State 的 FP32 Head argmax |
|---|---:|---:|---:|---:|
| 实际 Cache，Q=2 | 31.625 | 31.625 | 429 | 5980 |
| 实际 Cache，Q=1 | 31.500 | 31.500 | 429 | 5980 |
| 逐 Token AR 重建 Cache，Q=2 | 31.625 | 31.500 | 5980 | 5980 |
| 逐 Token AR 重建 Cache，Q=1 | 31.500 | 31.375 | 5980 | 5980 |
| 原提议/验证/crop 重放 Cache，Q=2 | 31.625 | 31.625 | 429 | 5980 |

实际 Cache 与完整投机轨迹重放 Cache 在 **36 层 K/V 上逐元素相同**，对应 Hidden States 和 Logits 也相同。逐 Token AR Cache 则从第 0 层就有 K/V 差异。后者的前向分块历史不同；“逐 Token AR 数值不等”本身不能证明 Cache 被错误写入。

对实际 Q=2 的同一最终 Hidden State，真实 FP32 Head GEMM 得到：5980 为 **31.637157440185547**，429 为 **31.575721740722656**，差约 **0.061436**，没有并列。重新执行 BF16 Head 的全部 Logits 与现场逐元素一致；其输出将两个候选都舍入为 31.625。直接 `torch.argmax` 与官方 one-hot 概率 argmax 都为 **429**，随后该候选被接受。这也纠正了旧临时诊断日志的 `argmax:5980` 字段。

## 可支持的解释及限制

本例的直接分叉机制已得到控制实验支持：不同计算分块/Cache 计算历史产生不同数值，实际投机路径的 BF16 Head 输出又合并了两个相近候选的分数，导致低索引候选胜出。不能进一步把全部差异归因于某一个 Attention Kernel、cuBLAS 算法或 reduced-reduction 开关，因为本轮没有逐算子隔离这些因素。

本轮未发现长度/位置不变量失效，现场也可从原轨迹逐元素重建，没有观察到随机 Cache 污染；同路径重放一致仍不能独立排除掩码等共同实现因素，不能宣称对所有 Cache 状态完成了正确性证明。

**FP32 Head 目前仅在相同 Hidden State 的局部对照中恢复了 5980，不等于整段生成已经修复。** 下一步对所有算法和各自 AR 基线统一采用 BF16 主干＋FP32 Target Head，保持 Draft BF16、原阈值/输入/采样，重新执行三条输入、32 Token 的完整一致性回归。该新精度配置必须另行归档，不能覆盖旧 BF16 失败结果或沿用旧速度。
