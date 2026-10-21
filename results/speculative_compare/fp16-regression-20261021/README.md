# FP16 Target 短回归：20/21 个投机请求执行一致

原始包：`speculative-fp16-regression-20261021-010942Z-records.zip`，SHA256 及提取文件指纹见 [SOURCES.json](SOURCES.json)。本目录的原始 JSON/JSONL、环境及日志不修改；[分叉汇总](divergence/divergence-analysis.json) 为离线派生分析。

配置为单张 A100 PCIe 40GB、HF Qwen3-8B FP16 主干＋真正 FP32 Target Head、三个官方 BF16 Draft、SDPA Eager。关闭 FP16 reduced-precision reduction，BF16 reduced reduction 保持启用；无 CUDA Graph。复用原正式输入中的 `humaneval:19`、`humaneval:4`、`mt-bench:9`，每条实际生成 256 Token，忽略 EOS，重复一轮。

共完成 30 次请求执行：9 次配对 AR、21 次投机。`errors=[]`、`precision_errors=[]`、`ar_divergences=[]`。**六个投机模式各 3/3 一致；Confidence 0.8 为 2/3，因此整体正确性仍未通过。**

| 模式 | 三请求累计生成时间 s | 输出 tokens/s | 相对本进程 AR | 完整一致请求 |
|---|---:|---:|---:|---:|
| EAGLE3 AR | 31.590 | 24.311 | 1.000× | 基线 |
| EAGLE3 | 13.836 | 55.506 | 2.283× | 3/3 |
| DFlash AR | 32.180 | 23.866 | 1.000× | 基线 |
| DFlash | 12.324 | 62.320 | 2.611× | 3/3 |
| DSpark AR | 31.300 | 24.537 | 1.000× | 基线 |
| DSpark No Markov | 17.845 | 43.037 | 1.754× | 3/3 |
| DSpark Markov Only | 10.212 | 75.206 | 3.065× | 3/3 |
| DSpark Fixed | 10.249 | 74.931 | 3.054× | 3/3 |
| DSpark Confidence 0.5 | 10.302 | 74.549 | 3.038× | 3/3 |
| DSpark Confidence 0.8 | 11.851 | 64.802 | 2.641×，未合格 | 2/3 |

吞吐为 768 / 三请求累计耗时；包括每请求 Prefill 和完整生成，不是并发 Serving 吞吐，也不是仅 Decode 的计时。单轮短回归不替代十二条、三轮正式实验。

## 唯一分叉

`dspark_conf_0.8` 的 `mt-bench:9` 在输出索引 **24** 首次不同：AR Token **3381**，投机 Token **1083**。公共输出前缀 24 Token；其他二十组投机输出的公共前缀均为完整 256 Token。

根据保存的接受轨迹，分叉发生在第 5 次 Verification，Pending 对应输出索引 19，Target 计算前有效 KV 长度 72，输入 Q=8，对应 Target 预测行索引 4；该 Token 来自被接受的 Draft 前缀。

这轮没有保存 Target Top-2、Hidden States 或 KV 张量，**无法从 Token ID 推断此处 Logits 是否并列，也不能确认 Cache 错误或精度误差**。历史 BF16 `humaneval:19` 的分数并列不能直接解释此处。

与旧 BF16 正式结果在相同三条输入上的子集对照：前六种模式原来均为 0/3，本轮均为 3/3；Confidence 0.8 原来为 1/3，本轮为 2/3。机器、精度及归约配置不完全相同，不能将变化全部归因于 FP16 单一变量，也不能按不同请求总数直接比较总体通过率。

## 后续

保留此轮及 BF16 原结果。新增 [归档 Target 适配路径](../../../experiments/speculative_compare/LEGACY_TARGET.zh.md)，按用户要求复用旧 Mini-vLLM 稳定 Target 算术，保留当前七种官方 Draft 模式；后续短回归为 21/21，[十二条三轮正式结果](../legacy-target-formal-20261021/README.md) 为各模式 36/36 一致。这不等于原 HF 计算路径的分叉已修复。
