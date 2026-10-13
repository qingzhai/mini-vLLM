# Chunked Prefill 与 P/D 混合调度

整理自历史通过的 `bfloat16-fp32head-graph` 实验。直接使用仓库已有 ChunkScheduler/ChunkRuntime，不运行早期 patch/rollback 脚本。

```bash
export WORK_ROOT="$HOME/minivllm-work"
export MODEL_PATH="$WORK_ROOT/models/Qwen3-8B"
export RUN_DIR="$WORK_ROOT/results/chunked-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python experiments/chunked/bench_chunked.py
```

固定 BF16 主干和 KV、FP32 RMS 累加、实际 FP32 LM Head、普通引擎 Decode CUDA Graph，关闭 TF32 和 BF16 reduced reduction。默认 `CHUNK_LOGITS_DIAGNOSTICS=1` 保留原 top-5 诊断，计时因此属于诊断运行；可设为 0 另测，但不能将不同诊断配置直接混比。该入口限制 `CORE_DTYPE=bfloat16`，不自动改用其他精度掩盖错误。

负载为两个短请求先 Decode，随后加入两个长请求，每请求生成 32 greedy tokens。对照 full Prefill 与 chunk256/decode-first 混合调度，检查没有中间 chunk 提前采样、实际 mixed steps、四条完整输出一致和页回收。随后单独验证 Prefix Cache 加 chunk64 的起点、缓存命中和重复输出一致。

所有计算、阈值和诊断规则沿用已验证源代码，路径改为可配置。这里的 P/D 混合调度不是双 GPU 的 1P1D 分离，DFlash/TP/DP/AWQ 也没有同时开启。小规模通过不代表历史 Flash Prefill 数值问题普遍解决，吞吐是否提升以本次数据为准。

输出为 `RUN_DIR/<model>-bfloat16-fp32head-graph-details.json` 与 `-summary.csv`，包含请求时间、完整 token IDs、调度 trace、Graph 检查和组合验证。已有 details 文件拒绝覆盖。
