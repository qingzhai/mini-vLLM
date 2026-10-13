# Prefix Cache 与 LRU

该入口整理自最后使用的 Prefix v2 实验，使用仓库已有 BlockManager、Paged KV Cache 和 Prefix Attention，不应用运行时源码补丁。

```bash
export WORK_ROOT="$HOME/minivllm-work"
export MODEL_PATH="$WORK_ROOT/models/Qwen3-8B"
export RUN_DIR="$WORK_ROOT/results/prefix-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python experiments/prefix/bench_prefix.py
```

使用 BF16 主干和普通引擎 CUDA Graph。先执行四组冻结 Q/K/V 的算子测试：BF16/FP16、16/32 query heads、8 KV heads、D=128、混合前缀及查询长度、乱序物理页与尾部 NaN 哨兵。参考为独立 FP32 attention 公式，阈值保持原样。

随后在四个共享长 system prompt 上对照 cache off/on，各生成 16 greedy tokens；独立 seed 请求在计时外。缩小逻辑分配器至至多 16 页以触发 LRU，实际 GPU 缓冲区大小仍按引擎原分配器决定。

硬校验要求缓存命中、全部输出一致、LRU 驱逐及页回收。模型首步 logits 的逐元素差异另行按原 `rtol=0.02/atol=0.15` 报告；若阈值失败会保留 WARNING，不能将功能通过等同于一般数值等价。未修复历史 Flash Prefill 路径的所有数值分叉；正式 ShareGPT 算术一致配置位于独立实验。

输出为 `RUN_DIR/<model>-prefix-v2-details.json` 与 `-summary.csv`，包含逐请求时间、缓存 token 数、数值诊断及 LRU 统计。已有 details 文件拒绝覆盖。
