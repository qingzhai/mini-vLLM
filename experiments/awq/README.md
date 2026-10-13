# AWQ W4A16

加载 Qwen 官方 `Qwen/Qwen3-8B-AWQ`，使用自实现 Triton packed GEMM，在 tile 内解包 int4 权重并反量化至 FP16，FP32 累加。保留原合并 QKV/GateUp、完整 checkpoint 消费检查及独立 FP32 参考算子测试。本实验不生成 AWQ 权重，不包含校准或任务质量评测。

通过仓库统一入口启动，或在已配置环境中直接执行：

```bash
export WORK_ROOT="$HOME/minivllm-work"
export MODEL_PATH="$WORK_ROOT/models/Qwen3-8B"
export AWQ_MODEL_PATH="$WORK_ROOT/models/Qwen3-8B-AWQ"
export RUN_DIR="$WORK_ROOT/results/awq-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 python experiments/awq/run.py
```

权重须预先下载；该入口不安装、不下载。复现权重版本为 `4da05a8edb55c6046cce958586c33b61da07bb79`。若模型下载器保存了 `source.json`，默认从 AWQ 模型目录读取，也可设置 `AWQ_SOURCE_PATH` 指定文件；提供 metadata 时会校验 repo/revision，缺少时在报告中明确标记为未验证下载版本。

FP16 baseline 与 AWQ 各自在全新子进程运行，避免上一模式的模型和 patch 留在内存。两边使用 FP16 激活、真实 FP32 LM Head、FP32 RoPE 初始化、eager、64 个 KV 页；关闭 Graph、Prefix、Chunked。四种合成 short/long × batch1/4 负载，各生成 32 tokens，预热后测三次取中位数。沿用旧实验的诊断同步和计时方式，没有合并到 DFlash Graph 的执行器。

验收覆盖首层 QKV/Down 的 M=1/4/33 算子参考对照、有限 logits、各模式重复输出一致及页回收；量化输出不要求等同原 FP16 模型。性能以 CSV 的实测比值为准，权重大小下降不等于吞吐提升。

输出位于 `RUN_DIR`：两份 details JSON、两份运行日志、`awq-comparison.csv/json` 和汇总日志。已有文件拒绝覆盖。模型权重和虚拟环境不进入代码仓库。
