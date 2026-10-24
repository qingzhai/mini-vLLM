# 安装与复现实验

先选择实验，再选择环境和模型。普通引擎、DFlash＋Graph、多卡和三种 Draft 对照保留独立入口；只想学习或核对已有数字时，阅读[正式结果](../results/README.md)并做离线复算即可，无需 GPU 重跑。

## 1. 选择环境与所需文件

GPU 实测为 Linux、Python 3.11、PyTorch 2.9.1+cu128、Triton 3.5.1、A100-PCIE-40GB。旧实验使用 Python 3.11.14，新 ShareGPT 与三方法实验为 3.11.12。

| 实验 | Transformers | 需要的模型与数据 | GPU | 后续入口 |
|---|---|---|---:|---|
| S1 ShareGPT 四组 | 4.57.3，建议 `venv-sharegpt` | Qwen3-8B、固定 100 请求输入 | 1 | [四组操作说明](SHAREGPT_FACTORIAL.zh.md) |
| G1 DFlash＋Target Graph | 4.57.3，原引擎环境 | Qwen3-8B、z-lab b16 Draft、内置固定 Prompt | 1 | `scripts/run.py dflash-graph` |
| M1 Single/TP2/DP2/1P1D | 4.57.3，原引擎环境 | Qwen3-8B、固定 ShareGPT 输入 | 2 | `scripts/run.py distributed` |
| Q1 AWQ | 4.57.3，原引擎环境 | Qwen3-8B、官方 Qwen3-8B-AWQ | 1 | `scripts/run.py awq` |
| D1 EAGLE-3/DFlash/DSpark | **5.10.2，独立 `venv-deepspec`** | Target、三个官方 Draft、固定 DeepSpec 源码与十二输入 lock | 1 | [Target 适配与独立运行包](../experiments/speculative_compare/LEGACY_TARGET.zh.md) |

原引擎环境按根 README 的 `requirements-gpu.txt` 安装；D1 不套用它的 Transformers 版本。已有匹配环境直接复用，不重复安装 PyTorch/CUDA。S1 的环境脚本可从镜像继承 Torch，使用阿里云 PyPI 补非 GPU 依赖；其他平台先核实基础 Python 路径再执行。

统一用 `WORK_ROOT` 存模型、缓存与新结果，放在 Git 仓库之外；`scripts/run.py` 默认使用仓库 `.work`。常见数据盘为恒源云 `/hy-tmp`、AutoDL `/root/autodl-tmp`，按真实挂载设置。Qwen 主目录末级名称保留 `Qwen3-8B`，不要把其他环境的 venv 直接搬过来当成已恢复。

## 2. 准备模型与输入

### 已有文件时

保留本地权重、Config、Tokenizer 和生成过的 workload，先用相应准备入口校验复用。S1/M1 不需要三个 Draft；G1 的 z-lab b16 与 D1 的 DeepSpec Draft 也不是同一份模型。

### 缺少文件时

原引擎下载入口为 `scripts/download_models.py`；国内可设置 `HF_ENDPOINT=https://hf-mirror.com`。S1 的[恢复入口](SHAREGPT_FACTORIAL.zh.md#prepare-model-env)支持按已归档 SHA256 从 ModelScope 恢复原 Target。模型与数据准备允许联网，正式 GPU 执行按离线模式运行，缺失文件会报错。

`data/models.json` 保留历史 Config/Tokenizer 哈希。历史基础 Qwen 的完整 checkpoint commit 未保存；后续下载器会记录解析到的 commit，但不能补写为历史版本。新三方法的完整权重文件指纹在正式 lock 中保留，复现时应核对实际文件。官方 AWQ/Draft 的 revision 与实现源码哈希按对应实验记录核验。

### 固定 ShareGPT 输入

```bash
export WORK_ROOT=/hy-tmp/minivllm-work
export HF_ENDPOINT=https://hf-mirror.com
# S1 四组：复用既有相同输入；缺失才获取固定版本的数据集。
python scripts/prepare_sharegpt_factorial.py --work-root "$WORK_ROOT"
# S0 历史三组或 M1 多卡：采用历史文件清单的严格校验。
# python scripts/prepare_workload.py --work-root "$WORK_ROOT"
```

只执行要复现的那条准备命令。固定 seed=42：40 条短 Prompt＋40 条长 Prompt，再追加 20 次完整重放，共 100 请求。首条 Human 消息经 Qwen Chat Template，关闭 thinking；50% short/50% long，每请求 greedy 真实输出 64 Token、忽略 EOS。四组准备严格核对 Token、顺序、Config 和生成配置，Tokenizer 文件清单差异单独记录；这不关闭输出比较。离线原始 `train.json` 可用 `--dataset-file` 指定。

## 3. 原引擎与四组实验入口

| 命令 | 配置 |
|---|---|
| `python scripts/run.py sharegpt-factorial` | S1：四组统一 Decode-first、各三轮，BF16＋FP32 Head |
| `python scripts/run.py sharegpt` | S0：历史三种调度各一轮，保留其不同优先级 |
| `python scripts/run.py dflash-graph` | G1：AR/DFlash × Eager/Graph，各三轮、每请求 256 输出 |
| `python scripts/run.py distributed` | M1：单卡及三种双卡模式，burst/stream、每请求 64 输出 |
| `python scripts/run.py awq` | Q1：FP16 与官方 W4A16，short/long × batch1/4，各三轮 |
| `python scripts/run.py prefix` / `chunked` | 前缀、LRU、分块与组合的功能检查 |
| `python scripts/run.py decode` / `prefill` | 局部 Attention 算子检查，不代替完整模型实验 |

`--dry-run` 只打印计划。额外参数写在 `--` 后，例如：

```bash
python scripts/run.py sharegpt-factorial --dry-run
python scripts/run.py sharegpt-factorial -- --repeats 3
python scripts/run.py distributed -- --modes single,dp2,pd
```

S1 从环境准备到后台执行的完整顺序见[四组协议](SHAREGPT_FACTORIAL.zh.md)。所有性能表均限定自身模型、精度、负载与基线；不会将多组实验的加速比相乘。

## 4. 三种 Draft 的独立流程

D1 不在 `scripts/run.py` 的子命令列表里。统一入口文档只负责导航，实际调用链是官方生成循环 → `LegacyTarget` → Target Eager，无 Graph、串行 Batch=1。

依次准备固定提交 DeepSpec 源码、Qwen3-8B 与三个本地 Draft、Transformers 5.10.2 环境、本机路径与指纹正确的十二输入 lock；然后构建并上传短回归包和正式包。短回归通过后再执行正式三轮，完整可复制命令在 [LEGACY_TARGET.zh.md 的公开仓库复现部分](../experiments/speculative_compare/LEGACY_TARGET.zh.md#从公开仓库生成运行包)。不要复制别人的绝对路径 lock，或运行会重新安装/下载的旧 `start.sh`。

正式范围是 GSM8K/HumanEval/MT-Bench 各四条，256 输出、三轮；它不是 100 请求 ShareGPT 并发 Serving。只核对既有结果时，用[正式报告里的离线分析命令](../results/speculative_compare/legacy-target-formal-20261021/README.md)，无需恢复全部权重。

## 5. 启动、判断完成与保存

每次自动创建新结果目录；显式 `--run-dir` 必须为空。用 nohup 启动并给日志使用不同名称：

```bash
mkdir -p "$WORK_ROOT/results"
LAUNCH_LOG="$WORK_ROOT/results/sharegpt-launch-$(date -u +%Y%m%d-%H%M%SZ).log"
nohup python -u scripts/run.py sharegpt-factorial > "$LAUNCH_LOG" 2>&1 < /dev/null &
echo "PID: $!"
echo "日志: $LAUNCH_LOG"
tail -f "$LAUNCH_LOG"
```

`Ctrl+C` 在这里结束的是 `tail`，后台实验继续。看到无新日志时先检查进程与末尾日志，不重复启动。

- S1 完成标志：`EXIT STATUS: 0`、`SHAREGPT FINAL CHECKS PASS`，并生成 `*-records.zip`。非零退出保留现场，不能只挑成功行。
- D1 完成标志与 ZIP 指针由独立运行脚本输出；同时检查报告的一致性结论，而不是只看速度。
- 关机前下载完整 records ZIP。它保留配置、环境、源码快照与原始记录；GitHub 精选摘要不能代替完整备份。
- 迁移还需单独保存数据盘中的模型、输入和其他成果。实例镜像与 OSS workspace 的具体内容不同，按自己的备份清单恢复。

## 6. 本地检查与历史状态

```bash
python scripts/check_cpu.py --report .work/cpu-checks.json
python scripts/check_cpu.py --tensor
python scripts/check_cpu.py --transport
```

默认检查包含原始核心/共享算术、发布源码清单、历史输出参考、调度与页状态、四组协议、三方法测量协议及权重恢复合约。`--tensor` 需要 CPU Torch，模拟张量、Cache、Graph 缓冲等；`--transport` 检查 localhost RPC。它们不执行 GPU CUDA Graph/NCCL，也不构成新 GPU 性能测量。

2026-10-24 的 S1 已完成新四组 GPU 正式测量；G1、D1、M1、Q1 的 GPU 结论继续对应各自历史归档源码。本次文档整理不重新跑 GPU。当前检查范围与基准提交见 [VALIDATION.json](VALIDATION.json)，早期目录合并检查另行标为历史记录。
