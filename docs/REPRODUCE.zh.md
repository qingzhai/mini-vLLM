# 安装和复现实验

## 环境与模型

GPU 路径实测 Linux、Python 3.11（历史实验 3.11.14，新 ShareGPT 四组 3.11.12）、Torch 2.9.1+cu128、Triton 3.5.1、Transformers 4.57.3、NVIDIA A100-PCIE-40GB。先按根 README 安装 `requirements-gpu.txt`。CPU 合约测试可使用 CPU Torch；没有执行 CUDA/NCCL 时不得将合约测试结果写成 GPU 正确性。

统一使用 `WORK_ROOT` 保存模型、缓存和新结果，默认是仓库下的 `.work`。已有 AutoDL 目录可设置为原来的工作目录，不需要重新下载模型。模型位置也可用 launcher 的 `--model`、`--draft`、`--awq-model` 参数单独指定。普通引擎按目录名选择实现，Qwen 主模型目录的末级名称应保留 `Qwen3-8B` 或 `Qwen3-0.6B`。

`data/models.json` 保留历史 config/tokenizer 哈希。AWQ 和 Draft 固定为已测 revision；历史基础 Qwen checkpoint 的完整 commit 没有保存，因此没有补写一个推测的历史 commit。首次新下载基础模型时，下载器解析并记录具体 commit，校验已保留的元数据哈希；这约束 tokenizer/config，但不能证明完整权重与历史下载逐字相同。若知道原基础 checkpoint commit，可使用 `--revision` 明确指定。

下载脚本默认访问 Hugging Face；国内网络可设置 `HF_ENDPOINT=https://hf-mirror.com`。运行实验时禁止联网拉取缺失模型，缺文件会报错。官方 Draft 的 Python 代码按已记录 SHA256 核对。

## 数据准备

```bash
export WORK_ROOT="$HOME/autodl-tmp/minivllm-work"
export HF_ENDPOINT=https://hf-mirror.com
python scripts/prepare_workload.py
```

使用固定 dataset commit 和文件 SHA256，按 seed=42 的顺序选取 40 条短 prompt、40 条长 prompt，再追加 20 条受控完整 prompt 重放。共 100 请求，50% short、50% long；这不是自然线上前缀分布。只使用首条 human 消息，Qwen chat template 关闭 thinking。

发布仓库不带原始 ShareGPT 对话或输入 token 列表。准备脚本在本地生成 workload，然后逐项核对 `data/sharegpt-selection.json` 的 source ID、row、长度、token 哈希、tokenizer/config 哈希及总请求哈希。允许用 `--dataset-file` 指定离线原始 train.json。Distributed 和 ShareGPT 入口共用此固定 workload。新的四组入口使用 `python scripts/prepare_sharegpt_factorial.py --work-root "$WORK_ROOT"`，要求全部输入 Token 和模型 config 精确相同，但单独记录 Tokenizer 文件清单差异；原归档的差异仅为空下载日志，详见[四组协议](SHAREGPT_FACTORIAL.zh.md)。

## 统一入口

| 命令 | GPU | 运行配置 |
|---|---:|---|
| `python scripts/run.py dflash-graph` | 1 | FP16 主干、FP32 Head、BF16 Draft；四模式各三轮，256 输出 tokens |
| `python scripts/run.py sharegpt` | 1 | BF16 主干、FP32 Head、batch 8；三种调度各一轮，64 输出 tokens |
| `python scripts/run.py sharegpt-factorial` | 1 | 同输入与精度，Full/Chunk-only/Prefix-only/Chunk+Prefix 各三轮；统一 Decode-first，64 输出 tokens |
| `python scripts/run.py distributed` | 2 | FP16 主干、FP32 Head；single/TP2/DP2/1P1D 各两组负载 |
| `python scripts/run.py awq` | 1 | FP16 与官方 W4A16 独立进程对照 |
| `python scripts/run.py prefix` | 1 | BF16，前缀缓存、LRU 与参考算子检查 |
| `python scripts/run.py chunked` | 1 | BF16 主干、FP32 Head、Graph；分块混合调度及前缀组合检查 |
| `python scripts/run.py decode` | 1 | 独立 Decode attention 算子 benchmark |
| `python scripts/run.py prefill` | 1 | 独立 Prefill attention 算子 benchmark |

使用 `--dry-run` 只打印启动计划。额外实验参数写在 `--` 之后。例如：

```bash
OUTPUT_TOKENS=256 REPETITIONS=3 python scripts/run.py dflash-graph
python scripts/run.py distributed -- --modes single,dp2,pd
python scripts/run.py sharegpt -- --repeats 1
python scripts/run.py decode -- --batch-sizes 1 4 --seq-lens 128 1024 --iters 20 --warmup 5
```

每次自动创建不同结果目录；显式 `--run-dir` 必须为空。控制台输出同步写入 run.log。成功或失败均保存 exit-status 和 records ZIP，包含依赖的唯一核心源码、共享工具和实验入口。运行归档不上传到 GitHub，也不包含权重或 venv。

需保持 SSH 断开后继续运行时，在已激活环境中使用：

```bash
mkdir -p "$WORK_ROOT/results"
nohup python -u scripts/run.py dflash-graph > "$WORK_ROOT/results/dflash-launch.log" 2>&1 < /dev/null &
tail -f "$WORK_ROOT/results/dflash-launch.log"
```

`Ctrl+C` 结束 tail 查看；任务仍在后台。关闭实例前先保存已完成的 records ZIP 和代码。

## 正确性与整理验证

```bash
python scripts/check_cpu.py
python scripts/check_cpu.py --tensor
python scripts/check_cpu.py --transport
```

检查内容包括原始核心与共享算术字节一致、源码摘要防篡改、ShareGPT 参考输出和 runtime guard、调度/页所有权、CPU KV pack/remap、Graph 静态元数据与输出所有权、实际投机状态机的接受/拒绝/回滚合约。Graph 测试在 CPU 上模拟捕获与回放，不会执行真正 CUDA Graph。传输测试使用 localhost TCP；GPU NCCL 迁移仍以归档 GPU 记录及实际重跑为准。

本次发布主要变更为目录合并、参数化路径、启动和归档流程，核心 25 个 Python 源文件与归档一致。整理没有重新租卡运行；GPU 性能数字对应历史 records ZIP 内源码。若修改算法、模型精度、采样、缓存或形状，需要重新做对应 GPU 正确性检查后再比较性能。
