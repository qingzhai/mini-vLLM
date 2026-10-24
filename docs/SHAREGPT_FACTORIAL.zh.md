# ShareGPT 的 Chunk 和 Prefix 四组消融

状态：**2026-10-24 已完成四组各三轮 GPU 测量与离线归档审查**。跨模式、跨轮、历史参考输出及 KV 回收检查通过；完整条件、三轮范围与配对比值见[正式报告](../results/sharegpt/factorial-20261024/README.md)。旧三组保持原样，不与本轮拼表。

## 只改变两个开关

| 模式 | Chunk | Prefix | 未缓存 Prefill 窗口 | 调度优先级 |
|---|---|---|---|---|
| Full Prefill | 关 | 关 | 一次计算完整剩余 Prompt | Decode-first |
| Chunk-only | 开 | 关 | 每请求本步最多 256 Token | Decode-first |
| Prefix-only | 关 | 开 | 一次计算完整未命中后缀 | Decode-first |
| Chunk＋Prefix | 开 | 开 | 每请求本步最多 256 Token | Decode-first |

四组使用同一 `FactorialScheduler`、同一请求补位策略。Full/Prefix-only 在剩余预算放不下整段未缓存 Prompt 时推迟，禁止为了塞满预算隐式切块；它们也可以和 Decode 出现在同一步，不能把 mixed step 一概当成开启 Chunk。旧 `ChunkScheduler`、模型和算子均未修改，新调度器位于[实验目录](../experiments/sharegpt/factorial_scheduler.py)。

保留历史的 100 条 ShareGPT 派生输入：80 个不同 Prompt＋20 次完整 Prompt 重放、50 短/50 长、同时入队，每请求 greedy 实际输出 64 Token、忽略 EOS。驻留上限 8、单步 Token Budget 2048、330 个 KV 页、页长 256；BF16 主干＋FP32 Head、固定算术、全阶段逐 Query PagedAttention，纯 Decode 保留 Graph。没有 Draft、AWQ 或多卡。

每组先用 16 条真实输入预热，正式四组各测三轮，轮间旋转顺序；每次 trial 重建冷前缀缓存。三轮并未让四个模式完全平衡每个执行位置，不宣称是完整拉丁方设计。

## 验收与报告

- 四组同轮逐 Token 比较，并检查 Full 跨轮一致；继续对照历史已通过的输出。Prefix-only 使用历史 Full 的输出参考，但绝不使用它的旧耗时作为本轮性能基线。
- 保留部分 Prefill 不采样、Full 无部分 Prefill、Token Budget、实际缓存命中数、重复请求输出和全部页回收检查。本轮已在指定 GPU 负载通过验收；CPU 模拟与离线归档审查不替代新配置的 GPU 验证。
- 若正确性不通过，仍保留完整原始结果和时间，报告标为不合格；不会跳过失败请求或关闭比较。
- `summary.csv` 保留 TTFT、TPOT、E2E、总输出吞吐、峰值显存和三轮范围。`factorial-effects.json/csv` 单独给出四项同轮配对吞吐比中位数：Full→Prefix-only、Chunk-only→Chunk＋Prefix、Full→Chunk-only、Prefix-only→Chunk＋Prefix。

比值大于 1 才是吞吐提升；效果可以为负。吞吐中位数之比和配对比值中位数分别标注。这里测的是引擎完整 Prefill＋Decode，含排队，不含 HTTP、分词、加载、预热或 Graph 诊断。输入的 20% 重放比例不等于实际 Token 复用率。

## 新机器需要什么

1. 当前仓库生成的代码包，包含原引擎、实验入口、历史输出参考和数据选择哈希；不要覆盖 OSS 恢复出的旧代码目录。
2. `Qwen3-8B` 原始未量化权重、config 和 tokenizer；本轮不需要三个 Draft，也不需要 AWQ 权重。
3. 原 100 请求 `sharegpt-workload.json`，或固定版本 ShareGPT `train.json`。准备脚本先复用本地正确输入，缺失时才下载数据并按原选择生成；禁止以随机或合成输入代替。
4. Python 3.11、PyTorch 2.9.1+cu128、Triton 3.5.1、Transformers 4.57.3。原 DeepSpec 的 Transformers 5.10.2 环境不能直接用于本轮。环境脚本新建 `venv-sharegpt`，继承镜像预装 Torch，不重新安装 Torch/CUDA。

本轮按照单 A100 40GB 配置；保留原 330 页预算时不能直接换小显存卡而仍称为相同实验。50GB 数据盘只保留这轮模型、代码、输入和环境可以使用；留 100GB 更方便恢复旧文件和下载缓存。不要额外下载三个 Draft 占空间。

OSS workspace 是保留旧代码、结果和恢复元数据的便利备份，**不是本轮必须重新下载的模型包**。使用前核对自己的归档内容；历史会话只确认过 `workspace.tar.gz` 完成上传，不能据此认定模型包和最终清单齐全。不要调用要求全部模型包完整的旧 `restore.sh`。恢复环境还要求镜像中的基础 Python/Torch 路径兼容。

## 按顺序操作

### 1. 本地打包和上传

在仓库根目录执行：

```bash
BUNDLE="/tmp/minivllm-sharegpt-$(date -u +%Y%m%d-%H%M%SZ).zip"
python3 scripts/build_sharegpt_factorial_bundle.py --output "$BUNDLE"
echo "$BUNDLE"
```

构建器拒绝覆盖已有 ZIP，因此每次使用不同文件名。将 `$BUNDLE` 上传到服务器 `/hy-tmp/`；下面将实际文件名设置为 `BUNDLE` 再解包，包内目录名固定不变：

```bash
BUNDLE=/hy-tmp/你上传的文件名.zip
python3 -m zipfile -e "$BUNDLE" /hy-tmp/
export WORK_ROOT=/hy-tmp/minivllm-work
export REPO=/hy-tmp/minivllm-sharegpt-factorial-v1
```

如果是 AutoDL，按实际挂载把工作目录和解包目录换为 `/root/autodl-tmp/`，不修改协议。

### 2. 可选恢复 OSS workspace

若本机工作目录已经完整，跳过本步。空数据盘恢复时先登录 OSS，将 `OSS_BACKUP` 换成自己保存的备份目录名：

```bash
oss login
export OSS_BACKUP=你的备份目录名
if [ -e "$WORK_ROOT" ]; then
  echo "工作目录已存在，跳过恢复，避免覆盖"
else
  mkdir -p /hy-tmp/minivllm-restore
  oss cp "oss://$OSS_BACKUP/workspace.tar.gz" /hy-tmp/minivllm-restore/ && \
    tar -xzf /hy-tmp/minivllm-restore/workspace.tar.gz -C /hy-tmp
fi
mkdir -p "$WORK_ROOT"
```

这里只恢复 workspace；当前代码继续使用独立的 `$REPO`，不要运行恢复出来的旧启动脚本。

<a id="prepare-model-env"></a>

### 3. 准备 Qwen3-8B 和环境

已有原权重时下载器只校验复用；缺失时从 ModelScope 获取目标文件，并逐文件核对已归档 SHA256，不下载 Draft：

```bash
cd "$REPO"
python3 experiments/speculative_compare/restore_model_weights.py \
  --root "$WORK_ROOT" \
  --lock "$REPO/results/speculative_compare/legacy-target-formal-20261021/lock.json" \
  --models target --workers 2

export BASE_PYTHON=/usr/local/bin/python
# 若要复用已有的匹配环境，先设置其 Python 绝对路径。
# export SHAREGPT_PYTHON="$WORK_ROOT/venv-core/bin/python"
if [ -z "${SHAREGPT_PYTHON:-}" ]; then
  bash "$REPO/scripts/setup_sharegpt_env.sh"
  export SHAREGPT_PYTHON="$WORK_ROOT/venv-sharegpt/bin/python"
fi
```

如果新镜像原始解释器路径不同，将 `BASE_PYTHON` 改为已安装上述 Torch 的那个 Python。脚本默认用阿里云 PyPI，只安装非 GPU 依赖；不调用 PyTorch 下载源，解析计划中若出现替换 GPU 依赖会停止。已有 `venv-core` 满足精确版本时，也可通过 `SHAREGPT_PYTHON` 使用它，跳过环境安装。

### 4. 准备原 ShareGPT 输入

```bash
export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"${SHAREGPT_PYTHON:-$WORK_ROOT/venv-sharegpt/bin/python}" "$REPO/scripts/prepare_sharegpt_factorial.py" \
  --work-root "$WORK_ROOT"
```

脚本先找工作目录的 `data/sharegpt-workload.json` 和旧 `results/sharegpt*/workload.json`，只接受与历史选择哈希一致的输入。文件在其他地方时用 `--reuse-workload /实际路径/workload.json`。若没有输入，则复用已缓存的原 `train.json`，否则下载固定 revision；离线文件可用 `--dataset-file /实际路径/train.json`。该步骤不加载模型到 GPU。

四组协议要求全部请求的实际 Token、顺序、选择元数据、生成设置、模型 config 和词表大小与历史记录一致。Tokenizer 文件清单可能因下载来源或额外文件而不同，也可能是文件内容变化；仅凭当前请求相同不能宣称两个 Tokenizer 普遍等价。准备阶段打印逐文件差异，实际哈希保留在 `workload.json`，正式报告在 `workload_identity` 中再次记录；不改写为历史哈希，也不跳过输入或输出一致性检查。旧协议默认仍要求 Tokenizer 文件清单完全一致。

本轮归档确认唯一差异为额外的空下载日志 `tokenizer.json.download.curl.log`；四个真实 Tokenizer 文件和 config 哈希均未变化。

### 5. 后台正式运行

```bash
mkdir -p "$WORK_ROOT/results"
LAUNCH_LOG="$WORK_ROOT/results/sharegpt-factorial-launch-$(date -u +%Y%m%d-%H%M%SZ).log"
nohup bash "$REPO/scripts/run_sharegpt_factorial.sh" > "$LAUNCH_LOG" 2>&1 < /dev/null &
echo "任务 PID: $!"
echo "日志: $LAUNCH_LOG"
tail -f "$LAUNCH_LOG"
```

默认四组各三轮，不额外先跑一套 GPU Smoke。只做已有的模型预热与 Graph 键预热，将诊断排除在计时外。2026-10-24 本轮十二次正式生成累计约 27.3 分钟，不含加载、预热与其他准备；未来机器仍建议预留 30～60 分钟，这不是完成时间保证。出现错误先保留新目录，不重复启动覆盖现场。

完成后输出 `EXIT STATUS` 和 `Records:` 的新 ZIP 路径。按 Ctrl+C 只退出 `tail`；下载 `*-records.zip` 后再关机。新运行需要先核验自己的完整结果，再更新结论；旧三组与 2026-10-24 四组的记录保持独立。

本地 CPU 验证入口：`python3 experiments/sharegpt/verify_factorial_cpu.py`，使用明确标记的合成 Token，只检查调度、页状态和报表约束，不生成 GPU 性能数据。
