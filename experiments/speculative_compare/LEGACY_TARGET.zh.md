# 原 Mini-vLLM Target 适配与正式对照复现

当前状态：短回归 21/21、十二条三轮正式对照 252/252 投机输出一致。性能与消融见 [正式报告](../../results/speculative_compare/legacy-target-formal-20261021/README.md)。下文保留从失败定位到正式测量的协议，复现时使用原 Target 入口，不启动旧 HF 失败路径。

2026-10-21 的 [HF FP16 短回归](../../results/speculative_compare/fp16-regression-20261021/README.md) 已有 20/21 组投机输出逐 Token 一致，仅 Confidence 0.8 的 `mt-bench:9` 在索引 24 分叉。本入口按用户要求对齐旧版已经通过实验的 Target 计算路径；不关闭检查、不删除输入、不改置信阈值。

## 复用了什么

`legacy-source-manifest.json` 锁定 31 个原仓库文件，独立包逐文件按原内容附带并校验。核心源码没有为了这轮实验修改。

- `experiments/common/target.py::Target / install_arithmetic`：原 Qwen3-8B 加载器和模型；FP16 主干，真实 FP32 Head GEMM；Linear/Silu 以 32 行为固定算术块，RMS 依维度使用原 32/1024 行配置。
- `src/myvllm/layers/attention.py`：全阶段逐 Query PagedAttention。每个 Query 的有效长度为绝对位置＋1；验证窗口中未来候选的 KV 不能被较早的 Query 读取。
- `experiments/dflash_graph/graph_executor.py::TargetGraphExecutor`：复用 **Eager** 分支，采集第 1/9/17/25/33 层原始输出；本入口不 Capture 或 Replay Graph。
- `experiments/dflash_graph/state.py::PagePool`：原分页分配、trim、release。Target KV 分配 64 页，每页 256 Token；受控最大长度 2048。
- 关闭 TF32、FP16/BF16 reduced-precision reduction；Draft 的 SDPA 使用原实验的 MATH 后端。

旧代码的 Head 保留原 FP16 Parameter，并额外持有完整 FP32 权重副本，额外 **2,489,319,424 bytes**。这与 HF 直接替换为 FP32 Head 的净参数增量不同；报告分别记录注册参数、额外权重与整个峰值 GPU 显存，不能混用。

## 适配层和状态边界

`legacy_target.py::LegacyTarget` 对接官方 DeepSpec 的 Target 调用接口。官方提议、验证、连续前缀接受、校正 Token 选择和 Draft Cache 更新函数体不修改。

1. 仅在本进程把 `deepspec.eval.base_evaluator.DynamicCache` 重新绑定为 `Target.new_cache()`。三个 Draft 模块自己导入的 Transformers DynamicCache 保持原状，不能用 Target 页池替代 Draft KV。
2. `LegacyCache.tokens` 保存已计算 Target KV 的 Token 历史。Forward 先写入完整 Pending＋候选窗口；官方循环 `crop(start)` 后，逻辑历史截到有效长度，`pool.trim()` 释放越界整页。保留页内的无效旧槽位由 Query 有效长度隔离，后续计算会覆盖。
3. 特征按官方 `hidden_states[layer_id+1]` 接口暴露，只有当前三个 Draft 实际使用的五层及最终归一化输出有效；其他层为 `None`，不是通用 HF 模型替代接口。官方 Draft 原更新函数负责截取有效特征和裁剪自己的 Cache。
4. 请求结束或生成异常时在 finally 中 `release_cache()`，检查所有页回收。适配器只支持串行 Batch=1，拒绝多个活跃 Target Cache。
5. Transformers 5 的 Qwen3 配置接口与旧版不同；构造原 Target 的短暂单线程区间，直接读取已校验 checkpoint 原 `config.json` 的 `rope_theta` 等字段，随后恢复 AutoConfig 工厂。没有改权重、配置文件或旧算子。
6. 三个官方 Draft 仍为 BF16，在 `draft.fc` 入口将 Target FP16 特征转 BF16；不改变 Target KV、接受规则或训练权重。

这是一条明确标记为 `legacy-target-official-draft-hybrid` 的新执行路径。旧 z-lab DFlash 是 b16、15 个候选，本轮 DeepSpec 官方 Draft 是 7 个候选；Transformers、驱动、Prompt 也不同。因此只说 **复用旧 Target 算术**，不能说整个旧实验完全相同。每个 Draft 进程重新执行自己的 AR，不能借用历史 HF 或旧 DFlash 的基线。

## 测量和证据

`run_legacy_target_regression.sh` 全程离线，复用原正式 lock、四个本地模型、环境和三条原输入；256 输出、一轮、全部七种模式及配对 AR，不自动长测。它不调用 `start.sh`、`setup.sh` 或下载脚本。

本轮每次 Target Forward 保存有效 Cache 长度、Query Token、Block Table、每行真实 argmax 与 Top-2 分数。`analyze_divergence.py` 按接受轨迹找到首次分叉对应的验证行，对照 AR 分数；Top-k 并列排序和真实 argmax 分开保存，缺失的分数不猜测。没有直接保存全量 KV 张量，若需判定 Cache 内容差异仍需独立控制实验。

收集分数包含 CPU 同步，原算术还使用 MATH Draft Attention，**本轮时间是诊断时间，不作为正式加速结论**。正确性仍要求七种模式全输出一致、跨进程 AR 一致、精度与源码匹配、完整请求执行和页回收；没有放宽为仅 Top-1 大多数一致或近似 Token 匹配。

本地已检查分页跨页裁剪、Pending/验证行映射、配置工厂恢复、特征层布局及官方接受循环的 CPU 模拟。随后返回的 [GPU 短回归](../../results/speculative_compare/legacy-target-regression-20261021/README.md) 为 21/21 组投机输出一致、跨 worker AR 一致且 Target 页全部回收；只代表本三条请求、一轮通过。

## 启动

解压独立包到现有 WORK_ROOT，然后执行：

```bash
export WORK_ROOT=/hy-tmp/minivllm-work
bash "$WORK_ROOT/minivllm-legacy-target-v1/run_legacy_target_regression.sh"
```

结果在新的 `results/speculative-legacy-target-regression-*`，原结果不覆盖。固定下载指针为 `$WORK_ROOT/speculative-legacy-target-latest-records.zip`，`QUALIFIED GREEDY COMPARISON: True` 仅表示本三请求配置通过。

## 十二条输入、三轮正式测量

`minivllm-legacy-target-formal-v1.zip` 单独解压，不覆盖回归包。`run_legacy_target_formal.sh` 读取已通过的回归目录，`prepare_legacy_formal.py` 核对原三十条执行记录、计算源码与完整十二条输入，再继承模型路径、权重和官方源码指纹、种子。只扩大请求集合/轮数、关闭 Top-2 收集，不改变模型算术或一致性规则。

```bash
export WORK_ROOT=/hy-tmp/minivllm-work
bash "$WORK_ROOT/minivllm-legacy-target-formal-v1/run_legacy_target_formal.sh"
```

默认通过目录从 `results/speculative-legacy-target-latest.txt` 读取；需要时可显式设 `REGRESSION_DIR=/实际通过的回归目录`。本轮十二条输入是历史原 GSM8K / HumanEval / MT-Bench 各四条，不是 ShareGPT；每条实际生成 256 Token、忽略 EOS、三轮，共 360 次请求执行和 92,160 个输出 Token（不计预热）。七种投机模式各有 36 次完整输出检查，三种 Draft 进程重新测各自 AR。计时仍含每请求 Prefill 和完整生成，不是纯 Decode 或并发 Serving 测量。

关闭的是额外 `.topk(2)` 与分数导出。官方验证仍需要每行完整 FP32 Logits，所以执行器的 `diagnostic=True` 返回 Logits 接口保留；不要把该内部参数误认为仍开启 Top-2 收集。精度、Attention、固定行数、Draft MATH、接受和 Cache 更新完全保留。完整 Token、接受轨迹、峰值显存、环境与源码继续归档。

按短回归总生成计时 269.42 秒乘以执行次数比值 12，粗估约 54 分钟，另有加载与预热；不同 Prompt 和移除采集均会改变耗时，建议预留 1～1.5 小时。没有增加额外 GPU Smoke/回归、重新安装或下载。

新目录为 `results/speculative-legacy-target-formal-*`，固定下载指针为 `speculative-legacy-target-formal-latest-records.zip`。旧 HF、FP16 及本轮回归记录保留。2026-10-21 返回的 [正式结果](../../results/speculative_compare/legacy-target-formal-20261021/README.md) 为七种投机模式各 36/36 输出一致、跨进程与跨轮 AR 一致、Target 页全部回收；本配置正式测量通过，不代表任意输入和所有路径的全局数值等价。

## 从公开仓库生成运行包

以下步骤供未来复现使用，本次归档不会自动执行 GPU。环境要求及三个 Draft 的固定 revision 见正式报告和 lock；已有服务器继续使用现有模型和环境。历史记录中的绝对路径只是当时的路径，换机后需要生成自己的 lock，不能直接拷贝 lock 然后忽略权重校验。

在本仓库根目录生成两个独立包。构建器将 31 个原 Target 文件按固定 SHA256 附带，两个包也包含同一份十二条输入；ZIP 已被 Git 忽略。

```bash
python scripts/build_legacy_target_bundle.py --output /tmp/minivllm-legacy-target-v1.zip
python scripts/build_legacy_target_bundle.py --formal --output /tmp/minivllm-legacy-target-formal-v1.zip
```

把两包上传到服务器的 `WORK_ROOT`。服务器需要 Python 3.11、匹配的实验环境、固定提交 DeepSpec 源码，以及四个本地模型。三个 Draft 文件夹需包含下载时生成的 `download_manifest.json`，不通过改造旧 lock 绕过缺失文件。权重恢复与校验入口另见本目录 `restore_model_weights.py`；本地文件缺失时它可能下载，使用前先查看 `--help`。本流程不运行旧 `start.sh`。

如果原十二条输入的本机 lock 仍在，直接设置 `SOURCE_LOCK=/实际路径/prepared/lock.json`。仅缺少本机 lock 时，先从本地文件准备，不做 GPU 测量：

```bash
export WORK_ROOT=/hy-tmp/minivllm-work
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHON="$WORK_ROOT/venv-deepspec/bin/python"
COMPARE=/实际仓库路径/experiments/speculative_compare
PREPARED="$WORK_ROOT/results/speculative-formal-local-$(date -u +%Y%m%d-%H%M%SZ)/prepared"
"$PYTHON" "$COMPARE/prepare.py" \
  --source "$WORK_ROOT/third_party/DeepSpec" \
  --target "$WORK_ROOT/models/Qwen3-8B" \
  --drafts-root "$WORK_ROOT/models" \
  --per-dataset 4 --seed 20261019 --output "$PREPARED"
export SOURCE_LOCK="$PREPARED/lock.json"
```

这一步要求分词结果与归档十二条输入相同，后续选择器会核对。如果不同，应检查 tokenizer、官方数据文件及配置，不能换一批输入后继续称为复现。

然后解包并先运行短回归。只有短回归退出状态为零，才接正式三轮；正式入口还会重新核对回归原始记录、源码和精度配置。

```bash
export WORK_ROOT=/hy-tmp/minivllm-work
PYTHON="$WORK_ROOT/venv-deepspec/bin/python"
"$PYTHON" -m zipfile -e "$WORK_ROOT/minivllm-legacy-target-v1.zip" "$WORK_ROOT"
"$PYTHON" -m zipfile -e "$WORK_ROOT/minivllm-legacy-target-formal-v1.zip" "$WORK_ROOT"
bash "$WORK_ROOT/minivllm-legacy-target-v1/run_legacy_target_regression.sh" && \
  bash "$WORK_ROOT/minivllm-legacy-target-formal-v1/run_legacy_target_formal.sh"
```

保持 SSH 会话或用 `tmux`/`nohup` 启动并保存输出；正式测量本次生成计时约 49 分钟，未来机器需另外预留核验、预热和加载时间。所有运行写入新目录，不覆盖旧结果。只想学习或复算已完成数据时，使用正式报告中的 CPU 离线分析命令，无需租 GPU 重跑。
