# EAGLE3 / DFlash / DSpark：DeepSpec 参考对照

状态：**原 Mini-vLLM Target 适配的正式对照已通过。** [十二条、三轮正式结果](../../results/speculative_compare/legacy-target-formal-20261021/README.md) 共 360 次请求执行、92,160 输出 Token，七种投机模式均 36/36 与配对 AR 一致。EAGLE3 / DFlash / DSpark Markov Only 相对 AR Eager 为 2.934× / 3.479× / 3.898×；Markov 开关带来 84.68% 配对吞吐收益，而 Confidence 0.5/0.8 相对 Fixed 下降 1.97%/18.16%。[适配及运行入口](LEGACY_TARGET.zh.md) 仍为串行 Batch=1，无 Graph、无普通 Serving 调度器。原 HF BF16、FP32 Head、HF FP16 失败和后续回归结果均保留；原 DFlash＋Graph 正式结果保持不变。

## 当前结论与历史记录

当前可使用的性能结论以 [原 Target 正式报告](../../results/speculative_compare/legacy-target-formal-20261021/README.md) 为准。复现入口见 [LEGACY_TARGET.zh.md](LEGACY_TARGET.zh.md)，离线分析用 `analyze_ablation.py`；不需要重新运行 GPU 即可查看逐请求一致性、候选接受与配对速度。

| 阶段 | 状态 | 可以得出什么 |
|---|---|---|
| [HF BF16 Smoke](../../results/speculative_compare/smoke-20261019/README.md) | Confidence 0.8 的 HumanEval 输出分叉 | 官方接口与 GPU 能运行；不作为正式性能结论 |
| [原始异常诊断](../../results/speculative_compare/diagnostic-20261019/README.md) | 原分叉复现，真实 BF16 Head 出现并列 | 同一 Hidden 的 FP32 Head 消除该次并列；不能外推全部分叉根因 |
| [BF16 主干＋FP32 Head 短回归](../../results/speculative_compare/fp32head-regression-20261019/README.md) / [旧正式测量](../../results/speculative_compare/formal-20261019/README.md) | 仍有完整生成分叉 | 保留全部性能和输出，未获得对 AR 的合格加速结论 |
| [HF FP16 短回归](../../results/speculative_compare/fp16-regression-20261021/README.md) | 20/21 一致 | 只改精度仍未全部通过，不能包装为已修复 |
| [原 Target 短回归](../../results/speculative_compare/legacy-target-regression-20261021/README.md) | 21/21 一致 | 原三条输入、同一数值与 Cache 适配通过 |
| [原 Target 正式对照](../../results/speculative_compare/legacy-target-formal-20261021/README.md) | 252/252 一致，跨进程/跨轮 AR 一致 | 本次方法对照与 DSpark 推理消融通过 |

原 Target 路径使用 FP16 主干、真实 FP32 Head、固定形状算术和逐 Query PagedAttention，Draft 保持 BF16 MATH。改动是组合配置，不能归因成某一个 BF16 问题，也没有修补一个已确诊的官方验证索引错误。[逐请求分叉分析](../../results/speculative_compare/divergence-20261019/README.md) 保留旧路径的首处分叉地图。

历史审查和失败定位协议见 [AUDIT.zh.md](AUDIT.zh.md)。下方固定条件、安装与首次启动部分描述的是 **历史 HF 参考路径**；不要把 `start.sh`、`run_gpu.sh` 或 `run_formal.sh` 当作当前通过路径的入口。已有环境、源码和本地权重时无需重复下载、安装或执行旧失败实验。

## 实验要回答什么

1. 同一 Qwen3-8B Target 上，自回归 Draft（EAGLE3）、并行 Draft（DFlash）、带轻量顺序校正的 Draft（DSpark）各需要多少验证轮，最终是否更快？
2. DSpark 使用同一 checkpoint 时，Markov 修正和 confidence head 各有什么开销与收益，置信度截断能否减少验证工作？

第一项是**方法对照**；第二项是**推理策略消融**。三种 checkpoint 的训练和容量不同，不是严格匹配训练条件的结构消融。EAGLE3 使用官方参考实现的单链候选生成与验证，没有实现生产引擎中的候选树搜索。

## 历史 HF 参考路径的固定条件

| 项目 | 设定 |
|---|---|
| 官方源码 | [DeepSpec](https://github.com/deepseek-ai/DeepSpec)，commit `005e03b81cec38b7da6399833d609ee89a2587f2` |
| Target | `Qwen/Qwen3-8B`，BF16，SDPA，非 thinking |
| Draft | `deepseek-ai/eagle3_qwen3_8b_ttt7`、`deepseek-ai/dflash_qwen3_8b_block7`、`deepseek-ai/dspark_qwen3_8b_block7` |
| 候选预算 | 每轮最多 7 个候选，不含 Pending；完整验证输入最多 8 个 Token |
| 并发 | batch=1，请求顺序执行；三种 Draft 分进程加载 |
| GPU | 单张 A100 PCIe 40GB；实际硬件、驱动和运行环境随各阶段记录归档 |
| 输入 | 官方仓库自带 GSM8K / HumanEval / MT-Bench，各随机固定 4 条合格输入；仅取第一轮用户消息 |
| 输入筛选 | Qwen chat template 后长度 1～1536 Token；不截断 Prompt；固定 seed=20261019，不放回抽样 |
| 输出 | temperature=0，每条实际生成 256 Token，忽略 EOS；不是 PAD 补齐 |
| 重复 | 所有模式先各预热一次 32 Token，正式测量 3 轮；轮内配对模式轮换顺序 |
| 图与编译 | 不主动启用 CUDA Graph 或 `torch.compile`；使用官方 SDPA Eager 评测路径 |

官方权重的不可变 revision 在准备阶段解析并保存；可以复用已有本地 Qwen3-8B，届时不声称知道它的历史 revision，而是保存全部权重及元数据的 SHA256。每次正式启动重新核验这些内容。三种方法共用锁定后的同一份 Target 和输入 Token。

DeepSpec 使用 Transformers 5.10.2，本项目旧环境使用 4.57.3。安装到独立的 `venv-deepspec`，不升级 `venv-core`，不在这个环境中安装本仓库的 `pyproject.toml` 依赖。首次依赖解析记录完整 `pip freeze`。

安装默认只使用清华 PyPI 镜像，避免为普通依赖访问跨境 PyTorch 源。可用 `DEEPSPEC_PACKAGE_INDEX` 指定其他兼容镜像；所有依赖版本要求保持原样。安装完成后检查 PyTorch 2.9.1、实际 CUDA runtime 12.8 和 GPU 可用性，不满足则停止。旧运行包同时使用 PyPI 与 PyTorch 额外源：`unsafe-best-match` 只能解决首源遮蔽，无法解决 PyTorch 源超时，需要更新安装命令。

若云镜像已经提供匹配的 PyTorch 2.9.1+cu128 和 Triton 3.5.1，可指定镜像原始解释器 `PYTHON_BIN` 并设置 `DEEPSPEC_REUSE_SYSTEM_TORCH=1`。安装器用 `--system-site-packages --without-pip` 让实验环境读取预装包，再由 pip 安装其余固定版本依赖；不使用 uv 重新解析这些继承包。新增包安装到实验环境中，不修改基础环境。该环境能看到基础环境的其他包，不属于完全隔离；完整 `pip freeze`、Torch 实际路径、Python 基础路径和 `pyvenv.cfg` 随报告记录。此模式不适用于无法确认版本的其他镜像。

## 对照矩阵

| 进程 | 模式 | 意义 |
|---|---|---|
| EAGLE3 | AR / EAGLE3 | 同一 Target、同一驻留 Draft 下的配对比较 |
| DFlash | AR / DFlash | 官方 DeepSpec DFlash；不是原先的 z-lab b16 |
| DSpark | DSpark no Markov | 推理时关闭 Markov 和 confidence head，完整验证；保留同一 backbone 权重 |
| DSpark | DSpark Markov only | 开启 Markov，关闭 confidence head，完整验证 |
| DSpark | AR / DSpark fixed | 阈值 0，仍计算 confidence head，验证完整候选 |
| DSpark | DSpark confidence 0.5 / 0.8 | 保留同一 Markov、confidence 权重，仅改变置信度截断阈值 |

阈值是预先选定的探索配置，不根据测试结果挑出一个阈值再称为普遍最佳。这里使用官方 `_confident_prefix_length()` 的逐位置 sigmoid 阈值截断，**不是论文中结合负载与引擎吞吐模型的完整调度器**。

no Markov → Markov only 比较已训练 checkpoint 在推理时启用顺序修正的变化；Markov only → fixed 比较计算 confidence head 但不截断的开销；fixed → 0.5/0.8 比较验证长度策略。关闭 Markov 时必须同时关闭依赖它的 confidence head。每次生成后恢复原模块，保留模块的 Python 引用使全部权重始终驻留 GPU；没有通过移除权重制造显存收益。关闭模块属于推理时消融，不能将其冒充重新训练的 DFlash，不能推断“训练时移除该结构”的效果。

每种 Draft 进程内都有自己的 AR 基线，AR 时 Draft 保持驻留，模型加载在计时之外。不同进程的 AR 时延、输出和驻留参数量均保留，避免把环境漂移或不同 Draft 大小隐藏在一个基线数字里。报告也检查不同进程、不同轮次的 AR 输出是否一致。

## AutoDL 操作

先把这份更新后的项目或本目录上传到服务器。无需覆盖已有结果，模型和环境继续放在仓库外的 `WORK_ROOT`。

```bash
export WORK_ROOT=/root/autodl-tmp/minivllm-work
export PATH="$HOME/.local/bin:$PATH"
source "$WORK_ROOT/venv-core/bin/activate"

# 在服务器上本目录的实际位置执行；只安装独立环境、下载官方源码，不占用 GPU 计算。
cd /实际项目路径/experiments/speculative_compare
bash setup.sh

# 后台执行并立即保存日志；首次会下载三个 Draft。
mkdir -p "$WORK_ROOT/results"
LAUNCH_LOG="$WORK_ROOT/results/speculative-launch-$(date -u +%Y%m%d-%H%M%SZ).log"
nohup bash run_gpu.sh > "$LAUNCH_LOG" 2>&1 < /dev/null &
echo "PID: $!"
echo "LOG: $LAUNCH_LOG"
tail -f "$LAUNCH_LOG"
```

看到最终 `ARCHIVE:` 和 `QUALIFIED GREEDY COMPARISON:` 后，按 Ctrl+C 退出 `tail -f`；这只停止看日志。不要在运行中重复执行同一任务。

默认下载端点为 `https://hf-mirror.com`；可通过 `HF_ENDPOINT` 改为官方端点。源码通过 GitHub 下载；网络失败会退出并保留下载缓存，修复网络后再运行。下载与准备不计入推理时间。模型文件、环境和缓存需要额外磁盘空间，首次建议预留至少 50 GB 可用空间；三种 Draft 不同时驻留 GPU。

要先做最小 GPU 接口检查，可运行 `PER_DATASET=1 REPEATS=1 OUTPUT_TOKENS=32 bash run_gpu.sh`。这会产生独立目录，属于小规模检查，不能与默认三轮正式测量混为一组。准备和计时分别执行的完整参数见 `python prepare.py --help`、`python benchmark.py --help`。

## 恒源云新机启动

新机使用 Python 3.11 镜像，数据盘路径 `/hy-tmp`，工作目录为 `/hy-tmp/minivllm-work`。先确认实际挂载和空间，再上传启动包；新建机器无需 `venv-core`。模型、独立环境、下载缓存和临时文件均放在数据盘。

合并启动包应同时包含本目录和 `minivllm-deepspec-offline-v1/` 离线源码安装器。解压至 `WORK_ROOT`，完成旧模型迁移后，在本目录运行 `bash start.sh`。它先检查至少 35 GiB 可用空间、安装并核验固定版本离线源码，随后建立独立环境、小规模 GPU 检查通过后执行正式三轮。GitHub 不可达不影响离线源码安装；依赖和未缓存的模型仍需联网。

旧机可迁移 `models/Qwen3-8B` 与完整的 `cache/huggingface/hub/models--deepseek-ai--eagle3_qwen3_8b_ttt7` 缓存目录。Hub 缓存包含 `blobs`、`refs`、`snapshots`，只复制符号链接而不复制其目标文件会导致缓存失效。不要迁移整个旧虚拟环境；新机重建后保存实际 `pip freeze`。

## 结果怎么解释

`results/speculative-compare-<时间>-<随机值>/measured/` 保存：

- `lock.json`、`workload.json`：模型 revision 或本地哈希、官方源码哈希、数据行号、输入 Token。
- `*-records.jsonl`：每次生成后立即落盘，保存完整输出 Token、验证轮数、候选长度、接受长度和最终截断量。
- `*-environment.json`、`pip-freeze.txt`、`nvidia-smi.txt`：参数容量、精度、软件与硬件。
- `summary.csv`、`report.json`：三轮统计、配对正确性及跨进程 AR 一致性。
- `harness-source/`、`deepspec-source/`：实际评测脚本、官方源码、数据文件和许可。
- `measured-records.zip`：在 `measured/` 同级，含以上记录，不含模型或环境；启动日志在外层 `results` 中。

每轮吞吐 = 全部样本实际输出 Token 数 / 全部样本完整生成耗时之和，再报告三轮中位数。计时包括初始 Prefill、Draft 初始化、Draft Forward、Target 验证、接受与 Cache 更新、必要的主机同步；排除模型加载、Tokenization、输入上卡、日志及输出落盘。没有测 TTFT、ITL 或并发服务吞吐。

`median_paired_speedup` 是各轮“该方法吞吐 / 同进程 AR 吞吐”的中位数，不能与本项目旧 FP16＋Graph 的加速比相乘。不同中位数直接相除与配对比值的中位数也可能略有差异。

`verified_acceptance_ratio` 的分子是验证通过的候选总数，分母是实际送入验证的候选总数；最后一轮可能超出剩余输出预算，超出部分在输出中截断，原始轨迹保留该额外工作。`output_tokens_per_verify` 按实际输出减去初始 Prefill 产生的首 Token 计算，避免把超额接受当成有效吞吐。

所有模式均调用 Target 的 `output_hidden_states=True`；AR 使用与官方投机循环相同的 temperature=0 概率构造及采样工具。此为参考路径的受控比较，不代表一个优化到极限的 AR 服务实现。

`QUALIFIED GREEDY COMPARISON: True` 要求全部任务完成、各模式逐 Token 对齐配对 AR、跨进程和跨轮 AR 一致。否则退出码为 2，仍打包报告；数值分叉与运行失败分别保留，不能删掉不一致样本后宣传速度。BF16 的不同 Query 形状可能改变接近并列的 argmax；报告仅定位首次分叉，**没有自动证明分叉属于数值误差而非状态错误**。

忽略 EOS 是固定工作量实验，不是自然对话停止策略，也没有执行 GSM8K 答案评分或 HumanEval 测试用例。12 条输入用于首轮探索，不足以声称复现论文全部任务结论。

## 源码关系和后续工作

本目录 `benchmark.py` 的 HF 参考分支调用官方 `build_models()`；原 Target 分支加载官方 Draft 并接 `LegacyTarget`。两者均调用官方 `generate_one_sample()`，显式初始化评测对象以跳过 `BaseEvaluator.__init__()` 的多进程样本分发与官方整套数据循环。Target 适配仅重绑定基类的 Target Cache 工厂，保留官方的 EAGLE3 单层约束、提议与接受函数体，官方源文件不修改。

重点追踪官方路径：

- `deepspec/eval/eagle3/evaluator.py`：顺序提议、Draft Cache crop/extend。
- `deepspec/eval/dspark/evaluator.py`：DFlash/DSpark 共用的并行提议接口。
- `deepspec/eval/dspark/draft_ops.py`：候选采样、置信度前缀截断。
- `deepspec/eval/base_evaluator.py`：Target 验证、拒绝采样、Cache crop。
- `deepspec/modeling/dspark/qwen3/modeling.py`：Markov head 和 confidence head 的调用。

原 Mini-vLLM Target 串行适配已有正式 GPU 正确性证据。后续候选包括并发批量验证、候选长度扫描和 Graph 分桶；当前没有这些新增结果，也没有 EAGLE3 树验证、在线负载感知调度、Draft 训练或严格控制训练条件的结构消融。不要将本轮配对 AR Eager 加速比与历史 Graph 结果相乘。

CPU 检查：`python verify_cpu.py`。这些检查只验证抽样、指纹、统计和失败判定，不证明 GPU 接口兼容或模型正确性。
