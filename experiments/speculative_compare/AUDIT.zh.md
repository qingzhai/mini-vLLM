# EAGLE3 / DFlash / DSpark 审查与继续实验

审查日期：2026-10-19。本文区分本地已审查源码、用户交接信息、尚待导入的服务器记录。没有启动 GPU Benchmark，也没有改变历史结果。

**更新：下方第 1～7 节保留首次收集前的审查，最新证据核对及交付见第 8 节。**

## 1. 当前证据与版本差异

- 本地 `README.md`、`protocol.py`、`prepare.py`、`benchmark.py` 的原计划是独立 DeepSpec reference，batch=1，BF16 SDPA Eager；不是将三种 Draft 接入 Mini-vLLM。
- 原输入为 DeepSpec 自带 GSM8K / HumanEval / MT-Bench，各固定 4 条，共 12 条；输出 256 Token、temperature=0、忽略 EOS、三轮。Smoke 是各 1 条、输出 32 Token、单轮。
- 用户交接报告已完成 A100 Smoke，除 `dspark_conf_0.8` 的 `humaneval:19` 外均对齐 AR；这些原始报告尚未导入本地，不能把口述加速比当成已独立审查的正式数据。
- 服务器 `prepare.py` 已支持 `--drafts-root`，本地版本没有该参数。同步前必须收集实际脚本、锁文件、源码状态和诊断记录；不把本地旧文件直接上传覆盖。
- 现有 `start.sh` 会执行安装、准备和后续运行，`run_gpu.sh` 会重新准备模型。恢复阶段不执行这两个入口。已有环境、权重、锁文件均应复用。
- 已审查的官方 DeepSpec 为 `005e03b81cec38b7da6399833d609ee89a2587f2`；服务器是否仍是该版本、是否修改过源码，需要实际 Git 状态和文件指纹确认。

## 2. 消融变量是否成立

| 模式 | Markov | Confidence head | 实际验证候选长度 | 对照目的 |
|---|---|---|---|---|
| EAGLE3 | 自回归 Draft | 不适用 | 最多 7 | 方法对照；官方单链候选，不是候选树服务 |
| DFlash | 无 | 无 | 7 | 与另两个官方 checkpoint 对照 |
| DSpark No Markov | 关闭 | 关闭 | 7 | 同一个已训练 DSpark checkpoint 的推理时结构关闭 |
| DSpark Markov Only | 开启 | 关闭 | 7 | 相对 No Markov，观察顺序修正及其开销 |
| DSpark Fixed | 开启 | 开启、阈值 0 | 7 | 相对 Markov Only，仅观察预测 confidence 的额外开销 |
| DSpark Confidence 0.5/0.8 | 开启 | 开启 | 0～7，遇首个低置信位置截断 | 相对 Fixed，观察可变验证长度 |

源码：本地 `benchmark.py::worker()` 内的 `generate()`；官方 `deepspec/eval/dspark/draft_ops.py::build_dspark_proposal()`、`_confident_prefix_length()`；`deepspec/modeling/dspark/qwen3/modeling.py::sample_draft_tokens()`、`predict_confidence_step()`。

Markov Only 和 Fixed **不是两个候选策略**。在同一状态和确定性计算下，它们应得到相同候选、接受轨迹和输出；后者只多计算 confidence，阈值 0 不截断。两者耗时很接近不代表代码没切模式，应记录实际 head 调用次数和轨迹校验，而不是根据加速比判断。待服务器脚本回收后核实没有改成完全等价的两个分支。

Confidence 0.5/0.8 当前仅为逐位置阈值截断，**没有**论文的负载感知引擎性能模型、在线最优验证长度选择或多请求 Serving 调度。No Markov 是对训练好的 DSpark 推理时关闭模块，不等于重新训练的 DFlash。不同 checkpoint 的方法对照不能解释成严格控制训练条件的结构消融。

## 3. ShareGPT 与既有 Mini-vLLM 结果如何衔接

先纠正历史负载：`results/sharegpt/summary.json` 的平均 Prompt 是 **788.89 Token**，不是约 1200；100 请求中短/长各 50 条，长度为 256～512 / 1024～1536，80 条唯一 Prompt 加 20 条完整重放。每条输出 64 Token，忽略 EOS，最多同时驻留 8 条，所有请求同时入队。

建议分开保存两个协议，不把它们混入同一张总加速比表：

1. **原计划参考对照**：保留 GSM8K / HumanEval / MT-Bench、256 输出、三轮。作为最初协议和补充领域输入，先解决原 Smoke 的失败样本。没有任务答案评分，不能称为 GSM8K/HumanEval 准确率评估。
2. **拟新增 ShareGPT 性能对照，待确认后执行**：复用历史完整 100 条输入 Token，64 输出、忽略 EOS、相同 Qwen3 Target 和 Draft 常驻条件，batch=1 顺序处理、每请求新建 Target/Draft Cache。三轮并保留相同请求顺序或配对轮换模式。重复 Prompt 不跨请求复用 Prefix，避免把缓存命中混入算法收益。报告每请求时延及整批服务时间之和，不报告队列 TTFT 或并发 Serving 吞吐。

优先导入已有 `workload.json` 并校验 Token 指纹、tokenizer/config 指纹，不重新随机抽 100 条，也不重新套 chat template。如果历史工作负载文件没迁到新服务器，先从本地历史归档找原文件；缺失时显式记录，不能靠平均长度重造“同一负载”。`data/sharegpt-selection.json` 是选择元数据，不能单独替代完整输入 Token 文件。

历史 Mini-vLLM 是不同精度、算术、Attention、Graph 与并发调度路径。统一输入只能建立联系，不能直接拿两套运行时的 tokens/s 宣称纯算法加速或性能回退。

## 4. Greedy 异常：已知与未知

交接记录指出真实 Verification 中 5980 和 429 的 BF16 Logits 都为 31.625，而干净 AR Cache 下 5980 更高。`torch.argmax()` 在并列最大时选择首次出现的索引，因此 429 小于 5980，可以解释这一步接受 429 的直接机制。官方 `deepspec/utils/sampling.py::logits_to_probs()` 在 temperature=0 时正是先 argmax，再构造 one-hot。

这还不能证明 Cache 正确，也不能证明 bug 仅在 LM Head。把已舍入的 BF16 Logits `.float()` 不会恢复丢失的信息；固定 seed 或更改并列选择规则也不能修复两条路径的不同输入数值。

官方状态更新的静态审查：

- `base_evaluator.py::generate_decoding_sample()` 的 `start` 指向 Pending，进入 Verification 时 Target Cache 应有效缓存 `[0,start)`，Pending 尚未写入。
- `verify_draft_tokens()` 一次输入 Pending 加候选；Causal Target 会计算候选各位置的 KV。验证通过 `a` 个候选后，主循环裁到 `start+a+1`，输出多出一个新的 Pending。有效历史长度等于有效 Target KV 长度加一。
- `dspark/evaluator.py::_update()` 仅保留 `accepted_draft_tokens+1` 个 Target features。
- `draft_ops.py::forward_dspark_draft_block()` 将 Draft Cache 裁到提议前 `start`，下一轮用刚提交的 features 补入。Target Cache、Draft Cache、features 是三个状态，不能只检查其中一个长度。

静态阅读暂未发现明确的 off-by-one，但服务器实际代码与张量尚未核对，不能据此宣布排除实现错误。

## 5. 最小复现与分层诊断顺序

先收集已有诊断再增补，避免重复运行完整 Smoke。最小输入固定 `humaneval:19` 的已锁定 Token、32 输出、原始 seed、BF16 SDPA、阈值 0.8；保留 AR、DSpark 输出和每轮提议/接受/裁剪轨迹。所有新增运行建独立目录。

1. **核对复现条件**：模型完整哈希、DeepSpec 版本、实际运行脚本、精度与 SDPA 后端、实际 warmup/seed。对照已归档记录里的真实 seed，不能只固定另一个 seed。
2. **先验证状态不变量**：每层 Target/Draft Cache 长度、`start`/position_ids、已提交 Token、Pending、feature 长度，标记被拒绝后是否越界残留。不能仅凭一次 `get_seq_length()` 判断全层 Cache 正确。
3. **同历史、同 Query、不同 Cache 历史**：在分叉前保存实际 Cache；用完全相同 Token 历史构造逐 Token AR Cache；再用记录的多 Token 验证及 crop 分区重放出另一份 Cache。逐层比较 K/V 尺寸、有限性、max/mean 差异，并定位最早出现差异的位置和层。
4. **同一 Cache 的独立克隆、不同 Query 形状**：分别执行 Q=1 和 Q=2，前者输入 `[6337]`，后者 `[6337,429]`；两次不能共享会被原地更新的 Cache。比较第一个位置的 Logits 和 Hidden States。纯干净单次 Prefill 与逐 Token/分块历史不同，不能把这些条件一并改变后归因于 Q。
5. **局部精度实验**：保存相同最终归一化 Hidden State，分别进行原 BF16 Head 和真实 FP32 Head GEMM；必要时关闭 BF16 reduced-precision reduction、固定 SDPA 后端。逐项控制，记录改动。如果只有 FP32 Head 缩小分叉，它仍不能证明早期层/Cache 完全一致。
6. **修复与回归**：若是状态错误，修状态并保留复现断言；若是精度敏感性，选择所有方法共用的 Target 算术配置，重新配对 AR 并检查逐 Token 一致。任何受控配置都先小规模验证，不能承诺 FP32 Head 一定通过。原始 BF16 失败结果保留，不能升级为通过。

整模型 FP32 在 40GB 上加 Draft、KV、激活可能超显存，不能作为默认方案；先做局部、单位置 Head 实验。诊断带来的采样、clone、同步和重算不计为正式性能数据。

只有 32 Token 复现恢复一致不等于 256 Token 正式负载或所有阈值都正确。按“失败样本 → 原 3 条 Smoke 全模式 → 待定正式协议”的顺序扩大。不能删样本、移除阈值 0.8 或放松输出一致条件来取得合格加速比。

## 6. 指标与尚缺实现

当前 `benchmark.py` 保存完整输出、端到端生成时间、候选长度、接受长度、Target 调用数、接受率、有效输出/验证轮和峰值 allocated 显存。现有 `output_tokens_per_s` **包括 Prefill、Draft 初始化和 Decode**，不是纯 Decode Tokens/s。

拟增补（服务器版本核对后实施）：

- 模型加载、分词、输入上卡在计时外；请求 Prefill、Draft 初始化和接受/裁剪均纳入端到端时间。
- TTFT 从请求实际开始计算到首 Token 可用，须声明使用同步观测还是 CUDA Event。batch=1 没有服务器排队时延。
- Decode wall time 从首 Token 后到固定输出完成；Decode Tokens/s 用剩余的 N−1 个 Token 除以上述时间。端到端 Tokens/s 仍用 N / 总生成时间。分界处同步会改变运行开销，应将带细分计时的诊断与主性能计时分开或统一对全部模式启用。
- 记录每请求 Latency、三轮每轮总时长/吞吐、配对 Speedup、逐请求分布。100 请求三轮不构成高精度尾延迟估计。
- 每轮 proposed、verified accepted、有效输出推进分别记录；最终多生成的 Token 截断量照旧保留。Acceptance Ratio = 验证通过候选 / 实际验证候选；有效 Output Tokens per Verification = (最终输出 N−1) / 验证次数。两个指标分子不同。
- Proposal Length 与 Accepted Draft Length 的均值、分布及按位置接受率；置信度缩短验证不一定节省 Draft backbone 的完整 7 位置计算，须以真实调用为准。
- 当前 `proposal_lengths` 是实际送入验证的候选长度；应额外记录 Draft 完整预测的候选数，不能将这两个长度都叫作同一个“生成长度”。
- 记录 Markov/confidence 实际调用次数及 Markov Only/Fixed 轨迹是否一致，防止误设模式；始终保留相同模型权重驻留，显存不能与“关闭计算”混淆。
- 另开短的 profile 运行统计 Draft/Target CUDA Event 时间；不在所有正式测量中逐步 `synchronize()`。GPU Event 时间与 CPU 采样/提交开销不能相加冒充完整端到端时间。

目前缺少服务端真实版本、原始 Smoke/诊断和历史 ShareGPT 完整输入；没有实施新的精度方案、完整 GPU 回归或正式长测，没有新的正式加速比。

## 7. 成本与执行门槛

只读归档不加载模型、不安装、不下载，不需要 GPU Benchmark。导入记录后，先依据实际 Smoke 的 `total_s` 估算运行成本，再给出精确任务矩阵。

原 12 条、256 输出、三轮矩阵为 360 次请求执行（含三个独立 AR 对照）；新增 ShareGPT 100 条、64 输出、三轮矩阵为 3000 次请求执行。不能因为输出更短，就把新增负载看作与原计划相同的成本。另需模型加载与预热时间；在缺少原始逐请求耗时前不报可靠分钟数。

长测须在正确性回归通过、负载/精度/计时协议确认后才启动。若迁移精度或增补 ShareGPT，会建立新锁文件和新结果目录，保留旧目录与原异常。后续结果只能支持已经测得的参考路径，不自动支持 Mini-vLLM Serving 或原 DFlash＋Graph 的结果。

## 8. 归档导入后的实证核对与下一步

已读取 `deepspec-audit-20261019-080548Z.zip`：244 个条目，manifest 没有 Missing/Skipped；收集条目的 SHA256 与归档实际内容一致，所有文件收集时均稳定。服务器 DeepSpec Git 工作树干净，92 个源码文件与 lock 一致。导入时的 `benchmark.py`、`protocol.py` 与服务器逐字节相同，已合并服务器的 `prepare.py::--drafts-root`，没有替换服务器源码或执行安装器。

已归档 [Smoke 原始数据](../../results/speculative_compare/smoke-20261019/README.md)。所有 worker 完成，只有 `dspark_conf_0.8` 在 `humaneval:19` 索引 26 分叉；跨 worker AR 无分叉。三条 Prompt 长度为 100、114、44，平均 86 Token。所有方法所有模式的实际计时合计约 27.07 s（不含预热/模型加载/准备），这只反映 Smoke 的计算量，不直接外推长上下文或 256 输出的成本。

Markov Only 与 Fixed 三条输入的完整输出、提议长度、接受长度完全一致，符合实现预期。Confidence 0.5 相对 Fixed 吞吐约低 2.05%；其候选验证总数减少，不代表端到端性能一定提高。Confidence 0.8 未通过，保留结果但不认领其 2.062× 为合格收益。

**修正第 4 节中交接结论的证据强度**：旧 `DIAG_REAL_R5` 日志实际写 `argmax:5980`，同一行两个候选同为 31.625，输出却为 429；归档没有生成该字段的临时诊断代码，也没有交接中“干净 AR Cache Q=1/Q=2”的原始张量/脚本。因此目前可确认分叉及该日志矛盾，不能把那个字段视为真正 argmax 的实证。Top-k 首项在并列时不代表 argmax，需直接记录 `torch.argmax`、并列集合与官方 one-hot 概率。新的诊断不会改变并列选择策略。

新增 `diagnose_greedy.py` / `run_diagnostic.sh`，固定 32 输出，默认原 BF16 reduced-reduction 设置：

1. 核验已有源码、工作负载及权重哈希，不重新准备或下载。保留原 warmup，以及失败样本前的请求/模式顺序；先获得没有 hook 的实际输出并与旧记录比较。
2. 对该样本单独挂内存中的诊断 wrapper，调用原 `verify_draft_tokens()` 和生成循环；追踪历史提交、Pending、各层 Target/Draft KV 及 features 的长度关系，比较带/不带诊断的输出和接受轨迹。
3. 第 5 轮独立保存 Cache；比较同 Cache Q=1/完整 Q、同 Query 下实际 Cache/逐 Token 重建 Cache/完整验证与 crop 重放 Cache。重放包含最终被拒绝的候选计算，不能只重放接受 Token 的拼接。
4. 在相同最终 Hidden State 上比较 BF16 与真实 FP32 Head 计算，保存小型观察张量及逐层差异；这些计算不改变此次生成，也不作为性能测量。
5. `DIAGNOSTIC COMPLETE` 只是采集完成，必须先检查 `REPRODUCED HISTORICAL FAILURE`、`TRACE MATCHES UNTRACED`，再解释控制矩阵。此轮不自动转入长测，不宣称修复已通过。

本地已跑诊断的 CPU 张量测试，以及使用微型 CPU Target/Cache 和审查过的官方 verification/crop 循环的完整控制流检查；它们不能代替服务器上的 Qwen GPU 复现。新脚本没有修改官方 DeepSpec、旧 benchmark 或旧结果。

实现接口另核对了 Transformers 5.10.2 的固定版本源码：[DynamicCache / DynamicLayer](https://github.com/huggingface/transformers/blob/v5.10.2/src/transformers/cache_utils.py) 使用各层 `keys` / `values` 并由 `crop()` 截取有效历史；[Qwen3ForCausalLM.forward](https://github.com/huggingface/transformers/blob/v5.10.2/src/transformers/models/qwen3/modeling_qwen3.py) 在不计算 loss 时不会自动把 LM Head Logits 转为 FP32。这解释了为什么需要重新进行 FP32 GEMM，而非仅对现成 BF16 Logits 做 `.float()`。

采集没有找到历史 ShareGPT 完整工作负载文件，本机原先提及的历史 ZIP 也已不在原 Downloads 路径。当前 `data/sharegpt-selection.json` 只有请求选择/哈希元数据，没有输入 Token；不能直接生成“与历史完全相同”的 100 条新负载。新增 ShareGPT 需先恢复旧完整 workload，或在独立新协议下显式重建并核对其历史指纹，不能擅自替换原计划的三类输入。

## 9. 真实 Cache/Query 控制结果与精度回归候选

已核对返回的 `dspark-greedy-diagnostic-records.zip`，其中三份 Python 源码与发出的诊断包一致，lock/workload 与 Smoke 相同。结果见 [诊断归档](../../results/speculative_compare/diagnostic-20261019/README.md)。原失败、相同诊断轨迹、相同 AR 公共历史三项都已通过复现核对。

- 真实 argmax 和官方 one-hot 概率 argmax 均为 429，旧临时日志的 `argmax:5980` 已被纠正。
- Target 36 层 KV 与按完整投机轨迹重放的 KV 逐元素一致；该 Query 的 Hidden States、Logits 也逐元素一致。逐 Token 重建历史的 KV 与现场存在数值差异。同路径重放不能完全排除共同的实现因素，但本例未观察到长度/位置错误或随机污染。
- 实际 Cache 的 Q=1 与 Q=2 都将 5980/429 变成 BF16 并列；逐 Token AR Cache 的两个 Query 形状均选择 5980。仅修改当前 Q 不能消除该历史路径的分叉。
- 实际 Q=2 同一 Hidden State 的 FP32 Head 计算为 5980=31.63715744、429=31.57572174，均被 BF16 Head 输出为 31.625。所有五个 Cache/Query 控制中的局部 FP32 Head 都选 5980。

因此新增候选配置是 **BF16 Target 主干＋真实 FP32 Target Head**，仍用 BF16 Draft。`precision.py::configure_target_head()` 在内存中将已加载的 Head 权重转为 FP32，输入也转 FP32，关闭 Head 的 autocast，直接返回 FP32 GEMM 输出；不写磁盘，不改阈值或接受规则，不将 BF16 Logits `.float()` 冒充 FP32 计算。

在第 8 节诊断之后，`benchmark.py` 增加显式 `--target-head checkpoint|fp32`、`--full-bf16-reduction`。默认仍为原 checkpoint Head；本轮候选只设置前者 fp32，保留原 reduced-reduction 设置。所有三个 worker 在任何预热/AR/投机调用前统一应用配置，记录头部权重、计算、输出 dtype、实际 reduction/TF32 开关及参数字节变化。汇总若发现各 worker 或记录的精度不一致，不允许判定为合格对照。

在当前 untied Qwen3-8B 中，这种替换增加的模型参数存储约 1.159 GiB；这不是峰值显存的承诺，实际峰值仍由各 worker 记录。Draft 权重始终保持常驻，未以删除模块制造显存收益。

`run_precision_regression.sh` 复用旧 lock，只运行原三条输入、32 输出、单轮、全部模式（保留 Confidence 0.8）。它使用独立目录，不执行任何准备/下载/安装，也不自动启动 256 Token 或 ShareGPT 长测。即使通过，也只证明该配置在这个短回归中的一致性；256 输出、更多输入及正式性能仍需另测。旧 BF16 失败结果和加速比不覆盖、不混用。

当前已完成局部原因诊断、候选实现与 CPU 控制流/张量检查，**整段 GPU 精度回归尚待执行**。后续按实际回归结果决定是否需要进一步控制主干归约精度，不能提前宣布全局修复。

## 10. 短回归失败后，按用户要求完成正式测量

FP32 Head 短回归返回后，七种投机模式在 `humaneval:19` 索引 15 与 AR 分叉；运行、精度、跨 worker AR 检查均无错误，数据已归档。用户明确要求直接长测，不再增加前置检查，故沿用同一配置完成原十二条输入、256 输出、三轮正式测量，没有放松一致性条件或剔除失败样本。

[正式报告](../../results/speculative_compare/formal-20261019/README.md) 包含 360 次执行、92,160 个输出 Token，累计生成计时 37.85 分钟。所有模式完整、无运行或精度错误、AR 跨进程/轮一致；但相对 AR 全序列一致性未通过。返回的 harness 源码与发出的正式包逐字节一致，官方源码与 lock 的 92 个指纹一致。

相对 AR 的 EAGLE3 / DFlash / DSpark Markov Only 吞吐观测比分别为 2.545× / 3.129× / 3.448×，不能认领为已验收的无损加速。另一方面，五种不截短投机模式的完整输出彼此在 36 组配对中完全一致；同一个 DSpark checkpoint 的 No Markov → Markov Only 可独立观察到同输出吞吐比 1.886×。这项内部消融结论与相对 AR 失败分别报告。

Confidence 0.5/0.8 相对 Fixed 的配对吞吐比中位数为 0.99532 / 0.83138，未获吞吐收益；它们与 Fixed 也只有 30/36、24/36 组完整输出一致。0.8 接受率虽高，实际有效推进更短、验证轮数更多，不能据其接受率宣称更快或任务质量更好。当前没有新分叉的逐点证明、任务评分、纯 Decode 计时或 Serving 测量。

本轮整理完成后不自动追加 GPU 排查或重测；原始失败、代码快照和全部逐请求记录保留。
