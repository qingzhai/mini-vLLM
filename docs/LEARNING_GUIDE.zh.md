# 从代码建立项目画像

这个项目是在开源 MinivLLM 基础上做的 **Qwen3 推理系统复现、扩展与受控实验**。它把请求调度、分页 KV、投机验证、CUDA Graph 和双卡执行连接起来，并记录哪些配置输出一致、哪些优化在实测中有效。它适合作为推理系统或 AI Infra 实习项目；现有仓库不是把所有功能同时打开的生产服务。

先形成下面四个认识，再读具体实现：

- 模型负责计算下一个 token 的分布；推理系统负责决定本轮处理哪些请求、哪些 token，KV 写到哪里，以及请求何时完成。
- Prefill 计算已有输入的 KV 并得到首个输出；Decode 用已有 KV 推进生成。减少重复计算、CPU 启动开销和请求间干扰是三类不同优化。
- 仓库有四类实验执行路径。它们共用模型和部分算子，但调度状态、页所有权与 Graph 入口不同；新增三方法对照复用原 Target Eager，不进入普通 Serving 调度器。
- 正确性先于吞吐：完整生成对照、同历史数值检查、页回收与实际执行计数分别提供不同证据。

整体证据见 [项目总结](PROJECT_SUMMARY.zh.md)，实际启动入口见 [scripts/run.py](../scripts/run.py#L25)。

## 第一次阅读按这个顺序

先看 [README 的请求流程](../README.md)，然后按下表读函数。**第一遍只读前六项，能手推状态变化后，再看多卡和量化。** 不必先读模型初始化、进程启动、权重下载或 Benchmark 的报表代码；第一次也不需要启动 GPU。

| 顺序 | 打开的源码与函数 | 读完要能回答 |
|---|---|---|
| 1 | [sequence.py](../src/myvllm/engine/sequence.py) 的 `Sequence`；[llm_engine.py](../src/myvllm/engine/llm_engine.py) 的 `add_prompt()`、`step()`、`generate()` | Prompt 长度为何固定？输出追加在哪？谁决定继续或结束？ |
| 2 | [block_manager.py](../src/myvllm/engine/block_manager.py) 的 `allocate()`、`mark_computed()`、`deallocate()`，再回头读 `_cached_prefix()`；[scheduler.py](../src/myvllm/engine/scheduler.py) 的 `schedule()`、`postprocess()` | 逻辑页对应哪张物理页？何时能共享？归还引用为何不等于清空内容？ |
| 3 | [chunk_scheduler.py](../src/myvllm/engine/chunk_scheduler.py) 的 `schedule()`；[chunk_runtime.py](../src/myvllm/engine/chunk_runtime.py) 的两个执行函数；[model_runner.py](../src/myvllm/engine/model_runner.py) 的 `prepare_prefill()`、`prepare_decode()`、`run()` | 本轮算哪些 Token？KV 写哪里？哪些请求允许采样？ |
| 4 | [qwen3.py](../src/myvllm/models/qwen3.py) 的各层 `forward()` 与 `compute_logits()` | ID 如何变成特征，再得到下一 Token 的词表分数？KV 在 Attention 的哪一环使用？ |
| 5 | [state.py](../experiments/dflash_graph/state.py) 的 `Request`、`Window`、`accept_prefix()`、`commit()`；[dflash_runtime.py](../experiments/dflash_graph/dflash_runtime.py) 的 `generate()`、`propose()` | Pending 与候选如何对齐？接受后哪些输出、KV、特征有效？拒绝后哪些页归还？ |
| 6 | [graph_runtime.py](../experiments/dflash_graph/graph_runtime.py) 的 `target_forward()`；[graph_executor.py](../experiments/dflash_graph/graph_executor.py) 的 `build_query_metadata()`、`_fill()`、`capture_all()`、`run()` | 固定的是地址还是 Token？四类元数据怎样更新？为何返回值必须 clone？ |
| 7 | [distributed/runtime.py](../experiments/distributed/runtime.py) 的 `run_load()`，先读内部 `admit()`、`schedule()`、`receive()`；[worker.py](../experiments/distributed/worker.py) 的 `main()`；[target.py](../experiments/common/target.py) 的 `forward()`、`pack()`、`unpack()` | DP 请求归哪卡？1P1D 何时传 KV？D 怎样按自己的物理页读取它？ |
| 8 | [linear.py](../src/myvllm/layers/linear.py) 的四种 Parallel Linear；[awq_loader.py](../experiments/awq/awq_loader.py) 的 `load_awq_model()`；[awq_linear.py](../experiments/awq/awq_linear.py) 的 `forward()`、`_gemm()` | TP 切哪一维，何时通信？INT4 怎样变为参与矩阵乘的权重？ |

上表这 18 个文件的核心部分已在代码行上一行加了简短中文注释。其中请求状态、块管理、调度、DFlash/Graph 和 AWQ 的小文件逐行覆盖；引擎、ModelRunner、Qwen、多卡控制器及共享 Target 则覆盖表中要读的关键函数，不要求先通读整个文件。空行、已有注释和多行字符串正文不额外插入注释。

每读一个函数，先做三件事：找输入，找被修改的状态，再找返回值。遇到模型或 CUDA 调用，第一遍先把它看成“接收哪些张量、写哪些 KV、返回哪些特征”，不要立刻跳进所有底层实现。

建议在纸上连续手推下面三段，确认自己理解了调用顺序：

1. Prompt `[10,20,30]` 经 Prefill 预测 40：序列长度为 4，有效 KV 只有前三个位置；下一轮输入 40 才写它自己的 KV。
2. Pending=P、候选 A/B/C、Target 后继预测 A/B/X/Y：新增输出 A/B/X，保留 KV P/A/B，X 留作下一轮 Pending。完整推演见 [面试问答第 12 题](INTERVIEW.zh.md)。
3. 页长 256、页表 `[4,9]`：逻辑位置 255 写 slot 1279，位置 256 写 slot 2304。换成 Graph 后地址缓冲不变，这些数值仍须每轮更新。

读完前六项后，再用 [项目总结](PROJECT_SUMMARY.zh.md) 对照性能数字，用 [面试材料](INTERVIEW.zh.md) 练口述。不要从结果中的加速倍数反推自己已经理解实现。

## 四类路径要分清

| 路径 | 调用关系 | 管理什么 | 使用场景与边界 |
|---|---|---|---|
| 普通引擎 | `LLMEngine.step` → `Scheduler`/`ChunkScheduler` → `ModelRunner` → Qwen3 | `Sequence`、waiting/running、`BlockManager` | 读基础推理、Prefix、Chunk、普通 Decode Graph；ShareGPT 正式入口额外挂统一算术参考配置 |
| DFlash＋Target Graph | benchmark → `GraphDFlashRuntime.generate` → `DFlashRuntime.generate` → `TargetGraphExecutor.run` | 独立 `Request`、`PagePool`、Draft cache、验证窗口 | 绕过普通 `engine.step`，独占 Target/KV；Graph 只覆盖 Target 前向，不组合 Prefix/Chunk/TP |
| 分布式实验 | benchmark → `Session`/`run_load` → torchrun worker → 公共 `Target.forward` | 控制器请求队列、各 worker 页表、在途任务和 KV handoff | 同一控制器比较 Single/TP2/DP2/1P1D；本轮关闭 Graph/DFlash/AWQ/Prefix/Chunk |
| EAGLE3/DFlash/DSpark 对照 | `benchmark._worker` → 官方 `generate_one_sample` → `LegacyTarget.forward` → 原 Target Eager | 独立 Target PagePool、官方 Draft Cache、验证和接受轨迹 | 串行 Batch=1、最多七候选；复用旧 Target 算术，不启用 Graph，不是并发 Serving |

源码入口：[普通引擎](../src/myvllm/engine/llm_engine.py#L83)、[Graph 适配器](../experiments/dflash_graph/graph_runtime.py#L13)、[分布式控制器](../experiments/distributed/runtime.py#L172)。公共 [Target](../experiments/common/target.py#L124) 是 Qwen3-8B、FP16 主干＋FP32 Head 的受控执行器，不能当成所有模型通用的服务 API。

普通引擎示意：

```text
prompt → tokenizer → Sequence → waiting
                          ↓ schedule()
                  本轮 token 窗口 + 页表
                          ↓ ModelRunner
        token IDs / slot_mapping / context_lens / block_tables
                          ↓ Qwen3 + Attention
                    KV 写入 → logits → 采样
                          ↓ postprocess()
             追加 token → 继续 running / 完成并归还页
```

## 八步阅读顺序

新增的 ShareGPT 四组消融沿用普通引擎，在实验层使用 [FactorialScheduler.schedule()](../experiments/sharegpt/factorial_scheduler.py)，四组统一 Decode-first，只有 Chunk/Prefix 两个开关不同。读完基础调度和 Chunk 后，对照它怎样在关闭 Chunk 时整段接纳或推迟未缓存 Prefill，避免隐式切块；再读 [settings()/trial_order()/paired_effects()](../experiments/sharegpt/factorial_protocol.py) 与[正式结果](../results/sharegpt/factorial-20261024/README.md)。Prefix-only 已完成，不再属于待补实验；历史三组入口仍保留。

按下面顺序读，不必从头通读全部源码。每一步用一个小例子讲明白，再进入下一步。

### 1. 跟踪一个请求的一生

读 [LLMEngine.add_prompt/generate/step](../src/myvllm/engine/llm_engine.py#L127)、[Sequence](../src/myvllm/engine/sequence.py#L24)、[Scheduler.schedule/postprocess](../src/myvllm/engine/scheduler.py#L59)，最后看 [Qwen3Attention.forward](../src/myvllm/models/qwen3.py#L74) 和 [compute_logits](../src/myvllm/models/qwen3.py#L405)。

读后应能解释：输入如何入队，本轮是谁挑出的，`num_prompt_tokens` 与生成长度有何区别，为什么模型前向和调度器是两个角色，完成条件如何触发页归还。

小练习：用 2 个短 prompt、各输出 4 tokens，在临时副本打印 `seq_id/status/num_tokens/num_cached_tokens/block_table`。画出 Prefill 后、一次 Decode 后、完成后三个状态；不要把打印留在正式计时代码里。基础模型需要 GPU，小对象状态跟踪可以在 CPU 做。

注意：普通 `Scheduler` 的 `max_num_sequences` 控制本轮选中数量；`ChunkScheduler` 和分布式控制器另外明确限制全局驻留请求。不能把所有路径里的同名 batch 配置理解成同一种约束。

### 2. 先弄清 KV 地址，再看 Prefix/LRU

读 [BlockManager.compute_hash/_cached_prefix/allocate/mark_computed/deallocate](../src/myvllm/engine/block_manager.py#L45)、[prepare_prefill](../src/myvllm/engine/model_runner.py#L271)、[store_kvcache](../src/myvllm/layers/attention.py#L67) 和 [paged_attention_decode](../src/myvllm/layers/attention.py#L397)。

读后应能手算：逻辑位置 `p` 对应 `block_table[p // block_size] * block_size + p % block_size`；页表把逻辑连续序列映射到不连续物理页。KV 字节数按 `2 × 层数 × tokens × KV heads × head_dim × dtype字节` 计算；Qwen3-8B FP16 的 1024 tokens KV 为 144 MiB。

Prefix 哈希包含上一块哈希和本块 token；命中还检查 token 内容与 `ready`。先 pin 命中页，再分配缺失页，避免命中页在本次分配里被驱逐。只复用已计算完成的整页，至少保留一个未缓存 token，最后可写页保持私有。空闲队列按释放/复用次序实现 LRU 管理，活跃页由引用计数保护；没有任意共享可写尾页的 copy-on-write 实现。

小练习：把 `block_size` 设成 4，用 3 个物理页手推两条共享前 4 个 token 的请求；依次分配、发布、释放、再申请。记录 `ref_count/ready/free_block_ids`，再改掉前一块的 token，说明后一块内容即使相同也不应沿用旧前缀身份。这些管理逻辑可以在 CPU 阅读和构造对象验证。

### 3. 理解 Chunk 为什么不能“每块都采样”

读 [ChunkScheduler.schedule](../src/myvllm/engine/chunk_scheduler.py#L45)、[run_chunked/chunked_step](../src/myvllm/engine/chunk_runtime.py#L11)，对照 [prepare_prefill](../src/myvllm/engine/model_runner.py#L271)。

读后应能解释：`_scheduled_start/end` 是本轮窗口；Prefill 还没处理完时，只更新 KV 和 `num_computed_tokens`，不追加输出 token。只有最后一个 Prefill 块和 Decode 请求进入采样列表。部分 Prefill 返回 `-1` 占位，调度层断言不会误采样；`mark_computed(..., end)` 只发布真正写完的整页。

小练习：用 600-token prompt、chunk=256 画出 `[0,256)`、`[256,512)`、`[512,600)`；插入一个 Decode 请求，标注每轮 token budget、是否 mixed、采样行号及可复用整页数。再把 chunk 改为 128，对照 trace 中部分块和混合步骤数量；不要假设块更小就更快。

这条路径验证了连续批处理和混合调度。正式 ShareGPT 结果里，Chunk 单独比完整 Prefill 慢，必须同时看执行次数与输出指标，见 [ShareGPT 结果](../results/sharegpt/README.md)。

### 4. 学会读正确性与指标，而不是只看 PASS

读 [canonical_runtime.install_model/install_runner_hooks](../experiments/sharegpt/canonical_runtime.py#L46)、[fixed_rows_forward](../experiments/common/fixed_shape.py#L11)、[request_metrics](../experiments/sharegpt/metrics.py#L15)、[分布式 metrics/compare_tokens](../experiments/distributed/common.py#L58)。

读后应能解释：为什么不同 batch/query 形状可能改变浮点归约和近似相等 logits 的 argmax；为什么固定算术、FP32 Head、统一 Attention 路径用于建立受控参考，而不是生产最快配置；为什么同历史 logits 检查不能替代完整自由生成一致。

小练习：设请求到达为 0 s，首 token 在 2 s，最后 token 在 5 s，共 4 个输出，手算 TTFT=2 s、E2E=5 s、TPOT=1 s。再在一份结果副本改掉一个输出 token，确认比较函数能报告第一处分叉。不能通过放宽断言把分叉结果变成“完全一致”。

历史原 Flash Prefill 在完整生成中出现分叉，统一参考路径通过并不等于旧路径已修复。AWQ 也只有算子参考、有限值生成和页回收验证，没有模型质量评测。

### 5. 用一个 4-token 小例子讲清投机接受与回滚

先读纯 Python 的 [Request/Window/accept_prefix/commit](../experiments/dflash_graph/state.py#L63)，再读 [DFlashRuntime.propose/generate](../experiments/dflash_graph/dflash_runtime.py#L148)。

关键不变量是 `len(req.tokens) == req.computed + 1`：末尾保留一个已知但尚未写入 Target KV 的 token。Draft 的 16-token block 包含它和 15 个候选；Target 一次因果前向给出逐位置预测。`block[1:]` 与 `posterior[:-1]` 从头比较，到首个不一致处停止，接受连续前缀，再追加对应 Target token。

读后应能解释：为什么首 token 由初始 Target Prefill 得到；为什么收集 DFlash 特征时用层输出的 `x + residual`；为什么每个请求有自己的 Draft cache；为什么 Draft 在本版本逐请求执行，Target 验证则打包一个 batch。

小练习：待计算 token 为 10，`block=[10,20,30,40]`、`posterior=[20,99,88,77]`。手算只接受 20，再追加 99，推进 2；说明被拒绝的后续 KV 即使物理内存里还在，也必须由逻辑长度屏蔽。跟踪 [PagePool.trim](../experiments/dflash_graph/state.py#L38) 只归还超出需要的整页，和下一轮 Draft cache/特征长度如何衔接。

本实现验收 greedy 固定输出；没有实现随机采样的概率修正，也没有与 Prefix/Chunk/TP 的联合调度。

### 6. 看懂 Graph 的“不变”和“每轮都变”

读 [TargetGraphExecutor.build_query_metadata](../experiments/dflash_graph/graph_executor.py#L33)、[_fill/_forward/capture_all/run](../experiments/dflash_graph/graph_executor.py#L198)，以及 [GraphDFlashRuntime.target_forward](../experiments/dflash_graph/graph_runtime.py#L47)。

不变的是捕获图用到的 tensor 地址、形状和计算结构；变化的是这些缓冲区中的 token IDs、slot mapping、context lengths、block tables。每次 replay 前四类输入全部更新。捕获 batch 1/2 × query 1/16 四张图；初始 Prefill 图外，其他形状可以 eager fallback，正式受控负载要求验证前向 fallback=0。

读后应能解释：Python hook 在捕获时执行，replay 重放已记录的 tensor 运算，不会重新跑 Python hook；返回的 features/hidden/picks 要 clone，避免下一次 replay 写同一地址后污染请求保留状态。

小练习：阅读 [Graph CPU 合约测试](../experiments/dflash_graph/verify_graph_executor_cpu.py#L124)，给页边界场景增加 `start=255, query=16`，检查 slot/table/context 全部更新；保留上一次 features，再跑一次，确认保留值不变。CPU mock 可检查逻辑与所有权，真实 Graph 正确性仍需 GPU 同历史对照。

结果讲述优先比较 [AR Graph 与 DFlash Graph](../results/dflash_graph/README.md)，单请求 2.99×、batch=2 三请求 1.52×，不要只挑 AR eager 当基线。

### 7. 把 TP、DP 和实际 KV 迁移对应到代码

读 [ModelGroup/Target](../experiments/common/target.py#L24)、[严格分片加载](../experiments/common/tp_support.py#L8)、[Session/run_load](../experiments/distributed/runtime.py#L20)、[worker 的 send_kv/recv_kv](../experiments/distributed/worker.py#L172) 和 [Target.pack/unpack](../experiments/common/target.py#L263)。量化作为补充看 [AWQLinear/_gemm](../experiments/awq/awq_linear.py#L20)。

读后应能解释：TP 是一个模型跨两卡做分片计算；DP 是两个完整副本各接请求；本项目 1P1D 两卡也各有完整权重，P 做 Prefill、D 做 Decode，需要迁移全部 prompt KV。DP/PD 的模型组为单 rank，防止完整副本误用 WORLD 被分片；KV 传输仍走 WORLD。

小练习：手推一个 257-token prompt，故意让源、目的页号不同，检查 pack→send/recv→unpack 后的逻辑 KV；参考 [CPU pack/remap 测试](../experiments/distributed/verify_tensors_cpu.py#L11)。再改变两个 DP 副本驻留数，说明新请求路由到谁。没有 GPU 也能先阅读这些测试，GPU 结果见 [分布式摘要](../results/distributed/README.md)。

本项目没有 PP。PD 首 token 由 P 卡发出，迁移完成且收到两端 ACK 后释放 P 页，D 再处理这个待计算 token。handoff 是同步协议，不要描述为异步传输与计算重叠。AWQ 的 tile 内解包没有保证速度提升，所测实现更慢，见 [量化结果](../results/awq/README.md)。

### 8. 比较三种 Draft，再解释 DSpark 的正负收益

先读 [三方法正式结果](../results/speculative_compare/legacy-target-formal-20261021/README.md)，记住条件：十二条唯一输入、每条 256 输出、三轮、单张 A100、串行请求。252/252 是七种投机模式各 36 次完整生成一致，不是 252 条不同 Prompt。相对的是各 Draft 进程自己测的 AR Eager；旧 DFlash＋Graph 的 2.99×/1.52× 相对 AR Graph，两组不能混用。

代码按下面四层阅读：

1. [protocol.py](../experiments/speculative_compare/protocol.py) 的 `MODES` 与 `summarize()`，再读 [benchmark.py](../experiments/speculative_compare/benchmark.py) 的 `_worker()`：确认哪些开关发生变化、AR 是否共用精度与驻留 Draft、同轮如何配对、计时包括什么。
2. [legacy_target.py](../experiments/speculative_compare/legacy_target.py) 的 `LegacyTarget.forward()`、`LegacyCache.crop()`、`release_cache()`：看官方接口如何映射到已有分页 Target。Target Cache 与 Draft DynamicCache 分开，只改变前者的工厂；每个 Query 按绝对位置限制可见 KV，异常也释放页。
3. 固定提交 DeepSpec 的 `deepspec/eval/eagle3/evaluator.py::_propose()`、`deepspec/eval/dspark/draft_ops.py::forward_dspark_draft_block()/build_dspark_proposal()`：EAGLE3 单链逐步提议，DFlash 并行猜块，DSpark 先并行算块特征，再做轻量 Markov 顺序修正。源码由运行包归档，官方地址与提交见 [来源说明](../THIRD_PARTY_NOTICES.md)。
4. [analyze_ablation.py](../experiments/speculative_compare/analyze_ablation.py) 与 [analyze_divergence.py](../experiments/speculative_compare/analyze_divergence.py)：分别复算配对消融和定位首个不同 Token。先看分母、执行次数、实际输出配额，再解释加速比。

小练习：DSpark Fixed 每次送验 7 个候选，接受率约 56%，平均实际输出 4.83；Confidence 0.8 只送验约 3.52 个，接受率升至 83%，实际输出却降到 3.87，需要更多验证轮，配对吞吐下降约 18.16%。把“接受率高”与“每轮推进多”分开讲。Confidence 在完整块特征和候选产生后截断，不能说它少执行了相同比例的 Draft Transformer。

同一 checkpoint 的 No Markov→Markov Only 配对吞吐提高 84.68%；Markov Only 与 Fixed 的输出和接受轨迹相同，但后者仍计算 Confidence Head，只是不截断。它们不是两组独立结构收益，也不能把 No Markov 称为重新训练的 DFlash。三种官方 checkpoint 训练和容量不同，属于方法对照；同一 DSpark 的开关才是推理策略消融。

最后读 [精度与 Cache 排查](../experiments/speculative_compare/LEGACY_TARGET.zh.md)：真实 BF16 Logits 并列只解释一个具体现场；仅 FP32 Head 和 HF FP16 都未完全通过，原 Target 组合路径才通过本轮正式负载。不要背成“BF16 一律不正确，换 FP16 就解决”。这轮没有任务答案评分、树验证、Graph、Serving 并发或逐阶段耗时 Profiling。

## 投入重点与验收方式

优先把 **请求生命周期 → 页地址/所有权 → 投机接受/回滚 → Graph 动态元数据** 连成一条可手推的链。这是最有区分度、也最容易被追问的部分。随后解释 DP/1P1D 的对照与负面结果，再深入 Triton 内核；无需先补所有论文、其他模型或 PP。

建议完成三个可展示动作：手推一次跨页投机回滚；在源码指出每轮更新的四类 Graph 输入；用一个统一结果表同时讲清吞吐提升、显存下降和没有提升的实验。能讲清这三件事，比背功能名更接近实习面试要求。

若要动手改代码，在临时分支或副本中做上面的练习。仓库源码 manifest 会检测改动；确认改动后使用 [update_manifest.py](../scripts/update_manifest.py) 更新。CPU 检查入口是 `python scripts/check_cpu.py`；安装 CPU PyTorch 后可加 `--tensor`，本地 RPC 检查可加 `--transport`。这些不等同重新完成 GPU 验证，具体启动参数见 [复现说明](REPRODUCE.zh.md)。

先前阅读注释的整理没有重新执行 GPU Benchmark。历史实验的原始 SHA256 不改写：[source-annotations.json](../data/source-annotations.json) 记录新增注释的行号，校验器仅移除这些注释后恢复原文件，再与原始哈希比较；当前文件仍由发布 Manifest 校验。任意原代码行被修改时仍会失败。新增三方法数据来自单独归档的 GPU 正式测量，本次文档整理与离线复算不构成新一轮 GPU 验收。

继续练口头表达可看 [面试问答与讲述稿](INTERVIEW.zh.md)。
