# AI Infra 模拟面试复盘：从状态到性能

这份材料依据最近九道源码级练习题整理，分数是练习中的反馈，不是正式面试成绩。前八题有反馈，第九题尚未作答。目标是先能独立手推请求、页、Token 与 GPU 数据，再用对照实验解释收益和边界；详细理论与源码上下文仍以[原 32 题](INTERVIEW.zh.md#questions)为准。

优先补 **第 6 题 Graph、 第 7 题 AWQ、 第 8 题性能分析**。每题先独立回答，再遮住参考答案画状态，最后回源码核对；不要把可能原因背成 Profiling 已证实的结论。

| 题 | 主题 | 练习反馈 | 深入原问答 |
|---|---|---:|---|
| [1](#mock01) | 项目架构 | 5.5/10 | [1](INTERVIEW.zh.md#q01)、[2](INTERVIEW.zh.md#q02) |
| [2](#mock02) | 请求与元数据 | 7.5/10 | [2](INTERVIEW.zh.md#q02)、[3](INTERVIEW.zh.md#q03) |
| [3](#mock03) | 跨页扩容 | 8.5/10 | [3](INTERVIEW.zh.md#q03)、[5](INTERVIEW.zh.md#q05) |
| [4](#mock04) | DSpark 消融 | 7.5/10 | [9](INTERVIEW.zh.md#q09)、[15](INTERVIEW.zh.md#q15) |
| [5](#mock05) | 1P1D 与 ITL | 6.5/10 | [24](INTERVIEW.zh.md#q24)、[25](INTERVIEW.zh.md#q25) |
| [6](#mock06) | DFlash＋Graph | 4.5/10 | [12](INTERVIEW.zh.md#q12)、[13](INTERVIEW.zh.md#q13)、[17](INTERVIEW.zh.md#q17)、[18](INTERVIEW.zh.md#q18) |
| [7](#mock07) | AWQ 与 Triton | 3/10 | [26](INTERVIEW.zh.md#q26) |
| [8](#mock08) | Chunk/Prefix 和性能定位 | 6.5/10 | [6](INTERVIEW.zh.md#q06)、[8](INTERVIEW.zh.md#q08)、[29](INTERVIEW.zh.md#q29) |
| [9](#mock09) | 双卡系统设计 | 尚未回答 | [20](INTERVIEW.zh.md#q20)、[23](INTERVIEW.zh.md#q23)、[32](INTERVIEW.zh.md#q32) |

<a id="mock01"></a>
## 1. 项目整体架构：先说清一条请求

**考查与纠错。** 你已经提到了调度和 Qwen Forward，缺口是 Tokenizer 和 Embedding 的顺序，以及模块间谁调用谁。Tokenizer 把文本编码为 Token ID；Embedding 在模型内部把 ID 映射为向量。原版已有 Paged KV、基础 Continuous Batching 和模型结构，本项目在其上扩展前缀、Chunk 等，并在独立实验路径接入投机、Graph、多卡和 AWQ。

**源码入口。** [LLMEngine.add_prompt()/step()](../src/myvllm/engine/llm_engine.py)、[Sequence](../src/myvllm/engine/sequence.py)、[Scheduler.schedule()/postprocess()](../src/myvllm/engine/scheduler.py)、[BlockManager](../src/myvllm/engine/block_manager.py)、[ModelRunner.run()](../src/myvllm/engine/model_runner.py)、[Qwen3](../src/myvllm/models/qwen3.py)。

**口述参考（60～90 秒）。** 我基于 MinivLLM 扩展 Qwen3-8B 推理。普通请求先经过 Tokenizer 得到 ID，封装成 Sequence 放入等待队列；Scheduler 选择本轮请求，BlockManager 分配或复用 KV 页，ModelRunner 构造 Token 和页元数据，Qwen3 内的 Embedding 再把 ID 映射成向量。模型前向写 KV、算 Logits，采样后由 postprocess 更新序列；完成就归还页引用，否则进入下一轮。项目在已有分页和基础调度上增加 Prefix、Chunk，并在独立执行路径验证 DFlash＋Graph、多卡和 AWQ。它们分别验收，我会按实验配置报告结果。

**手推。** 文本「你好」→假设 ID `[10,20]` →模型 Embedding 得到两行向量→Prefill 写两个位置的 KV→采样得到 30；Tokenizer 本身不会产生模型隐藏向量。Token ID 仅为示意。

<a id="mock02"></a>
## 2. Sequence、Scheduler、BlockManager、ModelRunner

**考查与纠错。** 把请求状态、资源管理和地址元数据分开。Block Table 记录逻辑页→物理页，不决定可分配容量；Slot Mapping 是当前新 KV 写入槽；Context Length 限制每个 Query 能看到的有效上下文。

**源码入口。** [Sequence.append_token()](../src/myvllm/engine/sequence.py)、[Scheduler.schedule()](../src/myvllm/engine/scheduler.py)、[BlockManager.allocate()/append()](../src/myvllm/engine/block_manager.py)、[prepare_prefill()/prepare_decode()](../src/myvllm/engine/model_runner.py)。

**口述参考（60～90 秒）。** Sequence 保存 Prompt、生成 Token、状态与页表；Scheduler 从队列选本轮请求，BlockManager 管理物理页引用和扩容；ModelRunner 将被选请求变成 GPU 前向的输入。Block Table 用于读取历史时映射物理页，Slot Mapping 指定本轮写入位置，Context Length 给每个 Query 指定有效长度。一个 100 Token Prompt 做完 Prefill 得到 A 后，Sequence 已有 101 个 Token，但 Target 有效 KV 只有 Prompt 的 100 个；下一轮以 A 为输入，写 A 的 KV，再预测 B。

**手推。** 写出状态 `(已知 Token=101, 已计算 KV=100, 下轮输入=A)`；Decode 处理 A 后，变成 `(已知 Token=102, 已计算 KV=101, 下轮输入=B)`。KV 有效长度不应直接等同 `len(Sequence)`。

<a id="mock03"></a>
## 3. Paged KV 动态扩容：别漏掉新生成 Token

**考查与纠错。** 已生成的 X 虽尚未计算自己的 KV，仍已加入 Sequence，计算页需求时长度是 9。`can_append()` 判断现有页是否足够或空闲页是否存在；`append()` 使用空闲队列头补页。

**源码入口。** [Sequence.num_blocks/append_token()](../src/myvllm/engine/sequence.py)、[BlockManager.can_append()/append()](../src/myvllm/engine/block_manager.py)、[prepare_decode()](../src/myvllm/engine/model_runner.py)。

**口述参考（60～90 秒）。** 假设页大小 4，Prompt 长 8，Prefill 已生成 X。此时序列长度为 9，页需求是 `ceil(9/4)=3`。原页表只有 `[5,7]` 两页，空闲队列是 `[2,9]`，因此 can_append 返回真，append 取队头 2，把页表扩成 `[5,7,2]`。下轮 X 位于逻辑位置 8，也就是第三逻辑页、页内偏移 0，所以写入物理 slot `2×4+0=8`。页号只说明物理位置，不代表 X 在 Prefill 时已写 KV。

**手推。** 将空闲队列改为空：现有页不足，`can_append()` 返回 False；是否抢占或报错取决于实际调度路径，不能把普通 Scheduler 和 Chunk 路径混写。

<a id="mock04"></a>
## 4. DSpark：接受率上升为何反而变慢

**考查与纠错。** Markov Head 利用前一个已选 Draft Token，对当前位置 Base Logits 加轻量低秩偏置，建立候选间顺序依赖；它不修改 Target 接受规则。Confidence Head 是辅助分数，不是 LM Softmax 概率，阈值只截短连续送验前缀，不保证候选正确。

**源码入口。** [benchmark.py](../experiments/speculative_compare/benchmark.py)、[LegacyTarget](../experiments/speculative_compare/legacy_target.py)；官方固定提交的 Markov/Confidence 实现入口见[第 15 题](INTERVIEW.zh.md#q15)。结果见[正式报告](../results/speculative_compare/legacy-target-formal-20261021/README.md)。

**口述参考（60～90 秒）。** 我在同一 DSpark Checkpoint 下对比推理开关。Fixed 平均送验 7 个候选，接受率 55.90%，每次验证实际输出 4.826 Token，三轮 1902 次验证，吞吐 63.225 Token/s。Confidence 0.8 平均只送 3.520 个，接受率升到 83.21%，但实际输出降到 3.873，需要 2370 次验证，吞吐降到 51.741 Token/s，配对下降 18.16%。完整 Draft Backbone 已先计算候选块，截短窗口不能等价于省掉同等比例的 Draft 计算。应同时看候选数、实际推进、轮次和耗时。

**手推。** 平均每轮推进减少时，相同输出预算需要更多轮，即使被送验的候选更准确，也可能因为额外 Draft/Target 调用而更慢。多请求动态验证长度是后续方向，当前没有实现或实测；提到 DeepInfer 时应说明它只是借鉴方向。

<a id="mock05"></a>
## 5. 1P1D：迁移哪些数据，改善哪个时延

**考查与纠错。** P 完整计算 Prompt KV 并产生首 Token A，A 自身 KV 尚未写入。`pack()` 收集的是全层 Prompt K/V Tensor；控制命令和页表等元数据由控制路径管理，不能说它们一起打包进 KV Tensor。D 依据自己的物理页表 `unpack()`；同步交接完成后 P 释放页，D 输入 A 开始 Decode。

**源码入口。** [Target.pack()/unpack()](../experiments/common/target.py)、[worker.main()](../experiments/distributed/worker.py) 的 `send_kv/recv_kv`、[run_load()](../experiments/distributed/runtime.py)。

**口述参考（60～90 秒）。** 两张卡各持一份完整模型。P 卡做 Prompt Prefill、产生首 Token，按本地页表把 36 层 Prompt K/V 打包，经 NCCL Send/Recv 发送；D 卡先分配自己的页，再按目的页表写入。两端确认同步交接后 P 释放页，D 以首 Token 开始 Decode，不重算 Prompt。Burst 中 DP2→1P1D 吞吐 30.303→30.360 Token/s，最大 ITL 2157.99→207.92 ms，但 P99 为 115.33→142.57 ms。这里体现极端停顿改善，不能说所有尾延迟改善；当前 Handoff 不是异步重叠。

**手推。** 源页表 `[5,7]` 与目的页表 `[2,9]` 不同，传输逻辑顺序的 KV 内容，D 把相同逻辑位置写到自己的页；控制消息传页号等元数据，不把它们当成 KV 张量。后续先看两卡时间线、pack/send/recv/unpack 和控制等待，再评估异步方案；尚无对应 Profiling 结论。[正式结果](../results/distributed/README.md)。

<a id="mock06"></a>
## 6. DFlash＋CUDA Graph：从 32 个 Query 到回滚

**考查与纠错。** B=2、Q=16 表示两请求各输入 Pending＋15 个候选，共 32 个 Query 位置。图捕获固定计算图、缓冲地址与形状；Token 和页元数据内容在每次 Replay 前更新。Replay 会重新执行 Kernel。图只覆盖 Target Forward；初始 Prefill、Draft、接受和回滚在图外。

**源码入口。** [build_query_metadata()/_fill()/capture_all()/run()](../experiments/dflash_graph/graph_executor.py)、[target_forward()](../experiments/dflash_graph/graph_runtime.py)、[accept_prefix()/commit()/PagePool.trim()](../experiments/dflash_graph/state.py)、[propose()](../experiments/dflash_graph/dflash_runtime.py)。

**口述参考（60～90 秒）。** 我为 B=1/2、Q=1/16 捕获四个 Target 桶。B=2、Q=16 时，两条请求各含一个 Pending 和 15 个候选；下一轮只要仍符合对应图的形状、页表容量及约束，即使 Token、物理页号和上下文长度不同，也能在原地址的 `input_ids`、`slot_mapping`、`context_lens`、`block_tables` 中写入新值再 Replay。返回的 Picks、Hidden 和 Features 要 clone，防止下次回放覆盖请求保留状态。验证拒绝后，commit 更新有效 `req.computed`，trim 释放不再需要的整页；末页中的无效内容通过有效长度隔离。Draft DynamicCache 另行裁剪。正式验证 Fallback=0 只针对受测 Target 验证路径。

**手推。** Pending=P，候选 A/B/C，Target 后继为 A/B/X/Y：接受 A、B，输出 A/B/X，Target 有效 KV 保留 P/A/B，X 是下轮 Pending。C 的 KV 如果在保留页内可以不清零，但下一轮 Context Length 不得让 Attention 读取它。Draft Cache 的裁剪与 Target `pool.trim()` 是两个过程；不要把回滚说成只改 Block Table。原实验相对 AR Graph 的吞吐为 2.99×/1.52×，条件见[正式报告](../results/dflash_graph/README.md)。

<a id="mock07"></a>
## 7. AWQ W4A16：形状、解包与定位方案

**考查与纠错。** Activation-aware 指用校准 Activation 帮助选择权重缩放；本项目加载官方量化 Checkpoint，没有自行完成校准。W4A16 是 4-bit 权重、16-bit 激活。真实收益是存储/峰值显存下降，所测吞吐下降。

**源码入口。** [AWQLinear.__init__()/forward()/_gemm()/dequantize_reference()](../experiments/awq/awq_linear.py)、[load_awq_model()](../experiments/awq/awq_loader.py)、[check_operators()](../experiments/awq/bench_awq.py)。

**口述参考（60～90 秒）。** 以输入维 K=256、输出维 N=64、Group Size=128 为例，八个 INT4 打包到一个 INT32，因而 qweight 为 `[256,8]`、qzeros 为 `[2,8]`；K 分两组，每组每输出通道一个 FP16 Scale，所以 scales 为 `[2,64]`。内核按 AutoAWQ 的交错顺序移位并与 15，取出 q 和 zero，按 `(q-zero)×scale` 在 Tile 内还原成 FP16 权重，`tl.dot` 用 FP32 累加。当前 BM/BN/BK 为 16/64/32，M=1 时可能利用不足，但无法单凭 Tile 判唯一瓶颈。权重及元数据存储下降 62.77%，short-b1 吞吐却从 12.111 降到 3.429 Token/s。

**手推与后续排查。** `64/8=8` 打包列、`256/128=2` 组；若 q=9、zero=7、scale=0.5，则权重为 1。先比较解包结果和浮点参考，再设计 M=1/4/16/64 的 AWQ 与 FP16 GEMM Microbenchmark；随后用 PyTorch Profiler、Nsight Systems/Compute 观察 Kernel 次数、时间、利用率与访存。这是待执行方案，不是已有 Profiling 测量。[正式结果](../results/awq/README.md)。

<a id="mock08"></a>
## 8. Chunk＋Prefix：怎样解释结果、怎样查慢

**考查与纠错。** Chunk 改变 Prefill 窗口和调度混合，单独开启不减少总 Prompt 计算；Prefix 复用 ready 的连续完整前缀页，减少重复 Prefill。吞吐下降本身不能证明 Scheduler CPU 开销是唯一瓶颈。

**源码入口。** [ChunkScheduler.schedule()](../src/myvllm/engine/chunk_scheduler.py)、[run_chunked()/chunked_step()](../src/myvllm/engine/chunk_runtime.py)、[FactorialScheduler.schedule()](../experiments/sharegpt/factorial_scheduler.py)、[paired_effects()](../experiments/sharegpt/factorial_protocol.py)。

**口述参考（60～90 秒）。** 最新 100 请求四组统一 Decode-first，各三轮。四组吞吐中位数依次为 Full 43.817、Chunk-only 43.576、Prefix-only 53.688、Chunk+Prefix 46.675 Token/s。按同轮配对比值中位数，Prefix 在无 Chunk/有 Chunk 时提升 16.21%/16.61%，Chunk 则分别下降 9.75%/8.88%。Full 也可能与 Decode 混合，关键区别是未缓存后缀能否拆成多个窗口。下一步应对照 Forward 次数、每步 Token 和 Mixed Steps，再拆分 CPU 元数据准备与 GPU Kernel 时间，并结合实际命中量判断原因。没有 Profiling 前只能给待验证假设。

**手推。** 预算 512、未缓存后缀 700：Full 必须推迟整段，Chunk 可以处理最多 256；是否更快取决于后续调度与执行成本。20 条受控重放请求不等于 20% Token 命中；本轮实际复用 13,568/78,889≈17.20%。旧三组 +20.23% 是单次不同协议，不能拼入新表。[新实验报告](../results/sharegpt/factorial-20261024/README.md)。

<a id="mock09"></a>
## 9. 双 A100 多请求系统设计：参考答案

**状态。** 本题尚未回答，没有原始回答、错误清单或练习评分。下面是设计参考，不代表现有功能已经组合上线。

**源码入口。** [run_load()](../experiments/distributed/runtime.py)、[Target.pack()/unpack()](../experiments/common/target.py)、[FactorialScheduler](../experiments/sharegpt/factorial_scheduler.py)、[Graph 执行器](../experiments/dflash_graph/graph_executor.py)。

**口述参考（60～90 秒）。** 我先确定吞吐、单请求延迟和尾部停顿目标，固定输入、到达率和资源预算。Qwen3-8B 能完整驻留单张 A100 时，吞吐优先先测两个独立副本的 DP2；如果最严重的出字停顿是主要目标，再与同步 KV Handoff 的 1P1D 在相同双卡条件下比较。TP2 可分摊显存，但本项目受测吞吐更低且有 1/48 完整生成分叉，不能默认加卡更快。再测前缀重用率和 Chunk 对预算、排队的影响；Graph 和投机要看具体入口支持的 B/Q 形状。上线前检查逐 Token、页泄漏、请求隔离、EOS、取消及延迟分布，并保留同资源基线。

**手推。** 相同 burst 条件下，先把 32 个请求分给 DP2 两副本，再与 1P1D 的 P/D 队列、全层 KV 迁移作比较；总吞吐接近不代表 Max/P99 ITL 相同。不同实验实现尚未统一集成，不能把 Prefix、Graph、投机和多卡的独立倍数相乘。取消请求与自然 EOS 等仍需专门验证。

## 轮换训练计划

1. 第一轮：依次讲 1～3 题，画出 Token ID→Embedding、Sequence/KV 长度和跨页 slot；不看文档复述。
2. 第二轮：集中练 6、7 题，各用一个例子说明四个 Graph 缓冲、接受回滚，以及 W4A16 的形状、解包和慢路径假设。
3. 第三轮：练 4、5、8 题，每次同时说基线、负载、正负结果与未测项；把“可能”与“证实”明确分开。
4. 第四轮：独立回答 9 题，先提出资源和验收条件，再选择方案；之后回到[32 题](INTERVIEW.zh.md#questions)抽查。每轮记录首次遗漏，重新回答后再扩展追问。
