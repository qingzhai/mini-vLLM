# 项目面试讲述与技术问答

这份材料用来练习口头讲述，并能沿源码和结果继续展开。**性能主模型是 Qwen3-8B；项目基于开源 MinivLLM，使用官方 Qwen 权重、EAGLE3/DFlash/DSpark Draft 和 AWQ Checkpoint。** Qwen3-0.6B 只做过较早的基础功能检查，不与 8B 的正式性能评测并列。

阅读时区分三种说法：**理论原理**解释一种方法为什么可能有效；**项目实际实现**说明本仓库做了什么；**实验观察**只描述给定配置和负载测到的结果。下文的示意 Token、页号、地址和时间轴用于手推，不是新增实验数据。完整口径见 [项目总结](PROJECT_SUMMARY.zh.md)，源码阅读顺序见 [学习指南](LEARNING_GUIDE.zh.md)。

## 1. 30 秒项目介绍

> 我基于 MinivLLM 扩展 Qwen3-8B 的缓存与调度，重点做 DFlash 批量验证和 CUDA Graph，单请求吞吐约为 AR Graph 的三倍。还统一对照了 EAGLE-3、DFlash、DSpark：Markov 有收益，Confidence 截断未加速；多卡也有独立评测。它是受控实验原型。

这段只口述一个近似倍数：对应旧 DFlash＋Graph 单请求 2.987×，不对应新三方法对照。详细三请求、多卡和消融数字留给追问。证据：[原 DFlash＋Graph](../results/dflash_graph/README.md)、[新三方法对照](../results/speculative_compare/legacy-target-formal-20261021/README.md)。

## 2. 2 分钟项目介绍

> 我基于开源 MinivLLM，围绕 Qwen3-8B 做推理执行和对照实验。模型和 Draft 使用官方权重，我主要完成调度扩展、执行适配、状态管理与验证。
>
> 缓存部分扩展 Prefix、LRU 和 Chunked Prefill，处理共享页、部分 Prefill 不采样和请求补位。ShareGPT 派生 100 请求中，同一 Chunk 配置开启 Prefix，吞吐提高约 20%；Chunk 单独却更慢。
>
> 主要工程优化是 DFlash 加 Target CUDA Graph：实现批量验证、连续前缀接受、KV 回滚及固定缓冲。相对已有 Graph 的 AR，固定单请求和三请求总吞吐约为 2.99 倍、1.52 倍，三轮输出和接受轨迹一致。
>
> 另一路把官方 EAGLE-3、DFlash、DSpark 接到同一 Target，做十二条输入、三轮对照。相对配对 AR Eager，吞吐约为 2.93、3.48、3.90 倍，最后一个对应 Markov Only，252 组投机输出一致。同一 DSpark 开启 Markov 提高约 85%；Confidence 截断提高了接受率，却降低每轮推进和总体吞吐。这组是 Batch=1、无 Graph，不能和前一组混算。
>
> 多卡里 DP2 提高系统总吞吐，1P1D 搬全层 KV、减少最大出字停顿；TP2 和 AWQ 只获得显存或存储收益，没有加速。数值排查保留了 Logits 与 Cache 重放证据，最终组合配置在受测负载通过，尚未确定所有分叉的唯一原因。各路径独立验收，项目仍是实验原型。

先讲缓存和 Graph 的工程工作，再讲三方法对照与消融；不要把固定算术的 AR Eager 称为生产最优基线，也不要把三种官方 checkpoint 比较称为匹配训练条件的结构消融。

## 3. 5 分钟项目介绍

> 我希望能从一条 Prompt 跟到 GPU 前向和 KV 地址，所以以开源 MinivLLM 为基础，围绕 Qwen3-8B 做缓存与调度扩展、DFlash、Target Graph 和双卡对照。模型、预量化 Checkpoint 和 Draft 都来自官方。
>
> 一条普通请求先分词成为 Sequence，进入等待队列。调度器根据请求数、Token Budget 和 KV 容量选窗口；ModelRunner 准备 Token、写缓存用的 Slot Mapping、读缓存用的 Block Table 和有效长度。Prefill 写入输入的 KV，最后位置预测首个输出。之后每轮输入刚生成但还没计算过的 Token，读取历史 KV，再预测下一个。结束时归还页引用，让等待请求补位。
>
> Prefix 避免重复算前缀：用包含历史的链式 Hash 找到已写完的整页，以引用计数保护共享页；引用归零先留缓存，覆盖物理页才驱逐。Chunk 把长 Prefill 拆小，让 Decode 穿插。600 个输入分成 256、256、88 三轮，前两轮只建 KV，最后才采样答案。100 请求对照里，Prefix 相对同一 Chunk 配置有效，但 Chunk 单独更慢，相对完整 Prefill 的平均时延也未全面改善。
>
> DFlash 是主要优化。轻量 Draft 利用 Target 已算出的五层特征，对带 MASK 的块一次预测 15 个候选。多个请求的 Draft 在本项目仍逐个调用，Target 验证才合批。并行猜测不等于直接决定答案，最终由 Target 因果前向验证。
>
> 比如真实 Pending 是 P，Draft 猜 A、B、C。Target 输入 P、A、B、C，后继预测 A、B、X、Y。前两个正确，第三个错误，所以新输出 A、B、X，提交 KV 为 P、A、B，X 是下一轮 Pending。错误候选可能已写入物理缓存，但必须恢复有效长度，归还超出整页，屏蔽页内残留；Draft 临时块缓存和 Target 新特征也分别处理。
>
> CUDA Graph 减少 Target 前向的重复提交开销。B 是请求数，Q 是每请求本轮输入数，捕获 B 为 1/2、Q 为 1/16 的四种组合。地址和形状固定，输入内容每轮更新；输出 clone，避免下次 Replay 覆盖保留特征。Replay 仍算模型，不是复用旧答案。Draft、初始 Prefill 和接受回滚都在图外。
>
> AR 与 DFlash 各分 Eager、Graph，共四项对照，各测三轮。主要基线是 AR Graph：单请求 256 个输出，从 5.51 秒到 1.85 秒；三请求、驻留两个，共 768 输出，从 11.65 秒到 7.65 秒。24 次运行共 12,288 输出一致，八组同历史检查精确一致，正式验证没有 Fallback。这是自己的执行器对照，不是生产 vLLM 对比。
>
> 多卡比较 Single、TP2、DP2、1P1D。TP 切模型矩阵，DP 是两个完整副本分请求；1P1D 也是两份模型，GPU 0 Prefill、GPU 1 Decode，通过 NCCL 搬全部 36 层 Prompt KV，按目的页表重映射。P 生成首 Token，D 计算它自己的 KV，不重算 Prompt。DP2 总吞吐提高、每卡效率下降；1P1D 对 DP2 吞吐持平，最大出字间隔从约两秒到约 0.2 秒，P99 未全面改善。TP2 显存约减半但更慢，还有一条生成分叉；AWQ 也更慢，只有存储和显存收益。
>
> 后来我把官方 EAGLE-3、DFlash、DSpark 接到同一 Target，统一最多七个候选的预算，做十二条输入、三轮方法对照与 DSpark 推理消融。这里 Batch=1、无 Graph，相对配对 AR Eager 分别约 2.93、3.48、3.90 倍，最后一个对应 Markov Only，252 组投机输出一致。Markov 带来约 85% 吞吐提升；Confidence 0.8 的接受率从约 56% 升到 83%，每次验证实际输出却从 4.83 降到 3.87，所以整体更慢。
>
> 排查时，我抓到过 BF16 Head 将两个候选都算成 31.625 的现场；只换 FP32 Head、再换 FP16 都未完全通过。最后复用旧 Target 的固定形状、统一逐 Query Attention 和精度控制，完整正式对照才通过。我能证明这个组合在已测负载有效，尚不能把全部分叉归因于 BF16。各路径独立验收，没有全部联合启用；后续应先做 Profiling 与更多负载，再考虑 Draft 合批或异步 Handoff。

### 可展开的技术细节

正文不必背函数名，追问时沿这几条线展开：

| 对方想深挖什么 | 先手推什么 | 再打开什么 |
|---|---|---|
| 请求与页管理 | 600-token Prompt 分块、A 完成后 C 补位 | `ChunkScheduler.schedule()`、`chunked_step()`，见第 3～8 题 |
| 投机正确性 | P/A/B/C → A/B/X；三种状态分别处理 | `accept_prefix()`、`commit()`、`propose()`，见第 9～15 题 |
| Graph 的动态输入 | 地址不变、内容变化；跨 256-token 页边界 | `build_query_metadata()`、`_fill()`、`run()`，见第 16～19 题 |
| 多卡的价值 | 一个模型切两卡 vs 两份模型；257-token KV 重映射 | `run_load()`、`Target.pack()/unpack()`，见第 20～25 题 |
| 三种 Draft 与消融 | 区分方法对照和同 checkpoint 开关，手算接受率与推进长度 | 第 9、15、29～31 题；[三方法正式报告](../results/speculative_compare/legacy-target-formal-20261021/README.md) |
| 实验是否可信 | 明确基线、工作量、计时边界、次数和未通过项 | 第 26～32 题与文末正式数据表 |

## 4. 项目技术问答

共 32 题。先练每题的“30 秒回答”，再用例子解释状态变化，最后到源码核对。六组内容对应不同执行路径，不能把一种路径的页池、精度或 Graph 配置套到另一种路径。

### A. 项目背景与基础推理

#### 1. 这个项目与直接调用 vLLM 有什么区别？你具体做了什么？

**面试官问题：** 框架和模型都来自开源，你的工作在哪里？它是生产服务吗？

**30 秒回答：** 我基于 MinivLLM 的模型、算子和引擎结构做系统扩展与验证，重点是完整前缀页管理、Chunk 调度、DFlash 接受与回滚、Target Graph，以及双卡对照。我的价值是能把请求、Token、KV 所有权和执行路径连起来，并用明确基线说明收益与代价。它是可复现的实验原型，没有生产 HTTP 服务，也没有宣称替代生产 vLLM。

**深入解释：** 分页、基础 Continuous Batching、Qwen 模型和并行 Linear 等已有开源基础，不能全部说成独立创新。DFlash 算法和 Draft、AWQ 权重也有官方来源。本项目在这些基础上完成适配、状态管理、数值调试、对照和归档。面试中“我完成了集成与验证”和“我发明或训练了它”是两种完全不同的贡献。

**项目实际实现：** 普通引擎是 [LLMEngine.step()](../src/myvllm/engine/llm_engine.py) → Scheduler/ChunkScheduler → ModelRunner；旧 DFlash 是 [GraphDFlashRuntime](../experiments/dflash_graph/graph_runtime.py) → TargetGraphExecutor，独占 KV；多卡是 [Session/run_load()](../experiments/distributed/runtime.py) → worker → Target；新三方法是官方 DeepSpec 生成循环 → [LegacyTarget](../experiments/speculative_compare/legacy_target.py) → 原 Target Eager。来源见 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)，项目范围见 [PROJECT_SUMMARY](PROJECT_SUMMARY.zh.md)。

**具体例子：** ShareGPT 用 BF16 普通引擎检查 Prefix/Chunk；DFlash 四项对照用 FP16 Target 独立运行时；分布式实验关闭 Graph/DFlash/Prefix/Chunk。它们共用部分模型与工具，但不是给同一个统一服务依次叠加所有开关。

**常见追问与边界：** Qwen3-8B 是主要性能模型，0.6B 仅有早期基础功能检查。仓库整理后的目录做过 CPU 合约检查，没有重新跑 GPU；归档性能对应当时保存的源码。若问个人贡献，应按自己实际参与且能解释的工作回答，不把第三方实现或辅助工具工作全部说成独立手写。

#### 2. Prefill 与 Decode 有什么区别？为什么 KV 可以复用？

**面试官问题：** 首 Token 是哪一步得到的？每轮 Decode 是否重新算整个历史？

**30 秒回答：** Prefill 处理已有 Prompt，各位置通过因果 Attention 建立自己的 K/V，最后一个位置预测首 Token。普通 Decode 输入这个刚生成的 Token，算它的新 Q/K/V，读取历史 KV 后预测下一个。历史 Token 不变，在同一模型和兼容配置下它们的 K/V 可以复用，因此不必重算历史层前向；但每个新 Query 仍需要对历史 KV 做 Attention。

**深入解释：** KV Cache 缓存的是每层、每个已计算位置的 Key 和 Value，不是整段文本答案或下一轮 Logits。历史 K/V 不依赖未来 Token，因果性让缓存成立。Prefill 同时处理多个已有位置，普通 AR Decode 通常每请求处理一个位置；DFlash Target Verification 一次有多个 Query，也保持因果边界，不能把“非初始 Prefill”一概理解为一行输入。

**项目实际实现：** [LLMEngine.add_prompt()/step()](../src/myvllm/engine/llm_engine.py) 建立请求并执行；[ModelRunner.prepare_prefill()/prepare_decode()](../src/myvllm/engine/model_runner.py) 构造窗口和元数据；[Qwen3Attention.forward()、Qwen3ForCausalLM.compute_logits()](../src/myvllm/models/qwen3.py) 负责 QKV、Attention 与词表输出。正式参考路径的 Attention 适配另见 [canonical_runtime.install_model()](../experiments/sharegpt/canonical_runtime.py) 和 [install_arithmetic()](../experiments/common/target.py)。

**具体例子：** Prompt 为 `[10,20,30]`，Prefill 输入三项，写完它们的 KV，最后一行预测 40。此时序列已知 `[10,20,30,40]`，但有效 KV 只有前三项。下一轮输入 40，写它自己的 KV，读取前四个位置后预测 50。普通生成中末 Token 也是“已知但尚未计算”的状态，DFlash 将它显式称为 Pending。

**常见追问与边界：** 复用 KV 不等于 Attention 与历史长度无关。常见的“Prefill 偏计算、Decode 偏带宽”是特定规模下的性能倾向，不能代替本项目 Profiling。当前没有用 Profiling 证明每个阶段的唯一瓶颈。



### B. KV Cache、Prefix、Chunk 与 Continuous Batching

#### 3. Paged KV Cache 如何寻址，究竟节省了什么？

**面试官问题：** 一条请求的 KV 不连续，你怎么知道新 Token 写在哪里、历史 Token 又从哪里读？分页能把 KV 本身变小吗？

**30 秒回答：** 我把每条请求的逻辑 Token 序列和实际 GPU 缓存页分开。Block Table 记录逻辑页对应哪个物理页，Slot Mapping 指定本轮新 K/V 的写入位置；Attention 读取历史时根据页表和有效长度寻址。分页减少的是按最大长度预留连续空间的浪费，还支持整页前缀共享，不会减少每个有效 Token 的 KV 字节数。

**深入解释：** 对逻辑位置 `p`，地址关系是 `slot = block_table[p // block_size] × block_size + p % block_size`。写入路径先计算当前 Token 的 K/V，再按照 Slot Mapping 存入缓存；读取路径对每个历史位置查 Block Table，且只读取 Context Length 范围内的有效位置。Slot Mapping 解决“这轮写哪里”，Block Table 解决“历史在哪里”，二者不是同一个数组。

**项目实际实现：** [ModelRunner.prepare_prefill / prepare_decode](../src/myvllm/engine/model_runner.py) 生成这些元数据；[Attention.forward、store_kvcache_kernel、paged_attention_decode_kernel](../src/myvllm/layers/attention.py) 分别执行写入和分页读取；[Sequence.block_table](../src/myvllm/engine/sequence.py) 保存每条请求的映射。正式 ShareGPT 在 [canonical_runtime.install_model](../experiments/sharegpt/canonical_runtime.py) 中统一使用 per-query PagedAttention，不能把它说成原 Flash Prefill 路径。

**具体例子：** 假设页大小 4、页表为 `[7,2]`，逻辑位置 5 属于第二个逻辑页，对应物理页 2、页内偏移 1，所以写入 slot 9；逻辑位置 0 则写入 slot 28。物理页不必相邻。Qwen3-8B 有 36 层、8 个 KV Heads、Head Dim 128，FP16/BF16 都按 2 bytes 计算，KV 为 `2 × 36 × tokens × 8 × 128 × 2`：每 Token 144 KiB，1024 Tokens 为 144 MiB，不含权重和工作区。

**常见追问与边界：** 这里按 KV Heads 计算，不能误用 32 个 Query Heads。页内最后未填满的空间仍会浪费；页表与元数据也有开销。分页能力基于已有 MinivLLM 框架，不是本项目从零发明的机制。

#### 4. Prefix Caching 为什么需要链式块 Hash？

**面试官问题：** 两条请求中某块 Token 完全一样，能否直接共享这块 KV？Hash 是怎么找到 KV 的？

**30 秒回答：** 不能只比较当前块，因为因果 Attention 的隐藏状态还依赖此前全部上下文。项目把上一块 Hash 与本块 Token 一起计算 Hash，沿前缀顺序查找；命中后还检查页已经计算完成、Hash 和 Token 内容一致。Hash 只是定位已有物理缓存页的索引，不能从 Hash 还原 KV。

**深入解释：** 可以写成 `H0=Hash(Block0)`、`H1=Hash(H0,Block1)`、`H2=Hash(H1,Block2)`。这是把前文依赖传递到后续块标识中。只有从开头连续匹配的完整页才构成可复用前缀；中间首次缺失就停止，不能跳到后面的相似块。还必须区分“页已分配”和“页的完整 KV 已写完”：Chunk 尚未计算的页即使 Token 已知，也不能拿去复用。

**项目实际实现：** [BlockManager.compute_hash / _cached_prefix](../src/myvllm/engine/block_manager.py) 使用 8-byte BLAKE2b 摘要，逐页查 `hash_to_block_id`，验证 `ready`、Hash 和本块 Token；[mark_computed](../src/myvllm/engine/block_manager.py) 只发布已写完的完整页。`_cached_prefix()` 最多检查 `(len(seq)-1)//block_size` 页，至少留一个未缓存 Token 继续执行，并保持需要写入的尾页私有。

**具体例子：** 页大小设为 2，请求一是 `[X,Y,A,B]`，请求二是 `[U,V,A,B]`。最后块虽然都是 `[A,B]`，但它们分别在 `[X,Y]` 和 `[U,V]` 后面，K/V 可能不同。链式 Hash 会把不同前文带入第二块标识，所以不会因为 `[A,B]` 相同就共享。若请求三是 `[X,Y,A,B,C]`，已经就绪的前两整页可以复用，从位置 4 的 C 继续算。

**常见追问与边界：** 同一模型、位置编码及兼容运行配置是讨论复用的前提；项目没有跨模型缓存池。它没有支持任意可写尾页共享的 Copy-on-Write。发布页后保留的是 GPU 中已算好的 K/V，Hash 自身没有模型计算能力。

#### 5. Release、Cache Reuse 和 LRU Eviction 有什么区别？

**面试官问题：** 请求结束时你把 KV 页释放了，为什么下一条请求还能命中？LRU 会不会把正在使用的页驱逐？

**30 秒回答：** 释放的是请求对物理页的引用，不一定清空页内容。引用计数降为零后，页进入空闲队列；已经就绪的完整页仍保留内容和 Hash 映射。前缀命中时重新增加引用并移出空闲队列，Hash 映射继续保留。只有这个物理页被分给不相同的新内容时，才删除旧 Hash 并计为驱逐，正在引用的页不会被这样重分配。

**深入解释：** `ref_count` 是活跃请求的页引用数；`ready` 表示完整页 KV 已经计算完成；`free_block_ids` 管理当前无引用、可再分配的页；`hash_to_block_id` 负责定位可复用页。空闲不等于内容无效。项目从空闲队列头取页，释放时放到尾部，前缀页按逆序释放使其更晚进入队列；这是按释放与重新引用次序管理的空闲页 LRU，而非每次 Attention 读取都更新时间戳的精确全局 LRU。

**项目实际实现：** [BlockManager.deallocate / _deallocate_block](../src/myvllm/engine/block_manager.py) 减引用并保留 ready 页内容；`allocate()` 先固定全部命中页，再分配未命中页，避免后续分配误驱逐刚命中的页；`_allocate_block()` 才删除指向该物理页的旧 Hash 映射、增加 `evictions` 并 `reset()`。**前缀缓存命中并重新引用时不会删除其 Hash 映射。**

**具体例子：** A、B 共享物理页 7 时，引用数为 2。A 结束后变为 1，页仍由 B 使用；B 结束后变为 0，页 7 入空闲队列，Hash 仍在。C 命中同一前缀后把它重新引用为 1，保留 Hash。若没有命中且内存压力使 D 把页 7 分给新内容，才淘汰页 7 的旧缓存并删除旧映射。

**常见追问与边界：** `all_pages_released` 只说明活跃页引用全部归还，不表示预分配 CUDA 显存池已退还给系统，也不表示缓存被清空。ShareGPT 正式实验驱逐数为 0；LRU 压力验证来自另外的早期小页池实验，不能混成同一次实验。

#### 6. Chunked Prefill 为什么前几个 Chunk 不能采样？

**面试官问题：** Prompt 分成三块之后，每块最后位置都有 Logits，为什么不能都生成一个输出？返回的 `-1` 又是什么？

**30 秒回答：** 前几个 Chunk 还没有处理完整用户输入，它们末尾对应的是 Prompt 中间位置，预测的是后面的输入，而不是完整 Prompt 后的答案。所以部分 Prefill 只写 KV、推进计算进度；最后一个 Chunk 才采样首个输出。代码返回的 `-1` 只是“不采样”哨兵，不是 PAD，也不会追加到请求的 Token 序列里。

**深入解释：** Chunk 切的是本轮计算窗口，KV 页切的是内存存储，两者概念不同。调度器记录 `_scheduled_start/end`；本轮窗口处理完后更新 `num_computed_tokens`。只有 Prefill 已到 `len(seq)` 或本轮属于 Decode 的请求，才选取对应最后一行 Hidden State 经过 LM Head 和采样。部分 Prefill 的 KV 可以供后续 Chunk 读取，但只有完整且已计算的页能发布为 Prefix-ready。

**项目实际实现：** [ChunkScheduler.schedule](../src/myvllm/engine/chunk_scheduler.py) 选择窗口和 Decode-first 顺序；[run_chunked](../src/myvllm/engine/chunk_runtime.py) 用累计行号选出可采样请求，其余输出初始化为 `-1`；`chunked_step()` 断言部分 Prefill 返回 `-1`、尚无 Completion，再调用 `mark_computed(seq, computed_tokens=end)` 发布有效完整页。混合 Prefill/Decode 使用一次 `run_chunked` 前向；纯 Decode 保留现有 Graph 路径。

**具体例子：** 600-token Prompt、Chunk Size 256，窗口为 `[0,256)`、`[256,512)`、`[512,600)`。第一轮只完成前 256 个输入，第二轮到 512，二者不能向用户输出答案；第三轮处理剩余 88 个输入，末位置预测第一个生成 Token。采样前已有 600 个 Token 的 KV；追加首个输出后序列长度为 601，这个新 Token 的 KV 留到下一轮计算。

**常见追问与边界：** Chunk 的目的是让长 Prefill 可以与 Decode 交错，是否更快取决于负载及执行器。Chunk Scheduler 当前只支持单卡 Qwen3 路径，不能说已和 TP 或 DFlash 联合调度。把 `-1` 当 PAD 输入模型或把每个 Chunk 都算作一个输出，都会破坏状态与指标。

#### 7. Continuous Batching 如何补位，完成后的 KV 去哪里？

**面试官问题：** 两条请求长短不同，为什么不等整个 Batch 结束？驻留上限、每轮 Token Budget 和 KV 容量有什么关系？

**30 秒回答：** Continuous Batching 在每轮重新选择活跃请求，短请求结束就能让等待请求补位，不必等最慢请求结束。项目的 Chunk 路径同时限制驻留请求数、单步 Token Budget 和可分配 KV 页。完成后先减少页引用，下一轮才接新请求；完整 ready 页仍可能作为前缀缓存留下，因此回收所有权和清空物理内容是两件事。

**深入解释：** 三种限制不能混淆：驻留上限限制同时管理多少请求；Token Budget 限制一次 Forward 的工作量；页池决定这些请求的 KV 能否存下。`decode_first=True` 时先为 Decode 各安排一个 Token，再用剩余 Budget 安排 Prefill Chunk。某请求到达停止条件后从 Running 移除，释放的引用可用于下一轮入驻；缓存命中还可能跳过一段 Prefill，但不保证任何负载都达到最高吞吐。

**项目实际实现：** [ChunkScheduler.schedule](../src/myvllm/engine/chunk_scheduler.py) 明确用 `len(running) < max_num_sequences` 约束入驻；[Scheduler.postprocess](../src/myvllm/engine/scheduler.py) 检查 EOS、输出长度、上下文长度，并调用 `BlockManager.deallocate()`。普通 `Scheduler.schedule()` 的 `max_num_sequences` 主要限制本步选中数量，不能据此宣称它也实施同样严格的总驻留约束。DFlash 的补位由独立 [DFlashRuntime.generate](../experiments/dflash_graph/dflash_runtime.py) 实现，并不调用这里的 `engine.step()`。

**具体例子：** 设驻留上限为 2，Running 为 A、B，Waiting 为 C。A 本轮生成最后一个 Token 后被移除，页引用减少；下一轮 C 才进入 Running，与仍在 Decode 的 B 共存。如果 Token Budget 为 257，B 的 Decode 占 1，C 最多获得 256-token Prefill Chunk。若 C 命中旧完整页会直接从已缓存位置开始；否则需要空闲页，不能只看有空位就无条件接入。

**常见追问与边界：** 这不是每一层都重新组 Batch，也不是承诺请求绝不等待。原 Scheduler 有抢占路径；当前 Chunk Scheduler 页耗尽时显式报错，不要混写成已实现同样的抢占策略。正式 ShareGPT 的驻留上限 8 是 ChunkScheduler 路径的实际约束。

#### 8. ShareGPT 三种调度实验怎么设计，Prefix 的提升相对谁？

**面试官问题：** 你是不是只跑了 Prefill？20% 共享前缀是哪里来的？为什么 Chunk 单独更慢还要保留？

**30 秒回答：** 这是完整 Prefill 加 Decode 实验：100 条请求同时入队，每条真实生成 64 个 Token，最多驻留 8。80 个不同 ShareGPT Prompt 后追加 20 个受控完整 Prompt 重放，并非数据天然有 20% 共享率。Chunk 开 Prefix 后吞吐从 43.67 到 52.51 tokens/s，提高 20.23%；这个基线是 Chunk only。相对 Full Prefill 只提高 8.63%，平均 TTFT 和 E2E 反而更高，我会同时报告。

**深入解释：** Full Prefill 用 Chunk Size 2048、Prefill-first，让本实验最长 1536-token Prompt 一次读完；Chunk only 改为 Size 256、Decode-first；Chunk+Prefix 在相同 Chunk 配置上只开缓存。因此第一组对照同时改变分块和优先级，第二组才隔离 Prefix。各模式冷启动自己的块管理器，使用相同输入、BF16 主干、FP32 Head、固定算术及 per-query PagedAttention，纯 Decode 使用 Graph。

**项目实际实现：** [prepare_data.build_requests / validate_workload](../experiments/sharegpt/prepare_data.py) 固定版本、seed 42、首条非空 Human 消息和 Qwen Chat Template，选 40 短+40 长，再重放各 10 条；[bench_sharegpt.main 的 trial](../experiments/sharegpt/bench_sharegpt.py) 执行到所有请求完成、记录时间和 Trace，验收 100/100 序列一致、有限 Logits、页引用全部归还。每模式正式测一次。

**具体例子：** 重放请求占比是 `20/100=20%`；实际复用输入是 `13,568/78,889=17.20%`，因为请求长度不同、按完整页复用且保留未缓存尾部，两者分母不同。正式总耗时 Full/Chunk/Chunk+Prefix 分别是 132.397/146.539/121.881 s，吞吐分别为 48.339/43.674/52.510 tokens/s。Chunk 形成 267 个部分 Prefill 块、188 个混合步骤，改变了各步的工作组成；部分块更多不等于总 Forward 次数更多。混合路径与调度成本可能影响耗时，尚无 Profiling 定位。

**常见追问与边界：** 上述成本是合理解释，尚无 Profiling 证明唯一瓶颈。`max_tokens=64, ignore_eos=True` 固定输出工作量；遇到 EOS 后仍真实前向与采样，不用 PAD 填满。计时包含排队，排除分词、模型加载、预热及 HTTP；一次测量不能说明稳定误差范围。尚无 Prefix-only 对照，也未证明原 Flash Prefill 完整模型分叉已修复。


### C. DFlash、Pending、验证与 KV 回滚

#### 9. EAGLE3、DFlash、DSpark 有什么区别？你做的是三种消融吗？

**面试官问题：** 三种方法如何提出候选？哪些工作是你的，哪些来自官方？

**30 秒回答：** 三者都由 Draft 提议、Target 因果验证，最终只接受连续正确前缀。EAGLE3 在本轮官方单链实现中逐步产生候选；DFlash 通过 Masked Parallel Prediction 一次算出候选块；DSpark 先并行计算块特征，再用依赖前一个 Token 的轻量 Markov Head 顺序修正候选，Confidence Head 可以截短送验前缀。我使用官方训练好的三个 checkpoint，把它们接入同一稳定 Target 并完成对照。三个 checkpoint 是方法比较；同一 DSpark 的 Markov/Confidence 开关才是推理策略消融。

**深入解释：** 普通自回归 Draft 通常要依次执行小模型步骤，DFlash 则一次 Forward 对带 MASK 的块共同预测未来位置。Draft 可以用非因果块内信息猜测；Target 仍必须保持因果性，不能提前看到待预测位置之后的候选。DSpark 并没有对每个候选再完整运行一遍 Draft Transformer：昂贵的块特征先并行算出，之后由较轻的 Markov Head 依据前一个 Token 修正。Confidence 在完整提议后计算并截断，不是提前少运行同比例的 Draft Backbone。

| 路径 | Draft 与候选预算 | Target 执行 | 正式比较对象 |
|---|---|---|---|
| 旧 DFlash＋Graph | z-lab b16，15 个候选＋Pending | B=1/2、Q=1/16，Eager/Graph 四项对照 | 主要相对 AR Graph，2.99×/1.52× |
| 三种 Draft 对照 | DeepSpec 官方 EAGLE3、DFlash、DSpark，最多 7 个候选＋Pending | 串行 B=1，Q=1..8，原 Target Eager | 各自配对 AR Eager，2.934×/3.479×/3.898×（Markov Only） |

**项目实际实现：** [benchmark._worker()](../experiments/speculative_compare/benchmark.py) 加载官方 Draft 并保留其 `generate_one_sample()`；[LegacyTarget.forward()/LegacyCache.crop()](../experiments/speculative_compare/legacy_target.py) 接入原 Target 和分页 Cache。官方固定提交中的 `deepspec/eval/eagle3/evaluator.py::_propose()` 是单链提议；`deepspec/eval/dspark/draft_ops.py::forward_dspark_draft_block()` / `build_dspark_proposal()` 负责 DFlash/DSpark 块提议；`deepspec/modeling/dspark/qwen3/modeling.py::sample_draft_tokens()` 选择是否调用 Markov Head；`base_evaluator.py::verify_draft_tokens()` / `generate_decoding_sample()` 验证、接受并裁剪 Target Cache。外部源码版本和 checkpoint 指纹见 [lock.json](../results/speculative_compare/legacy-target-formal-20261021/lock.json)，完整源码在原 records ZIP 中。

**具体例子：** 每轮七个候选时，DSpark No Markov 平均接受约 1.56 个 Draft Token，Markov Only 约 3.91 个；完成同样 36 次请求，验证次数从 3615 降到 1902，配对吞吐提高 84.68%。这是同一已训练 checkpoint 的推理开关效应。关闭 DSpark 的 Markov 后，权重仍是 DSpark，不能把它当作重新训练的官方 DFlash。

**追问：Markov Head 为什么可能提高候选质量？** 并行块特征在选出具体候选前已算好，Markov 用刚选出的前一个 Token 修正当前位置分数，增加候选间的顺序依赖。本轮 checkpoint 是 `vanilla`、rank=256：低维 Embedding 查前一个 Token，经 Linear 投到词表，作为偏置加到并行基础 Logits；不是额外重跑一遍 Transformer。示意上，先根据 Pending=P 修正并选出 A，再用 A 修正下一位，选出 B；Target 仍逐位置验证，不能因为经过修正就直接接受。

源码为官方固定提交的 [VanillaMarkov.apply_step_logits()/sample_block_tokens()](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/markov_head.py)，实际配置见 [dspark-environment.json](../results/speculative_compare/legacy-target-formal-20261021/dspark-environment.json)。84.68% 是本轮同权重推理开关的实测收益，不保证每个 Prompt 的候选都变好，也不代表 Markov 在所有模型上必然有效。

**常见追问与边界：** 当前 EAGLE3 没有树搜索/树验证，三方法没有匹配训练数据和模型容量；所有新对照都是 B=1、无 Graph，没有进入普通 Serving 调度器。官方代码包含概率接受工具，但本项目正式验收只用 temperature=0，不证明随机采样分布一致。下文第 10～14 题的 15 候选、Q=16、MASK ID 和请求补位，主要对应旧 Graph 路径；新路径的候选预算与 Cache 适配单独标注。官方模型和算法不能说成由本项目训练或发明。

#### 10. Target Hidden States 是什么？DFlash 实际用了哪些层？

**面试官问题：** `target_hidden` 与 `noise_embedding` 有什么区别？为什么 Draft 不需要重新跑完整的 Qwen3-8B？

**30 秒回答：**

> Token ID 是词表编号，Embedding 是这个编号查出来的向量，Hidden State 是经过模型层处理、融合上下文后的向量，Logits 才是输出词表分数。DFlash 复用 Target 已经算出的中间层 Hidden States作为上下文。本项目采集零基索引 1、9、17、25、33 五层，也就是第 2、10、18、26、34 层，拼接后传给 Draft。`noise_embedding` 则是 Pending 加 MASK 块的输入向量，负责本轮要预测的位置。

**深入解释：**

不同深度特征提供不同层次的上下文表示，但本项目没有做逐层贡献消融，不能断言某一层专门负责语法或推理。官方 Draft 用 `fc` 把五层拼接特征投影回 Hidden Size，再做归一化；每个 Draft Attention 层由这些上下文特征生成 K/V，当前块的 Hidden States 生成 Q 和块内 K/V。Draft 自己仍要计算，只是沿用大模型已经提取的上下文，而不用为每轮候选重新执行 Target 的全部 36 层历史计算。

**项目实际实现：**

[DFlashRuntime.__init__()/propose()](../experiments/dflash_graph/dflash_runtime.py) 检查层索引并注册 hook。MinivLLM 层返回 `(x, residual)`，采集时必须取 `x + residual`，对应完整层输出，不能只拿尚未合并残差的 `x`。Graph 路径由 [TargetGraphExecutor.__init__()/_forward()](../experiments/dflash_graph/graph_executor.py)接管 hook 和拼接；[GraphDFlashRuntime.__init__()](../experiments/dflash_graph/graph_runtime.py)移除基础运行时的重复 hook。官方 [DFlashDraftModel.forward()/Qwen3DFlashAttention.forward()](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/blob/9b41424b7109f9c5413454f481b09a82b85333f4/modeling_dflash.py)实现特征投影及 K/V 注入。

**具体例子：**

Qwen3-8B 每层 Hidden Size 是 4096；一个位置的五层特征拼接为 `5×4096=20480` 维，再投影为 4096 维。初始 Prefill 产生整个 Prompt 的特征；后续只把有效推进位置的新特征交给 Draft，旧上下文已在这个请求自己的 Draft Cache 中。

**常见追问与边界：**

Hidden State 不是 Token ID，也不是已经确定的未来答案。首次 Prefill、各轮 Target 验证都真实执行；“复用特征”不能说成“不再调用 Target”。本项目固定使用官方层选择，没有训练特征投影或比较其他层组合。

#### 11. Pending Token、MASK 和 PAD 有什么区别？为什么保留一个 Pending？

**面试官问题：** 这三种“占位”是不是同一件事？为什么生成了一个 Token，却还没有它自己的 Target KV？

**30 秒回答：**

> Pending 是已经确定、真实输出的 Token，只是还没有把这个 Token 自己经过 Target 后的 KV 写入缓存。MASK 是 Draft 对未来未知位置的预测占位，本项目 ID 为 151669。PAD 是补齐计算形状的填充概念，这里固定行数计算使用零向量补行，并不把 PAD Token 放进真实 Attention。三者用途完全不同：Pending 是已知内容，MASK 是待猜的位置，补行是计算布局。

**深入解释：**

一个位置的 logits 预测它的后继 Token。初始 Prefill 把 Prompt 的 KV 写完，并根据最后一个 Prompt 位置预测首输出 `P`；这个 Forward 没有以 `P` 为输入，所以没有 `P` 自己的 KV。下一轮输入 `P` 才会补上它的 KV并预测后继。投机验证也保留这个时序：确认若干候选后，再补一个 Target 预测，最末尾那个真实输出留到下一轮作为输入。

**项目实际实现：**

[DFlashRuntime.generate()](../experiments/dflash_graph/dflash_runtime.py)在 Prefill 后先设 `computed = len(prompt)` 再追加首 Token；[Request/Window/commit()](../experiments/dflash_graph/state.py)维护 `len(req.tokens) == req.computed + 1`。`propose()` 把最后一个 Token 放到块首，其余 15 位初始化为 MASK。正式配置的 MASK ID 由 [benchmark.main()](../experiments/dflash_graph/benchmark.py)检查。用于固定行数的 [fixed_rows_forward()](../experiments/common/fixed_shape.py)是零向量补行、算完裁回真实行；这些补行不进入 KV、Attention 或采样。

**具体例子：**

Prompt 有 100 个 Token。Prefill 后 `tokens` 长度为 101，`computed=100`，第 101 个 `P` 是 Pending。下一轮 Draft 输入 `[P, MASK, …, MASK]`，Target 验证的则是 `[P, A, …]` 的真实候选 ID，Target 不接收这些 MASK。输出只剩 3 个名额时，当前实现仍验证 Q=16，再把提交输出截到 3 个，不把尾部 13 个变成 PAD。

**常见追问与边界：**

Pending 在逻辑上没有已提交 KV；此前投机写入的同位置物理数据即使残留，也不算有效状态。Chunked Prefill 返回的 `-1` 又是另一个“不采样”哨兵，不是 Pending、MASK 或 PAD。

#### 12. 怎样手推一轮 DFlash 验证？为什么只接受连续正确前缀？

**面试官问题：** Pending 为 P，候选 A、B、C，Target 后继预测 A、B、X、Y；最终输出和 KV 分别保留什么？

**30 秒回答：**

> Target 输入是 `[P,A,B,C]`，各行预测的是下一位置，所以拿候选 `[A,B,C]` 对齐预测 `[A,B,X]`。A、B 连续匹配，C 与 X 不同，于是接受 A、B，补上 Target 的 X，这轮新增输出是 A、B、X。KV 只提交 P、A、B；X 是下一轮 Pending。C 后面的预测建立在错误历史上，不能跳过 C 再接受后续候选。

**深入解释：**

验证一次计算多个位置，但保持因果可见范围：P 行没有看 A，A 行只看 P 与 A，B 行只看 P、A、B。因此 B 行预测的 X 是正确历史下的 Target 后继；C 行预测的 Y 已经使用错误候选 C，不可作为正确续写。连续前缀一旦断开，后续匹配即使碰巧相等，也不是在同一正确历史下成立。

**项目实际实现：**

[accept_prefix()](../experiments/dflash_graph/state.py)比较 `block[1:]` 与 `posterior[:-1]`，首个不一致处停止。`commit()` 追加接受候选和 `posterior[accepted]`，按剩余输出配额截断，并令 `computed` 增加实际新增输出数。Target 一次前向由 [GraphDFlashRuntime.target_forward()](../experiments/dflash_graph/graph_runtime.py)及 [TargetGraphExecutor._forward()](../experiments/dflash_graph/graph_executor.py)完成；每行 `context_lens=position+1` 保持因果边界。

**具体例子：**

设进入本轮时 `computed=n`，`tokens=[历史n个, P]`：

| Target 输入位置 | 当前 Token | 该行后继预测 | 处理 |
|---|---|---|---|
| n | P | A | 接受候选 A |
| n+1 | A | B | 接受候选 B |
| n+2 | B | X | 候选 C 错误，输出 X |
| n+3 | C | Y | 错误历史上的预测，丢弃 |

提交后 `tokens=[历史, P,A,B,X]`，`computed=n+3`；有效 KV 对应 `[历史,P,A,B]`，仍满足 `len(tokens)=computed+1`。本轮新增三个输出，但接受的 Draft Token 只有两个。实际块是 16，这个 4 Token 例子只是同规则的缩小演示。

**常见追问与边界：**

若 15 个候选全匹配，还能补上末行的 Target 预测，一轮最多新增 16 个输出；若第一个候选就错，也至少新增一个 Target Token。数学上的 greedy 等价还要求正确数值与缓存状态，不能只凭接受公式就宣布整套实现无误。

#### 13. 拒绝候选后回滚什么？为什么不必立即清零整块 KV？

**面试官问题：** Target 已经计算错误候选并写入 KV，下一轮会不会读到错误状态？Target KV、Draft KV 和 Hidden Features 怎样分别处理？

**30 秒回答：**

> 回滚的关键是恢复有效长度和所有权，而不是把显存全部清零。Target 先算整块，所以拒绝候选的 KV 可能已写入；提交后只允许读到 `req.computed` 确认的范围，超出范围的整页由 `pool.trim()` 归还，最后一页中的残留由逻辑长度屏蔽。Draft Cache 要丢掉临时噪声块，Target Features 也只保留本轮有效推进的位置。三种状态分别管理，不能只回滚一个长度。

**深入解释：**

Target KV 是大模型所有层的 K/V；Draft KV 是 Draft 自己的上下文投影与注意力缓存；Target Features 是选定五层的新上下文表示。拒绝后，Target 的有效输入前缀应停在 Pending 和已接受候选，不能包含错误候选。Draft 本轮处理的是 MASK 块，不应把其临时 K/V 当成已确认上下文永久保留；下一轮应以 Target 实际确认的 Features 补进 Draft。

**项目实际实现：**

[commit()](../experiments/dflash_graph/state.py)更新 `computed`；[PagePool.trim()](../experiments/dflash_graph/state.py)按 `ceil(computed/block_size)` 保留所需页、释放尾部整页。[DFlashRuntime.generate()](../experiments/dflash_graph/dflash_runtime.py)用 `features[:advance]` 保留有效位置。特别注意：`propose()` 中 `draft_cache.crop(req.computed)` 在 **Draft Forward 后、Target 验证前**执行，裁掉临时噪声块；它不是在 `commit()` 后对 Target KV 做裁切。下一次 `propose()` 断言 `features长度 = computed − DraftCache长度`，保证只补入尚未进入 Draft Cache 的已确认特征。完成及异常清理走 `PagePool.release()`。

**具体例子：**

每页 256 Token，原 `computed=255`，验证 16 位暂时占到第二页。若首候选错误，本轮仅推进 1，提交后 `computed=256`，第二页可整页归还。如果原 `computed=250`，提交后 253，错误候选还落在第一页；不需要清零保留页的 253 以后的物理位置，因为后续各 Query 的 `context_lens` 不会把它们计入历史，下一轮真实写入还会覆盖对应槽位。

**常见追问与边界：**

把错误值清零却仍用错误长度读它们，一样会破坏 Attention。反过来，有残留值但长度/页表正确可以安全继续。该 PagePool 是 DFlash 独占的页池，不启用普通 BlockManager 的 Prefix 共享；不能声称此处完成了共享页投机回滚。

#### 14. DFlash 的“批量”具体批了什么？Draft 和 Target 都多请求合批了吗？

**面试官问题：** 并行预测 15 个候选与 batch=2 是一回事吗？三个请求怎样补位？

**30 秒回答：**

> 这里有两种批量：一个请求一次提出多个未来位置，这是 Token Block 内的并行；多个请求的候选块放进一次 Target Forward，这是请求之间的合批。本项目 Draft 仍按活跃请求逐个调用，每个请求有自己的 Draft Cache；Target 验证能合批。三请求实验最多同时驻留两个，请求完成归还页后，第三个请求再补位，不能说 Draft 与 Target 都实现了完全并行多请求计算。

**深入解释：**

合批前，两个请求可能有不同历史长度、不同物理页，但都带 16 位验证块。模型输入展平为 32 行，每行配自己的 context length、slot 与页表，因此不会互相读上下文。Target 一次大调用处理这 32 行。当前 Draft 提议却是 Python 循环中的两个独立 Forward；这部分串行成本和补位时的初始 Prefill 都仍计入总耗时。

**项目实际实现：**

[DFlashRuntime.generate()](../experiments/dflash_graph/dflash_runtime.py)中的 `blocks = [self.propose(r) ... for r in active]` 顺序提议；随后一次 `target_forward(active, blocks, pool)` 验证。每个 [Request](../experiments/dflash_graph/state.py)独立持有 `draft_cache/features/computed`。[GraphDFlashRuntime.target_forward()](../experiments/dflash_graph/graph_runtime.py)建立 Window 并在返回后按 Query 长度切回各请求。它自己的 waiting/active 队列独占 KV，没有调用普通 `engine.step()`。

**具体例子：**

活跃 A、B，等待 C：先做 `Draft(A)`，再做 `Draft(B)`，然后合并两块为 `Target(B=2,Q=16)`。A 完成后释放自己的页；下轮 C 进行图外初始 Prefill 并加入活跃队列，再与 B 一起验证。三请求正式实验观察到 1 次仍有活跃请求时的补位，Target 总调用从 AR 的 512 次降到 DFlash 的 144 次。

**常见追问与边界：**

“那 batch=2 的吞吐是单请求速度吗？”不是，结果用三个请求共 768 个输出除整组耗时。Draft 合批、跨请求异步 Draft/Target 重叠，以及与 Prefix/Chunk 的联合调度都没有验证。

#### 15. 候选数、接受率和平均推进长度怎样影响速度？Confidence 为什么反而慢？

**面试官问题：** 旧 DFlash 接受率约 31% 仍能加速，新 DSpark 接受率升到 83% 却更慢，为什么？

**30 秒回答：**

> 候选数量只是预测预算，实际收益取决于每轮确认多少输出，以及 Draft 和验证各花多少时间。连续正确前缀之外的候选全作废；输出配额充足时还会补一个 Target Token，尾轮则按配额截断，所以接受率与平均推进长度不同。旧 Graph 单请求场景提出 675 个候选、接受 211 个，接受率约 31.26%；45 轮验证完成首 Token 以外的 255 个输出，平均每轮推进约 5.67 个。这减少了 Target 调用，但还要付 Draft 与多位置验证成本。

**深入解释：**

可用近似式理解一轮生成成本：`平均每输出成本 ≈ (Draft耗时 + Target验证耗时 + 状态管理开销) / 平均推进长度`。这是机制分析，不是本项目测出的逐算子耗时。候选一多，最早错误之后的工作可能浪费；接受率也会随 Prompt 类型改变。较低接受率仍可能比逐 Token Target 更便宜，但不保证胜过已经优化的 AR Graph。

**项目实际实现：**

[commit()](../experiments/dflash_graph/state.py)分别记录 `accepted_draft`、`proposed_draft` 与 `advance`；[DFlashRuntime.generate()](../experiments/dflash_graph/dflash_runtime.py)累计接受率和 `average_verified_advance`。以 [PROJECT_SUMMARY](PROJECT_SUMMARY.zh.md)及 [正式 DFlash Graph 结果](../results/dflash_graph/summary.json)为准，单请求 Target 总调用 256→46，其中首次 Prefill 1 次、验证 45 次。

**具体例子：**

第 12 题 P/A/B/C 的例子接受 2/3 个候选，但新增 3 个输出，因此不能用“接受率 2/3”替代“推进 3”。三请求场景正式接受率为 `536/3465=15.47%`，仍有有效推进；但 DFlash Eager 的 40.93 tokens/s 低于 AR Graph 的 65.91 tokens/s。加上 Target Graph 后才达到 100.38 tokens/s，即相对 AR Graph 约 1.52×。

**常见追问与边界：**

新 DSpark 对照正好提供了一个反例：

| 同一 DSpark checkpoint | 推理开关 | 平均送验候选数 | 已送验候选接受率 | 实际输出/验证 | 三轮验证次数 | 吞吐 tokens/s |
|---|---|---:|---:|---:|---:|---:|
| No Markov | 两个 Head 均关闭 | 7.000 | 22.29% | 2.539 | 3615 | 34.283 |
| Markov Only | 开 Markov、关 Confidence | 7.000 | 55.90% | 4.826 | 1902 | 63.180 |
| Fixed | 两个 Head 均开，阈值 0、不截断 | 7.000 | 55.90% | 4.826 | 1902 | 63.225 |
| Confidence 0.5 | 两个 Head 均开，阈值 0.5 | 5.711 | 66.66% | 4.722 | 1944 | 61.839 |
| Confidence 0.8 | 两个 Head 均开，阈值 0.8 | 3.520 | 83.21% | 3.873 | 2370 | 51.741 |

Confidence 0.8 截掉低置信位置及其后缀，送验集合和接受率分母随之改变；本轮这个集合的接受率升高，但每轮有效输出变少，Target 验证轮数增多，配对吞吐比 Fixed 下降 18.16%。0.5 也下降 1.97%。原官方 `_propose()` 先跑完整 Draft Block，`build_dspark_proposal()` 再计算 confidence、裁短送验前缀，不能按送验长度比例推算节省了 Draft 计算。

**追问：Confidence Head 是什么，等于 Token Softmax 概率吗？** 它是独立学习的候选接受可靠度辅助预测器；本轮将各位置 Hidden 与前一个 Token 的 Markov Embedding 拼接，经 Linear 输出一个标量，再用 sigmoid 变成 0～1 的分数。Token Softmax 则由 LM Head 的整张词表 Logits 归一化，是另一种输出。0.8 只是策略阈值，不表示该 Token 已有经过本项目校准的 80% 接受概率；本轮 temperature=0 的采样概率还是 one-hot，也不会让 Confidence 自动变成 1。

官方实现为 [predict_confidence_step()](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py)、[AcceptRatePredictor.forward()](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py) 和 [_confident_prefix_length()](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/draft_ops.py)。例如候选 A/B/C 的分数为 `[0.95,0.60,0.90]`，阈值 0.8 只送验 A；不能跳过 B 再送 C，因为只保留首个低置信位置之前的连续前缀。这是规则演示，不是新增测量。

“Markov Only 和 Fixed 是否重复？”两者计算不同：前者关闭 confidence head，后者计算 confidence 但阈值为零、不截断。本轮 36 组输出、提议与接受轨迹逐项相同；速度差约 0.06%，不足以声称明显收益。所有头权重仍驻留 GPU。开关由 [generate()](../experiments/speculative_compare/benchmark.py) 设置并在 finally 恢复；指标由 [summarize()](../experiments/speculative_compare/protocol.py) 及 [analyze_ablation.analyze()](../experiments/speculative_compare/analyze_ablation.py) 复算。

**追问：这是否否定了高并发 Confidence Scheduling？** 不能。本轮只有串行 Batch=1、两个固定阈值，没有根据队列、活跃请求和 Target 成本动态分配预算。高并发时，缩短某请求的验证窗口可能把预算留给其他请求，而候选不足也可能降低 GPU 利用率；这是待验证的机制权衡。应在相同输入、到达率、总 Token 预算及正确性要求下，对比固定长度和负载感知动态策略的系统吞吐与尾延迟。本轮不能据接受率或单请求速度推断那种策略有效或无效。

不能用候选数、接受率或 Target 调用减少比例直接推导端到端加速，也不能乘另一场实验的 Graph 倍数。最后一轮可能超额计算，因此“平均接受 Draft 数＋1”与扣掉初始首 Token 后的实际输出/验证不同。没有块长扫描或 Draft/Target 分段计时，不能说某个块长普遍最优，也不能给唯一瓶颈的时间占比。

### D. CUDA Graph、静态缓冲与形状分桶

#### 16. Eager Execution 与 CUDA Graph 有什么区别？Replay 是复用结果吗？

**面试官问题：** CUDA Graph 为什么能加速，和 Kernel Fusion、减少 FLOPs 有什么关系？

**30 秒回答：**

> Eager 每次 Forward 都要从 CPU 端经过框架调度并提交一串 GPU 操作。小 batch 解码中，许多 Kernel 本身很短，反复提交的开销会占比较大。CUDA Graph 先 Capture 这段 GPU 工作及依赖，再 Replay 已记录的执行图，减少重复提交成本。Replay 仍真实执行本轮 GPU 计算，读取新 Token 与新 KV；它不是缓存上一轮答案，也不自动减少 FLOPs 或把所有算子融合成一个 Kernel。

**深入解释：**

Kernel Launch Overhead 是组织和提交 GPU 工作的成本，不等同 Kernel 实际计算时间。Graph 把可重复的提交结构固定下来；Kernel Fusion 则把多个操作合为更少 Kernel，可能同时减少中间显存访问，两者可以配合但不是同一技术。本项目保留同一计算路径比较 Eager 与 Graph，改善多少由实测决定。Graph 原理参照 [PyTorch 2.9 CUDA Graph 文档](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs)。

**项目实际实现：**

[TargetGraphExecutor.capture_all()](../experiments/dflash_graph/graph_executor.py)为各桶分配固定缓冲，在独立 stream 预热三次，再通过 `torch.cuda.graph` Capture `_forward()`；`run()` 更新输入后调用 `graph.replay()`。每张图使用独立私有内存池。Capture 时 Python hook 执行并记录相应张量操作，Replay 不重新调用 Python hook；特征拼接等已记录的 GPU 操作仍执行。

**具体例子：**

普通 AR 每请求输出 256 个 Token，首 Token 来自 Prefill，后续 255 次 Forward 都要解码。Graph 模式实际 Replay 255 次，并没有省掉这 255 次模型计算。正式单请求 AR Eager 25.416 s、AR Graph 5.513 s，体现这个受控执行器的收益；但没有 Profiling，不能把二者差值全部定量归为某一种 Kernel Launch。

**常见追问与边界：**

Capture 本身有成本：四张图共约 3.533 s、allocated 增量约 77.39 MiB，未计入正式生成耗时。这个比值不等于任意生产引擎的 Graph 收益；生产引擎已有不同算子、融合与调度优化。

#### 17. 序列长度一直变化，静态输入缓冲到底固定什么？

**面试官问题：** Graph 输入地址不变，是不是 Token、历史长度和页表也不能变？

**30 秒回答：**

> 固定的是捕获所引用的缓冲区地址和形状，里面的数据可以在每轮 Replay 前替换。比如第一次 `input_ids=[123]` 在地址 `0x1000`，第二次把 456 拷到同一个地址，Graph 就计算新 Token。本项目每轮一起更新 `input_ids`、`slot_mapping`、`context_lens`、`block_tables`；只更新 Token 而忘了长度或页表，会把新内容写错位置或读到错误历史。

**深入解释：**

`input_ids` 决定本轮输入；`slot_mapping` 决定每个输入 Token 的 K/V 写到哪个物理槽；`context_lens` 同时决定有效历史长度及绝对位置，代码用它减 1 得到 RoPE 位置；`block_tables` 把每个 Query 的逻辑页映射为物理页。序列变长时，可以在固定长度页表中替换内容，空尾部统一为 `-1`。地址不变与数据不变是两件事，替换张量对象不能代替原地址上的 `copy_()`。

**项目实际实现：**

[build_query_metadata()](../experiments/dflash_graph/graph_executor.py)逐位置产生四类元数据并校验页号/范围；`_allocate()` 建 GPU 缓冲，`_fill()` 每次建立 CPU staging 后对四类目的缓冲完整 `copy_()`，包括页表尾部。`_forward()` 将缓冲交给上下文，使用全阶段 per-query PagedAttention。[Qwen3DecoderLayer.forward()](../src/myvllm/models/qwen3.py)在该路径按 `context_lens-1` 求位置。

**具体例子：**

每页 256，物理页表 `[4,9]`。输入 Query 起点 255、长度 2，其位置为 255、256，`context_lens=[256,257]`，`slot_mapping=[4×256+255,9×256]=[1279,2304]`；两行都携带补齐至宽度 8 的 `[4,9,-1,…]`。本例演示跨页地址公式，Q=2 会走 Eager；正式跨页诊断使用已捕获 Q=16。

**常见追问与边界：**

`0x1000` 只是说明固定地址的假设示例，不是实测地址。动态长度没有消失，只是变成固定缓冲里的值。该执行器是进程内顺序使用，不能让另一个引擎同时修改模型上下文或 KV 池。

#### 18. B/Q 四个 Graph 桶怎样选择？为什么返回值必须 clone？

**面试官问题：** B、Q 分别是什么？Q=16 是补到 16 的 PAD 吗？保存上轮 Features 为什么会被下一轮覆盖？

**30 秒回答：**

> B 是本轮请求数，Q 是每个请求实际输入的 Token 数。当前捕获 `(1,1)、(2,1)、(1,16)、(2,16)`：Q=1 用于 AR Decode，Q=16 用于 DFlash Target 验证，后者是 Pending 加 15 个真实候选，并非 PAD。每桶有固定输入与输出地址，Replay 会重写这些输出，所以返回的 Picks、Hidden 和 Features 要 clone 成独立张量，避免下一轮覆盖仍被请求保存的特征。

**深入解释：**

不能只按展平后的总行数选图：B=2/Q=1 与 B=1/Q=2 都是 2 行，但请求边界和验证语义不同。`build_query_metadata()` 先确定每请求 Query 长度是否相同，再构造 `(B,Q)`；非统一长度没有对应图桶。Graph 返回输出引用若直接保留，下一次同桶 Replay 写入原地址，上轮 `req.features` 也会悄悄变成新值，这是一类很隐蔽的状态错误。

**项目实际实现：**

[TargetGraphExecutor.capture_all()/run()](../experiments/dflash_graph/graph_executor.py)分别缓存四桶，依据元数据 `key` 选图；`run()` 对 `picks/features/hidden` 全部 `clone()`，诊断需要时另 clone logits，Eager 也采用相同所有权策略。[GraphDFlashRuntime.target_forward()](../experiments/dflash_graph/graph_runtime.py)再将独立结果切回各请求。基础运行时仍在每轮提议完整 16 位块，`commit()` 最后按输出限额裁剪，尾轮不缩短 Q。

**具体例子：**

活跃 A、B 时，DFlash 块展平为 32 行，选 B=2/Q=16。A 完成后只剩 B，下一轮选 B=1/Q=16。AR 的同一变化则从 B=2/Q=1 切至 B=1/Q=1。复制前后地址不同，使 A 留下的特征快照不会因 B 后续 Replay 改写。

**常见追问与边界：**

固定行数 Linear 的内部补行与 Graph Q 分桶不同；PAD 行在返回 Attention 前已裁掉。更多 B/Q、非统一 Query 和动态 Shape 图不是本项目已验证能力；图桶数量也会增加捕获时间和显存成本。

#### 19. Fallback=0 证明什么？怎样确认 Graph 没破坏计算和投机状态？

**面试官问题：** 有没有因为 Shape 不支持而偷偷走 Eager？输出一样是否足够？初始 Prefill 和 Draft 进图了吗？

**30 秒回答：**

> Fallback 是请求使用 Graph、但形状不满足已捕获条件时退回 Eager。正式实验每个 Target 验证步都有对应 Replay，fallback=0，说明这些验证没有绕开图；不代表整个系统每一步都在图里。初始 Prefill、Draft、接受判定和回滚都在图外。我同时检查同历史张量、完整生成 Token，以及 DFlash 的接受/回滚轨迹，避免只看最终文字。

**深入解释：**

三个证据回答不同问题：同历史前向隔离一次 Eager/Graph 数值差异；完整生成验证误差是否在连续迭代中累积；接受轨迹验证投机状态机是否以相同方式推进。理论上最终文本相同也可能掩盖内部推进变化，所以本项目进一步比对接受数量、Target 调用、补位、回滚页与页回收。测试覆盖有限负载，不等于证明任意 Shape 或任意 Prompt 正确。

**项目实际实现：**

[TargetGraphExecutor.run()](../experiments/dflash_graph/graph_executor.py)统计 Replay、Eager 与 fallback；已支持的桶如果根本未 Capture 会直接报错，不会静默掩盖。[GraphDFlashRuntime.target_forward()](../experiments/dflash_graph/graph_runtime.py)明确令初始 Prompt 使用 Eager。[benchmark.graph_diagnostics()](../experiments/dflash_graph/benchmark.py)做 8 组同历史检查；`compare_generation(..., compare_trace=True)`比较完整 Token 和投机轨迹，`main()`断言 Replay 数等于验证次数、fallback=0，并在正式四模式实验中逐轮验收。

**具体例子：**

8 组来自 B=1/2、起点约 250/505、Q=1/16 的组合，覆盖 256/512 页边界；logits、Hidden、Features、Picks 都精确相等，最大差为 0。正式两种负载、四模式、各三轮共 24 次运行，48 次请求执行、12,288 个输出 Token 全部匹配，DFlash Eager/Graph 接受轨迹也一致，结束时页全部归还。

**常见追问与边界：**

fallback=0 的范围是这些正式 Target 验证前向。初始 Prefill 即使恰好长 1 或 16 也强制在图外；Draft Graph、随机采样、更多负载及全部优化联合配置都未验证。三轮中位数比单次更能观察波动，但仍不是广泛生产负载的统计结论。


### E. TP / DP / 1P1D / NCCL

#### 20. TP、DP、PP 和 P/D 分离分别拆什么？怎样比较才公平？

**面试官问题：** 你做了哪些多卡方案？单卡与双卡放在同一张表里，比较是否公平？

**30 秒回答：** TP 拆一个模型的矩阵，由两卡合作算同一请求；DP 放两个完整副本，各接不同请求；PP 拆模型层；P/D 分离拆请求的 Prefill 和 Decode 阶段。本项目实现了 TP2、DP2 和单机 1P1D，没有 PP。Single 对双卡衡量扩展收益，三种双卡方案之间比较相同资源下的部署取舍，所以我同时报告 GPU 数、总吞吐和每 GPU 吞吐，不把双卡总吞吐提升当作单请求或每卡加速。

**深入解释：** TP 可以让单卡放不下的模型跨卡部署，但同一次前向要通信。DP 要求每张卡能放完整模型，适合分散请求。PP 的典型边界是不同层之间传激活；1P1D 两卡都保留完整 36 层模型，只是执行阶段不同，传的是请求的 KV，因而不能说成 Pipeline Parallel。本轮固定相同模型、FP16 主干和 FP32 Head、输入、64 输出 tokens、到达时间、全局最多 4 个驻留请求，并让 KV 数据缓冲区合计为 2.25 GiB。它控制了工作量和部分资源，但 Single 只有一张卡，不能声称硬件资源完全相同。

**项目实际实现：** [Session.__enter__()](../experiments/distributed/runtime.py) 为 Single 启动 1 个 worker，为另外三种模式启动 2 个；[benchmark.main()](../experiments/distributed/benchmark.py) 固定配置，[run_load()](../experiments/distributed/runtime.py) 控制全局驻留；[metrics()](../experiments/distributed/common.py) 分别计算总吞吐与 `tokens / total_s / gpu_count`。

**具体例子：** burst 中 Single 是 24.012 tokens/s，DP2 总吞吐是 30.303，增加 26.20%；除以两张卡后却只有 15.151 tokens/(GPU·s)，低于单卡的 24.012。说明这轮增加副本可以更快完成整组请求，但没有线性翻倍，也没有提高每卡效率。

**常见追问与边界：** 为什么没做 PP？本次范围是单机双卡可运行的三种部署路径，PP 未实现。两组负载每模式只正式测一次，不能从 0.1% 的差异下稳定优劣结论；本轮也关闭了 Graph、DFlash、AWQ、Prefix、Chunk，不与其他表相乘计算加速比。

#### 21. TP2 的 Column/Row Parallel 怎样落实到 Qwen3？为什么更慢？

**面试官问题：** 你说矩阵切分，具体切哪一维？QKV、O Projection、FFN 分别怎么处理？

**30 秒回答：** Column Parallel 按输出特征切，每卡算不同输出；Row Parallel 按输入特征切，每卡算相同输出的一部分，再 AllReduce 求和。Qwen3 的 Q/K/V 和 FFN Gate/Up 用 Column，O 和 Down 用 Row。TP2 把 Q heads 从 32 分成每卡 16，把 KV heads 从 8 分成每卡 4，也分摊权重、Head 和 KV 显存。但小负载下新增通信和运行时成本可能超过计算节省，所以本轮显存约减半，吞吐却下降。

**深入解释：** 使用 `Y = X Wᵀ` 约定，PyTorch 权重存成 `[out_features, in_features]`。所谓 Column Parallel 是切 Y 的输出列，对应权重的第 0 维；Row Parallel 是切输入，对应权重第 1 维，别被命名误导。Column 的输出保持分片，使本地 Attention heads 或 FFN 激活能继续计算；Row 把各卡部分和归约后恢复复制的 hidden。不能说每层 Column 都立刻 AllGather，否则会掩盖“分片一路传到 Row”的意义。

**项目实际实现：** [ColumnParallelLinear、QKVColumnParallelLinear、MergedColumnParallelLinear 和 RowParallelLinear](../src/myvllm/layers/linear.py) 定义分片和加载；`RowParallelLinear.forward()` 调用求和 AllReduce。[Qwen3Attention.__init__()/forward()、Qwen3MLP](../src/myvllm/models/qwen3.py) 连接 QKV→本地 Attention→O，以及 Gate/Up→SiluAndMul→Down。[load_tensors()](../experiments/common/tp_support.py) 将官方分离权重映射到合并模块并严格检查漏载；正式执行器的 FP32 Head 通过 [install_arithmetic()](../experiments/common/target.py) AllGather 词表 logits。

**具体例子：** 把 4 维输入分为 `x₀=[x0,x1]`、`x₁=[x2,x3]`。Row 的两卡各算 `y₀=x₀W₀ᵀ`、`y₁=x₁W₁ᵀ`，AllReduce 得 `y=y₀+y₁`。本轮 burst 的峰值 allocated 从 Single 20.080 GiB 变成 TP2 每卡 10.098 GiB；总吞吐却是 24.012→21.732 tokens/s。

**常见追问与边界：** 通信确实存在，PHB 拓扑、小 batch、同步/RPC 和参考算术路径都是可能成本，但没有 kernel profiling，不能断言 NCCL 是唯一瓶颈。TP 同历史 logits 容差通过，完整自由生成仅 47/48 序列一致；该条分叉原因尚未定位，不能写“完全等价”。这些并行算子的基础来自原框架，本项目工作包括严格加载、显式通信组、受控执行与验证。

#### 22. NCCL 做什么？AllReduce、AllGather、Send/Recv 有什么区别？

**面试官问题：** 不要只说“用了 NCCL”，它在你的代码里传什么、怎样避免 DP 被误分片？

**30 秒回答：** NCCL 是 NVIDIA Collective Communications Library，提供 GPU 间集合通信和点对点通信。AllReduce 把各卡部分结果归约，让每卡得到同一份完整结果；AllGather 收集不同分片但不求和；Send/Recv 是指定两端的点对点传输。本项目 TP 的 Row Linear 用 AllReduce、FP32 Head 用 AllGather；DP 没有频繁的模型层间跨卡通信；1P1D 用 Send/Recv 搬 Prompt 的全层 KV。

**深入解释：** 通信必须限定参与者，也就是 process group。TP2 的模型组包含两张卡，因为它们共同计算一个模型；DP2 和 1P1D 虽然 WORLD 有两卡，每个完整模型副本的模型组却只有自己。否则 `get_world_size()` 返回 2，会让副本错误地加载一半权重，还可能等待另一张执行不同请求的卡参与 collective。通信原理见 [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) 与 [Point-to-point](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)。

**项目实际实现：** [worker.main()](../experiments/distributed/worker.py) 初始化 WORLD，并为 DP/PD 创建 `[0]`、`[1]` 两个单 rank 组。[ModelGroup、bind_model_group()](../experiments/common/target.py) 只把仓库模型模块的 `dist` 绑定到模型组，不修改整个 `torch.distributed`。正式 Head 的 AllGather 在 `install_arithmetic()`；原 [ParallelLMHead.forward()](../src/myvllm/layers/embedding_head.py) 则是 Gather 到 rank 0，不能把两条实现混写。worker 的 `send_kv/recv_kv` 分支直接用 WORLD 中的 rank 0/1。

**具体例子：** AllReduce：卡 0 有 `[1,2]`、卡 1 有 `[3,4]`，求和后两卡都是 `[4,6]`。AllGather：两卡各有词表的一段 `[1,2]`、`[3,4]`，结果是 `[1,2,3,4]`。Send/Recv：P 把一个请求的 KV tensor 发给 D，其他请求的 KV 不参与求和。

**常见追问与边界：** CPU 控制器 RPC 只传命令、token 和元数据，实际 KV 是 GPU tensor 经 NCCL 传输，不是通过 Python socket 搬 KV。Send/Recv 两端的形状、数量、dtype 要一致。现有同步实现没有声称使用异步传输、通信计算重叠或特殊 NCCL 算法调优。

#### 23. DP2 怎样路由请求？为什么两个副本没有让吞吐翻倍？

**面试官问题：** DP2 的两个副本怎样接请求、各自怎样管理 KV？提高的是哪种吞吐？

**30 秒回答：** DP2 在两张卡各放一个完整 Qwen3-8B，各自维护 KV 池。控制器把新请求发给驻留请求较少的副本，并让同一请求始终留在该副本。全局最多驻留 4 个，每副本最多 2 个。它提高的是多请求总吞吐，实测比单卡约高 19%～26%；并没有把一个请求拆给两卡，也没有保证单请求更快或吞吐翻倍。

**深入解释：** 普通推理 DP 的请求和 KV 独立，不像训练 DP 每步同步梯度。路由时数的是 resident requests，包含已接纳但尚未完成的请求，不只数当前 Forward 中的请求。每个 worker 一次做一条完整 Prefill，其他时候把本卡 ready 请求合批 Decode；结束后释放本卡页所有权，再给等待请求补位。副本运行可以并行，但控制器/RPC 开销、Prefill 与 Decode 干扰、批次大小与到达模式都会影响收益。这些是机制上的可能原因，不能替代 profiling 结论。

**项目实际实现：** [run_load() 内的 admit()/schedule()/receive()](../experiments/distributed/runtime.py) 用 `min(candidates, key=(驻留数, rank))` 确定性路由；`cap//2` 限制每副本驻留，`active[rank]` 与 `pools[rank]` 独立。[PagePool.allocate()/release()](../experiments/distributed/common.py) 管页所有权。分布式实验按 Prompt+最大输出预留页，不调用普通 BlockManager 的 Prefix/LRU，这两种管理路径不要混淆。[worker.main()](../experiments/distributed/worker.py) 为每卡构建 `tp=1` 的完整 Target。

**具体例子：** 卡 0 驻留 A、B，卡 1 驻留 C；D 到达时路由到卡 1。A 完成后释放它在卡 0 的页，E 可以补到卡 0，其余请求不等待整个 batch 结束。burst 32 请求共 2048 输出 tokens，Single 用 85.290 s，DP2 用 67.585 s，总吞吐 24.012→30.303；stream 是 24.929→29.787 tokens/s。

**常见追问与边界：** 当前按请求数路由，没有按剩余 token、Prompt 长度或预计算力负荷预测，两个“各 2 请求”的副本不一定工作量均衡。DP2 两组共 48 请求逐 token 与 Single 一致、页全部归还；每负载仅一次正式测量。双卡每卡吞吐更低，不能说资源效率也提升了。

#### 24. 单机 1P1D 怎样迁移 KV？首 Token 和 Pending Token 在哪里？

**面试官问题：** D 会不会重新 Prefill？物理页号不同怎么办？P 何时能释放页？

**30 秒回答：** GPU 0 的 P 完成 Prompt Prefill 并生成首 token；这个 token 已知，但它自己的 KV 还没算。P 把 Prompt 的全部 36 层 K/V 打包，经 NCCL 传给 GPU 1 的 D，D 按自己的页表写入，然后从首个待计算 token 开始 Decode，不重新算 Prompt。控制器收到发送端和接收端两份完成响应后才释放 P 页。这是同步 handoff，不是异步流水线，也不是 PP。

**深入解释：** 两卡都持有完整模型。接纳时先预留 D 的目标页和 P 的 Prompt 页；P 输出首 token 后，历史 KV 只覆盖 Prompt。token ID 由控制器保留，不在迁移的 KV tensor 中。D 的第一次 Decode 读取迁移历史，输入该首 token，写入它自己的 KV，产生第二个输出。逻辑位置连续不代表两卡物理页号一致，因此传输按逻辑 token 次序打包，再按目的页表重映射。迁移开始要等两 worker 空闲，先提交接收和发送，之后等待两端响应；ACK 指 worker result，不是另外的 NCCL ACK 报文。

**项目实际实现：** [Target.slots()/flat_cache()/pack()/unpack()](../experiments/common/target.py) 将页式缓存与 `[2,36,prompt_tokens,8,128]` FP16 tensor 互转。[worker.main() 的 send_kv/recv_kv 分支](../experiments/distributed/worker.py) 调用 `dist.send()/recv()` 并同步 GPU；[run_load() 的 schedule()/receive()](../experiments/distributed/runtime.py) 等两端完成才 `pools[0].release()`、置 D `ready=True`。`pd_contract` 断言 D Prefill=0、P Decode=0；预热校验源、接收及重映射后的 SHA256 相同。

**具体例子：** 257-token Prompt 在 P 用页 `[5,9]`，在 D 用 `[31,30]`。逻辑位置 256 从 P slot `9×256` 读取，写到 D slot `30×256`。迁移字节是 `2×36×257×8×128×2=37,896,192`，不包括首 token 的 KV；首 token 在 D 下一次前向才计算。

**常见追问与边界：** P 的 Prefill 与 D 对其他请求的 Decode 可由独立 worker 执行，但 KV handoff 仍同步，没有传输计算重叠。平均 handoff 13.03/12.42 ms 包括打包、等待、传输和写入，不能直接当纯链路带宽。两负载总吞吐与 DP2 近似持平，价值主要在最严重出字停顿；逻辑页归还后预分配显存池仍驻留。

#### 25. TTFT、TPOT、ITL 怎样区分？1P1D 改善了什么？

**面试官问题：** 你说最大 ITL 降了约九成，为什么 P99 没全面变好？是不是挑指标？

**30 秒回答：** TTFT 是入队到首 token，E2E 是入队到末 token，TPOT 是首 token 之后各间隔的平均值，ITL 是每一对相邻 token 的实际间隔。Max ITL 关注最严重一次停顿，P99 关注分布靠后的百分位，两者不能互换。本轮 1P1D 相对 DP2 把最大 ITL 从约 1.8～2.2 s 降到约 0.2 s，但 burst 的 P99 反而从 115.33 到 142.57 ms，我同时报告这个负面结果。

**深入解释：** 同卡执行长 Prefill 时，正在 Decode 的请求可能暂时得不到执行机会；把阶段放在不同卡可隔离这种干扰，但同时新增 KV 迁移、控制器等待等成本，所以不是每个间隔都会缩短。这里“阶段隔离”有调度实现作支撑，具体每个 stall 的因果仍需 trace/profiling 分析，不能凭指标确定唯一原因。P99 是所有请求相邻输出间隔合并后的第 99 百分位，约 99% 间隔不超过它；本实现排序后线性插值。Max 则直接取最大观测值。burst 有 32×63=2016 个样本，少数极大值减少，不意味着第 99 百分位也必然减少。

**项目实际实现：** [metrics()/percentile()](../experiments/distributed/common.py) 用控制器收到 token 的 monotonic 时间计算，含排队、RPC 和运行时，不含 HTTP 网络。[run_load().receive()](../experiments/distributed/runtime.py) 为 token 记录时间；[正式结果](../results/distributed/summary.csv) 分别保留全体/短请求的 P99 和 Max，不能只挑最有利的一列。

**具体例子：** 入队为 0 s，四个 token 在 `[2.0,2.1,2.2,3.2]` s 到达：TTFT=2 s，E2E=3.2 s，ITL=`[0.1,0.1,1.0]` s，TPOT=`(3.2−2.0)/3=0.4` s，Max ITL=1 s。若整组 2 请求共 8 输出 tokens，完成窗口为 4 s，则总输出吞吐为 `8/4=2 tokens/s`，请求吞吐为 `2/4=0.5 req/s`；用了两卡时每 GPU 吞吐为 `2/2=1 tokens/(GPU·s)`。这三种吞吐不能混单位；整组耗时也不是请求 E2E 的均值。以下为正式全体 ITL，单位 ms：

| 负载 | DP2 P99 | 1P1D P99 | DP2 Max | 1P1D Max |
|---|---:|---:|---:|---:|
| burst | 115.33 | 142.57 | 2157.99 | 207.92 |
| stream | 150.29 | 143.25 | 1792.52 | 200.24 |

**常见追问与边界：** 结论是本次两负载最大停顿下降，不是所有尾延迟全面改善。每负载只正式测一次，P99 也是有限样本统计，没有稳定置信区间。投机解码一次返回多个 token 的可见时间与逐 token Decode 不同，不能拿平均 TPOT 代替完整 ITL 分布。

### F. AWQ、实验设计、正确性与边界

#### 26. AWQ 为什么叫 Activation-aware？W4A16 怎样执行，为何只有内存收益？

**面试官问题：** 既然主要量化 Weight，Activation 起什么作用？你做了校准吗，INT4 怎样进入 GEMM？

**30 秒回答：** AWQ 是 Activation-aware Weight Quantization，属于低比特权重量化。Activation 统计用来判断哪些输入通道对输出更敏感，指导离线缩放与量化，但在线 W4A16 仍是 4-bit 权重、16-bit 激活。本项目使用官方 Qwen3-8B-AWQ，没有自行训练或校准；我完成 checkpoint 适配、Triton 解包 GEMM、参考检查和推理对照。权重及元数据减少 62.77%，但当前推理更慢，所以没有量化加速结论。

**深入解释：** 理论上，线性层某输入通道激活幅度大，同样的权重误差更容易放大到输出。AWQ 观察离线校准输入的 Activation，寻找通道缩放，让重要权重更好地被低比特表示；对应逆缩放保持量化前函数等价。它不是把“重要权重全部保留 FP16”，也不意味着在线每个 token 重跑校准。[AWQ 论文](https://arxiv.org/abs/2306.00978) 的这套方法与本项目消费现成 checkpoint 要分开讲。

**项目实际实现：** [Reader、merged()、load_awq_model()](../experiments/awq/awq_loader.py) 检查 AWQ、4bit、group128、zero_point、GEMM 格式，消费 903 个张量并替换 144 个合并模块。每个 int32 打包 8 个 4-bit 值；[AWQLinear.forward() 与 Triton _gemm()](../experiments/awq/awq_linear.py) 在 GEMM tile 内移位、`&15` 解包权重和 zero point，按输入维 group 查 scale，执行 `w≈(q−z)×scale`，转 FP16 后与 FP16 Activation 做 dot、FP32 累加。没有常驻完整反量化权重；`dequantize_reference()` 只供小算子对照。

**具体例子：** 某组 `q=9,z=8,scale=0.1`，近似权重为 0.1；与激活 2.0 相乘贡献约 0.2。三种内存口径必须分开：

| 口径 | FP16→AWQ | 解释 |
|---|---|---|
| 权重及量化元数据 | 16,381,470,720→6,098,479,104 bytes，−62.772% | 包含未量化参数、scale/zero 等 |
| 上述加额外 FP32 Head | 18,870,790,144→8,587,798,528 bytes，−54.492% | 本执行器保留原 Head 并增加 FP32 副本 |
| short-b1 峰值 allocated | 19.873→10.314 GiB | 含 KV、临时张量等运行时占用，不是纯权重 |

short-b1 每请求生成 32 tokens，耗时 2.642→9.332 s，吞吐 12.111→3.429 tokens/s；其他三种 case 也更慢。

**常见追问与边界：** 更少权重读取不保证抵消解包/反量化、tile 利用率与其他开销；没有 profiling，不能给唯一瓶颈结论。[check_operators()/weight_storage()](../experiments/awq/bench_awq.py) 完成 6 项解包参考检查和存储统计，各 case 三轮中位数，另验有限值生成和页回收。它没有证明任务质量无损，也没有要求量化输出与 FP16 完全一致；未量化 KV，不支持本次 AWQ 与 TP/Graph 的联合性能结论。


#### 27. 为什么用四模式对照？2.99× 和 1.52× 究竟相对谁？

**面试官问题：** 你的收益来自投机算法还是 CUDA Graph？为什么不只展示最大的加速比？

**30 秒回答：** 我比较 AR Eager、AR Graph、DFlash Eager 和 DFlash Graph，分别观察固定算法时开 Graph、固定执行方式时开投机，以及联合收益。主要对照是已经开启 Graph 的 AR，而不是较慢的 AR Eager。单请求和三请求的 2.99×、1.52× 都相对 AR Graph，使用同一个受控执行器、相同输出工作量，不能说成相对 vLLM 或所有负载都能达到的加速。

**深入解释：** 四模式形成一个 2×2 对照：

| 模式 | 候选提议与 Target 验证 | Target 后续前向方式 |
|---|---|---|
| AR Eager | 每轮输入一个 Pending、输出一个 Token | Eager |
| AR Graph | 同上 | Q=1 Graph |
| DFlash Eager | Draft 提议后，Target 验证整个块 | Eager |
| DFlash Graph | 同上 | Q=16 Graph |

AR Graph 对 AR Eager 观察 Graph 在 AR 中的收益；DFlash Graph 对 DFlash Eager 观察 Graph 在投机中收益；DFlash Graph 对 AR Graph 才回答“已有 Graph 后，加投机还有没有用”。两种技术有相互作用，不能假设收益独立。相同负载和统一分母下可直接算比值，不能把其他实验的加速比拿来相乘。

**项目实际实现：** [benchmark.MODES、main()/summarize()](../experiments/dflash_graph/benchmark.py) 控制两开关、旋转运行顺序并汇总中位数；四模式都经 [GraphDFlashRuntime](../experiments/dflash_graph/graph_runtime.py) 和同一个 [TargetGraphExecutor](../experiments/dflash_graph/graph_executor.py)。都采集五层特征、驻留同一 Draft 和四张图，让开关之间的比较保持一致。AR 模式不调用 Draft，但仍保留其资源；这不是独立 AR 部署的最低显存基线。

**具体例子：** 单请求输出 256 Token：AR Graph 5.513 s、46.43 tokens/s；DFlash Graph 1.846 s、138.70 tokens/s，`138.70/46.43≈2.987`。三请求共 768 Token：11.653→7.651 s、65.91→100.38 tokens/s，约 1.523×。三请求 DFlash Eager 只有 40.93 tokens/s，低于 AR Graph 的 65.91；所以“投机本身普遍优于已经优化的 AR”不成立。

**常见追问与边界：** 计时包含初始 Prefill、Draft、Target、元数据、输出读取、队列及接受回滚，排除加载、捕获、预热和诊断。所有阶段的 per-query Attention 与固定算术是自己的参考配置；没有生产 vLLM 对照，没有把 Graph 的全部改善归给投机算法。

#### 28. 为什么固定输出长度并忽略 EOS？自然生成 EOS 后怎么继续？

**面试官问题：** 64 或 256 个输出是实际生成还是 PAD 补齐？不同实验为何用不同长度？

**30 秒回答：** 固定输出长度是为了控制各模式完成相同工作量，避免某个模式早停后看起来更快。`ignore_eos=True` 关闭 EOS 停止条件，EOS 若被选中仍是一个真实生成 Token，下一轮继续以前向和 argmax 得到后继，直到输出配额用完，不用 PAD 填满。ShareGPT 和多卡每请求 64 个输出，DFlash 四模式每请求 256 个，长度不同所以只能在各自表内比较。新三方法对照也实际输出 256 个 Token，通过 `stop_token_ids=None` 保持固定工作量，不能误认成与 100 请求 ShareGPT 相同的负载。

**深入解释：** EOS 是模型词表中的特殊 Token；“看到 EOS 就结束”由生成控制逻辑实施，不是 GPU 无法继续计算。忽略 EOS 会让续写可能超出自然回答的结束点，这适合固定工作量的系统测量，不等于自然对话体验或任务质量评测。DFlash 一轮可提交多个输出，最后用剩余配额截断，不能把多算但未提交的候选计入输出吞吐。

**项目实际实现：** ShareGPT 的 [bench_sharegpt.main()](../experiments/sharegpt/bench_sharegpt.py) 设置 `SamplingParams(max_tokens=64, ignore_eos=True)`，Greedy 实际用 argmax；[Scheduler.postprocess()](../src/myvllm/engine/scheduler.py) 仅在不忽略 EOS 时因 EOS 结束。DFlash [generate()](../experiments/dflash_graph/dflash_runtime.py) 按 `len(outputs)==max_new_tokens` 结束，[commit()](../experiments/dflash_graph/state.py) 按 remaining 截提交。多卡的 [run_load().receive()](../experiments/distributed/runtime.py) 按 `config['max_new_tokens']` 检查完成；AWQ [bench_awq.main()](../experiments/awq/bench_awq.py) 另用每请求 32 个输出。

**具体例子：** 要求输出 4 个 Token，模型依次选出 `[A,EOS,B,C]`。忽略 EOS 时这四项均经过真实生成，输出计数为 4；正常停止策略则在第二项结束。DFlash 还剩 3 个配额、当前能接受并补出 6 项时，只提交前 3 项，吞吐分子只加 3。

**常见追问与边界：** 256 个输出能观察更多轮投机推进，64 个输出限制大量请求实验的总工作量，这是两种设置的用途解释；没有长度消融证明它们最优。长输出改变 Prefill 占比、接受率演化和请求补位时机，不能把 256-token 固定 Prompt 实验的倍数直接套到 64-token ShareGPT。当前未完成自然 EOS、变长输出的正式对照。

#### 29. 三轮中位数就可信吗？怎样复现，计时包含什么？

**面试官问题：** 你怎样避免把首次编译、预热、缓存状态或机器波动当成优化收益？

**30 秒回答：** 我先固定模型、精度、输入 Token、输出长度、到达规则、并发和 KV 预算，再把加载、预热、Graph 捕获及诊断排除在计时外。DFlash 四模式各正式测三轮、旋转顺序并取中位数，还保留范围与源码记录；ShareGPT 和多卡每配置只有一次正式测量，我只报告当次观察，不声称有稳定置信区间。重复测量、正确性与可追溯性提供不同证据，不能相互代替。

**三方法正式测量的具体口径：** 十二条唯一输入在每轮各执行一次，GSM8K / HumanEval / MT-Bench 各四条，平均输入 120.33 Token；每请求 256 输出，三轮。七种投机模式各 36 次生成，加上三个 Draft 进程各自 36 次 AR，共 360 次执行、92,160 输出 Token。252 是投机执行次数，不是 252 条独立 Prompt。每轮吞吐为 `12×256 / 十二请求累计耗时`，再取中位数；加速比取同轮配对比值中位数，不能把每个 Prompt 的加速比直接平均。所有 252 组投机输出和跨进程/跨轮 AR 均一致。

**项目实际实现补充：** [validate_inputs()/_worker()/collect_report()](../experiments/speculative_compare/benchmark.py) 锁定权重、源码与输入，轮换模式顺序并保存原始 Token；[prepare_legacy_formal.prepare()](../experiments/speculative_compare/prepare_legacy_formal.py) 核对通过回归的计算源码，仅扩大请求和轮数、关闭 Top-2。新路径计时包括 Prefill、Draft、Target、接受/回滚和必要主机同步；未测 Serving TTFT/ITL，也未做任务答案评分。见 [正式报告](../results/speculative_compare/legacy-target-formal-20261021/README.md)。

**深入解释：** 首次 Triton 编译、Graph 捕获、初始化和正式生成不是同一种成本。若目标是生成稳态性能，应预热后测；若目标是冷启动，则要另设指标，不能偷偷混入某个模式。请求时延包含排队，总耗时从开始处理整组到全部完成，二者不能互换。三轮中位数能减弱一个异常值影响，但样本仍少，既没有覆盖多台机器，也不构成生产负载保证。

**项目实际实现：** [benchmark.main()/summarize()](../experiments/dflash_graph/benchmark.py) 在正式前完成 Capture、8 组诊断和四模式预热，正式关闭有限值逐步诊断并旋转顺序；[DFlashRuntime.generate()](../experiments/dflash_graph/dflash_runtime.py) 同步 GPU 后开始计时，结束再同步。ShareGPT [trial()](../experiments/sharegpt/bench_sharegpt.py) 重建块管理器，以每模式冷前缀缓存测量，逐步同步记录 Token 时间；多卡 [Session/run_load()](../experiments/distributed/runtime.py) 包含控制器/RPC 与迁移。数据版本与选择由 [prepare_data](../experiments/sharegpt/prepare_data.py) 固定；[SOURCES.json](../results/dflash_graph/SOURCES.json) 等记录原始归档 SHA256，[repro.verify_source()/snapshot_sources()](../experiments/common/repro.py) 负责发布源码完整性和快照。

**具体例子：** DFlash Graph 单请求三轮范围为 1.837～1.846 s，中位数约 1.846 s；三请求范围为 7.634～7.753 s，中位数 7.651 s。四图捕获另耗 3.533 s，不计入这组生成耗时。ShareGPT 的 121.881 s 是一次完整 100 请求生成窗口，不能称为“三轮平均”；分布式 0.1% 左右差异也不能据单次运行判定稳定胜负。

**常见追问与边界：** 数据集与 Draft/AWQ 固定 revision，基础模型保留 config/tokenizer 哈希，但历史完整基础 Checkpoint commit 未被归档，复现完整性仍有这个缺口。模型加载、分词、HTTP/网络均不在正式生成时延中。源码 Manifest 证明文件版本，不能证明代码没有 bug；新目录整理后的 CPU 检查也不等于重新完成 GPU 实验。

#### 30. 为什么使用固定算术、FP32 Head 和 per-query Attention？项目都是 FP16 吗？

**面试官问题：** Qwen 常用 BF16，你为什么改成 FP16？参考配置是否牺牲了速度？

**30 秒回答：** 项目不是统一 FP16：ShareGPT 是 BF16 主干加 FP32 Head，DFlash 和多卡是 FP16 Target 加 FP32 Head，官方 Draft 保持 BF16。不同 Batch/Query 形状可能改变低精度计算，近似相等的 Logits 会发生 argmax 分叉。当前通过配置用固定算术和统一 Attention 建立可比较的参考，这可能增加成本；没有同配置 BF16 对 FP16 消融，不能说 FP16 本身更快或更正确。

**深入解释：** 固定算术不等于所有 Tensor 都改成 FP32。Linear 等按固定行数分段、补行再裁回，使 AR 与多位置验证更接近同一计算方式；FP32 Head 处理最终词表分数；每个 Query 用自己的因果长度，避免不同 Prefill 路径的数值差异干扰调度比较。它是验证策略，也是性能条件：不能将其测量当成优化充分的 A100 推理上限，或把参考路径通过写成旧 Flash Prefill 已修复。

**项目实际实现：** [fixed_rows_forward()/fixed_linear()](../experiments/common/fixed_shape.py) 实现固定行数与裁切；[Target.__init__()/install_arithmetic()](../experiments/common/target.py) 固定 FP16、初始化 RoPE 精度、替换 Attention 并保留 FP32 Head；[canonical_runtime.install_model()](../experiments/sharegpt/canonical_runtime.py) 则明确断言 BF16 Attention/KV，安装独立参考适配。[AWQLinear.forward()](../experiments/awq/awq_linear.py) 要求 FP16 激活，不能只换一个 dtype 参数就声称全部支持 BF16。

**具体例子：** AR 本轮只有 1 个真实行，DFlash 验证有 16 行；固定 Linear 行数 32 时，两者都在该算子内部补到 32 再裁回真实行。补行不会成为 Token，不进入 Attention/KV/采样。FP32 LM Head 的额外权重副本占 2,489,319,424 bytes，因此权重存储、运行峰值都要纳入这个成本。

**三种 Draft 对照如何接 FP16：** 三个官方 Draft 仍为 BF16，Target 的五层特征在 [install_draft_feature_bridge()](../experiments/speculative_compare/precision.py) 的 `draft.fc` 边界转换为 BF16，不更改 Target KV。HF FP16 短回归曾达到 20/21，仍有 Confidence 0.8 的 `mt-bench:9` 在索引 24 分叉。最终 [LegacyTarget](../experiments/speculative_compare/legacy_target.py) 复用旧 Target 的固定 Linear/RMS/激活、逐 Query PagedAttention、真实 FP32 Head，并关闭 TF32 和两种 reduced reduction；Draft SDPA 限定 MATH。组合配置通过十二输入三轮，未证明 FP16 是唯一原因。新 AR 同样执行此配置，不借用旧 HF 基线。

**常见追问与边界：** BF16 与 FP16 都是 16 bit，选择 FP16 不是节省一半显存。当前 FP16 Target 是已验收配置的延续，数值通过也依赖算术与状态修正，不是单靠 dtype。若迁移 BF16，要检查 dtype 断言、KV、算子和 Draft 特征转换，再重新做完整生成对照；该最终配置的 BF16/FP16 性能与质量消融未完成。

#### 31. 生成分叉时，怎么区分 KV 状态错误与浮点数值差异？

**面试官问题：** 为什么不把容差放宽？你怎么证明 Graph 没有静默写错缓存？

**30 秒回答：** 先保存完整 Token 和 Trace，找到第一处分叉，再在相同历史上比较输入、位置、页表、有效 KV、Hidden 和 Logits。状态错位、拒绝 KV 泄漏、旧 Graph 元数据或输出覆盖都要先排除，然后才看算术形状、精度和 Top-2 Margin。极小 Margin 只说明数值误差可能改变 argmax，不能证明没有状态 bug。同历史检查负责定位，完整自由生成负责验证多轮累积行为。

**深入解释：** 分叉之后两个模式的历史已经不同，后续 Logits 大幅不同没有定位意义；必须回到共同历史。Teacher-forced 检查是强制给相同前缀，隔离一次前向的差异；自由生成则让各模式使用自己的输出，能暴露多轮缓存与调度错误。页全部归还只验证所有权账目，不能说明页内容正确。放宽浮点容差可用于预先约定的数值验收，但不能让不同 Token 变成“逐 Token 完全一致”。

**为什么 AR 与多 Token Verification 会数值不同？** 即使因果可见范围正确，Q=1 与 Q>1 也可能使矩阵乘和 Attention 选择不同计算分块或归约顺序；浮点加法不满足结合律，低精度舍入会改变近似并列的 argmax。输出 Token 相同也不代表此前写入的 K/V 逐元素相同，差异还可能沿后续步骤累积。因此要同时控制当前 Query 形状和历史 Cache 的计算轨迹；这是机制解释，具体归因仍需逐项控制，不能断言每个分叉都是 BF16 或某个 Kernel 导致。

**项目实际实现：** [compare_generation()/graph_diagnostics()](../experiments/dflash_graph/benchmark.py) 分别比完整 Token、投机轨迹和同历史张量；[worker.probe()](../experiments/distributed/worker.py) 做 TP 的固定历史 Logits 检查；[compare_tokens()](../experiments/distributed/common.py) 另报完整序列一致率。Graph 的 [run()](../experiments/dflash_graph/graph_executor.py) 每次覆盖四类输入并 clone 输出，是防止陈旧状态和所有权污染的实际措施。

**具体例子：** 一个纯数值示例：同一历史下两个最大 Logits 为 10.000 与 9.999，微小扰动就可能交换大小；这解释了“误差小却 Token 不同”的可能性，但还需排除错页、错位置等原因。正式 TP 同历史检查 `max_abs=0.08344、relative_RMSE=0.001803`，通过预设 `0.35/0.01` 阈值；完整生成却只有 47/48 序列一致。这两个结果并不矛盾，后者也没有被前者抹掉。

**实际数值排查案例：** 旧 Smoke 的 `humaneval:19` 第 5 轮，Target 在位置 139/140 验证 `[6337,429]`。实际 BF16 Head 对 Token 5980 与 429 均算出 31.625，真实 argmax 为 429；相同 Hidden State 的 FP32 Head 得分分别为 31.637157、31.575722，恢复为 5980。按原提议/验证/crop 历史重放，36 层 K/V 与现场逐元素相同；按 AR 单步重建却不同，支持“分块数值历史和 Head 并列参与了这个位置分叉”的解释，未发现随机 Cache 污染。诊断记录见 [局部控制实验](../results/speculative_compare/diagnostic-20261019/README.md)。

只提高 Head 后，新的短回归在索引 15 仍分叉；HF FP16 短回归也只达 20/21。后者没有保存失败位置的 Top-2，不能把旧索引 26 的并列套用过去。最终接回稳定 Target 后，短回归 21/21、正式 252/252；这是组合配置的实测修复，不是对原 HF 所有分叉的单因素归因，也不是证明任何输入都与原 HF AR 相同。原始失败均保留。

**Cache 适配怎么保证语义：** `LegacyCache.get_seq_length()` 以已计算 Token 历史界定有效 KV；官方循环的 `crop(start)` 进入 `pool.trim()`，裁掉拒绝候选越界整页；每个 Query 的 `context_lens=position+1` 隔离保留页中的无效尾部。仅 Target Cache 工厂被适配，Draft DynamicCache 和有效 Target 特征截取仍由官方循环处理，请求 finally 释放 Target 页。本轮没有修补官方验证的某个已确诊索引错误。

**常见追问与边界：** 当前 Graph 八组同历史检查精确一致、24 次完整生成通过，是已测形状和负载的证据。TP 那条生成分叉尚未定位；旧 Flash Prefill 完整模型分叉也未被参考配置证明修复。有限值、文本合理、局部算子 allclose、无页泄漏，都不能单独替代完整模型验收。

#### 32. 哪些结论已经成立，哪些必须留给后续？

**面试官问题：** 如果要继续做，先做什么？为什么现在还不能作为生产系统部署？

**30 秒回答：** 当前较完整的是 DFlash 与 Target Graph 四项对照，以及新三方法十二输入三轮对照；Prefix 在同一 Chunk 配置上有收益，DP2 提升双卡总吞吐，1P1D 减少本轮最大停顿。负面结果也明确：Chunk 单独更慢，TP2 更慢且有生成分叉，AWQ 更慢；新 DSpark 的 Confidence 截断也未提高整体吞吐。我会先补代表性负载、重复测量和 Profiling，再针对证据改热点，而不是继续叠加未经联合验证的功能。

**深入解释：** 算法机制成立、代码存在、一次运行通过和生产可用是不同层次。生产需要更多模型/长度/并发/到达模式、自然 EOS、取消与故障处理、服务接口和长期资源测试。普通引擎、旧 Graph、分布式和新 Draft 适配各自验收，不代表 DFlash、Graph、Prefix、Chunk、AWQ、TP、DP、PD 可以任意组合。当前“速度不错”的结论必须带上模型、精度、硬件、工作量、次数和比较对象。

**项目实际实现：** [PROJECT_SUMMARY 的证据与边界](PROJECT_SUMMARY.zh.md) 与 [results 摘要](../results/README.md) 是当前结论依据；[scripts/run.py](../scripts/run.py) 提供独立入口，不是联合服务配置。源码里的分支或开关，不能代替对应 GPU 运行证据。

**具体例子：** 若下一步研究 Prefix，应补 Full+Prefix 的 Prefix-only 对照，而不是把 Chunk+Prefix 当成它；若研究 1P1D，应单独设计异步 Handoff 与同步方案的相同双卡对照；若研究 DFlash，应先区分 Draft 串行提议、Target 验证和 CPU 状态开销，再决定是否做 Draft 合批或 Graph。这些是后续方案，不是已有成果。

**常见追问与边界：** 未完成项包括：Prefix-only；更多并发/到达/输入输出长度及自然 EOS；双卡多轮统计；完整 Profiling；Draft 合批/Graph；异步 KV Handoff；PP；其他模型正式适配；所有功能联合验收；AWQ 自主校准/任务质量；最终参考配置 BF16/FP16 对照；生产 HTTP 服务与生产 vLLM 性能对比。说“尚未验证”比用其他表的数字推导不存在的结果更准确。

## 5. 实验结果与局限

### 5.1 正式配置与比较对象

| 实验 | 模型与精度 | 负载与测量次数 | 可以回答的问题 |
|---|---|---|---|
| DFlash＋Graph | Qwen3-8B FP16 Target＋FP32 Head，BF16 官方 Draft；单 A100 40GB | 固定 1 请求/3 请求，每请求 256 输出；四模式各 3 轮中位数 | 本执行器已有 AR Graph 后，投机是否仍有收益 |
| ShareGPT | Qwen3-8B BF16＋FP32 Head；单 A100 40GB | 100 请求同时入队、驻留≤8、每请求 64 输出；每模式 1 次 | Full、Chunk、Chunk+Prefix 在固定调度负载下的取舍 |
| Distributed | Qwen3-8B FP16＋FP32 Head；Single 一卡，其余双 A100 PHB | burst 32/stream 16，驻留总上限4、每请求64输出；每模式每负载1次 | 加卡扩展收益，以及相同双卡资源下的部署取舍 |
| AWQ | 官方 Qwen3-8B-AWQ，FP16 激活＋FP32 Head；单 A100 40GB | short/long × batch1/4，每请求32输出；各3轮指标中位数 | 本量化实现的存储、显存与性能成本 |
| 三种 Draft | Qwen3-8B 原 Target FP16＋FP32 Head，官方 BF16 Draft，MATH，无 Graph | GSM8K/HumanEval/MT-Bench 各4条，B=1，每条256输出，3轮 | 方法对照和同 DSpark checkpoint 的 Markov/Confidence 推理消融 |

各实验都使用已说明的参考算术，关闭项和计时口径不同。以下表格只在同一实验内比较，不横向拼出一个“最终总加速比”。

### 5.2 DFlash 四模式：主要成果与对照

| 模式 | 单请求总耗时 s | 单请求 tokens/s | 三请求总耗时 s | 三请求总 tokens/s |
|---|---:|---:|---:|---:|
| AR Eager | 25.416 | 10.07 | 51.343 | 14.96 |
| AR Graph | 5.513 | 46.43 | 11.653 | 65.91 |
| DFlash Eager | 5.372 | 47.65 | 18.763 | 40.93 |
| DFlash Graph | 1.846 | 138.70 | 7.651 | 100.38 |

主结论是相对 AR Graph 为 **2.987× / 1.523×**，总耗时降低 **66.52% / 34.34%**。三请求驻留上限 2，共 768 输出，吞吐是总输出而非单条速度。24 次运行、48 次请求执行、12,288 输出 Token 匹配，八组同历史前向精确一致；正式 Target 验证实际 Replay 且 Fallback=0，页全部归还。Draft 和初始 Prefill 仍在图外，资源同驻留，不能外推生产引擎优势或最低部署显存。[原始精选指标](../results/dflash_graph/summary.json)

### 5.3 ShareGPT：Prefix 的基线不可换

| 配置 | 总耗时 s | 平均 TTFT s | 平均 E2E s | 平均 TPOT ms | 总输出 tokens/s |
|---|---:|---:|---:|---:|---:|
| Full Prefill | 132.397 | 64.675 | 69.913 | 83.134 | 48.339 |
| Chunk only | 146.539 | 72.091 | 80.356 | 131.181 | 43.674 |
| Chunk+Prefix | 121.881 | 68.250 | 75.134 | 109.267 | 52.510 |

Prefix 对 **Chunk only** 吞吐 **+20.23%**；对 Full 为 **+8.63%**，但平均 TTFT **+5.53%**、E2E **+7.47%**，TPOT 也没有保持不变。三配置均完成完整 Prefill+Decode 并逐 Token 一致。20% 是受控重放请求占比，实际缓存复用为 **13,568/78,889=17.20%**。每模式一次，Full→Chunk 还同时改变优先级，未完成 Prefix-only。[原始精选指标](../results/sharegpt/summary.json)

### 5.4 多卡：同时报告总吞吐与每卡效率

| 负载 | 模式 | GPU 数 | 总输出 tokens/s | tokens/(GPU·s) | 最大 ITL ms | 全体 P99 ITL ms | 序列一致 |
|---|---|---:|---:|---:|---:|---:|---|
| burst | Single | 1 | 24.012 | 24.012 | 4400.130 | 1864.866 | 基线 |
| burst | TP2 | 2 | 21.732 | 10.866 | 4914.372 | 2056.610 | 31/32 |
| burst | DP2 | 2 | 30.303 | 15.151 | 2157.990 | 115.328 | 32/32 |
| burst | 1P1D | 2 | 30.360 | 15.180 | 207.916 | 142.570 | 32/32 |
| stream | Single | 1 | 24.929 | 24.929 | 3077.076 | 619.347 | 基线 |
| stream | TP2 | 2 | 21.820 | 10.910 | 3531.997 | 705.923 | 16/16 |
| stream | DP2 | 2 | 29.787 | 14.893 | 1792.517 | 150.294 | 16/16 |
| stream | 1P1D | 2 | 29.749 | 14.875 | 200.235 | 143.253 | 16/16 |

DP2 对 Single 总吞吐 **+26.20% / +19.49%**，每卡效率下降；1P1D 对 DP2 吞吐近似持平，最大 ITL **−90.37% / −88.83%**，但 burst P99 上升。TP2 每卡峰值约 **20.08→10.10 GiB**，吞吐下降，完整生成 **47/48** 一致，分叉尚未定位。所有双卡结果每负载正式测一次；1P1D 为同步全层 KV Handoff，不能说成异步流水线。[原始精选指标](../results/distributed/summary.json)

### 5.5 AWQ：有存储收益，没有加速结论

权重及量化元数据 **16,381,470,720→6,098,479,104 bytes，−62.772%**；加额外 FP32 Head 后 **18,870,790,144→8,587,798,528 bytes，−54.492%**。short-b1 运行峰值 allocated **19.873→10.314 GiB**，这三种口径不能混成一个“显存减少比例”。short-b1 耗时 **2.642→9.332 s**、吞吐 **12.111→3.429 tokens/s**，四个 case 均更慢。903 个 Checkpoint 张量全部消费、六项量化 Linear 参考检查通过，不等于完成自主 Calibration、量化质量无损或生成逐 Token 一致验证。[原始精选指标](../results/awq/summary.json)

### 5.6 EAGLE3 DFlash DSpark 方法对照与推理消融

下表时间为每轮十二条请求完整生成累计耗时的三轮中位数，吞吐包含 Prefill，不是纯 Decode 或 Serving 吞吐。三个 Draft 进程各自 AR 的吞吐约 16.17～16.24 tokens/s，累计耗时约 189～190 秒；AR 也驻留相应 Draft。

| 模式 | tokens/s | 十二请求累计 s | 配对 AR Eager 加速 | 完整一致 |
|---|---:|---:|---:|---:|
| EAGLE3 | 47.260 | 65.002 | 2.934× | 36/36 |
| DFlash | 56.407 | 54.461 | 3.479× | 36/36 |
| DSpark No Markov | 34.283 | 89.608 | 2.111× | 36/36 |
| DSpark Markov Only | 63.180 | 48.623 | 3.898× | 36/36 |
| DSpark Fixed | 63.225 | 48.588 | 3.894× | 36/36 |
| Confidence 0.5 | 61.839 | 49.677 | 3.823× | 36/36 |
| Confidence 0.8 | 51.741 | 59.372 | 3.193× | 36/36 |

三轮配对 Markov 开关收益 **+84.68%**；Confidence 0.5/0.8 相对 Fixed 为 **−1.97%/−18.16%**。配对比值的中位数不一定等于两个中位数之比。EAGLE3/DFlash/DSpark 峰值 allocated 约 23.18/24.57/24.72 GiB，包含不同 Draft 驻留，不能解释为同一个独立纯 AR 部署的显存增量。

Fixed 的吞吐中位数略高（63.225 对 63.180），Markov Only 的配对加速比中位数略高（3.898× 对 3.894×），这是不同统计量，不能都称为同一个“最好模式”；微小计时差也不足以判断稳定优劣。

三个 checkpoint 训练和容量不同；本轮未实现树验证、并发 Serving、Graph 或任务答案评分。旧 z-lab b16 Graph 实验是另一组候选预算和负载，不能比较或相乘其倍数。HF 失败 → FP16 20/21 → 原 Target 回归 21/21 → 正式 252/252 的证据链完整保留，[详细结果与离线复算](../results/speculative_compare/legacy-target-formal-20261021/README.md)可供现场追问。

### 5.7 面试现场展示与诚实边界

优先展示 DFlash 四模式表，再讲一个 P/A/B/C 回滚例子和一次 Graph 元数据更新；分布式用同一张表同时说清总吞吐、每卡效率、Max 与 P99。被问到“为什么没提升”，先承认观察，再给可能机制和下一步验证方法，不把未 Profiling 的猜测说成结论。

必须保留的边界：普通引擎、旧 DFlash＋Graph、分布式控制器及新 Draft 串行适配不同；Qwen3-8B 是性能主模型；第三方基础、Draft 和 AWQ 有明确来源；正式配置没有验证所有功能联合启用；TP 分叉、旧 Flash Prefill 分叉未解决；AWQ 未加速；没有生产 HTTP/生产 vLLM 对照、PP、Draft Graph、异步 KV Handoff 或任务质量评测。完整数据与限制以 [PROJECT_SUMMARY.zh.md](PROJECT_SUMMARY.zh.md) 和 [results/](../results/README.md) 为准。

### 5.8 项目实验的局限与下一步

新三方法只在 Batch=1、无 Graph 下正式对照，尚未在统一 Graph/并发框架下重做新旧两套投机实验。固定形状补行增加计算成本，下一步应先 Profiling Target 执行器，再研究更高并发下的动态验证预算，同时保留逐 Token 与 Cache 状态检查。TP2 的生成分叉和通信/执行成本、AWQ 的 GEMM 性能与任务质量评测仍待处理；这些是后续研究方向，不是已有成果。
