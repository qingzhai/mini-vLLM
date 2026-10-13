# 项目面试讲述与问答

以下提纲对应当前仓库。讲述保留“基于开源 MinivLLM 扩展”“官方 Qwen/AWQ/DFlash 来源”和具体实验条件，不把第三方组件说成从零开发。项目画像与阅读顺序见 [学习指南](LEARNING_GUIDE.zh.md)，数字完整口径见 [项目总结](PROJECT_SUMMARY.zh.md)。

## 30 秒讲述稿

> 我做的是基于 MinivLLM 的 Qwen3 推理系统复现与优化，主要涉及分页 KV、前缀缓存、分块调度、DFlash 投机解码和 CUDA Graph，也做了双卡 DP 与单机 P/D 分离。比较明确的结果是：单 A100 40GB 上，DFlash＋Target Graph 相比已经开启 Graph 的普通解码，固定负载的单请求和 batch=2 三请求总吞吐达到 2.99 倍、1.52 倍，三轮输出逐 token 一致。双卡实验还验证了 DP 吞吐提升和 P/D 分离减少最大出字停顿。

对应证据：[Graph 四项对照](../results/dflash_graph/README.md)、[双卡对照](../results/distributed/README.md)。

## 2 分钟讲述稿

> 项目目标是把推理系统里几个常见优化真正跑通，并用相同输入和明确基线判断有没有收益。我基于 MinivLLM 保留 Qwen3 模型与算子基础，扩展分页 KV 的前缀复用、Chunked Prefill、DFlash 批量验证以及双卡控制器。
>
> 第一条工作是请求和 KV 生命周期：调度器决定本轮 token 窗口，页表映射物理 KV，只有完整且已经写完的前缀页可以共享，部分 Prefill 不能采样。ShareGPT 派生 100 请求实验里，同一 Chunk 配置开启 Prefix 后吞吐从 43.67 到 52.51 tokens/s，三种调度输出一致。但 Chunk 单独没有加速，和完整 Prefill 相比时延也不是全面改善。
>
> 第二条是 DFlash 与 Graph。Draft 每次给候选，Target 批量验证连续前缀，拒绝位置后的 KV 要回滚。我给 Target 捕获 batch 1/2、query 1/16 四张图，每轮更新 token、位置、slot 和页表，保留静态地址。四项对照各测三轮，联合模式对 AR Graph 单请求 5.513 到 1.846 秒，三请求 11.653 到 7.651 秒，输出和接受轨迹一致。Draft 还在图外，结果只代表固定负载和这个执行器。
>
> 第三条是双卡。DP2 相对单卡总吞吐提高约 19% 到 26%；1P1D 通过 NCCL 真正迁移全部 36 层 prompt KV，在近似 DP2 吞吐下把最大 token 间隔从约 1.8 到 2.2 秒降到约 0.2 秒。TP2 虽然显存减半但更慢，AWQ 也主要带来存储下降；我把这些负面结果一起保留，没有只挑加速数字。

源码落点：[BlockManager](../src/myvllm/engine/block_manager.py#L26)、[commit](../experiments/dflash_graph/state.py#L70)、[Graph executor](../experiments/dflash_graph/graph_executor.py#L70)、[分布式 run_load](../experiments/distributed/runtime.py#L171)。

## 5 分钟讲述稿

> **背景和边界。** 我选择轻量框架，是希望能跟到请求调度、KV 地址和模型前向，而不是只调用一个服务 API。项目基于开源 MinivLLM，范围限定 Qwen3，官方模型权重、AWQ checkpoint、DFlash Draft 都保留来源。仓库有普通引擎、DFlash Graph 独立运行时、分布式控制器三条路径，共用模型和部分算子，但没有声称全部功能一起启用。
>
> **基本执行链。** prompt 变成 Sequence 后排队，调度器选本轮窗口；ModelRunner 构造 input IDs、slot mapping、context lengths 和 block tables；模型算 logits，采样后追加 token，完成时归还页。物理 KV 不要求连续，通过逻辑页表寻址。前缀缓存的 hash 包含此前的前缀，每页还检查 ready 和 token；共享页有引用计数，最后可写页私有。Chunk 把长 Prefill 拆开与 Decode 混合，但只有最后块才输出首 token，这个约束直接写进采样行选择和状态断言。
>
> **数值与正确性。** 不同 query/batch 形状可能让低精度归约改变，接近的两个 logits 会交换 argmax。我没有只看文本是否看起来合理，而是比较完整 token；出现分叉后用相同历史定位到 KV、hidden、logits 的差异。当前受控执行器统一计算形状、FP32 Head 与 per-query PagedAttention，建立通过的参考配置。它不是最快的 Prefill 配置，旧 Flash Prefill 的完整模型分叉也没有被写成已修复。TP 的容差检查和自由生成一致率是分开报告的。
>
> **最主要的优化。** DFlash 生成一个包含待计算 token 和候选的 block，Target 一次前向给每个位置的预测。只接受从头连续匹配的候选，然后追加对应 Target token。状态保持末尾一个 pending token；拒绝后的逻辑 KV 长度要收缩，超出页归还，Draft cache 和 Target features 也按已接受进度衔接。
>
> Graph 进一步减少这个执行器的前向启动开销。捕获四种固定形状，每轮把新的 token、slot、context lengths、完整页表拷进固定地址，输出 clone 后交给请求保存，防止 replay 覆盖。正式 Graph 验证前向 fallback 为零，另做 8 组同历史前向精确一致检查。四种模式各三轮，单请求 256 tokens 的 AR Graph 是 5.513 秒，联合模式 1.846 秒；3 请求、并发 2、共 768 tokens 则是 11.653 到 7.651 秒。总共 12,288 个输出 token 全部匹配。Draft 和初始 Prefill 还在图外，不能把这个倍数外推到生产服务。
>
> **多卡和取舍。** 我比较了单卡、TP2、DP2、1P1D。DP 是两个完整副本按驻留数路由；TP 是一个模型的矩阵分片；1P1D 两卡各有完整模型，P 计算 prompt 后通过 NCCL send/recv 迁移全部层 KV，并在 D 的物理页表重新放置。首 token 由 P 发出，D 不重新 Prefill。DP2 总吞吐提高 19% 到 26%，但是每卡效率没有提升。P/D 分离主要减少极端出字停顿，吞吐和 DP2 基本相同，P99 也没有全面改善。TP 的显存约减半，但小负载上更慢；AWQ 权重存储减少 62.77%，当前 kernel 也更慢。我会把这些结果解释为不同瓶颈和成本下的权衡。
>
> **结论和后续。** 这个项目最大的收获是把 token、KV、调度和验证状态连在一起，并用公平对照判断优化。若继续做，我会优先扩展代表性负载、重复测量双卡结果、profiling 定位热点，再考虑 Draft Graph 或异步 KV handoff；这些都不属于当前已完成项。

核对讲述内容的源码：[普通 step](../src/myvllm/engine/llm_engine.py#L82)、[部分 Prefill 的处理](../src/myvllm/engine/chunk_runtime.py#L38)、[固定算术](../experiments/common/target.py#L59)、[Graph run](../experiments/dflash_graph/graph_executor.py#L202)、[KV pack/unpack](../experiments/common/target.py#L221)。

## 16 个常见追问

### 1. 这个项目和直接调用 vLLM 有什么区别？你做了什么？

回答提纲：基于已有轻量模型/算子基础，把页管理、Prefix/Chunk、DFlash 验证、Graph executor 和双卡实验控制器连起来；重点能说明状态与验证，不把开源框架归为从零自研。实际有三条执行路径，入口与资源所有权不同。没有生产 HTTP 服务、PP 或所有功能同时启用的配置。

源码：[LLMEngine](../src/myvllm/engine/llm_engine.py#L25)、[GraphDFlashRuntime](../experiments/dflash_graph/graph_runtime.py#L8)、[Session/run_load](../experiments/distributed/runtime.py#L20)。

### 2. Prefill 和 Decode 到底哪里不同？KV 为什么能复用？

回答提纲：Prefill 处理已有 prompt 的多个位置，因果 Attention 每个位置只能看当前及以前；其最后位置产生首 token。Decode 本轮通常只计算末尾待处理 token 的 Q/K/V，访问历史缓存 K/V 得到下一分布。KV 复用省去历史层状态的重复计算，不是缓存完整 logits。DFlash 验证本轮有多个 query，不能简单按“不是 Prefill 所以只有一行”理解。

源码：[prepare_prefill/prepare_decode](../src/myvllm/engine/model_runner.py#L270)、[Qwen3Attention.forward](../src/myvllm/models/qwen3.py#L73)、[Graph metadata](../experiments/dflash_graph/graph_executor.py#L25)。

### 3. Paged KV 怎么寻址，能省多少显存？

回答提纲：`physical_slot = block_table[position // block_size] * block_size + position % block_size`。分页减少为每条请求预留最大连续空间的浪费，也支持页粒度复用；页内尾部仍可能浪费，分页不会改变有效 token 的 KV 字节数。字节公式为 `2×layers×tokens×KV_heads×head_dim×bytes`。本 Qwen3-8B FP16：每 token 147,456 bytes，1024 tokens 为 144 MiB；GQA 按 8 个 KV heads 而不是 32 个 Q heads算。

源码：[slot 构造](../src/myvllm/engine/model_runner.py#L270)、[Target cache 分配](../experiments/common/target.py#L124)、[Target.slots](../experiments/common/target.py#L212)。

### 4. Prefix 为什么要链式 hash？LRU 会不会驱逐正在用的页？

回答提纲：相同当前块在不同前文下 KV 可能不同，因此 hash 纳入上一块 hash 和本块 token。命中还检查 `ready` 与 token 内容；只共享已经计算完成的整页，最后可写页私有。引用计数大于 0 的页不在空闲队列，不能分配给别人；释放为 0 后留存内容，直到被复用时删除 hash 索引。这里 LRU 按空闲页的释放/复用次序管理，不是每次访问都维护全局精确时钟。没有任意共享尾页的 copy-on-write。

源码：[compute_hash/_cached_prefix](../src/myvllm/engine/block_manager.py#L37)、[allocate/mark_computed/deallocate](../src/myvllm/engine/block_manager.py#L88)。

### 5. Chunked Prefill 为什么不能每个块都采样？

回答提纲：前几个块还没读完用户输入，其最后位置预测的是 prompt 的后续，不是最终 prompt 后的首个输出。部分块只写 KV 和更新计算进度，返回 `-1` 占位；最后块和 Decode 才选择 hidden 行计算 logits 并采样。只把已经写完的整页发布为 prefix-ready。

源码：[run_chunked 的 ready/indices](../src/myvllm/engine/chunk_runtime.py#L7)、[chunked_step 的部分块断言](../src/myvllm/engine/chunk_runtime.py#L38)、[ChunkScheduler](../src/myvllm/engine/chunk_scheduler.py#L25)。

### 6. Continuous batching 和普通固定 batch 有什么区别？

回答提纲：每轮重选仍活跃的请求，完成后腾出的驻留位置能补入新请求，不必等最慢请求把整批结束。还要受 token budget、KV 页和驻留上限约束。Chunk 路径可先安排 Decode，再用剩余 budget 安排 Prefill 块。DFlash 用独立 active/waiting 队列实现补位，没有复用普通 `engine.step` 调度。

源码：[Scheduler.schedule/postprocess](../src/myvllm/engine/scheduler.py#L35)、[ChunkScheduler.schedule](../src/myvllm/engine/chunk_scheduler.py#L25)、[DFlashRuntime.generate](../experiments/dflash_graph/dflash_runtime.py#L106)。

### 7. DFlash 候选怎么验收？为什么保证这轮 greedy 输出？

回答提纲：本实现使用官方 Draft，16-token block 的第 0 项是 Target 已知的 pending token，后 15 项为候选。Target 因果前向给出每个 query 的后继预测；从头比较 `block[1:]` 和 `posterior[:-1]`，只接受连续一致前缀，再补上拒绝位置的 Target token；全部接受时还能得到末尾 Target 预测。逻辑上的 greedy 等价仍需数值和完整生成测试证实。没有实现随机采样下的概率验收修正。

源码：[propose](../experiments/dflash_graph/dflash_runtime.py#L81)、[accept_prefix/commit](../experiments/dflash_graph/state.py#L61)、[compare_generation](../experiments/dflash_graph/benchmark.py#L24)。

### 8. 投机失败时回滚什么？把内存清零就够吗？

回答提纲：关键是不让拒绝部分参与后续计算，而不是要求立即清零。维护 `len(tokens)=computed+1`，computed 只前进到接受范围，输出末尾一个 token 等下轮计算；页表超出逻辑需要的整页归还，留在最后页的旧值由 context length 遮住并会被后续覆盖。请求的 Target features 只保留有效推进范围，Draft cache 按已计算长度裁切衔接。

源码：[commit](../experiments/dflash_graph/state.py#L70)、[PagePool.trim](../experiments/dflash_graph/state.py#L20)、[propose/generate](../experiments/dflash_graph/dflash_runtime.py#L81)。

### 9. CUDA Graph 要静态地址，序列长度一直变怎么做？

回答提纲：捕获固定 batch/query 形状和缓冲地址，长度与页表是缓冲内的动态数据。每轮替换 input IDs、slot mapping、context lengths、block tables，四类全部更新，页表尾部也要覆盖。当前捕获 B=1/2、Q=1/16，不是任意 shape 一张图。初始 Prefill 图外；正式受控验证没有 fallback。输出 clone 保证下一次 replay 不会覆盖已保留 features；Python hook 不在 replay 重新执行。

源码：[build_query_metadata/_fill](../experiments/dflash_graph/graph_executor.py#L25)、[capture_all/run](../experiments/dflash_graph/graph_executor.py#L154)、[Graph runtime](../experiments/dflash_graph/graph_runtime.py#L27)。

### 10. 为什么必须比较四种模式？2.99 倍具体相对谁？

回答提纲：AR eager、AR Graph、DFlash eager、DFlash Graph 分别识别图优化、投机算法和联合收益。2.99×/1.52× 是对 **AR Graph**，各三轮中位数，单 A100、固定负载；不能把旧对 AR eager 的倍数乘上 Graph 倍数。四模式同算术、同特征采集、相同 Draft/图驻留，计时包含初始 Prefill 和运行时开销，排除捕获/预热/诊断。这是受控实验执行器，不是生产引擎对比。

源码与证据：[benchmark.summarize/main](../experiments/dflash_graph/benchmark.py#L47)、[Graph 精选结果](../results/dflash_graph/summary.json)。

### 11. TP、DP、PP、P/D 分离分别拆什么？本项目做了哪些？

| 方式 | 拆分对象 | 本项目状态 |
|---|---|---|
| TP | 一个模型的矩阵/heads 分片，各卡共同完成同一请求 | 已做 TP2；Column 切输出，Row 切输入，Row 结果 all-reduce，Head logits all-gather |
| DP | 完整模型副本，各接不同请求 | 已做 DP2，副本独立 KV、按驻留数路由 |
| PP | 模型不同层/阶段放在不同设备，传中间激活 | 未做 |
| P/D 分离 | 同一请求 Prefill 和 Decode 分给不同 worker | 已做单机 1P1D；两卡完整权重，跨卡传 prompt KV |

DP/PD 的模型通信组为单 rank，防止被 WORLD=2 错误分片；KV send/recv 仍用 WORLD。不要把本地混合 P/D 批处理说成跨卡 P/D 分离。

源码：[Column/Row Parallel](../src/myvllm/layers/linear.py#L84)、[ModelGroup](../experiments/common/target.py#L24)、[worker 建组](../experiments/distributed/worker.py#L50)。

### 12. P/D 分离是真的传 KV，还是 D 重新算 prompt？

回答提纲：P 对完整 prompt 前向，先发首 token；按源页表 pack 全 36 层有效 prompt K/V，GPU tensor NCCL send/recv，D 按自己的页表 unpack。传输形状为 `[2,36,prompt_tokens,8,128]`、FP16；每请求字节为 `2×36×prompt_tokens×8×128×2`。D 用迁移的历史 KV 处理首个 pending token。两端 ACK 后 P 释放页；本轮 D Prefill=0、P Decode=0，预热还校验源/接收/重映射后 hash。

源码：[Target.pack/unpack](../experiments/common/target.py#L221)、[worker.send_kv/recv_kv](../experiments/distributed/worker.py#L110)、[handoff 和 pd_contract](../experiments/distributed/runtime.py#L171)。

### 13. 为什么 TP 更慢，P/D 分离也没明显提高吞吐？

回答提纲：实测结论先明确：TP2 对单卡吞吐下降、每卡显存约减半；DP2 总吞吐 +19%～26%，每卡效率下降；1P1D 对 DP2 吞吐近似持平，最大 ITL 大幅下降，P99 不全面改善。可能因素包括小 batch、通信、调度/RPC、同步 handoff 和现有 per-query 执行器，但没有 kernel profiling，不能把某个因素说成已经确定的唯一瓶颈。单次双卡正式测量也不足以给稳定误差范围。

源码与证据：[同步 handoff](../experiments/distributed/runtime.py#L171)、[分布式结果](../results/distributed/summary.csv)、[项目结果解释](PROJECT_SUMMARY.zh.md)。

### 14. W4A16 为什么显存下降但速度更慢？

回答提纲：权重 4 bit、激活 16 bit，仍有 scales/zeros、未量化权重、KV 和额外 FP32 Head，所以整机显存不会简单降到四分之一。当前 Triton GEMM tile 内解包量化权重再做 dot；减少读权重量不保证能抵消解包、tile 效率和其他开销，没有 profiling 不下唯一原因结论。实测权重及元数据减少 62.77%，含额外 FP32 Head 为 54.49%；short-b1 吞吐 12.11→3.43 tokens/s。做的是官方 checkpoint 推理和算子参考验证，没有自行校准或任务质量评测。

源码与证据：[AWQ _gemm/forward](../experiments/awq/awq_linear.py#L14)、[AWQ loader](../experiments/awq/awq_loader.py#L37)、[量化结果](../results/awq/summary.json)。

### 15. TTFT、TPOT、E2E、ITL、吞吐怎么计算？公平对照要固定什么？

回答提纲：TTFT=`首token时刻−arrival`，E2E=`末token时刻−arrival`，TPOT=`(末−首)/(输出数−1)`；ITL 是每对相邻 token 的差。总吞吐是总输出数/整组耗时，每卡吞吐再除 GPU 数。排队属于本实验时延，模型加载/分词/HTTP不属于。固定模型、精度、输入 token、输出长度/EOS规则、到达时间、驻留上限、KV预算、预热和计时边界；多卡比较同时报告 GPU 数和每卡吞吐，不能只报总吞吐。

ShareGPT 的 20.2% 指同一 Chunk 配置开 Prefix，不是对 full Prefill；20% 是受控完整 prompt 重放比例，实际 token 复用为 17.2%。这些比较对象必须讲清。

源码与证据：[metrics](../experiments/distributed/common.py#L58)、[build_cases](../experiments/distributed/common.py#L102)、[ShareGPT 结果](../results/sharegpt/summary.json)。

### 16. 出现生成分叉，你怎么判断状态 bug 还是数值问题？

回答提纲：保存完整 token，定位第一处分叉；固定该位置以前的相同历史，再比较输入/位置/页表、逻辑 KV、层特征、hidden、logits 和 top-2 margin。先排除错位、拒绝 KV 泄漏、未写完页被复用、Graph metadata 陈旧或输出缓冲覆盖，再检查计算形状和精度。极小 top-2 margin 说明微小误差可能改变 argmax，但本身不能证明没有状态 bug。

本项目统一固定计算形状、FP32 Head/RoPE 初始化和 per-query Attention 后建立通过配置；Graph 8 组同历史检查精确一致，还做 24 次完整生成对照。Teacher-forced 同历史检查用于定位，不能代替自由生成验证。TP2 当前完整序列是 47/48，不把容差 PASS 写成全部一致；原 Flash Prefill 的完整模型问题也仍单列边界。

源码与证据：[fixed_shape](../experiments/common/fixed_shape.py#L11)、[Target.install_arithmetic](../experiments/common/target.py#L59)、[graph_diagnostics](../experiments/dflash_graph/benchmark.py#L112)、[TP probe](../experiments/distributed/worker.py#L16)、[完整生成比较](../experiments/dflash_graph/benchmark.py#L24)。

## 展示什么，如何应对没有提升的结果

面试展示优先带三张表：

1. [Graph 四项对照](../results/dflash_graph/summary.csv)：明确相对 AR Graph 的 2.99×/1.52×，解释它与旧 eager 基线的区别。
2. [Single/TP2/DP2/1P1D](../results/distributed/summary.csv)：同时展示总吞吐、每卡效率、最大 ITL 和 P99，让听者看到吞吐、内存和停顿的不同权衡。
3. [ShareGPT 与 AWQ](PROJECT_SUMMARY.zh.md)：Prefix 同配置有效，Chunk 单独未加速；AWQ 降内存但更慢。用负面结果说明你有对照和边界意识。

源码现场展示可选 [commit](../experiments/dflash_graph/state.py#L70)、[Graph metadata/clone](../experiments/dflash_graph/graph_executor.py#L202)、[PD 迁移与页归还](../experiments/distributed/runtime.py#L171)，每段都能用一个具体例子说明状态变化。

如果被问“你独立做了哪些”，按实际工作和当前能解释的部分回答：开源模型/算子基础、官方预量化权重和 Draft 有明确来源；复现、系统扩展、实验调试、对照和整理是这个项目的工作，不用靠“从零自研”增加分量。
