import math
import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.models.llama import LlamaForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence
from myvllm.utils import *

# graph_compile_limits_v2
import torch._dynamo.config as dynamo_config
dynamo_config.recompile_limit = 64
dynamo_config.accumulated_recompile_limit = 256

class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        self.event = event

        # set distributed config
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        self.enforce_eager = config.get('enforce_eager', False)

        self.rank = rank
        dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)
        torch.cuda.set_device(rank)

        # set model
        path_str = self.config['model_name_or_path']
        model_name = Path(path_str).name
        match model_name:
            case "Qwen3-0.6B" | "Qwen3-8B":
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    num_heads=config['num_heads'],
                    head_dim=config['head_dim'],
                    scale=config['scale'],
                    num_kv_heads=config['num_kv_heads'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    qkv_bias=config['qkv_bias'],
                    base=config['base'],
                    max_position=config['max_position'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    tie_word_embeddings=config['tie_word_embeddings'],
                    block_size=self.block_size,
                )
            case 'Llama-3.2-1B-Instruct':
                self.model = LlamaForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    head_dim=config['head_dim'],
                    num_qo_heads=config['num_qo_heads'],
                    num_kv_heads=config['num_kv_heads'],
                    has_attn_bias=config['has_attn_bias'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    rope_base=config['rope_base'],
                    max_position_embeddings=config['max_position_embeddings'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    block_size=self.block_size,
                    tie_word_embeddings=config['tie_word_embeddings'],
                )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # Load weights in GPU (model moved to GPU before loading weights)
        self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step
        self._first_decode = False

        # warm up model so that we know peak memory usage
        self.warmup_model()
        # allocate kv cache
        self.allocate_kv_cache()
        # capture cuda graph for decoding
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device(f'cuda:{rank}')
        torch.set_default_dtype(self.default_dtype)

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__

    # only use read when rank != 0
    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # Check if process group exists before destroying
        if dist.is_initialized():
            dist.destroy_process_group()
    
    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args) # Unpack args when calling
            if method_name == 'exit':
                self.exit()
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: dict):
        if self.world_size > 1 and self.rank == 0: # will be called in main engine
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length
        seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.config['block_size']) for _ in range(batch_size)]
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # find all available memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem * self.config['gpu_memory_utilization']
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        # reserve some room for peak memory usage during model execution
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # find parameters to compute kv cache size
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads'] // self.world_size
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # check whether the current free memory can hold at least one block
        # compute the actual byte required of each block
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        
        # Synchronize max_cached_blocks across all ranks.
        # Each rank independently computed num_available_kv_blocks from its own
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead
        # (NCCL buffers, process-group state) so it often has less free memory than
        # workers. Without sync, the scheduler (which runs only on rank-0) would use
        # rank-0's local value and could allocate more blocks than some rank can hold,
        # causing an OOM on that rank during KV cache writes.
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )
            # all_reduce with MIN: every rank learns the most conservative limit,
            # i.e. the block count that even the most memory-constrained rank can serve.
            # This single agreed-upon value is then stored in config so the Scheduler
            # (initialized afterwards on rank-0) never allocates more blocks than any
            # rank can physically hold.
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        # allocate max possible kv cache for the model, instead for each sequence
        # this is the key for paged attention: one giant KV cache pool, divided into blocks
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values
        allocated_kv_cache = torch.zeros(2, self.config['num_layers'], self.config['max_cached_blocks'], self.block_size, num_kv_heads, head_dim, device=f'cuda:{self.rank}')
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1

    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache into consideration: 
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    # 把本轮窗口展平并构造 KV 元数据
    def prepare_prefill(self, seqs):
        # chunk_prepare_v1: exact scheduled Q window, independent of page boundaries.
        # 设置本轮输入 ID、KV 写入槽位、各请求 Query 长度、各请求历史终点
        input_ids, slots, qlens, klens = [], [], [], []
        # 设置 Query 累计偏移、历史长度累计偏移
        cu_q, cu_k = [0], [0]
        # 设置是否需要读取已缓存前缀
        has_history = False
        # 逐个处理请求
        for seq in seqs:
            # 有分块窗口时用窗口起点，否则跳过缓存前缀
            start = getattr(seq, "_scheduled_start", seq.num_cached_tokens)
            # 有分块窗口时用窗口终点，否则处理到序列末尾
            end = getattr(seq, "_scheduled_end", len(seq))
            # 本轮窗口必须是请求内的非空区间
            assert 0 <= start < end <= len(seq)
            # 设置是否需要读取已缓存前缀
            has_history = has_history or start > 0
            # 只打包本次尚待计算的 Token
            input_ids.extend(seq.token_ids[start:end])
            # 向各请求 Query 长度追加本次内容
            qlens.append(end - start)
            # 向各请求历史终点追加本次内容
            klens.append(end)
            # 用累计 Query 数标记不同请求的边界
            cu_q.append(cu_q[-1] + end - start)
            # 记录包含缓存前缀的完整历史长度
            cu_k.append(cu_k[-1] + end)
            # 只有页表已分配时才登记物理写入槽
            if seq.block_table:
                # 逐个处理绝对 Token 位置
                for position in range(start, end):
                    # 根据逻辑页索引找到物理页号
                    physical = seq.block_table[position // self.block_size]
                    # 物理页号乘页长，再加页内偏移得到写入槽
                    slots.append(physical * self.block_size + position % self.block_size)

        # 把元数据转换成指定设备张量
        def tensor(values, dtype):
            # 返回本次计算结果
            return torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True).to(
                # 补充当前字段或错误说明
                f"cuda:{self.rank}", non_blocking=True)

        # 先将各 Query 的页表设为空
        tables = None
        # 已有前缀 KV 时需要提供历史物理页表
        if has_history:
            # 任一请求有缓存历史时，整批都必须有页表
            assert all(seq.block_table for seq in seqs)
            # 设置页表固定宽度
            width = max(len(seq.block_table) for seq in seqs)
            # 设置各 Query 的页表
            tables = tensor([seq.block_table + [-1] * (width - len(seq.block_table))
                             # 将前述构造应用到每个元素
                             for seq in seqs], torch.int32)
        # 把本轮 KV 地址与有效长度传给模型
        set_context(is_prefill=True,
                    # 记录各请求在展平 Query 中的边界
                    cu_seqlens_q=tensor(cu_q, torch.int32),
                    # 记录各请求包含缓存前缀的历史终点
                    cu_seqlens_k=tensor(cu_k, torch.int32),
                    # 传入本批最长 Query 和历史长度
                    max_seqlen_q=max(qlens), max_seqlen_k=max(klens),
                    # 指定各输入位置的 KV 写入槽
                    slot_mapping=tensor(slots, torch.long),
                    # 指定有效历史长度
                    context_lens=None, block_tables=tables)
        # mixed_decode_metadata_v1: Q=1 Decode rows use the ordinary Decode kernel.
        # 设置混合批次中的 Decode 请求索引
        decode_ids = [i for i, seq in enumerate(seqs)
                      # 仅筛选调度器标记为 Decode 的请求
                      if not getattr(seq, "_scheduled_prefill", True)]
        # 混合批次确实含 Decode 时补充其读取元数据
        if decode_ids:
            # 混合批次中的 Decode 请求必须各占一行
            assert all(qlens[i] == 1 for i in decode_ids)
            # 导入 get_context 所需接口
            from myvllm.utils import get_context
            # 设置当前前向元数据
            context = get_context()
            # 检查页数量、容量或所有权约束
            assert context.block_tables is not None
            # 设置 context.chunk_decode_query_indices
            context.chunk_decode_query_indices = tensor([cu_q[i] for i in decode_ids], torch.long)
            # 设置 context.chunk_decode_block_tables
            context.chunk_decode_block_tables = context.block_tables.index_select(
                # 按 Decode 请求索引抽取对应页表行
                0, tensor(decode_ids, torch.long))
            # 设置 context.chunk_decode_context_lens
            context.chunk_decode_context_lens = tensor([klens[i] for i in decode_ids], torch.int32)
        # 返回指定类型的元数据张量
        return tensor(input_ids, torch.long)


    # prepare input data for decoding
    # 为每请求的末尾 Token 准备元数据
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        # 初始化本轮输入 ID
        input_ids = []
        # 初始化有效历史长度
        context_lens = []   
        # 初始化本轮 KV 写入位置
        slot_mappings = []  
        # 初始化逐请求或逐 Query 物理页表
        block_tables = []
        # 逐个处理请求
        for seq in seqs:
            # 普通 Decode 每请求只输入末尾待计算 Token
            input_ids.append(seq.last_token)
            # 本轮 Attention 可以看到当前 Token 在内的全部历史
            context_lens.append(len(seq))
            # 向本轮 KV 写入位置追加本次内容
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        # 设置所有请求的页表
        all_block_tables = [seq.block_table for seq in seqs]
        # 设置本批次最长页表
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        # 逐个处理i、请求
        for i, seq in enumerate(seqs):
            # 用无效页号补齐批次页表宽度
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            # 向逐请求或逐 Query 物理页表追加本次内容
            block_tables.append(block_table)
        # 设置本轮输入 ID
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        # 把本轮 KV 地址与有效长度传给模型
        set_context(
            # 指定是否包含 Prefill
            is_prefill=False,
            # 单 Token Decode 不需要变长 Query 累计边界
            cu_seqlens_q=None,
            # 历史有效长度改由 context_lens 提供
            cu_seqlens_k=None,
            # 此路径不使用 Prefill 的最大 Query 参数
            max_seqlen_q=0,
            # 此路径不使用 Prefill 的最大历史参数
            max_seqlen_k=0,
            # 指定各输入位置的 KV 写入槽
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            # 指定有效历史长度
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            # 指定逐请求或逐 Query 物理页表
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        # 结束当前调用或分组
        )
        # 返回本轮输入 ID
        return input_ids    

    # prepare the temperature
    # 构造采样温度张量
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        # 返回本次计算结果
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    # 将该调用设为无梯度推理
    @torch.inference_mode()
    # 按阶段选择 Eager 或普通 Decode 图
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        # Prefill 或显式 Eager 配置直接执行前向
        if is_prefill or self.enforce_eager:
            # 返回本次计算结果
            return self.model.compute_logits(self.model(input_ids))

        # 设置当前批大小
        bs = input_ids.size(0)
        # 检查是否可使用相应 Graph
        if bs not in self.graphs:
            # 状态不满足要求时立即报错
            raise ValueError(f"No CUDA Graph captured for batch={bs}")

        # 设置当前前向元数据
        context = get_context()
        # 设置固定地址张量
        buffers = self.graph_vars
        # 设置页表固定宽度
        width = context.block_tables.size(1)
        # 检查页表固定宽度是否满足本分支条件
        if width > buffers["block_tables"].size(1):
            # 状态不满足要求时立即报错
            raise ValueError("Sequence exceeds CUDA Graph block-table capacity")

        # 首次遇到某个 batch 和块表宽度时，顺带比较 eager 结果。
        # 设置本次 Graph 诊断键
        check_key = (bs, width)
        # 设置是否首次检查此形状
        need_check = check_key not in self._graph_checked
        # 先将Eager 参考结果设为空
        expected = None
        # 诊断开启时再做额外检查
        if need_check:
            # 首次检查此形状时计算 Eager 参考词表分数
            expected = self.model.compute_logits(
                # 计算 Eager 模型特征
                self.model(input_ids)
            # 复制参考结果，防止后续缓冲复用
            ).clone()

        # 更新静态地址中的输入 Token
        buffers["input_ids"][:bs].copy_(input_ids)
        # 更新新 KV 的物理写入位置
        buffers["slot_mapping"][:bs].copy_(context.slot_mapping)
        # 更新每请求当前有效历史长度
        buffers["context_lens"][:bs].copy_(context.context_lens)
        # 先清除旧页表尾部，避免残留有效页号
        buffers["block_tables"][:bs].fill_(-1)
        # 将新内容写入现有张量存储
        buffers["block_tables"][:bs, :width].copy_(
            # 把当前请求页表写入静态缓冲
            context.block_tables
        # 结束当前调用或分组
        )

        # 重放已捕获的模型计算，不复用旧答案
        self.graphs[bs].replay()
        # 设置词表分数
        logits = self.model.compute_logits(buffers["outputs"][:bs])

        # 诊断开启时再做额外检查
        if need_check:
            # graph_diagnostic_v2
            # 设置数值差异
            delta = (logits.float() - expected.float()).abs()
            # 输出本轮进度或诊断信息
            print("GRAPH DIAGNOSTIC:", {
                # 补充当前字段或错误说明
                "batch": bs,
                # 补充当前字段或错误说明
                "table_width": width,
                # 补充当前字段或错误说明
                "graph_finite": bool(torch.isfinite(logits).all().item()),
                # 补充当前字段或错误说明
                "eager_finite": bool(torch.isfinite(expected).all().item()),
                # 补充当前字段或错误说明
                "max_abs": delta.max().item(),
                # 补充当前字段或错误说明
                "mean_abs": delta.mean().item(),
                # 补充当前字段或错误说明
                "top1_match": bool(
                    # 比较各行词表 argmax 是否一致
                    (logits.argmax(-1) == expected.argmax(-1)).all().item()
                # 结束当前分组参数
                ),
            # 打印诊断信息并立即刷新输出
            }, flush=True)
            # 按普通引擎原有容差检查图与 Eager 结果
            torch.testing.assert_close(
                # 按此普通引擎路径既定容差比较 Logits
                logits, expected, rtol=1e-2, atol=5e-2
            # 结束当前调用或分组
            )
            # 设置最大绝对误差
            error = (logits.float() - expected.float()).abs().max().item()
            # 输出本轮进度或诊断信息
            print(
                # 报告普通 Decode 图已通过本次形状检查
                f"CUDA GRAPH CHECK PASS: batch={bs}, "
                # 报告页表宽度与最大数值差异
                f"table_width={width}, max_abs={error:.6f}",
                # 立即刷新诊断输出
                flush=True,
            # 结束当前调用或分组
            )
            # 把当前元素加入集合
            self._graph_checked.add(check_key)

        # 返回完整词表的预测分数
        return logits


    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    # 执行前向并返回本次结果
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 按 Prefill 或 Decode 选择处理方式
        if is_prefill:
            # 设置本轮输入 ID
            input_ids = self.prepare_prefill(seqs)
        # 处理上述条件不成立的情况
        else:
            # 设置本轮输入 ID
            input_ids = self.prepare_decode(seqs)
        # 设置词表分数
        logits = self.run_model(input_ids, is_prefill)
        # only sample when rank == 0
        # 先将完整逻辑 Token 序列设为空
        token_ids = None
        # 普通引擎只由主进程执行采样
        if self.rank == 0:
            # 根据词表分数和采样参数选择输出
            token_ids = self.sampler(logits, self.prepare_sample(seqs))
        # 清除本次全局前向元数据，避免影响后续调用
        reset_context()
        # 普通引擎只返回主进程的采样 ID
        return token_ids

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated onece and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = int(self.config.get(
            "max_num_sequences",
            self.config.get("max_num_seqs", 8),
        ))
        if not 1 <= max_bs <= 16:
            raise ValueError("This implementation captures batch sizes 1 through 16")

        max_len = int(self.config["max_model_length"])
        max_blocks = math.ceil(max_len / self.block_size)
        device = f"cuda:{self.rank}"
        dtype = next(self.model.parameters()).dtype
        self.model.eval()

        buffers = {
            "input_ids": torch.zeros(
                max_bs, dtype=torch.long, device=device
            ),
            # 捕获时不写真实 KV Cache。
            "slot_mapping": torch.full(
                (max_bs,), -1, dtype=torch.long, device=device
            ),
            # 使用有效位置，避免空 attention 和 position=-1。
            "context_lens": torch.ones(
                max_bs, dtype=torch.long, device=device
            ),
            "block_tables": torch.zeros(
                max_bs, max_blocks, dtype=torch.int32, device=device
            ),
            # model() 返回 hidden states，维度不是 vocab_size。
            "outputs": torch.empty(
                max_bs, self.config["hidden_size"],
                dtype=dtype, device=device,
            ),
        }

        self.graphs = {}
        self.graph_vars = buffers
        self._graph_checked = set()
        warmup_stream = torch.cuda.Stream(device=self.rank)

        # 捕获每个 batch，尾批也能直接使用对应图。
        for bs in range(max_bs, 0, -1):
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=buffers["slot_mapping"][:bs],
                context_lens=buffers["context_lens"][:bs],
                block_tables=buffers["block_tables"][:bs],
            )
            try:
                # 在捕获外完成编译和预热。
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    for _ in range(3):
                        buffers["outputs"][:bs].copy_(
                            self.model(buffers["input_ids"][:bs])
                        )
                torch.cuda.current_stream().wait_stream(warmup_stream)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    buffers["outputs"][:bs].copy_(
                        self.model(buffers["input_ids"][:bs])
                    )
                self.graphs[bs] = graph
                torch.cuda.synchronize()
                print(f"CUDA Graph captured: batch={bs}", flush=True)
            finally:
                reset_context()

    def run_chunked(self, seqs):
        from myvllm.engine.chunk_runtime import run_chunked
        return run_chunked(self, seqs)
