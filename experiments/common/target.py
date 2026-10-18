"""Common FP16 Qwen target, explicit TP group and stable arithmetic.

Uses the archived MinivLLM model/layers. All phases use the existing per-query
PagedAttention kernel. graph_executor and graph_runtime add target Graph and
DFlash execution externally; this base initializer does not capture graphs.
"""
from __future__ import annotations
import hashlib
import importlib
import inspect
from pathlib import Path
from types import MethodType
import sys
import torch
import torch.distributed as dist
from torch.nn import functional as F
from fixed_shape import fixed_rows_forward, fixed_linear
from tp_support import load_checkpoint

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE.parents[1] / 'src'))


class ModelGroup:
    """Scope only the repository's model collectives to its replica TP group.

    torch.distributed itself stays untouched. PD KV send/recv use WORLD.
    """
    def __init__(self, group):
        self.group = group

    def get_world_size(self, group=None):
        return dist.get_world_size(self.group if group is None else group)

    def get_rank(self, group=None):
        return dist.get_rank(self.group if group is None else group)

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        return dist.all_reduce(tensor, op, self.group if group is None else group, async_op)

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        return dist.all_gather(tensor_list, tensor, self.group if group is None else group, async_op)

    def gather(self, tensor, gather_list=None, dst=0, group=None, async_op=False):
        return dist.gather(tensor, gather_list, dst, self.group if group is None else group, async_op)

    def __getattr__(self, name):
        return getattr(dist, name)


def bind_model_group(group):
    adapter = ModelGroup(group)
    for name in ('myvllm.layers.linear', 'myvllm.layers.embedding_head', 'myvllm.models.qwen3'):
        module = importlib.import_module(name)
        module.dist = adapter
    return adapter


def install_arithmetic(model, group):
    from myvllm.layers.linear import LinearBase
    from myvllm.layers.attention import Attention, store_kvcache, paged_attention_decode
    from myvllm.utils import get_context
    counts = dict(linear=0, rms=0, activation=0, attention=0)
    for module in model.model.modules():
        name = type(module).__name__
        if name in ('LayerNorm', 'RotaryEmbedding', 'SiluAndMul'):
            attribute = 'rms_forward' if name == 'LayerNorm' else 'forward'
            function = getattr(type(module), attribute)
            raw = getattr(function, '_torchdynamo_orig_callable', inspect.unwrap(function))
            setattr(module, attribute, MethodType(raw, module))
        if isinstance(module, LinearBase):
            original = module.forward
            def linear(x, original=original):
                return fixed_rows_forward(original, x, 32)
            module.forward = linear
            counts['linear'] += 1
        elif name == 'LayerNorm':
            original = module.rms_forward
            rows = 1024 if module.weight.numel() <= 128 else 32
            def rms(x, original=original, rows=rows):
                return fixed_rows_forward(original, x, rows)
            module.rms_forward = rms
            counts['rms'] += 1
        elif name == 'SiluAndMul':
            original = module.forward
            def activate(x, original=original):
                return fixed_rows_forward(original, x, 32)
            module.forward = activate
            counts['activation'] += 1
        elif isinstance(module, Attention):
            def attention(q, k, v, module=module):
                context = get_context()
                assert q.ndim == k.ndim == v.ndim == 3
                assert q.dtype == k.dtype == v.dtype == torch.float16
                store_kvcache(k.contiguous(), v.contiguous(), module.k_cache, module.v_cache,
                              context.slot_mapping, module.block_size)
                out = paged_attention_decode(q, module.k_cache, module.v_cache,
                    context.block_tables, context.context_lens,
                    module.scale/(module.head_dim**0.5), module.num_heads,
                    module.num_kv_heads, module.head_dim, module.block_size)
                return out.reshape(q.shape[0], module.num_heads*module.head_dim)
            module.forward = attention
            counts['attention'] += 1
    head = model.lm_head
    weight = head.weight.detach().float().contiguous()
    def head_forward(x):
        context = get_context()
        if context.is_prefill:
            x = x[context.cu_seqlens_q[1:]-1].contiguous()
        logits = fixed_linear(x.float(), weight)
        if head.tp_size > 1:
            pieces = [torch.empty_like(logits) for _ in range(head.tp_size)]
            dist.all_gather(pieces, logits.contiguous(), group=group)
            logits = torch.cat(pieces, dim=-1)[..., :head.num_embeddings]
        return logits
    head.forward = head_forward
    expected = dict(linear=144, rms=145, activation=36, attention=36)
    assert counts == expected, counts
    return dict(modules=counts, fixed_linear_rows=32, head_extra_bytes=weight.numel()*4,
                attention='Existing per-query PagedAttention for Prefill and Decode',
                torch_compile=False, cuda_graph=False)


class Target:
    def __init__(self, path, local_rank, model_group, pages):
        from transformers import AutoConfig
        from myvllm.models.qwen3 import Qwen3ForCausalLM
        from myvllm.layers.rotary_embedding import RotaryEmbedding
        self.device = torch.device('cuda', local_rank)
        self.group = model_group
        self.adapter = bind_model_group(model_group)
        self.tp = self.adapter.get_world_size()
        self.tp_rank = self.adapter.get_rank()
        self.pages = pages
        hc = AutoConfig.from_pretrained(path, local_files_only=True)
        assert hc.num_hidden_layers == 36 and not hc.tie_word_embeddings, 'This package targets Qwen3-8B'
        assert (hc.hidden_size, hc.num_attention_heads, hc.num_key_value_heads, hc.head_dim) == (4096, 32, 8, 128)
        self.hc = hc
        for n in (hc.num_attention_heads, hc.num_key_value_heads, hc.intermediate_size, hc.vocab_size):
            assert n % self.tp == 0
        original = RotaryEmbedding.__init__
        def rope(module, *args, **kwargs):
            previous = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                original(module, *args, **kwargs)
            finally:
                torch.set_default_dtype(previous)
        RotaryEmbedding.__init__ = rope
        try:
            self.model = Qwen3ForCausalLM(vocab_size=hc.vocab_size, hidden_size=hc.hidden_size,
                num_heads=hc.num_attention_heads, head_dim=hc.head_dim,
                num_kv_heads=hc.num_key_value_heads, rms_norm_epsilon=hc.rms_norm_eps,
                qkv_bias=hc.attention_bias, base=hc.rope_theta, max_position=2048,
                intermediate_size=hc.intermediate_size, ffn_bias=False,
                num_layers=hc.num_hidden_layers, tie_word_embeddings=False,
                block_size=256).to(self.device).eval()
        finally:
            RotaryEmbedding.__init__ = original
        self.loading = load_checkpoint(self.model, path)
        assert self.loading['strict'] and (self.tp == 1 or self.loading['shard_loader_calls'] > 0)
        assert all(m.tp_size == self.tp for m in self.model.modules() if hasattr(m, 'tp_size'))
        assert self.model.lm_head.weight.shape[0] == hc.vocab_size//self.tp
        assert all(p.dtype == torch.float16 for p in self.model.parameters())
        self.arithmetic = install_arithmetic(self.model, model_group)
        self.cache = torch.zeros(2, hc.num_hidden_layers, pages, 256,
            hc.num_key_value_heads//self.tp, hc.head_dim, dtype=torch.float16, device=self.device)
        for i, layer in enumerate(self.model.model.layers):
            layer.self_attn.attention.k_cache = self.cache[0, i]
            layer.self_attn.attention.v_cache = self.cache[1, i]
        self.reset_counters()

    def reset_counters(self):
        self.counters = dict(prefill_calls=0, decode_calls=0, prefill_query_tokens=0,
                             decode_query_tokens=0, transfer_send_bytes=0, transfer_recv_bytes=0)

    # 将该调用设为无梯度推理
    @torch.inference_mode()
    # 执行当前模块前向
    def forward(self, seqs, prefill):
        # 导入 set_context, reset_context 所需接口
        from myvllm.utils import set_context, reset_context
        # 设置展平输入、KV 写入槽位、各 Query 的页表、各 Query 的有效长度、Query 累计偏移、历史累计偏移
        ids, slots, tables, lengths, cuq, cuk = [], [], [], [], [0], [0]
        # 逐个处理请求
        for seq in seqs:
            # 设置 Token 序列、KV 页数量或页表
            tokens, pages = seq['token_ids'], seq['block_table']
            # 设置窗口或分片起点、窗口或分片终点
            start, end = (0, len(tokens)) if prefill else (len(tokens)-1, len(tokens))
            # Query 窗口与页表宽度必须在受控配置范围内
            assert 0 <= start < end <= 2048 and 0 < len(pages) <= 8
            # 写入与读取必须落在当前分配的有效页范围内
            assert all(0 <= p < self.pages for p in pages) and end <= len(pages)*256
            # 向展平输入追加本次内容
            ids.extend(tokens[start:end])
            # 向Query 累计偏移追加本次内容
            cuq.append(cuq[-1] + end-start)
            # 向历史累计偏移追加本次内容
            cuk.append(cuk[-1] + end)
            # 逐个处理绝对 Token 位置
            for position in range(start, end):
                # 向各 Query 的页表追加本次内容
                tables.append(pages + [-1]*(8-len(pages)))
                # 每行只允许读取到自身位置，保持 Target 因果性
                lengths.append(position+1)
                # 向KV 写入槽位追加本次内容
                slots.append(pages[position//256]*256 + position % 256)
        # 把元数据转换成指定设备张量
        def tensor(data, dtype):
            # 返回指定类型的元数据张量
            return torch.tensor(data, dtype=dtype, device=self.device)
        # 把本轮 KV 地址与有效长度传给模型
        set_context(is_prefill=prefill,
            # Prefill 才需要提供各请求 Query 边界
            cu_seqlens_q=tensor(cuq, torch.int32) if prefill else None,
            # Prefill 才需要提供各请求历史边界
            cu_seqlens_k=tensor(cuk, torch.int32) if prefill else None,
            # 记录本轮最长 Query
            max_seqlen_q=max(b-a for a,b in zip(cuq,cuq[1:])),
            # 记录最大因果长度和逐 Token KV 写入槽
            max_seqlen_k=max(lengths), slot_mapping=tensor(slots, torch.long),
            # 指定有效历史长度
            context_lens=tensor(lengths, torch.int32), block_tables=tensor(tables, torch.int32))
        # 执行操作并确保后续清理
        try:
            # 设置词表分数
            logits = self.model.compute_logits(self.model(tensor(ids, torch.long)))
            # 此入口仅返回每请求最后一个位置的词表分数
            assert logits.shape == (len(seqs), self.hc.vocab_size)
            # 检查张量没有 NaN 或无穷值
            assert logits.dtype == torch.float32 and torch.isfinite(logits).all().item(), 'Non-finite logits'
            # 设置请求编号或图形状
            key = 'prefill' if prefill else 'decode'
            # 更新分阶段执行计数
            self.counters[key+'_calls'] += 1
            # 更新分阶段执行计数
            self.counters[key+'_query_tokens'] += len(ids)
            # 返回完整词表的预测分数
            return logits
        # 无论成功或异常都执行清理
        finally:
            # 清除本次全局前向元数据，避免影响后续调用
            reset_context()

    # 把逻辑 Token 位置转换成物理槽位
    def slots(self, pages, length):
        # 逻辑读取长度必须能被已有页表容纳
        assert 0 < length <= len(pages)*256
        # 返回指定类型的元数据张量
        return torch.tensor([pages[i//256]*256+i%256 for i in range(length)],
                            # 用整型槽位在当前 GPU 上执行索引
                            dtype=torch.long, device=self.device)

    # 把页和页内偏移展平成槽位维度
    def flat_cache(self):
        # 返回共享存储的槽位视图
        return self.cache.view(2, self.hc.num_hidden_layers, self.pages*256,
                               # 补充把页和页内偏移展平成槽位维度的字段
                               self.hc.num_key_value_heads//self.tp, self.hc.head_dim)

    # 按逻辑次序打包全层 Prompt KV
    def pack(self, pages, length):
        # 这里仅迁移完整副本的 KV，不支持 TP 分片迁移
        assert self.tp == 1
        # 按源页表取出全层有效 Prompt KV，排成连续传输张量
        return self.flat_cache().index_select(2, self.slots(pages, length)).contiguous()

    # 按目的页表写入迁移的全层 KV
    def unpack(self, packed, pages, length):
        # 这里仅迁移完整副本的 KV，不支持 TP 分片迁移
        assert self.tp == 1
        # 检查张量形状与本轮数据一致
        assert packed.shape == (2, self.hc.num_hidden_layers, length,
                                # 校验迁移张量的 KV Head 与 Head Dim
                                self.hc.num_key_value_heads, self.hc.head_dim)
        # 按目的页表重映射写入，目的物理页号可与源不同
        self.flat_cache().index_copy_(2, self.slots(pages, length), packed)

    def stats(self, rank):
        return dict(rank=rank, gpu=torch.cuda.get_device_name(self.device), tp=self.tp,
                    model_parameter_gib=sum(p.numel()*p.element_size() for p in self.model.parameters())/2**30,
                    fp32_head_extra_gib=self.arithmetic['head_extra_bytes']/2**30,
                    kv_gib=self.cache.numel()*self.cache.element_size()/2**30,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(self.device)/2**30,
                    counters=self.counters.copy())


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
