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

    @torch.inference_mode()
    def forward(self, seqs, prefill):
        from myvllm.utils import set_context, reset_context
        ids, slots, tables, lengths, cuq, cuk = [], [], [], [], [0], [0]
        for seq in seqs:
            tokens, pages = seq['token_ids'], seq['block_table']
            start, end = (0, len(tokens)) if prefill else (len(tokens)-1, len(tokens))
            assert 0 <= start < end <= 2048 and 0 < len(pages) <= 8
            assert all(0 <= p < self.pages for p in pages) and end <= len(pages)*256
            ids.extend(tokens[start:end])
            cuq.append(cuq[-1] + end-start)
            cuk.append(cuk[-1] + end)
            for position in range(start, end):
                tables.append(pages + [-1]*(8-len(pages)))
                lengths.append(position+1)
                slots.append(pages[position//256]*256 + position % 256)
        def tensor(data, dtype):
            return torch.tensor(data, dtype=dtype, device=self.device)
        set_context(is_prefill=prefill,
            cu_seqlens_q=tensor(cuq, torch.int32) if prefill else None,
            cu_seqlens_k=tensor(cuk, torch.int32) if prefill else None,
            max_seqlen_q=max(b-a for a,b in zip(cuq,cuq[1:])),
            max_seqlen_k=max(lengths), slot_mapping=tensor(slots, torch.long),
            context_lens=tensor(lengths, torch.int32), block_tables=tensor(tables, torch.int32))
        try:
            logits = self.model.compute_logits(self.model(tensor(ids, torch.long)))
            assert logits.shape == (len(seqs), self.hc.vocab_size)
            assert logits.dtype == torch.float32 and torch.isfinite(logits).all().item(), 'Non-finite logits'
            key = 'prefill' if prefill else 'decode'
            self.counters[key+'_calls'] += 1
            self.counters[key+'_query_tokens'] += len(ids)
            return logits
        finally:
            reset_context()

    def slots(self, pages, length):
        assert 0 < length <= len(pages)*256
        return torch.tensor([pages[i//256]*256+i%256 for i in range(length)],
                            dtype=torch.long, device=self.device)

    def flat_cache(self):
        return self.cache.view(2, self.hc.num_hidden_layers, self.pages*256,
                               self.hc.num_key_value_heads//self.tp, self.hc.head_dim)

    def pack(self, pages, length):
        assert self.tp == 1
        return self.flat_cache().index_select(2, self.slots(pages, length)).contiguous()

    def unpack(self, packed, pages, length):
        assert self.tp == 1
        assert packed.shape == (2, self.hc.num_hidden_layers, length,
                                self.hc.num_key_value_heads, self.hc.head_dim)
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
