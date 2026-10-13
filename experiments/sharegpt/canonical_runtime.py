"""Opt-in arithmetic reference; preserves scheduler, page ownership and Graph.

All query positions use the existing PagedAttention Decode kernel. This changes
Prefill arithmetic and performance, so results are not the old Flash benchmark.
"""
import inspect
from types import MethodType
import torch
from fixed_shape import fixed_rows_forward, install_fixed_shape_linears


def query_metadata(seqs, width, block_size, total_pages):
    tables, lengths, slots = [], [], []
    for seq in seqs:
        start, end = seq._scheduled_start, seq._scheduled_end
        table = list(seq.block_table)
        assert 0 <= start < end <= len(seq)
        assert 0 < len(table) <= width and end <= len(table)*block_size
        assert all(0 <= page < total_pages for page in table)
        padded = table + [-1]*(width-len(table))
        for position in range(start, end):
            tables.append(padded)
            lengths.append(position+1)
            slots.append(table[position//block_size]*block_size+position%block_size)
    return tables, lengths, slots


def unwrap_modules(model):
    counts = {'LayerNorm': 0, 'SiluAndMul': 0, 'RotaryEmbedding': 0}
    for module in model.model.modules():
        name = type(module).__name__
        if name not in counts:
            continue
        attribute = 'rms_forward' if name == 'LayerNorm' else 'forward'
        function = getattr(type(module), attribute)
        raw = getattr(function, '_torchdynamo_orig_callable', None)
        if raw is None:
            raw = inspect.unwrap(function)
        assert inspect.isfunction(raw) and raw is not function, (name, 'Expected compiled method')
        setattr(module, attribute, MethodType(raw, module))
        counts[name] += 1
    assert counts == {'LayerNorm': 145, 'SiluAndMul': 36, 'RotaryEmbedding': 36}, counts
    return counts


def install_model(model):
    from myvllm.layers.attention import Attention, store_kvcache, paged_attention_decode
    from myvllm.utils import get_context
    from fp32_head import install_fp32_head
    unwrapped = unwrap_modules(model)
    linear = install_fixed_shape_linears(model)
    counts = dict(attention=0, rms=0, activation=0)
    for module in model.model.modules():
        if isinstance(module, Attention):
            def forward(q, k, v, module=module):
                context = get_context()
                assert q.ndim == k.ndim == v.ndim == 3
                assert q.dtype == k.dtype == v.dtype == module.k_cache.dtype == module.v_cache.dtype == torch.bfloat16
                tables = context.canonical_tables if context.is_prefill else context.block_tables
                lengths = context.canonical_lengths if context.is_prefill else context.context_lens
                assert tables.shape == (q.shape[0], 8) and lengths.numel() == q.shape[0]
                store_kvcache(k.contiguous(), v.contiguous(), module.k_cache, module.v_cache,
                              context.slot_mapping, module.block_size)
                output = paged_attention_decode(q, module.k_cache, module.v_cache, tables, lengths,
                    module.scale/(module.head_dim**0.5), module.num_heads, module.num_kv_heads,
                    module.head_dim, module.block_size)
                return output.reshape(q.shape[0], module.num_heads*module.head_dim)
            module.forward = forward
            counts['attention'] += 1
        elif type(module).__name__ == 'LayerNorm':
            original = module.rms_forward
            rows = 1024 if module.weight.numel() <= 128 else 32
            def rms(x, original=original, rows=rows):
                return fixed_rows_forward(original, x, rows)
            module.rms_forward = rms
            counts['rms'] += 1
        elif type(module).__name__ == 'SiluAndMul':
            original = module.forward
            def activate(x, original=original):
                return fixed_rows_forward(original, x, 32)
            module.forward = activate
            counts['activation'] += 1
    assert counts == dict(attention=36, rms=145, activation=36), counts
    return dict(unwrapped=unwrapped, fixed_linear=linear, modules=counts,
                head=install_fp32_head(model), dtype='BF16 core + fixed-row FP32 LM Head',
                attention='Existing per-query paged_attention_decode for all phases',
                block_table_width=8, torch_compile=False, cuda_graph=True,
                prefill_flash_kernel_used=False, repository_source_modified=False)


def install_runner_hooks():
    from myvllm.engine.model_runner import ModelRunner
    from myvllm.utils import get_context
    assert not getattr(ModelRunner, '_sharegpt_canonical_hooks', False)
    original_capture = ModelRunner.capture_cudagraph
    original_prefill = ModelRunner.prepare_prefill
    original_decode = ModelRunner.prepare_decode

    def capture(self):
        assert self.world_size == 1 and self.block_size == 256
        assert self.config['max_model_length'] == 2048
        assert not getattr(self, '_canonical_ready', False)
        self._canonical_config = install_model(self.model)
        self._canonical_ready = True
        print('CANONICAL CONFIG:', self._canonical_config, flush=True)
        # Installation precedes capture: replay and eager use the same operators.
        return original_capture(self)

    def prefill(self, seqs):
        input_ids = original_prefill(self, seqs)
        if not getattr(self, '_canonical_ready', False):
            return input_ids  # Initial allocator warmup precedes KV allocation.
        context = get_context()
        tables, lengths, slots = query_metadata(seqs, 8, self.block_size,
                                               self.config['max_cached_blocks'])
        assert len(lengths) == input_ids.numel() == context.slot_mapping.numel()
        context.canonical_tables = torch.tensor(tables, dtype=torch.int32, device=input_ids.device)
        context.canonical_lengths = torch.tensor(lengths, dtype=torch.int32, device=input_ids.device)
        # The existing runner remains responsible for actual KV write addresses.
        # Keep the CPU construction for regression tests and bounds validation.
        assert len(slots) == len(lengths)
        return input_ids

    def decode(self, seqs):
        input_ids = original_decode(self, seqs)
        if getattr(self, '_canonical_ready', False):
            context = get_context()
            width = context.block_tables.shape[1]
            assert 0 < width <= 8
            context.block_tables = torch.nn.functional.pad(context.block_tables, (0, 8-width), value=-1)
        return input_ids

    ModelRunner.capture_cudagraph = capture
    ModelRunner.prepare_prefill = prefill
    ModelRunner.prepare_decode = decode
    ModelRunner._sharegpt_canonical_hooks = True
