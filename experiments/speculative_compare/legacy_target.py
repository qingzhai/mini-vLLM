"""Adapt the archived MinivLLM Target arithmetic to official DeepSpec draft loops.

This is a distinct hybrid backend, not an unchanged HF/DeepSpec reference run.
Only the Target cache factory in base_evaluator is rebound. Draft caches remain
Transformers DynamicCache, and official proposal/verification/acceptance code is
unchanged. Target generation remains batch=1 and Eager.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn


HERE = Path(__file__).resolve().parent


def archived_sources():
    root = HERE / 'legacy-source'
    if not root.exists():
        root = HERE.parents[1]
    manifest_path = HERE / 'legacy-source-manifest.json'
    expected = json.loads(manifest_path.read_text())
    for relative, digest in expected['sha256'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Archived Target source changed: ' + relative)
    return root, dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      files=len(expected['sha256']))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def checkpoint_config_context(path, auto_config):
    """Keep the archived constructor's raw config fields across HF 4.x/5.x."""
    path = Path(path).resolve()
    raw = json.loads((path / 'config.json').read_text())
    required = ('num_hidden_layers', 'hidden_size', 'num_attention_heads', 'num_key_value_heads',
                'head_dim', 'vocab_size', 'intermediate_size', 'rms_norm_eps',
                'attention_bias', 'tie_word_embeddings', 'rope_theta')
    if raw.get('model_type') != 'qwen3' or any(key not in raw for key in required):
        raise ValueError('Checkpoint is missing a field required by the archived Target constructor')
    original = auto_config.__dict__['from_pretrained']

    def load(source, *args, **kwargs):
        if Path(source).resolve() != path or args or kwargs != {'local_files_only': True}:
            raise ValueError('Unexpected config call during archived Target initialization')
        return SimpleNamespace(**raw)

    auto_config.from_pretrained = staticmethod(load)
    try:
        yield
    finally:
        auto_config.from_pretrained = original


@contextmanager
def legacy_arithmetic_context():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    old = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
           torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
           torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
           torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        # 与旧实验一致：Target 使用自定义 PagedAttention；Draft SDPA 使用 MATH。
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.set_float32_matmul_precision(old[0])
        torch.backends.cuda.matmul.allow_tf32 = old[1]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old[2]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old[3]
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(old[4])


class LegacyCache:
    """Logical Target history and original physical PagePool; no Draft KV here."""
    def __init__(self, pool):
        self.pool = pool
        self.tokens = []
        self.closed = False

    def get_seq_length(self, layer_idx=0):
        if self.closed:
            raise RuntimeError('Target cache already released')
        return len(self.tokens)

    def crop(self, max_length):
        if self.closed:
            raise RuntimeError('Target cache already released')
        end = max(0, len(self.tokens) + max_length) if max_length < 0 else max_length
        if end > len(self.tokens):
            raise ValueError('Cannot extend a Target cache by cropping')
        self.tokens = self.tokens[:end]
        if 0 in self.pool.tables:
            self.pool.trim(0, end)
        self.pool.check()

    def release(self):
        if not self.closed:
            if 0 in self.pool.tables:
                self.pool.release(0)
            self.pool.check()
            assert len(self.pool.free) == self.pool.pages
            self.tokens = []
            self.closed = True


def selected_hidden_states(features, hidden, layer_ids, layer_count):
    """Expose exactly the five raw layers consumed by the reviewed Drafts."""
    width = hidden.shape[-1]
    if features.shape[-1] != len(layer_ids) * width:
        raise ValueError('Target feature packing differs from the selected layers')
    states = [None] * (layer_count + 1)
    for index, layer in enumerate(layer_ids):
        states[layer + 1] = features[:, index*width:(index+1)*width].unsqueeze(0)
    states[-1] = hidden.unsqueeze(0)
    return tuple(states)


class LegacyTarget(nn.Module):
    def __init__(self, path, layer_ids, output, algorithm, logit_diagnostics=True):
        super().__init__()
        import torch.distributed as dist
        from transformers import AutoConfig
        root, self.source_identity = archived_sources()
        for folder in (root / 'src', root / 'experiments/common'):
            sys.path.insert(0, str(folder))
        target_module = load_module('_comparison_legacy_target', root / 'experiments/common/target.py')
        executor_module = load_module('_comparison_legacy_executor', root / 'experiments/dflash_graph/graph_executor.py')
        state_module = load_module('_comparison_legacy_state', root / 'experiments/dflash_graph/state.py')
        if not dist.is_initialized():
            rendezvous = Path(output).resolve() / f'{algorithm}-legacy-nccl-init'
            dist.init_process_group('nccl', init_method='file://' + str(rendezvous), rank=0, world_size=1,
                timeout=timedelta(seconds=180), device_id=torch.device('cuda', 0))
        if dist.get_world_size() != 1:
            raise ValueError('Legacy comparison backend is single GPU only')
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float16)
            # 新 Transformers 将 RoPE 字段收进 rope_parameters；旧构造器读原 rope_theta。
            # 仅此单线程构造期间桥接配置读取，数值来自已验证的原 config.json。
            with checkpoint_config_context(path, AutoConfig):
                self.legacy = target_module.Target(path, 0, dist.group.WORLD, 64)
        finally:
            torch.set_default_dtype(previous_dtype)
        self.model = self.legacy.model.model
        self.lm_head = self.legacy.model.lm_head
        self.config = self.legacy.hc
        self.layer_ids = tuple(map(int, layer_ids))
        if self.layer_ids != (1, 9, 17, 25, 33):
            raise ValueError('Expected the reviewed five Target feature layers')
        self.executor = executor_module.TargetGraphExecutor(self.legacy.model, self.layer_ids,
            max_length=2048, pages=64, block_size=256, max_batch=1)
        self.pool_class = state_module.PagePool
        self.current_cache = None
        self.trace = []
        self.logit_diagnostics = logit_diagnostics
        self.last_cache_released = True
        self.arithmetic = self.legacy.arithmetic
        self.eval()

    def new_cache(self):
        if self.current_cache is not None and not self.current_cache.closed:
            raise RuntimeError('Previous Target cache must be released before another request')
        self.current_cache = LegacyCache(self.pool_class(64, 256))
        self.last_cache_released = False
        return self.current_cache

    def release_cache(self):
        if self.current_cache is not None:
            self.current_cache.release()
        self.last_cache_released = True

    @torch.inference_mode()
    def forward(self, input_ids, past_key_values, use_cache=True, output_hidden_states=True,
                position_ids=None, logits_to_keep=0, **kwargs):
        if kwargs or not use_cache or not output_hidden_states:
            raise ValueError('Unsupported Target call outside the reviewed inference interface')
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError('Legacy Target adapter supports batch=1')
        cache = past_key_values
        if cache is not self.current_cache or cache.closed:
            raise ValueError('Target cache belongs to a different request')
        start = cache.get_seq_length()
        query = input_ids[0].tolist()
        end = start + len(query)
        if position_ids is not None and position_ids[0].tolist() != list(range(start, end)):
            raise ValueError('Target positions do not match the effective cache length')
        table = cache.pool.reserve(0, end)
        tokens = cache.tokens + query
        window = dict(token_ids=tokens, _scheduled_start=start, _scheduled_end=end, block_table=list(table))
        result = self.executor.run([window], use_graph=False, diagnostic=True)
        cache.tokens = tokens
        logits = result['logits'].unsqueeze(0)
        if self.logit_diagnostics:
            values, indices = result['logits'].topk(2, dim=-1)
            self.trace.append(dict(cache_length_before=start, query_ids=query,
                block_table=list(table), argmax=result['picks'].tolist(),
                top_ids=indices.tolist(), top_logits=values.tolist()))
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        states = selected_hidden_states(result['features'], result['hidden'], self.layer_ids,
                                        self.config.num_hidden_layers)
        return SimpleNamespace(logits=logits, hidden_states=states, past_key_values=cache)

    def precision_record(self):
        # 原 head 仍保留 FP16 Parameter；实际 GEMM 使用 install_arithmetic 捕获的 FP32 副本。
        return dict(requested='fp32', original_head_dtype='torch.float16',
            head_weight_dtype='torch.float32', head_compute_dtype='torch.float32', head_output_dtype='torch.float32',
            retained_checkpoint_head_dtype=str(self.lm_head.weight.dtype),
            backbone_dtype=str(next(self.model.parameters()).dtype),
            added_parameter_bytes=self.arithmetic['head_extra_bytes'],
            tf32=torch.backends.cuda.matmul.allow_tf32,
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            checkpoint_files_modified=False)

    def close(self):
        self.release_cache()
        self.executor.close()
