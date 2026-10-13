"""Fixed-shape target verification graphs for the archived MinivLLM Qwen model.

Only B in {1, 2} and uniform Q in {1, 16} are captured. Every query uses its
absolute position and the existing per-query PagedAttention implementation.
Long initial prefills and nonuniform windows use the identical eager forward.
Layer hooks execute during capture; their tensor operations are recorded in the
graph and Python hooks are never needed during replay.

This helper is process-local and sequential. The model, KV cache and global
myvllm context must not be used concurrently by another engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


def _get(view, name):
    return view[name] if isinstance(view, dict) else getattr(view, name)


def build_query_metadata(views, *, max_length=2048, pages=64,
                         block_size=256, max_batch=2):
    """Build and validate host metadata, including queries crossing KV pages.

    token_ids holds the complete prefix plus the pending verification window;
    _scheduled_start/end delimit only the rows evaluated by this forward.
    """
    if not views or len(views) > max_batch:
        raise ValueError(f'Expected 1..{max_batch} windows, got {len(views)}')
    width = math.ceil(max_length / block_size)
    ids, slots, lengths, tables, query_lengths = [], [], [], [], []
    for view in views:
        tokens = _get(view, 'token_ids')
        start, end = int(_get(view, '_scheduled_start')), int(_get(view, '_scheduled_end'))
        table = [int(p) for p in _get(view, 'block_table')]
        if not 0 <= start < end <= min(len(tokens), max_length):
            raise ValueError(f'Invalid query window [{start}, {end}) for {len(tokens)} tokens')
        if not 0 < len(table) <= width or end > len(table)*block_size:
            raise ValueError(f'Page table cannot address query end {end}: {table}')
        if len(set(table)) != len(table) or any(p < 0 or p >= pages for p in table):
            raise ValueError(f'Invalid or repeated physical pages: {table}')
        padded = table + [-1]*(width-len(table))
        query_lengths.append(end-start)
        ids.extend(int(t) for t in tokens[start:end])
        for position in range(start, end):
            lengths.append(position+1)
            slots.append(table[position//block_size]*block_size + position % block_size)
            tables.append(padded.copy())
    uniform_q = query_lengths[0] if len(set(query_lengths)) == 1 else None
    return dict(input_ids=ids, slot_mapping=slots, context_lens=lengths,
                block_tables=tables, query_lengths=query_lengths,
                batch=len(views), key=(len(views), uniform_q))


@dataclass
class _Buffers:
    input_ids: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    graph: Any = None
    outputs: dict | None = None
    private_allocated_bytes: int = 0


class TargetGraphExecutor:
    """Own four independent graph pools and return owned outputs from each run.

    run returns picks [N], features [N, len(layer_ids)*H], hidden [N, H],
    and logits [N, V] only when diagnostic=True (otherwise logits is None).
    Returned tensors are clones, so a subsequent graph replay cannot mutate a
    request's retained features. Eager uses the same output ownership policy.
    """

    def __init__(self, model, layer_ids, device='cuda:0', max_length=2048,
                 pages=64, block_size=256, max_batch=2):
        self.model = model
        self.device = torch.device(device)
        self.layer_ids = tuple(int(i) for i in layer_ids)
        self.max_length, self.pages = int(max_length), int(pages)
        self.block_size, self.max_batch = int(block_size), int(max_batch)
        if self.max_batch not in (1, 2) or self.pages < self.max_batch:
            raise ValueError('Graph experiment supports max_batch 1 or 2 and distinct dummy pages')
        if self.max_length < 16 or self.block_size < 16:
            raise ValueError('Graph experiment requires max_length and block_size >= 16')
        layers = model.model.layers
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError('Feature layers must be nonempty and unique')
        if any(i < 0 or i >= len(layers) for i in self.layer_ids):
            raise ValueError(f'Feature layer outside model: {self.layer_ids}')
        self.table_width = math.ceil(self.max_length / self.block_size)
        self._entries = {}
        self._captured_features = {}
        self._handles = []
        self._closed = False
        for index in self.layer_ids:
            def hook(module, args, output, index=index):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError('Expected MinivLLM (x, residual) decoder output')
                x, residual = output
                self._captured_features[index] = x + residual
            self._handles.append(layers[index].register_forward_hook(hook))
        self.reset_counters()

    def _allocate(self, rows):
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=self.device)
        return _Buffers(input_ids=empty((rows,), torch.int64),
                        slot_mapping=empty((rows,), torch.int64),
                        context_lens=empty((rows,), torch.int64),
                        block_tables=empty((rows, self.table_width), torch.int32))

    def _fill(self, buffers, metadata):
        # All four inputs are replaced on every call; no stale page-table tails.
        # Host staging and copies intentionally remain outside graph capture.
        for name in ('input_ids', 'slot_mapping', 'context_lens', 'block_tables'):
            destination = getattr(buffers, name)
            source = torch.tensor(metadata[name], dtype=destination.dtype,
                                  device='cpu')
            if source.shape != destination.shape:
                raise RuntimeError(f'Static input shape changed: {name} {source.shape} != {destination.shape}')
            destination.copy_(source)

    def _forward(self, buffers):
        from myvllm.utils import set_context, reset_context
        self._captured_features.clear()
        # is_prefill=False supplies exact absolute RoPE positions without the
        # model's host-side cu_seqlens.cpu().tolist() branch. Attention remains
        # per-query causal PagedAttention for Q=1, Q=16 and initial full prefill.
        set_context(is_prefill=False, slot_mapping=buffers.slot_mapping,
                    context_lens=buffers.context_lens,
                    block_tables=buffers.block_tables)
        try:
            hidden = self.model(buffers.input_ids)
            if set(self._captured_features) != set(self.layer_ids):
                raise RuntimeError('Target feature hooks did not capture every selected layer')
            features = torch.cat([self._captured_features[i] for i in self.layer_ids], dim=-1)
            logits = self.model.compute_logits(hidden)
            if logits.ndim != 2 or logits.shape[0] != buffers.input_ids.shape[0]:
                raise RuntimeError('Verification LM head must produce one logit row per input row')
            if logits.dtype != torch.float32:
                raise RuntimeError(f'Expected the existing FP32 LM head, got {logits.dtype}')
            picks = logits.argmax(dim=-1)
            return dict(picks=picks, features=features, hidden=hidden, logits=logits)
        finally:
            self._captured_features.clear()
            reset_context()

    @torch.inference_mode()
    def capture_all(self):
        """Capture four valid dummy shapes; caller must clear KV afterwards.

        Every graph gets a fresh private pool (no pool-sharing argument).
        Dummy windows start at zero and use distinct valid physical pages,
        so warmup/capture never accesses uninitialized prompt positions.
        """
        if self._closed:
            raise RuntimeError('Executor is closed')
        if self.device.type != 'cuda':
            raise RuntimeError('CUDA Graph capture requires an actual CUDA device')
        if any(entry.graph is not None for entry in self._entries.values()):
            raise RuntimeError('Graphs have already been captured')
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            stream = torch.cuda.Stream(device=self.device)
            for batch in range(1, self.max_batch+1):
                for query in (1, 16):
                    key = (batch, query)
                    buffers = self._allocate(batch*query)
                    views = [dict(token_ids=[0]*query, _scheduled_start=0,
                                  _scheduled_end=query, block_table=[b])
                             for b in range(batch)]
                    metadata = build_query_metadata(views, max_length=self.max_length,
                        pages=self.pages, block_size=self.block_size, max_batch=self.max_batch)
                    self._fill(buffers, metadata)
                    stream.wait_stream(current)
                    # Compile Triton and initialize GEMM state outside capture.
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            outputs = self._forward(buffers)
                        del outputs
                    current.wait_stream(stream)
                    torch.cuda.synchronize(self.device)
                    before = torch.cuda.memory_allocated(self.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = self._forward(buffers)
                    current.wait_stream(stream)
                    torch.cuda.synchronize(self.device)
                    buffers.graph, buffers.outputs = graph, outputs
                    buffers.private_allocated_bytes = max(0, torch.cuda.memory_allocated(self.device)-before)
                    self._entries[key] = buffers
                    print(f'TARGET CUDA GRAPH CAPTURED: batch={batch}, query={query}, '
                          f'rows={batch*query}, table_width={self.table_width}', flush=True)
        self.reset_counters()

    @torch.inference_mode()
    def run(self, views, use_graph=False, diagnostic=False):
        if self._closed:
            raise RuntimeError('Executor is closed')
        metadata = build_query_metadata(views, max_length=self.max_length,
            pages=self.pages, block_size=self.block_size, max_batch=self.max_batch)
        key = metadata['key']
        eligible = key[1] in (1, 16)
        buffers = self._entries.get(key) if eligible else None
        if buffers is None:
            if use_graph and eligible:
                raise RuntimeError(f'Graph {key} was not captured; call capture_all() before running')
            buffers = self._allocate(len(metadata['input_ids']))
            # Retain supported static shapes even in eager-only local tests.
            if eligible:
                self._entries[key] = buffers
        self._fill(buffers, metadata)
        if use_graph and eligible:
            if buffers.graph is None:
                raise RuntimeError(f'Graph {key} has not been captured')
            buffers.graph.replay()
            outputs = buffers.outputs
            self._graph_replays[self._key_name(key)] += 1
        else:
            outputs = self._forward(buffers)
            self._eager_calls += 1
            if use_graph:
                self._fallbacks += 1
        # replay writes the captured output addresses again. Own each returned
        # feature tensor so retained per-request features cannot silently change.
        result = {name: outputs[name].clone() for name in ('picks', 'features', 'hidden')}
        result['logits'] = outputs['logits'].clone() if diagnostic else None
        return result

    @staticmethod
    def _key_name(key):
        return f'b{key[0]}-q{key[1]}'

    def reset_counters(self):
        self._eager_calls, self._fallbacks = 0, 0
        self._graph_replays = {f'b{b}-q{q}': 0 for b in range(1, self.max_batch+1) for q in (1, 16)}

    def stats(self):
        captured = [(key, value) for key, value in self._entries.items() if value.graph is not None]
        return dict(graph_replays=self._graph_replays.copy(),
                    total_replays=sum(self._graph_replays.values()), eager_calls=self._eager_calls,
                    fallbacks=self._fallbacks,
                    captured_keys=[self._key_name(key) for key, _ in sorted(captured)],
                    private_graph_allocated_bytes=sum(value.private_allocated_bytes for _, value in captured),
                    private_graph_pools=True, feature_layers=list(self.layer_ids),
                    metadata_table_width=self.table_width,
                    output_ownership='cloned in eager and graph paths')

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured_features.clear()
        self._entries.clear()
        self._closed = True
