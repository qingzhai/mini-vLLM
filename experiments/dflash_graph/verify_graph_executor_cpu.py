"""CPU contracts only: real tensors plus a mock replay, never CUDA validation."""
from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
import unittest

import torch

from graph_executor import TargetGraphExecutor, build_query_metadata


_CONTEXT = None


def _set_context(**kwargs):
    global _CONTEXT
    _CONTEXT = SimpleNamespace(**kwargs)


def _reset_context():
    global _CONTEXT
    _CONTEXT = None


class FakeLayer(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.python_calls = 0

    def forward(self, x, residual):
        self.python_calls += 1
        residual = x if residual is None else x+residual
        return x*(1+self.index*0.01)+self.index, residual


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([FakeLayer(i) for i in range(5)])

    @staticmethod
    def embedding(ids, context):
        # Depend on every static input to detect incomplete buffer updates.
        base = (ids.float()+context.context_lens.float()*0.1
                +context.slot_mapping.float()*0.001
                +context.block_tables[:, 0].float()*0.03)
        return torch.stack([base, base+1, base+2], -1)

    def forward(self, ids):
        assert _CONTEXT is not None and _CONTEXT.is_prefill is False
        x, residual = self.embedding(ids, _CONTEXT), None
        for layer in self.model.layers:
            x, residual = layer(x, residual)
        return x+residual

    def compute_logits(self, hidden):
        value = hidden.float().sum(-1)
        return torch.stack([value, -value, hidden[:, 0].float()], dim=-1)

    def replay_formula(self, buffers):
        # Model the captured GPU tensor program without calling Python model
        # forwards/hooks. This is deliberately a mock, not a CUDA graph.
        context = SimpleNamespace(context_lens=buffers.context_lens,
            slot_mapping=buffers.slot_mapping, block_tables=buffers.block_tables)
        x, residual, features = self.embedding(buffers.input_ids, context), None, []
        for i in range(5):
            residual = x if residual is None else x+residual
            x = x*(1+i*0.01)+i
            features.append(x+residual)
        hidden = x+residual
        logits = self.compute_logits(hidden)
        return dict(picks=logits.argmax(-1), features=torch.cat(features, -1),
                    hidden=hidden, logits=logits)


class FakeGraph:
    def __init__(self, model, buffers):
        self.model, self.buffers, self.replays = model, buffers, 0

    def replay(self):
        self.replays += 1
        fresh = self.model.replay_formula(self.buffers)
        for name, tensor in fresh.items():
            self.buffers.outputs[name].copy_(tensor)


def window(start, end, table, token_offset=0):
    return SimpleNamespace(token_ids=[i+token_offset for i in range(end)],
        _scheduled_start=start, _scheduled_end=end, block_table=table)


class GraphExecutorCPUContractTests(unittest.TestCase):
    def setUp(self):
        self.previous_utils = sys.modules.get('myvllm.utils')
        utils = ModuleType('myvllm.utils')
        utils.set_context, utils.reset_context = _set_context, _reset_context
        sys.modules['myvllm.utils'] = utils
        self.model = FakeModel()
        self.executor = TargetGraphExecutor(self.model, range(5), device='cpu')

    def tearDown(self):
        self.executor.close()
        _reset_context()
        if self.previous_utils is None:
            sys.modules.pop('myvllm.utils', None)
        else:
            sys.modules['myvllm.utils'] = self.previous_utils

    def mock_capture(self):
        for batch in (1, 2):
            for query in (1, 16):
                views = [window(0, query, [b]) for b in range(batch)]
                metadata = build_query_metadata(views)
                buffers = self.executor._allocate(batch*query)
                self.executor._fill(buffers, metadata)
                buffers.outputs = self.executor._forward(buffers)
                buffers.graph = FakeGraph(self.model, buffers)
                self.executor._entries[(batch, query)] = buffers
        self.executor.reset_counters()

    def test_metadata_crosses_256_page_boundary(self):
        data = build_query_metadata([window(250, 266, [9, 3])])
        self.assertEqual(data['key'], (1, 16))
        self.assertEqual(data['context_lens'], list(range(251, 267)))
        self.assertEqual(data['slot_mapping'],
                         list(range(9*256+250, 10*256))+list(range(3*256, 3*256+10)))
        self.assertEqual(data['block_tables'], [[9, 3]+[-1]*6]*16)
        with self.assertRaises(ValueError):
            build_query_metadata([window(250, 266, [9])])
        with self.assertRaises(ValueError):
            build_query_metadata([window(0, 16, [64])])

    def test_mock_replay_updates_all_inputs_without_hooks(self):
        self.mock_capture()
        for batch in (1, 2):
            for query in (1, 16):
                key = (batch, query)
                buffers = self.executor._entries[key]
                addresses = [getattr(buffers, name).data_ptr() for name in
                    ('input_ids', 'slot_mapping', 'context_lens', 'block_tables')]
                views_a = [window(250, 250+query, [9+b, 3+b], 7+b) for b in range(batch)]
                expected = self.executor.run(views_a, use_graph=False, diagnostic=True)
                before_calls = sum(layer.python_calls for layer in self.model.model.layers)
                actual = self.executor.run(views_a, use_graph=True, diagnostic=True)
                after_calls = sum(layer.python_calls for layer in self.model.model.layers)
                self.assertEqual(before_calls, after_calls)
                for name in actual:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
                saved_features = actual['features'].clone()
                views_b = [window(5, 5+query, [20+b], 30+b) for b in range(batch)]
                updated = self.executor.run(views_b, use_graph=True, diagnostic=True)
                self.assertFalse(torch.equal(updated['features'], actual['features']))
                torch.testing.assert_close(actual['features'], saved_features, rtol=0, atol=0)
                self.assertEqual(addresses, [getattr(buffers, name).data_ptr() for name in
                    ('input_ids', 'slot_mapping', 'context_lens', 'block_tables')])
                self.assertTrue(torch.all(buffers.block_tables[:, 1:] == -1).item())
                self.assertEqual(buffers.context_lens.dtype, torch.int64)
                self.assertEqual(buffers.block_tables.dtype, torch.int32)
                self.assertEqual(actual['features'].shape, (batch*query, 15))
        stats = self.executor.stats()
        self.assertEqual(len(stats['captured_keys']), 4)
        self.assertEqual(list(stats['graph_replays'].values()), [2, 2, 2, 2])

    def test_nonuniform_and_long_prefill_fallback(self):
        self.mock_capture()
        result = self.executor.run([window(0, 20, [2])], use_graph=True)
        self.assertIsNone(result['logits'])
        self.assertEqual(result['picks'].shape, (20,))
        result = self.executor.run([window(0, 1, [2]), window(0, 16, [3])], use_graph=True)
        self.assertEqual(result['features'].shape, (17, 15))
        self.assertEqual(self.executor.stats()['fallbacks'], 2)
        self.assertEqual(self.executor.stats()['eager_calls'], 2)
        self.assertIsNone(_CONTEXT)

    def test_graph_requires_capture_and_close_removes_hooks(self):
        with self.assertRaises(RuntimeError):
            self.executor.run([window(0, 1, [2])], use_graph=True)
        with self.assertRaises(RuntimeError):
            self.executor.capture_all()
        self.executor.close()
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.model.layers))
        with self.assertRaises(RuntimeError):
            self.executor.run([window(0, 1, [2])])


if __name__ == '__main__':
    unittest.main(verbosity=2)
