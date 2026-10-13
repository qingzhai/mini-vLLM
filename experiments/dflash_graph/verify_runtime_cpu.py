"""Actual V4 state machine + graph adapter on CPU tensor oracles.

CUDA synchronization/statistics and graph launches are mocked. No Qwen, Triton,
NCCL, real Draft, or actual CUDA Graph executes in this contract test.
"""
from __future__ import annotations

from copy import deepcopy
from types import MethodType, ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

from benchmark import compare_generation
from graph_executor import TargetGraphExecutor, build_query_metadata
from state import PagePool, Request


_CONTEXT = None


def set_context(**values):
    global _CONTEXT
    _CONTEXT = SimpleNamespace(**values)


def reset_context():
    global _CONTEXT
    _CONTEXT = None


class FakeDynamicCache:
    def __init__(self):
        self.length = 0

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        self.length = length


class OracleLayer(torch.nn.Module):
    def forward(self, x, residual=None):
        return x, torch.zeros_like(x) if residual is None else residual


class OracleModel(torch.nn.Module):
    vocab_size = 128

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(OracleLayer() for _ in range(36))

    @staticmethod
    def embeddings(ids, context):
        return torch.stack([ids.float(), context.context_lens.float()], -1)

    def forward(self, ids):
        assert _CONTEXT is not None and not _CONTEXT.is_prefill
        x, residual = self.embeddings(ids, _CONTEXT), None
        for layer in self.model.layers:
            x, residual = layer(x, residual)
        return x

    def compute_logits(self, hidden):
        posterior = (hidden[:, 0].long()+1) % self.vocab_size
        vocabulary = torch.arange(self.vocab_size, dtype=torch.float32)
        return -(vocabulary[None, :]-posterior[:, None].float()).square()

    def replay_formula(self, buffers):
        context = SimpleNamespace(context_lens=buffers.context_lens)
        hidden = self.embeddings(buffers.input_ids, context)
        logits = self.compute_logits(hidden)
        return dict(picks=logits.argmax(-1), features=torch.cat([hidden]*5, -1),
                    hidden=hidden, logits=logits)


class OracleDraft(torch.nn.Module):
    block_size = 16
    target_layer_ids = [1, 9, 17, 25, 33]
    config = SimpleNamespace(dflash_config={'mask_token_id': 151669})

    def __init__(self):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))


class MockCapturedGraph:
    """Update captured output addresses without executing Python layer hooks."""
    def __init__(self, model, buffers):
        self.model, self.buffers = model, buffers

    def replay(self):
        values = self.model.replay_formula(self.buffers)
        for name, value in values.items():
            self.buffers.outputs[name].copy_(value)


def capture_mock_graphs(executor):
    for batch in (1, 2):
        for query in (1, 16):
            views = [dict(token_ids=[0]*query, _scheduled_start=0,
                          _scheduled_end=query, block_table=[b]) for b in range(batch)]
            buffers = executor._allocate(batch*query)
            executor._fill(buffers, build_query_metadata(views))
            buffers.outputs = executor._forward(buffers)
            buffers.graph = MockCapturedGraph(executor.model, buffers)
            executor._entries[(batch, query)] = buffers
    executor.reset_counters()


def draft_oracle(self, request):
    # Exercise retained target features and DynamicCache accounting, while
    # varying acceptance by request so completion/refill and rejection happen.
    assert request.features.shape[0] == request.computed-request.draft_cache.get_seq_length()
    assert torch.isfinite(request.features).all().item()
    request.draft_cache.crop(request.computed)
    accepted = {0: 15, 1: 2, 2: 0}[request.key]
    block = [request.tokens[-1]]
    for position in range(1, 16):
        token = (block[-1]+1) % self.model.vocab_size
        if position == accepted+1:
            token = (token+17) % self.model.vocab_size
        block.append(token)
    return block


class RuntimeCPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = {name: sys.modules.get(name) for name in
            ('myvllm', 'myvllm.utils', 'consistent_target', 'transformers')}
        myvllm, utils = ModuleType('myvllm'), ModuleType('myvllm.utils')
        utils.reset_context, utils.set_context = reset_context, set_context
        myvllm.utils = utils
        consistent = ModuleType('consistent_target')
        consistent.attach_query_metadata = lambda *args, **kwargs: None
        transformers = ModuleType('transformers')
        transformers.DynamicCache = FakeDynamicCache
        sys.modules.update({'myvllm': myvllm, 'myvllm.utils': utils,
                            'consistent_target': consistent, 'transformers': transformers})
        from graph_runtime import GraphDFlashRuntime
        cls.runtime_type = GraphDFlashRuntime

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    def setUp(self):
        self.model, self.draft = OracleModel(), OracleDraft()
        self.executor = TargetGraphExecutor(self.model, [1, 9, 17, 25, 33], device='cpu')
        capture_mock_graphs(self.executor)
        self.runtime = self.runtime_type(SimpleNamespace(model=self.model, pages=64),
                                         self.draft, self.executor)
        self.runtime.propose = MethodType(draft_oracle, self.runtime)
        self.patches = [patch('torch.cuda.synchronize', return_value=None),
                        patch('torch.cuda.reset_peak_memory_stats', return_value=None),
                        patch('torch.cuda.max_memory_allocated', return_value=0)]
        for value in self.patches:
            value.start()

    def tearDown(self):
        self.runtime.close()
        self.executor.close()
        for value in reversed(self.patches):
            value.stop()
        reset_context()

    @staticmethod
    def prompts():
        return [[1]*(length-1)+seed for length, seed in
                ((250, [3]), (251, [17]), (507, [41]))]

    def all_modes(self):
        outputs = {}
        for speculative in (False, True):
            for graph in (False, True):
                self.runtime.graph_enabled = graph
                label = ('dflash' if speculative else 'ar')+('_graph' if graph else '_eager')
                outputs[label] = self.runtime.generate(self.prompts(), max_new_tokens=64,
                    max_batch_size=2, speculative=speculative)
        return outputs

    def test_actual_generate_four_modes_parity_refill_rollback_and_graph_counts(self):
        outputs = self.all_modes()
        reference = outputs['ar_eager']
        for label, actual in outputs.items():
            self.assertTrue(compare_generation(reference, actual)['passed'], label)
            summary = actual['summary']
            self.assertEqual(summary['output_tokens'], 192)
            self.assertEqual(summary['max_verification_batch'], 2)
            self.assertEqual(summary['peak_active_requests'], 2)
            self.assertTrue(summary['all_pages_released'])
            execution = summary['target_execution']
            self.assertEqual(execution['fallbacks'], 0)
            expected_replays = summary['verification_batch_calls'] if label.endswith('_graph') else 0
            self.assertEqual(execution['total_replays'], expected_replays)
            self.assertEqual(summary['target_forward_calls'],
                             execution['eager_calls']+execution['total_replays'])
            for request, seed in zip(actual['requests'], (3, 17, 41)):
                self.assertEqual(request['token_ids'], [(seed+i+1) % 128 for i in range(64)])
        self.assertTrue(compare_generation(outputs['ar_eager'], outputs['ar_graph'], True)['passed'])
        self.assertTrue(compare_generation(outputs['dflash_eager'], outputs['dflash_graph'], True)['passed'])
        for label in ('dflash_eager', 'dflash_graph'):
            summary = outputs[label]['summary']
            self.assertGreater(summary['refill_while_active'], 0)
            self.assertGreater(summary['rollback_pages'], 0)
            self.assertGreater(summary['accepted_draft_tokens'], 0)
            self.assertLess(summary['target_forward_calls'], reference['summary']['target_forward_calls'])
        graph_counts = outputs['dflash_graph']['summary']['target_execution']['graph_replays']
        self.assertGreater(graph_counts['b1-q16'], 0)
        self.assertGreater(graph_counts['b2-q16'], 0)

    def test_runtime_retained_feature_ownership_and_initial_prefill_eager(self):
        self.runtime.graph_enabled = True
        pool = PagePool(64, 256)
        request = Request(0, self.prompts()[0], 250)
        picks, features = self.runtime.target_forward([request], [request.tokens], pool)[0]
        request.computed = len(request.tokens)
        request.tokens.append(picks[-1])
        initial = features.clone()
        block = [request.tokens[-1]]+list(range(15))
        _, first = self.runtime.target_forward([request], [block], pool)[0]
        retained = first.clone()
        _, second = self.runtime.target_forward([request],
            [[request.tokens[-1]]+list(range(30, 45))], pool)[0]
        self.assertFalse(torch.equal(first, second))
        torch.testing.assert_close(first, retained, rtol=0, atol=0)
        torch.testing.assert_close(features, initial, rtol=0, atol=0)
        stats = self.executor.stats()
        self.assertEqual(stats['eager_calls'], 1)
        self.assertEqual(stats['total_replays'], 2)
        pool.release(request.key)
        pool.check()

    def test_comparison_rejects_corrupt_token_trace_and_summary(self):
        outputs = self.all_modes()
        reference = outputs['dflash_eager']
        bad_token = deepcopy(outputs['dflash_graph'])
        bad_token['requests'][1]['token_ids'][7] += 1
        check = compare_generation(reference, bad_token, True)
        self.assertFalse(check['passed'])
        self.assertEqual(check['checks'][1]['first_mismatch'], 7)
        bad_trace = deepcopy(outputs['dflash_graph'])
        bad_trace['requests'][0]['trace'][0]['advance'] += 1
        self.assertFalse(compare_generation(reference, bad_trace, True)['passed'])
        self.assertTrue(compare_generation(reference, bad_trace, False)['passed'])
        bad_summary = deepcopy(outputs['dflash_graph'])
        bad_summary['summary']['rollback_pages'] += 1
        self.assertFalse(compare_generation(reference, bad_summary, True)['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
