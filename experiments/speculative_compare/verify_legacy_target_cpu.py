"""Check the hybrid Target interface and rollback with CPU fixtures, no GPU claims."""
import importlib.util
import ast
import json
import os
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace, ModuleType
import typing

import torch
from torch import nn

from legacy_target import (LegacyCache, LegacyTarget, archived_sources,
                           checkpoint_config_context, legacy_arithmetic_context, selected_hidden_states)


ROOT = archived_sources()[0]
spec = importlib.util.spec_from_file_location('_cpu_legacy_pages', ROOT/'experiments/dflash_graph/state.py')
pages = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pages
spec.loader.exec_module(pages)


class FakeExecutor:
    def run(self, windows, use_graph, diagnostic):
        assert not use_graph and diagnostic
        w = windows[0]
        ids = w['token_ids'][w['_scheduled_start']:w['_scheduled_end']]
        logits = torch.full((len(ids), 8), -1., dtype=torch.float32)
        picks = (torch.tensor(ids) + 1) % 8
        logits[torch.arange(len(ids)), picks] = 2.
        hidden = torch.tensor(ids, dtype=torch.float16).reshape(-1, 1).repeat(1, 2)
        return dict(logits=logits, picks=picks, hidden=hidden,
                    features=torch.cat([hidden + i for i in range(5)], dim=-1))


class LegacyTargetTests(unittest.TestCase):
    def target(self):
        target = LegacyTarget.__new__(LegacyTarget)
        nn.Module.__init__(target)
        target.pool_class = pages.PagePool
        target.current_cache = None
        target.last_cache_released = True
        target.trace = []
        target.logit_diagnostics = True
        target.executor = FakeExecutor()
        target.layer_ids = (1, 9, 17, 25, 33)
        target.config = SimpleNamespace(num_hidden_layers=36)
        return target

    def test_original_source_hashes_are_verified(self):
        root, identity = archived_sources()
        self.assertTrue((root/'src/myvllm/models/qwen3.py').exists())
        self.assertGreater(identity['files'], 10)

    def test_crop_crosses_pages_without_erasing_retained_page(self):
        cache = LegacyCache(pages.PagePool(4, 4))
        original = list(cache.pool.reserve(0, 9))
        cache.tokens = list(range(9))
        cache.crop(5)
        self.assertEqual(cache.tokens, list(range(5)))
        self.assertEqual(cache.pool.tables[0], original[:2])
        self.assertEqual(cache.pool.rollback_pages, 1)
        cache.release()
        self.assertEqual(len(cache.pool.free), 4)
        with self.assertRaises(RuntimeError):
            cache.get_seq_length()

    def test_forward_returns_correct_feature_indices_and_pending_crop(self):
        target = self.target()
        cache = target.new_cache()
        prompt = target(torch.tensor([[0, 1, 2]]), cache, logits_to_keep=1)
        self.assertEqual(prompt.logits.shape, (1, 1, 8))
        self.assertEqual(prompt.logits.argmax(-1).item(), 3)
        self.assertEqual(prompt.hidden_states[2].shape, (1, 3, 2))
        result = target(torch.tensor([[3, 4, 7]]), cache, position_ids=torch.tensor([[3, 4, 5]]))
        self.assertEqual(result.logits.argmax(-1).tolist(), [[4, 5, 0]])
        # 接受 4，修正为 5：保留 Target KV [Prompt,3,4]，5 作为 Pending。
        cache.crop(5)
        self.assertEqual(cache.tokens, [0, 1, 2, 3, 4])
        target(torch.tensor([[5]]), cache)
        self.assertEqual(cache.tokens, [0, 1, 2, 3, 4, 5])
        self.assertEqual(target.trace[-1]['cache_length_before'], 5)
        target.release_cache()
        self.assertTrue(target.last_cache_released)
        self.assertEqual(len(cache.pool.free), 64)

    def test_wrong_positions_and_overlapping_requests_fail(self):
        target = self.target()
        cache = target.new_cache()
        with self.assertRaises(RuntimeError):
            target.new_cache()
        with self.assertRaises(ValueError):
            target(torch.tensor([[1]]), cache, position_ids=torch.tensor([[9]]))
        target.release_cache()
        self.assertIsNot(target.new_cache(), cache)

    def test_feature_layout_and_precision_flags_restore(self):
        states = selected_hidden_states(torch.arange(20).reshape(2, 10), torch.ones(2, 2),
                                        (1, 9, 17, 25, 33), 36)
        self.assertEqual(states[10].tolist(), [[[2, 3], [12, 13]]])
        self.assertIsNone(states[1])
        before = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        with legacy_arithmetic_context():
            self.assertFalse(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
            self.assertFalse(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        self.assertEqual(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, before)

    def test_raw_config_compatibility_restores_hf_factory(self):
        class AutoConfig:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                return 'original'
        original = AutoConfig.__dict__['from_pretrained']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            values = dict(model_type='qwen3', num_hidden_layers=36, hidden_size=4096,
                num_attention_heads=32, num_key_value_heads=8, head_dim=128, vocab_size=151936,
                intermediate_size=12288, rms_norm_eps=1e-6, attention_bias=False,
                tie_word_embeddings=False, rope_theta=1000000)
            (path/'config.json').write_text(json.dumps(values))
            with checkpoint_config_context(path, AutoConfig):
                loaded = AutoConfig.from_pretrained(path, local_files_only=True)
                self.assertEqual(loaded.rope_theta, 1000000)
                with self.assertRaises(ValueError):
                    AutoConfig.from_pretrained(path/'other', local_files_only=True)
            self.assertIs(AutoConfig.__dict__['from_pretrained'], original)
            self.assertEqual(AutoConfig.from_pretrained(path), 'original')

    @unittest.skipUnless(os.environ.get('DEEPSPEC_SOURCE'), 'Optional: set DEEPSPEC_SOURCE for official loop fixture')
    def test_official_acceptance_loop_uses_native_target_cache(self):
        source = Path(os.environ['DEEPSPEC_SOURCE'])
        sampling = ModuleType('_legacy_fixture_sampling')
        exec((source/'deepspec/utils/sampling.py').read_text(), sampling.__dict__)
        namespace = dict(torch=torch, DynamicCache=None, SimpleNamespace=SimpleNamespace,
                         Any=typing.Any, Callable=typing.Callable)
        for name in ('logits_to_probs', 'sample_from_probs', 'sample_residual', 'gather_token_probs'):
            namespace[name] = getattr(sampling, name)
        names = {'DraftProposal', 'VerificationResult', 'verify_draft_tokens',
                 'generate_decoding_sample', 'has_stop_token', 'trim_output_ids'}
        parsed = ast.parse((source/'deepspec/eval/base_evaluator.py').read_text())
        body = ast.parse('from __future__ import annotations\nfrom dataclasses import dataclass').body
        body += [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        # 执行已核对的原函数体；只把 Target cache factory 换为分页适配器。
        exec(compile(ast.Module(body=body, type_ignores=[]), '<official-base-evaluator-fixture>', 'exec'), namespace)
        target = self.target()
        namespace['DynamicCache'] = target.new_cache

        def propose(output_ids, start, **kwargs):
            pending = int(output_ids[0, start])
            query = torch.tensor([[(pending + i) % 8 for i in range(4)]])
            query[0, 2] = (query[0, 2] + 2) % 8  # 强制在第二个候选拒绝并回滚。
            return namespace['DraftProposal'](3, query, torch.nn.functional.one_hot(query[:, 1:], 8).float())

        result = namespace['generate_decoding_sample'](target_model=target,
            input_ids=torch.tensor([[0, 1, 2]]), max_new_tokens=32,
            max_proposal_tokens=7, temperature=0., stop_token_ids=None,
            init_context=lambda **kwargs: None, propose=propose, update=lambda *args: None)
        self.assertEqual(result.output_ids[0, 3:].tolist(), [(3+i) % 8 for i in range(32)])
        self.assertTrue(all(n == 1 for n in result.accepted_draft_lengths))
        self.assertEqual(target.current_cache.tokens, result.output_ids[0, :len(target.current_cache.tokens)].tolist())
        target.release_cache()
        self.assertTrue(target.last_cache_released)


if __name__ == '__main__':
    unittest.main()
