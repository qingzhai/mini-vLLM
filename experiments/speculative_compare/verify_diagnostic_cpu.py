"""CPU tensor checks for diagnostic accounting; does not validate Qwen GPU math."""
import copy
import unittest
from types import SimpleNamespace

import torch

from diagnose_greedy import (commit_history, compare_caches, logit_details,
                             require_cache_length, tensor_difference)
from precision import configure_target_backbone, configure_target_head, install_draft_feature_bridge


class Layer:
    def __init__(self, keys, values):
        self.keys, self.values = keys, values

    def get_seq_length(self):
        return self.keys.shape[-2]


class DiagnosticTests(unittest.TestCase):
    def cache(self):
        return SimpleNamespace(layers=[Layer(torch.arange(20).reshape(1, 1, 5, 4),
                                                torch.ones(1, 1, 5, 4)) for _ in range(2)])

    def test_prefix_commit_retains_pending_and_only_accepted_candidates(self):
        self.assertEqual(commit_history([10, 11], [12, 13, 14, 99], 2), [10, 11, 12, 13, 14])
        self.assertEqual(commit_history([10], [12, 99], 0), [10, 12])
        with self.assertRaises(ValueError):
            commit_history([10], [12, 99], 2)

    def test_argmax_and_all_ties_are_separate_from_topk_order(self):
        row = logit_details(torch.tensor([0, 1, 31.625, 2, 31.625], dtype=torch.bfloat16), watch=(2, 4))
        self.assertEqual(row['argmax'], 2)
        self.assertEqual(row['tied_max_ids'], [2, 4])
        self.assertEqual(row['watched'], {'2': 31.625, '4': 31.625})

    def test_cache_invariants_check_every_layer(self):
        cache = self.cache()
        require_cache_length(cache, 5, 2)
        cache.layers[-1].values = cache.layers[-1].values[:, :, :3]
        # 键和值都必须对应完整有效历史。
        with self.assertRaises(ValueError):
            require_cache_length(cache, 5, 2)

    def test_independent_cache_clones_and_layer_differences(self):
        left = self.cache()
        right = copy.deepcopy(left)
        right.layers[1].keys[0, 0, 2, 1] += 2
        differences = compare_caches(left, right)
        self.assertTrue(differences[0]['keys']['exact'])
        self.assertFalse(differences[1]['keys']['exact'])
        self.assertEqual(differences[1]['keys']['max_abs'], 2)
        self.assertTrue(differences[1]['values']['exact'])
        self.assertFalse(tensor_difference(torch.zeros(2), torch.zeros(3))['shape_match'])

    def target(self):
        target = torch.nn.Module()
        target.config = SimpleNamespace(tie_word_embeddings=False)
        target.model = torch.nn.Embedding(2, 2, dtype=torch.bfloat16)
        target.lm_head = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            target.lm_head.weight.copy_(torch.tensor([[31.5, 10], [31.5, 12]], dtype=torch.bfloat16))
        return target.eval()

    def test_fp32_head_prevents_rounding_tie_without_changing_backbone(self):
        target = self.target()
        hidden = torch.tensor([[1, 1 / 128]], dtype=torch.bfloat16)
        self.assertEqual(target.lm_head(hidden).argmax().item(), 0)
        backbone_pointer = target.model.weight.data_ptr()
        original_weight = target.lm_head.weight.detach().clone()
        config = configure_target_head(target, 'fp32')
        result = target.lm_head(hidden)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.argmax().item(), 1)
        self.assertTrue(torch.equal(target.lm_head.weight, original_weight.float()))
        self.assertEqual(target.model.weight.dtype, torch.bfloat16)
        self.assertEqual(target.model.weight.data_ptr(), backbone_pointer)
        self.assertEqual(config['added_parameter_bytes'], original_weight.numel() * 2)
        self.assertFalse(config['checkpoint_files_modified'])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            self.assertEqual(target.lm_head(hidden).dtype, torch.float32)

    def test_checkpoint_control_is_unchanged_and_tied_heads_are_rejected(self):
        target = self.target()
        original = target.lm_head
        config = configure_target_head(target)
        self.assertIs(target.lm_head, original)
        self.assertEqual(config['added_parameter_bytes'], 0)
        target.config.tie_word_embeddings = True
        with self.assertRaises(ValueError):
            configure_target_head(target, 'fp32')

    def test_fp16_backbone_preserves_rope_buffer_and_bf16_draft_boundary(self):
        target = self.target()
        target.model.register_buffer('inv_freq', torch.tensor([0.123456789], dtype=torch.float32))
        original_buffer = target.model.inv_freq.clone()
        original_head = target.lm_head.weight.detach().clone()
        configure_target_backbone(target, 'fp16')
        self.assertEqual(target.model.weight.dtype, torch.float16)
        self.assertEqual(target.model.inv_freq.dtype, torch.float32)
        self.assertTrue(torch.equal(target.model.inv_freq, original_buffer))
        self.assertTrue(torch.equal(target.lm_head.weight, original_head))
        configure_target_head(target, 'fp32')
        draft = torch.nn.Module()
        draft.fc = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        draft_cache_dtype = draft.fc.weight.dtype
        install_draft_feature_bridge(draft)
        features = target.model(torch.tensor([0, 1]))
        self.assertEqual(features.dtype, torch.float16)
        expected = torch.nn.functional.linear(features.bfloat16(), draft.fc.weight)
        self.assertTrue(torch.equal(draft.fc(features), expected))
        self.assertTrue(torch.equal(draft.fc(input=features), expected))
        self.assertEqual(draft.fc.weight.dtype, draft_cache_dtype)
        self.assertEqual(target.lm_head(features).dtype, torch.float32)


if __name__ == '__main__':
    unittest.main(verbosity=2)
