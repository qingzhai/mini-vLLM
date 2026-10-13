#!/usr/bin/env python3
"""CPU Torch checks for the actual KV packing code and TP group adapter."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
import target


class TensorTests(unittest.TestCase):
    def test_actual_kv_pack_remap_partial_page(self):
        source = target.Target.__new__(target.Target)
        source.tp = 1
        source.pages = 4
        source.device = torch.device('cpu')
        source.hc = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=2, head_dim=4)
        # Keep values small so FP16 is finite even with a larger backing tensor.
        source.cache = (torch.arange(2*2*4*256*2*4) % 1000).reshape(2,2,4,256,2,4).half()
        dest = target.Target.__new__(target.Target)
        dest.tp, dest.pages, dest.device, dest.hc = source.tp, source.pages, source.device, source.hc
        dest.cache = torch.zeros_like(source.cache)
        for length in (1, 255, 256, 257, 511, 512):
            with self.subTest(length=length):
                packed = source.pack([3, 1], length)
                dest.unpack(packed, [0, 2], length)
                actual = dest.pack([0, 2], length)
                torch.testing.assert_close(actual, packed, rtol=0, atol=0)
                self.assertEqual(target.tensor_sha(actual), target.tensor_sha(packed))

    def test_group_adapter_uses_replica_group(self):
        replica = object()
        calls = []
        class FakeDist:
            def get_world_size(self, group):
                calls.append(('size', group))
                return 1
            def get_rank(self, group):
                calls.append(('rank', group))
                return 0
            def all_reduce(self, tensor, op, group, async_op):
                calls.append(('reduce', group))
            def all_gather(self, outputs, tensor, group, async_op):
                calls.append(('gather', group))
        with patch.object(target, 'dist', FakeDist()):
            adapter = target.ModelGroup(replica)
            self.assertEqual(adapter.get_world_size(), 1)
            self.assertEqual(adapter.get_rank(), 0)
            adapter.all_reduce(None)
            adapter.all_gather([], None)
        self.assertEqual([c[0] for c in calls], ['size', 'rank', 'reduce', 'gather'])
        self.assertTrue(all(group is replica for _, group in calls))


if __name__ == '__main__':
    unittest.main(verbosity=2)
