#!/usr/bin/env python3
"""CPU mechanism tests; no model weights, downloads or GPU claims."""
from __future__ import annotations
import json
from pathlib import Path
import time
import unittest
from common import PagePool, build_cases, compare_tokens, digest_json, metrics, warmup_case
from runtime import run_load


class FakeSession:
    """Deterministic autoregression and physical KV slots with paired transfer."""
    def __init__(self, mode):
        self.mode = mode
        self.clients = {r: None for r in ([0, 1] if mode in ('dp2', 'pd') else [0])}
        self.pending = {}
        self.kv = {r: {} for r in self.clients}
        self.counts = {r: dict(prefill_calls=0, decode_calls=0, transfer_send_bytes=0,
                              transfer_recv_bytes=0) for r in self.clients}

    def submit(self, rank, command):
        assert rank not in self.pending
        self.pending[rank] = command

    def collect(self, timeout=0):
        results = []
        # A transfer cannot complete unless BOTH endpoints have been posted.
        if {c['op'] for c in self.pending.values()} == {'send_kv', 'recv_kv'}:
            send, recv = self.pending[0], self.pending[1]
            length = send['length']
            assert length == recv['length']
            values = [self.kv[0][send['block_table'][i//256]*256+i%256] for i in range(length)]
            for i, value in enumerate(values):
                self.kv[1][recv['block_table'][i//256]*256+i%256] = value
            nbytes = 2*36*length*8*128*2
            sha = digest_json(values)
            for rank in (0, 1):
                command = self.pending.pop(rank)
                self.counts[rank]['transfer_send_bytes' if rank == 0 else 'transfer_recv_bytes'] += nbytes
                now = time.monotonic()
                message = dict(type='result', op=command['op'], worker_s=0., bytes=nbytes,
                               worker_begin_monotonic_s=now, worker_end_monotonic_s=now,
                               transport_wait_s=0., pack_s=0., unpack_s=0.)
                if command['diagnostic']:
                    message.update(sha256=sha, mapped_sha256=sha)
                results.append((rank, message))
            return results
        for rank in list(self.pending):
            command = self.pending[rank]
            if command['op'] != 'step':
                continue
            self.pending.pop(rank)
            tokens = []
            before = time.monotonic()
            for seq in command['seqs']:
                ids, pages = seq['token_ids'], seq['block_table']
                def slot(i):
                    return pages[i//256]*256+i%256
                if command['prefill']:
                    for i, value in enumerate(ids):
                        self.kv[rank][slot(i)] = value
                else:
                    # Detect a missing/incorrect KV migration or token boundary.
                    for i, value in enumerate(ids[:-1]):
                        assert self.kv[rank][slot(i)] == value
                    self.kv[rank][slot(len(ids)-1)] = ids[-1]
                tokens.append((sum(ids)+len(ids)*17) % 1000)
            self.counts[rank]['prefill_calls' if command['prefill'] else 'decode_calls'] += 1
            now = time.monotonic()
            results.append((rank, dict(type='result', op='step', tokens=tokens, worker_s=now-before,
                                      worker_begin_monotonic_s=before, worker_end_monotonic_s=now)))
        return results

    def stats(self):
        ranks = [dict(rank=r, peak_allocated_gib=0., counters=c.copy()) for r, c in self.counts.items()]
        if self.mode == 'tp2':
            ranks.append(dict(rank=1, peak_allocated_gib=0., counters=self.counts[0].copy()))
        return ranks


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fixtures import synthetic_workload
        cls.workload = synthetic_workload()
        cls.config = dict(global_batch=4, logical_kv_pages=64)

    def test_physical_pages_and_double_release(self):
        pool = PagePool(4, reverse=True)
        self.assertEqual(pool.allocate('a', 2), [3, 2])
        with self.assertRaises(RuntimeError):
            pool.allocate('b', 3)
        pool.release('a')
        self.assertTrue(pool.clean())
        with self.assertRaises(RuntimeError):
            pool.release('a')

    def test_metrics_include_queue_and_itl(self):
        r = dict(arrival_s=1., max_new_tokens=3, output_token_ids=[4,5,6], token_times_s=[3.,4.,7.])
        m = metrics([r], 8., 2)
        self.assertEqual(m['avg_ttft_s'], 2.)
        self.assertEqual(m['avg_e2e_s'], 6.)
        self.assertEqual(m['avg_tpot_ms'], 2000.)
        self.assertEqual(m['output_tokens_per_gpu_s'], 3/16)
        self.assertEqual(m['measured_itl_samples'], 2)

    def test_scheduler_and_handoff_across_all_modes(self):
        cases = build_cases(self.workload, 8, 8, 4, 0.00001)
        for case in [warmup_case(self.workload), *cases]:
            baseline = run_load(FakeSession('single'), case, self.config)
            for mode in ('tp2', 'dp2', 'pd'):
                with self.subTest(case=case['case'], mode=mode):
                    candidate = run_load(FakeSession(mode), case, self.config, diagnostic=(mode == 'pd'))
                    self.assertTrue(compare_tokens(baseline['requests'], candidate['requests'])['all_exact'])
                    self.assertTrue(candidate['all_pages_released'])
                    self.assertLessEqual(candidate['peak_active_requests'], 4)
                    if mode == 'pd':
                        self.assertEqual(candidate['pd_contract']['decode_prefill_calls'], 0)
                        self.assertEqual(len(candidate['transfers']), len(case['requests']))

    def test_token_comparison_detects_divergence(self):
        reference = [dict(request_id='x', prompt_token_ids=[1], max_new_tokens=2, output_token_ids=[2,3])]
        candidate = [dict(reference[0], output_token_ids=[2,4])]
        check = compare_tokens(reference, candidate)
        self.assertFalse(check['all_exact'])
        self.assertEqual(check['details'][0]['first_mismatch'], 1)

    def test_workload_preserves_source_tokens(self):
        cases = build_cases(self.workload)
        source = {r['request_id']: r for r in self.workload['requests']}
        for case in cases:
            for row in case['requests']:
                self.assertEqual(row['prompt_token_ids'], source[row['source_request_id']]['prompt_token_ids'])
        self.assertEqual(len(cases[0]['requests']), 32)
        self.assertEqual(len(cases[1]['requests']), 16)
        self.assertEqual(cases[1]['requests'][2]['bucket'], 'long')


if __name__ == '__main__':
    unittest.main(verbosity=2)
