"""Exercise real scheduling/cache state and report guards without Torch or GPU."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/common'))
from fixtures import synthetic_workload
from verify_cpu import load_actual
from factorial_protocol import MODES, PROTOCOL, paired_effects, settings, trial_order
from summarize import METRICS, write_summary
from workload_support import load_workload, tokenizer_metadata_diff
from prepare_data import request_sha256, token_sha256

Scheduler, _, Sequence, Params = load_actual(ROOT)
from factorial_scheduler import FactorialScheduler


def make_scheduler(mode):
    scheduler = FactorialScheduler(Scheduler(8, 2048, 330, 256, 999999))
    cfg = settings(mode)
    scheduler.chunk_enabled = cfg['chunk_enabled']
    scheduler.chunk_size = cfg['chunk_size']
    scheduler.block_manager.enable_prefix_caching = cfg['prefix_cache']
    return scheduler


def finish_step(scheduler, selected):
    ready = []
    for seq in selected:
        end = seq._scheduled_end
        seq.num_computed_tokens = end
        scheduler.block_manager.mark_computed(seq, computed_tokens=end)
        if not seq._scheduled_prefill or end == len(seq):
            ready.append(seq)
    scheduler.postprocess(ready, [17]*len(ready))


def report_fixture():
    trials = []
    rates = dict(zip(MODES, (100, 80, 110, 120)))
    for repeat in range(1, 4):
        for mode in trial_order(repeat):
            row = {key: 1.0 for key in METRICS}
            row.update(mode=mode, requests=100, output_tokens=6400,
                       output_tokens_per_s=rates[mode], total_s=6400/rates[mode],
                       all_pages_released=True, finite_logits=True, timing_clean=True,
                       mode_config=settings(mode))
            trials.append(dict(repeat=repeat, summary=row))
    return dict(protocol=PROTOCOL, completed=True, passed=True, repeats=3,
                modes={mode: settings(mode) for mode in MODES}, measured_trials=trials,
                all_exact_greedy=True, all_timings_clean=True, baseline_repeat_comparisons=[],
                greedy_comparisons=[dict(repeat=r, mode=m, exact_matches=100, all_exact=True)
                                    for r in range(1, 4) for m in MODES[1:]])


class FactorialContracts(unittest.TestCase):
    def test_tokenizer_provenance_difference_requires_exact_inputs_and_config(self):
        # A temporary synthetic historical record tests guards, never GPU performance.
        obj = synthetic_workload()
        obj.pop('fixture_is_synthetic')
        selected = copy.deepcopy(obj)
        for row in selected['requests']:
            row.pop('prompt_token_ids')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'data').mkdir()
            (root/'experiments/sharegpt').mkdir(parents=True)
            shutil.copy2(ROOT/'experiments/sharegpt/prepare_data.py', root/'experiments/sharegpt/prepare_data.py')
            (root/'data/sharegpt-selection.json').write_text(json.dumps(selected))
            source = root/'workload.json'
            obj['model']['tokenizer_files_sha256']['tokenizer_config.json'] = 'a'*64
            obj['model']['tokenizer_files_sha256']['tokenizer.json.metadata'] = 'b'*64
            source.write_text(json.dumps(obj))
            with patch('workload_support.REPO', root):
                with self.assertRaisesRegex(ValueError, 'tokenizer_files_sha256'):
                    load_workload(source)
                loaded = load_workload(source, require_tokenizer_file_identity=False)
                self.assertEqual(loaded, obj)
                differences = tokenizer_metadata_diff(loaded)
                self.assertEqual(set(differences), {'tokenizer_config.json', 'tokenizer.json.metadata'})
                self.assertIsNone(differences['tokenizer.json.metadata']['historical'])
                obj['model']['config_sha256'] = 'c'*64
                source.write_text(json.dumps(obj))
                with self.assertRaisesRegex(ValueError, 'config_sha256'):
                    load_workload(source, require_tokenizer_file_identity=False)
                obj['model']['config_sha256'] = selected['model']['config_sha256']
                # Change a non-replayed prompt and recompute internally consistent hashes.
                replayed = {row['repeated_from'] for row in obj['requests'][80:]}
                row = next(row for row in obj['requests'][:80] if row['request_id'] not in replayed)
                row['prompt_token_ids'][1] += 1
                row['prompt_sha256'] = token_sha256(row['prompt_token_ids'])
                obj['workload_sha256'] = request_sha256(obj['requests'])
                source.write_text(json.dumps(obj))
                with self.assertRaisesRegex(ValueError, 'historical workload'):
                    load_workload(source, require_tokenizer_file_identity=False)
                obj['fixture_is_synthetic'] = True
                source.write_text(json.dumps(obj))
                with self.assertRaisesRegex(ValueError, 'CPU fixtures'):
                    load_workload(source, require_tokenizer_file_identity=False)

    def test_switches_are_independent_and_priority_is_constant(self):
        matrix = {(settings(m)['chunk_enabled'], settings(m)['prefix_cache']) for m in MODES}
        self.assertEqual(matrix, {(False, False), (False, True), (True, False), (True, True)})
        self.assertTrue(all(settings(m)['decode_first'] for m in MODES))
        self.assertEqual(len({trial_order(r)[0] for r in range(1, 5)}), 4)

    def test_atomic_prefill_defers_instead_of_using_spare_budget(self):
        scheduler = make_scheduler('full_prefill')
        seqs = [Sequence([i+100]*length, 256, Params(max_tokens=4, ignore_eos=True))
                for i, length in enumerate((1536, 700))]
        for seq in seqs:
            scheduler.add_sequence(seq)
        selected, _ = scheduler.schedule()
        self.assertEqual(selected, [seqs[0]])
        self.assertEqual(scheduler.last_schedule['deferred_atomic_prefills'], 1)
        self.assertEqual(seqs[1].num_computed_tokens, 0)
        finish_step(scheduler, selected)
        selected, _ = scheduler.schedule()
        self.assertEqual(selected, [seqs[0], seqs[1]])
        self.assertEqual(seqs[1]._scheduled_end, 700)
        self.assertTrue(scheduler.last_schedule['mixed'])

    def test_impossible_atomic_budget_is_rejected(self):
        scheduler = make_scheduler('full_prefill')
        scheduler.max_num_batched_tokens = 1000
        scheduler.add_sequence(Sequence([20]*1000, 256, Params(max_tokens=4, ignore_eos=True)))
        with self.assertRaisesRegex(ValueError, 'atomic Prefill'):
            scheduler.schedule()

    def test_all_four_modes_complete_100_requests_and_release_pages(self):
        requests = synthetic_workload()['requests']
        for mode in MODES:
            with self.subTest(mode=mode):
                scheduler = make_scheduler(mode)
                seqs = [Sequence(r['prompt_token_ids'], 256, Params(max_tokens=64, ignore_eos=True,
                                 max_model_length=2048)) for r in requests]
                for seq in seqs:
                    scheduler.add_sequence(seq)
                partial = mixed = steps = 0
                while not scheduler.is_finished():
                    selected, _ = scheduler.schedule()
                    steps += 1
                    self.assertLess(steps, 10000)
                    self.assertLessEqual(len(scheduler.running), 8)
                    self.assertLessEqual(scheduler.last_schedule['tokens'], 2048)
                    mixed += int(scheduler.last_schedule['mixed'])
                    for seq in selected:
                        if seq._scheduled_prefill:
                            if settings(mode)['chunk_enabled']:
                                self.assertLessEqual(seq._scheduled_end-seq._scheduled_start, 256)
                            else:
                                self.assertEqual(seq._scheduled_end, seq.num_prompt_tokens)
                            partial += int(seq._scheduled_end < seq.num_prompt_tokens)
                    finish_step(scheduler, selected)
                self.assertTrue(all(len(seq.completion_token_ids) == 64 for seq in seqs))
                self.assertEqual(len(scheduler.block_manager.free_block_ids), 330)
                self.assertFalse(scheduler.block_manager.used_block_ids)
                self.assertEqual(partial > 0, settings(mode)['chunk_enabled'])
                self.assertGreater(mixed, 0)
                expected_hits = (sum(((r['prompt_tokens']-1)//256)*256 for r in requests[80:])
                                 if settings(mode)['prefix_cache'] else 0)
                self.assertEqual(scheduler.block_manager.stats['cache_hit_tokens'], expected_hits)
                print('CPU synthetic scheduling only:', mode, 'steps', steps, 'hits', expected_hits)

    def test_missing_duplicate_or_confounded_trials_are_rejected(self):
        report = report_fixture()
        with self.assertRaises(ValueError):
            paired_effects(report['measured_trials'][:-1], 3)
        with self.assertRaises(ValueError):
            paired_effects(report['measured_trials']+[report['measured_trials'][0]], 3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            report['measured_trials'][0]['summary']['mode_config']['decode_first'] = False
            (path/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'Executed mode'):
                write_summary(path)

    def test_paired_report_and_failed_qualification_are_separate(self):
        report = report_fixture()
        effects = paired_effects(report['measured_trials'], 3)
        self.assertAlmostEqual(effects[0]['median_paired_throughput_ratio'], 1.1)
        self.assertAlmostEqual(effects[1]['median_paired_throughput_ratio'], 1.5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'report.json').write_text(json.dumps(report))
            with redirect_stdout(io.StringIO()):
                rows = write_summary(path)
            self.assertEqual(len(rows), 4)
            data = json.loads((path/'factorial-effects.json').read_text())
            self.assertTrue(data['qualified_comparison'])
            report['passed'] = report['all_exact_greedy'] = False
            report['greedy_comparisons'][0]['all_exact'] = False
            (path/'report.json').write_text(json.dumps(report))
            with redirect_stdout(io.StringIO()):
                write_summary(path)
            data = json.loads((path/'factorial-effects.json').read_text())
            self.assertFalse(data['qualified_comparison'])
            self.assertEqual(len(data['effects']), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
