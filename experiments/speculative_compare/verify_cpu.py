"""Check comparison accounting and provenance without loading models."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from io import StringIO

from protocol import (MODES, first_mismatch, json_hash, model_fingerprint,
                      select_rows, summarize)
from benchmark import collect_report
from collect_audit import EVIDENCE_DIRS, collect


def records():
    rows = []
    for algorithm, modes in MODES.items():
        for mode in modes:
            for repeat in range(3):
                for case, seconds in (('short', 1.0), ('long', 3.0)):
                    rows.append(dict(algorithm=algorithm, mode=mode, repeat=repeat,
                        case=case, output_ids=[3, 4], output_tokens=2,
                        total_s=seconds if mode == 'ar' else seconds / 2,
                        peak_allocated_gib=20, verify_count=1,
                        proposal_lengths=[] if mode == 'ar' else [7],
                        accepted_draft_lengths=[] if mode == 'ar' else [1]))
    return rows


class ProtocolTests(unittest.TestCase):
    def report(self, rows):
        return summarize(rows, cases=['short', 'long'], repeats=3)

    def test_paired_aggregate_throughput(self):
        for row in self.report(records()):
            expected = 1 if row['mode'] == 'ar' else 2
            self.assertEqual(row['median_tokens_per_s'], expected)
            self.assertEqual(row['median_paired_speedup'], expected)
            self.assertTrue(row['paired_greedy_match'])

    def test_missing_duplicate_and_changed_tokens_never_pass(self):
        original = records()
        variants = [original[:-1], original + [copy.deepcopy(original[-1])]]
        changed = copy.deepcopy(original)
        changed[-1]['output_ids'] = [3, 5]
        variants.append(changed)
        for rows in variants:
            with self.subTest(length=len(rows)):
                self.assertFalse(self.report(rows)[-1]['paired_greedy_match'])

    def test_length_mismatch_is_not_a_matching_prefix(self):
        self.assertEqual(first_mismatch([1, 2], [1]), 1)
        self.assertEqual(first_mismatch([1], [1, 2]), 1)
        self.assertIsNone(first_mismatch([1], [1]))

    def test_selection_is_stable_without_replacement(self):
        left = select_rows(list(range(100)), count=12, seed=19)
        self.assertEqual(left, select_rows(list(range(100)), count=12, seed=19))
        self.assertEqual(len(set(left)), 12)
        with self.assertRaises(ValueError):
            select_rows([1], count=2, seed=0)

    def test_weights_and_token_order_have_content_fingerprints(self):
        self.assertNotEqual(json_hash([1, 2]), json_hash([2, 1]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'config.json').write_text('{}')
            with self.assertRaises(ValueError):
                model_fingerprint(path)
            (path / 'model.safetensors').write_bytes(b'original')
            before = model_fingerprint(path)
            (path / 'model.safetensors').write_bytes(b'modified')
            self.assertNotEqual(before, model_fingerprint(path))

    def test_paired_success_cannot_hide_cross_process_ar_divergence(self):
        rows = records()
        for row in rows:
            if row['algorithm'] == 'dflash':
                row['output_ids'] = [3, 5]
        self.assertTrue(all(x['paired_greedy_match'] for x in self.report(rows)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'fixture-records.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
            report = collect_report(path, [{'case': 'short'}, {'case': 'long'}], 3, [])
            self.assertFalse(report['qualified_greedy_comparison'])
            self.assertTrue(report['ar_divergences'])

    def test_archived_smoke_preserves_failure_and_file_fingerprints(self):
        folder = Path(__file__).resolve().parents[2] / 'results/speculative_compare/smoke-20261019'
        sources = json.loads((folder / 'SOURCES.json').read_text())
        for name, expected in sources['sha256'].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), expected)
        report = json.loads((folder / 'report.json').read_text())
        self.assertFalse(report['qualified_greedy_comparison'])
        self.assertEqual(report['errors'], [])
        self.assertEqual(report['ar_divergences'], [])
        failing = [row for row in report['summary'] if not row['paired_greedy_match']]
        self.assertEqual([row['mode'] for row in failing], ['dspark_conf_0.8'])
        self.assertEqual(failing[0]['mismatches'], [dict(repeat=0, case='humaneval:19', first_mismatch=26)])

    def test_archived_markov_and_fixed_trajectories_really_match(self):
        folder = Path(__file__).resolve().parents[2] / 'results/speculative_compare/smoke-20261019'
        rows = [json.loads(line) for line in (folder / 'dspark-records.jsonl').read_text().splitlines()]
        for case in {row['case'] for row in rows}:
            left = next(row for row in rows if row['case'] == case and row['mode'] == 'dspark_markov_only')
            right = next(row for row in rows if row['case'] == case and row['mode'] == 'dspark_fixed')
            for key in ('output_ids', 'proposal_lengths', 'accepted_draft_lengths'):
                self.assertEqual(left[key], right[key])

    def test_common_precision_is_required_in_fp32_comparison(self):
        rows = records()
        for row in rows:
            row['target_head'] = 'fp32'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'fixture-records.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
            (path / 'run-config.json').write_text(json.dumps(dict(target_head='fp32', full_bf16_reduction=False)))
            actual = dict(requested='fp32', head_weight_dtype='torch.float32',
                head_compute_dtype='torch.float32', head_output_dtype='torch.float32',
                backbone_dtype='torch.bfloat16', tf32=False, bf16_reduced_precision_reduction=True)
            for algorithm in MODES:
                (path / f'{algorithm}-environment.json').write_text(json.dumps(dict(target_precision=actual)))
            report = collect_report(path, [{'case': 'short'}, {'case': 'long'}], 3, [])
            self.assertTrue(report['qualified_greedy_comparison'])
            changed = dict(actual, head_compute_dtype='torch.bfloat16')
            (path / 'dspark-environment.json').write_text(json.dumps(dict(target_precision=changed)))
            report = collect_report(path, [{'case': 'short'}, {'case': 'long'}], 3, [])
            self.assertFalse(report['qualified_greedy_comparison'])
            self.assertTrue(report['precision_errors'])

    def test_fp16_protocol_rejects_bf16_backbone_and_missing_bridge(self):
        rows = records()
        for row in rows:
            row.update(target_head='fp32', target_backbone='fp16')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'fixture-records.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
            (path / 'run-config.json').write_text(json.dumps(dict(
                target_head='fp32', target_backbone='fp16', full_fp16_reduction=True)))
            actual = dict(requested='fp32', head_weight_dtype='torch.float32',
                head_compute_dtype='torch.float32', head_output_dtype='torch.float32',
                backbone_dtype='torch.float16', tf32=False,
                bf16_reduced_precision_reduction=True, fp16_reduced_precision_reduction=False)
            for algorithm in MODES:
                (path / f'{algorithm}-environment.json').write_text(json.dumps(dict(
                    target_precision=actual, draft_dtype='torch.bfloat16',
                    draft_feature_bridge={'boundary': 'draft.fc input'})))
            self.assertTrue(collect_report(path, [{'case': 'short'}, {'case': 'long'}], 3, [])['qualified_greedy_comparison'])
            changed = dict(actual, backbone_dtype='torch.bfloat16')
            (path / 'dflash-environment.json').write_text(json.dumps(dict(target_precision=changed)))
            failed = collect_report(path, [{'case': 'short'}, {'case': 'long'}], 3, [])
            self.assertFalse(failed['qualified_greedy_comparison'])
            self.assertTrue(any('not FP16' in e for e in failed['precision_errors']))

    def test_archived_diagnostic_preserves_observed_tie_and_exact_replay(self):
        folder = Path(__file__).resolve().parents[2] / 'results/speculative_compare/diagnostic-20261019'
        source = json.loads((folder / 'SOURCES.json').read_text())
        for name in source['included_in_repository']:
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), source['sha256'][name])
        report = json.loads((folder / 'report.json').read_text())
        self.assertFalse(report['performance_measurement'])
        self.assertTrue(report['reproduced_historical_failure'])
        self.assertTrue(report['trace_matches_untraced'])
        self.assertTrue(report['focus_history_matches_ar'])
        self.assertEqual(report['focus_round']['first_prediction']['argmax'], 429)
        self.assertEqual(report['live_same_hidden_fp32_head']['argmax'], 5980)
        replay = report['cache_comparisons']['live_vs_speculative_replay']
        self.assertEqual(len(replay), 36)
        self.assertTrue(all(row['keys']['exact'] and row['values']['exact'] for row in replay))


class AuditCollectionTests(unittest.TestCase):
    def test_collects_actual_bytes_without_weights_or_external_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in EVIDENCE_DIRS:
                (root / name).mkdir(parents=True)
            runtime = root / EVIDENCE_DIRS[0]
            source = runtime / 'prepare.py'
            source.write_text('# server version with local drafts\n')
            weights = runtime / 'unexpected.safetensors'
            weights.write_bytes(b'weights-must-not-be-archived')
            (runtime / 'outside.py').symlink_to('/etc/passwd')
            for model in ('Qwen3-8B', 'eagle3', 'dflash', 'dspark'):
                directory = root / 'models' / model
                directory.mkdir(parents=True)
                (directory / 'config.json').write_text('{}')
                (directory / 'model.safetensors').write_bytes(b'not-read')
            output = root / 'audit.zip'
            old_workload = root / 'data/sharegpt-workload.json'
            old_workload.parent.mkdir()
            old_workload.write_text(json.dumps({
                'workload_label': 'sharegpt-with-20pct-replayed-prompts',
                'requests': [{'prompt_token_ids': [1, 2]}],
            }))
            with redirect_stdout(StringIO()):
                info = collect(root, output, inventory=False)
            self.assertFalse(info['gpu_benchmark_run'])
            self.assertFalse(info['model_weights_rehashed'])
            self.assertEqual(info['missing'], [])
            self.assertEqual(len(info['skipped']), 2)
            self.assertEqual(info['sharegpt_workloads'], ['data/sharegpt-workload.json'])
            with zipfile.ZipFile(output) as archive:
                key = str(source.relative_to(root))
                data = archive.read('work/' + key)
                self.assertEqual(data, source.read_bytes())
                self.assertEqual(hashlib.sha256(data).hexdigest(), info['files'][key]['sha256'])
                self.assertFalse(any(name.endswith('.safetensors') for name in archive.namelist()))
                self.assertNotIn('work/' + EVIDENCE_DIRS[0] + '/outside.py', archive.namelist())
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                collect(root, output, inventory=False)
            self.assertEqual(output.read_bytes(), before)

    def test_missing_evidence_and_size_caps_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / EVIDENCE_DIRS[0]
            directory.mkdir()
            (directory / 'large.log').write_bytes(b'x' * 20)
            (directory / 'one.log').write_bytes(b'a' * 8)
            (directory / 'two.log').write_bytes(b'b' * 8)
            with redirect_stdout(StringIO()):
                info = collect(root, root / 'audit.zip', inventory=False,
                               max_file_bytes=10, max_total_bytes=10)
            self.assertIn(EVIDENCE_DIRS[-1], info['missing'])
            self.assertEqual(len(info['files']), 1)
            self.assertEqual({row['reason'] for row in info['skipped']},
                             {'weight or per-file limit', 'total size limit'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
