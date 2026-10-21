"""Check divergence accounting and prediction-row indexing on small CPU fixtures."""
import copy
import unittest
import json
from pathlib import Path
import tempfile

from analyze_divergence import analyze, first_difference, prediction_evidence, verification_site
from prepare_fp16_regression import prepare, CASES
from protocol import json_hash, sha256


def fixture():
    records = []
    for algorithm, mode in [('a', 'spec_a'), ('b', 'spec_b')]:
        for repeat in range(3):
            for case, output in [('pass', [1, 2, 3, 4]), ('fail', [1, 2, 9, 4])]:
                for name, ids in [('ar', [1, 2, 3, 4]), (mode, output)]:
                    records.append(dict(algorithm=algorithm, mode=name, case=case,
                        repeat=repeat, input_tokens=10, output_tokens=4, output_ids=ids,
                        proposal_lengths=[3], accepted_draft_lengths=[3]))
    return records


class DivergenceTests(unittest.TestCase):
    def test_prefix_statistics_and_pairwise_output_equality(self):
        report = analyze(fixture())
        for summary in report['summaries']:
            self.assertEqual(summary['matching_executions'], 3)
            self.assertEqual(summary['matching_unique_all_repeats'], 1)
            self.assertEqual(summary['common_prefix_mean'], 3)
            self.assertEqual(summary['failed_common_prefix_mean'], 2)
        self.assertEqual(report['pairwise'][0]['identical_outputs'], 6)
        self.assertEqual(report['pairwise'][0]['same_first_fork'], 3)
        self.assertTrue(report['ar_cross_process_identical'])

    def test_missing_pair_and_duplicates_fail(self):
        rows = fixture()
        with self.assertRaises(KeyError):
            analyze(rows[:-1])
        with self.assertRaises(ValueError):
            analyze(rows + [rows[0]])

    def test_repeats_are_retained_not_assumed_stable(self):
        rows = copy.deepcopy(fixture())
        rows[-1]['output_ids'] = [1, 8, 3, 4]
        report = analyze(rows)
        group = next(g for g in report['by_unique_request'] if g['algorithm'] == 'b' and g['case'] == 'fail')
        self.assertEqual(group['first_mismatch_by_repeat'], [2, 2, 1])
        self.assertFalse(group['speculative_output_stable'])

    def test_pending_prediction_row_and_truncated_final_output(self):
        row = dict(input_tokens=100, proposal_lengths=[7, 7], accepted_draft_lengths=[2, 4])
        self.assertEqual(verification_site(row, 0), {'phase': 'prefill'})
        first = verification_site(row, 3)
        self.assertEqual(first['target_prediction_row'], 2)
        self.assertEqual(first['output_origin'], 'target_correction_or_bonus')
        second = verification_site(row, 4)
        self.assertEqual(second['cache_length_before'], 103)
        self.assertEqual(second['target_prediction_row'], 0)
        self.assertEqual(second['round'], 2)
        self.assertEqual(first_difference([1, 2], [1]), 1)
        self.assertIsNone(first_difference([1, 2], [1, 2]))

    def test_subset_preserves_original_model_lock_and_detects_changed_workload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'results/speculative-formal-fixture/prepared'
            source.mkdir(parents=True)
            rows = [dict(case=name, input_ids=[i], input_tokens=1, input_ids_sha256=json_hash([i]))
                    for i, name in enumerate(CASES)]
            canonical = root / 'canonical.json'
            canonical.write_text(json.dumps(rows))
            (source / 'workload.json').write_text(canonical.read_text())
            lock = dict(models={'target': {'path': '/original/local/model', 'sha256': 'unchanged'}},
                        seed=19, source_sha256='unchanged', workload_sha256=sha256(canonical))
            (source / 'lock.json').write_text(json.dumps(lock))
            prepare(root, root / 'selected', canonical)
            result = json.loads((root / 'selected/lock.json').read_text())
            self.assertEqual(result['models'], lock['models'])
            self.assertEqual(result['source_sha256'], lock['source_sha256'])
            self.assertEqual(result['workload_sha256'], sha256(root / 'selected/workload.json'))
            canonical.write_text('[]')
            with self.assertRaises(ValueError):
                prepare(root, root / 'must-not-select', canonical)

    def test_top2_uses_real_argmax_and_correct_verification_row(self):
        record = dict(mode='spec', input_tokens=10, proposal_lengths=[3],
            accepted_draft_lengths=[3], target_prediction_trace=[
                dict(cache_length_before=0, query_ids=[0]*10, argmax=[1],
                     top_ids=[[1, 2]], top_logits=[[3., 2.]]),
                dict(cache_length_before=10, query_ids=[1, 2, 3, 4],
                     argmax=[2, 3, 4, 5], top_ids=[[2, 1], [9, 3], [4, 5], [5, 4]],
                     top_logits=[[3., 2.], [3., 3.], [3., 2.], [3., 2.]])])
        observed = prediction_evidence(record, 2, verification_site(record, 2))
        self.assertEqual(observed['target_call'], 1)
        self.assertEqual(observed['prediction_row'], 1)
        self.assertEqual(observed['top2_margin'], 0)
        self.assertEqual(observed['argmax'], 3)
        self.assertEqual(observed['top_ids'][0], 9)
        self.assertIsNone(prediction_evidence(dict(mode='ar'), 2))


if __name__ == '__main__':
    unittest.main()
