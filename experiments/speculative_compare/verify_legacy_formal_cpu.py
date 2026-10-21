"""Check formal-run preparation against the archived passed GPU regression."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from prepare_legacy_formal import COMPUTE_FILES, prepare


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class FormalPreparationTests(unittest.TestCase):
    def fixture(self, root):
        previous = root/'previous'
        measured = previous/'measured'
        shutil.copytree(ROOT/'results/speculative_compare/legacy-target-regression-20261021', measured)
        harness = root/'harness'
        harness.mkdir()
        source = measured/'harness-source'
        source.mkdir()
        for name in COMPUTE_FILES:
            shutil.copy2(HERE/name, harness/name)
            shutil.copy2(HERE/name, source/name)
        workload = ROOT/'results/speculative_compare/formal-20261019/workload.json'
        for folder in (harness, source):
            shutil.copy2(workload, folder/'formal-workload.json')
        return previous, harness

    def test_reuses_model_source_and_original_twelve_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous, harness = self.fixture(root)
            prepare(previous, root/'prepared', harness)
            before = json.loads((previous/'measured/lock.json').read_text())
            after = json.loads((root/'prepared/lock.json').read_text())
            for field in ('models', 'source', 'source_sha256', 'seed'):
                self.assertEqual(after[field], before[field])
            self.assertEqual(len(json.loads((root/'prepared/workload.json').read_text())), 12)
            self.assertEqual((root/'prepared/workload.json').read_bytes(), (harness/'formal-workload.json').read_bytes())

    def test_rejects_failed_regression_or_changed_compute_or_divergent_tokens(self):
        for failure in ('report', 'source', 'tokens'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                previous, harness = self.fixture(root)
                measured = previous/'measured'
                if failure == 'report':
                    path = measured/'report.json'
                    value = json.loads(path.read_text())
                    value['qualified_greedy_comparison'] = False
                    path.write_text(json.dumps(value))
                elif failure == 'source':
                    (harness/'legacy_target.py').write_text('changed')
                else:
                    path = measured/'dspark-records.jsonl'
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[-1]['output_ids'][24] += 1
                    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with self.assertRaises(ValueError):
                    prepare(previous, root/'must-not-start', harness)
                self.assertFalse((root/'must-not-start').exists())


if __name__ == '__main__':
    unittest.main()
