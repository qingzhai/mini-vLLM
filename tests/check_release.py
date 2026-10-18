"""Release migration contracts; uses neither GPU nor downloaded model weights."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import repro
from benchmark_support import load_proof
from fixtures import synthetic_workload
from workload_support import load_workload

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_original_core_and_shared_math_unchanged(self):
        records = json.loads((ROOT/'data/original-source-sha256.json').read_text())
        for name, expected in records.items():
            with self.subTest(source=name):
                original = repro.source_before_annotations(ROOT/name)
                self.assertEqual(hashlib.sha256(original).hexdigest(), expected)

    def test_reference_output_and_runtime_guards(self):
        selected = json.loads((ROOT/'data/sharegpt-selection.json').read_text())
        proof, rows = load_proof(ROOT/'data/sharegpt-reference', selected)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(proof['canonical_source_sha256']), 4)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            bad = {'completed': False}
            (p/'report.json').write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):
                load_proof(p, selected)

    def test_synthetic_inputs_rejected_by_formal_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'fixture.json'
            p.write_text(json.dumps(synthetic_workload()))
            with self.assertRaisesRegex(ValueError, 'CPU fixtures'):
                load_workload(p)

    def test_manifest_detects_changed_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'example.py').write_text('x = 1\n')
            digest = hashlib.sha256((root/'example.py').read_bytes()).hexdigest()
            (root/'source-manifest.json').write_text(json.dumps({'sha256': {'example.py': digest}}))
            with patch.object(repro, 'REPO', root):
                repro.verify_source()
                (root/'example.py').write_text('x = 2\n')
                with self.assertRaisesRegex(RuntimeError, 'changed'):
                    repro.verify_source()


if __name__ == '__main__':
    unittest.main(verbosity=2)
