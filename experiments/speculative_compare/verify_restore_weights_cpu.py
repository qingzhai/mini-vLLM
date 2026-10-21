"""Check original manifests, download integrity and conflict handling without network."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

from restore_model_weights import model_tasks, restore_file


class RestoreWeightsTests(unittest.TestCase):
    def task(self, folder):
        return dict(destination=folder/'model.safetensors', expected=hashlib.sha256(b'original').hexdigest(),
                    url='https://example.invalid/not-fetched', seed=folder/'missing', content=None)

    def test_original_draft_manifests_match_archived_hashes(self):
        lock = json.loads((Path(__file__).resolve().parents[2]/'results/speculative_compare/formal-20261019/lock.json').read_text())
        tasks = model_tasks(Path('/fixture'), lock)
        manifests = [t for t in tasks if t['content'] is not None]
        self.assertEqual(len(manifests), 3)
        for task in manifests:
            self.assertEqual(hashlib.sha256(task['content']).hexdigest(), task['expected'])
        lock['models']['target']['sha256']['../outside'] = 'unsafe'
        with self.assertRaises(ValueError):
            model_tasks(Path('/fixture'), lock)

    def test_accepts_only_original_bytes_and_reuses_without_download(self):
        with tempfile.TemporaryDirectory() as folder:
            task = self.task(Path(folder))
            calls = []
            def fetch(url, partial, stopped):
                calls.append(url)
                partial.write_bytes(b'original')
            restore_file(task, threading.Event(), fetch)
            restore_file(task, threading.Event(), fetch)
            self.assertEqual(len(calls), 1)
            self.assertEqual(task['destination'].read_bytes(), b'original')

    def test_changed_remote_file_is_never_promoted(self):
        with tempfile.TemporaryDirectory() as folder:
            task = self.task(Path(folder))
            def fetch(url, partial, stopped):
                partial.write_bytes(b'new version')
            with self.assertRaises(ValueError):
                restore_file(task, threading.Event(), fetch)
            self.assertFalse(task['destination'].exists())
            self.assertEqual((Path(folder)/'model.safetensors.download').read_bytes(), b'new version')

    def test_existing_mismatch_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            task = self.task(Path(folder))
            task['destination'].write_bytes(b'keep user file')
            with self.assertRaises(ValueError):
                restore_file(task, threading.Event())
            self.assertEqual(task['destination'].read_bytes(), b'keep user file')


if __name__ == '__main__':
    unittest.main()
