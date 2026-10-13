#!/usr/bin/env python3
"""Exercise the real authenticated RPC Session with CPU-only subprocesses."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from runtime import Session

FAKE_WORKER = '''
import os
from multiprocessing.connection import Client, wait
host, port = os.environ['BENCH_CONTROL_ADDRESS'].split(':')
mode = os.environ['BENCH_MODE']
n = 2 if mode in ('dp2', 'pd') else 1
conns = {}
for rank in range(n):
    c = Client((host, int(port)), authkey=bytes.fromhex(os.environ['BENCH_CONTROL_KEY']))
    c.send(dict(type='ready', rank=rank, pages=32, stats={}, arithmetic={}, loading={}))
    conns[c] = rank
while conns:
    for c in wait(list(conns), timeout=1):
        command = c.recv()
        rank = conns[c]
        if command['op'] == 'close':
            c.send(dict(type='closed', rank=rank))
            del conns[c]
            c.close()
        elif command['op'] == 'fail':
            c.send(dict(type='error', rank=rank, traceback='injected CPU worker error'))
        else:
            c.send(dict(type='result', rank=rank, op=command['op'], worker_s=0., value=command.get('value')))
'''


class TransportTests(unittest.TestCase):
    def test_two_independent_clients_and_error_propagation(self):
        real_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root/'cpu_worker.py'
            fake.write_text(FAKE_WORKER)
            config = root/'config.json'
            config.write_text(json.dumps(dict(logical_kv_pages=64)))
            def launch(command, **kwargs):
                return real_popen([sys.executable, '-u', str(fake)], **kwargs)
            with patch('runtime.subprocess.Popen', launch):
                with Session('dp2', config, root) as session:
                    self.assertEqual(set(session.clients), {0, 1})
                    answer = session.synchronous({0: dict(op='echo', value=10), 1: dict(op='echo', value=20)})
                    self.assertEqual(answer[0]['value'], 10)
                    self.assertEqual(answer[1]['value'], 20)
                    session.submit(1, dict(op='fail'))
                    with self.assertRaisesRegex(RuntimeError, 'injected CPU worker error'):
                        while session.pending:
                            session.collect()
                self.assertEqual(session.process.returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
