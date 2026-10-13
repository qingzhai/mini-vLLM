#!/usr/bin/env python3
"""Run CPU contracts in isolated processes; GPU correctness remains separate."""
import argparse
import ast
import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tensor', action='store_true', help='Include CPU Torch tensor and mocked Graph tests')
    p.add_argument('--transport', action='store_true', help='Include localhost TCP worker RPC test')
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    python_files = [f for f in ROOT.rglob('*.py') if '.git' not in f.parts and '.work' not in f.parts]
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    runs = [('tests', 'check_release.py', []),
            ('experiments/distributed', 'verify_cpu.py', [])]
    if args.transport:
        runs.append(('experiments/distributed', 'verify_transport_cpu.py', []))
    if args.tensor:
        if importlib.util.find_spec('torch') is None:
            p.error('--tensor requires PyTorch; CPU build is sufficient')
        runs += [('experiments/distributed', 'verify_tensors_cpu.py', []),
                 ('experiments/dflash_graph', 'verify_graph_executor_cpu.py', []),
                 ('experiments/dflash_graph', 'verify_runtime_cpu.py', [])]
    report = dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  actual_gpu_execution=False, ast_files=len(python_files), checks=[], skipped=[])
    with tempfile.TemporaryDirectory() as folder:
        if importlib.util.find_spec('xxhash') is not None:
            sys.path.insert(0, str(ROOT/'experiments/common'))
            from fixtures import synthetic_workload
            fixture = Path(folder)/'synthetic-workload.json'
            fixture.write_text(json.dumps(synthetic_workload()))
            runs.append(('experiments/sharegpt', 'verify_cpu.py', [str(ROOT), str(fixture)]))
        else:
            report['skipped'].append('Actual scheduler checks require xxhash; install project dependencies')
        for folder_name, file_name, extra in runs:
            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join(map(str, [ROOT/folder_name, ROOT/'experiments/common',
                ROOT/'experiments/sharegpt', ROOT/'src', ROOT]))
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            command = [sys.executable, str(ROOT/folder_name/file_name), *extra]
            print('CPU CHECK:', folder_name+'/'+file_name, flush=True)
            proc = subprocess.run(command, env=env, cwd=ROOT)
            report['checks'].append({'check': folder_name+'/'+file_name, 'exit_status': proc.returncode})
            if proc.returncode:
                raise SystemExit(proc.returncode)
    report['passed'] = True
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
