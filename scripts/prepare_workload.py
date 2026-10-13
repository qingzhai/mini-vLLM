#!/usr/bin/env python3
"""Regenerate the recorded ShareGPT selection; no GPU or model loading."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/common'))
from workload_support import load_workload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root', type=Path, default=Path(os.environ.get('WORK_ROOT', ROOT/'.work')))
    p.add_argument('--model', type=Path)
    p.add_argument('--dataset-file', type=Path)
    args = p.parse_args()
    work = args.work_root.expanduser().resolve()
    output = work/'data/sharegpt-workload.json'
    command = [sys.executable, str(ROOT/'experiments/sharegpt/prepare_data.py'),
               '--root', str(work), '--output', str(output)]
    if args.model:
        command += ['--model', str(args.model.expanduser().resolve())]
    if args.dataset_file:
        command += ['--dataset-file', str(args.dataset_file.expanduser().resolve())]
    subprocess.run(command, check=True)
    load_workload(output)
    print('Selection and all token hashes match the historical 100-request workload:', output)


if __name__ == '__main__':
    main()
