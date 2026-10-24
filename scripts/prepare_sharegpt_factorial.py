#!/usr/bin/env python3
"""Reuse an exact local 100-request workload, otherwise regenerate pinned data."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/common'))
from workload_support import load_workload, tokenizer_metadata_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, default=Path(os.environ.get('WORK_ROOT', ROOT/'.work')))
    parser.add_argument('--reuse-workload', type=Path)
    parser.add_argument('--dataset-file', type=Path)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    work = args.work_root.expanduser().resolve()
    destination = work/'data/sharegpt-workload.json'
    candidates = ([args.reuse_workload] if args.reuse_workload else
                  [destination, *sorted((work/'results').glob('sharegpt*/workload.json'), reverse=True)])
    selected = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            load_workload(candidate, require_tokenizer_file_identity=False)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            if args.reuse_workload or candidate == destination:
                raise ValueError('Existing workload is not the canonical ShareGPT input: ' + str(candidate)) from error
            continue
        selected = candidate
        break
    if selected:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if selected.resolve() != destination.resolve():
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(selected, destination)
        print('REUSED EXACT WORKLOAD:', selected, flush=True)
    else:
        if args.reuse_workload:
            raise FileNotFoundError(args.reuse_workload)
        command = [sys.executable, str(ROOT/'scripts/prepare_workload.py'), '--work-root', str(work),
                   '--allow-tokenizer-metadata-difference']
        if args.model:
            command += ['--model', str(args.model)]
        if args.dataset_file:
            command += ['--dataset-file', str(args.dataset_file)]
        subprocess.run(command, check=True)
    obj = load_workload(destination, require_tokenizer_file_identity=False)
    differences = tokenizer_metadata_diff(obj)
    print('TOKENIZER FILE DIFFERENCES:', json.dumps(differences, ensure_ascii=False), flush=True)
    print('Exact historical input Token IDs/config required; file differences retained, '
          'no general tokenizer equivalence claimed.', flush=True)
    model = args.model or work/'models/Qwen3-8B'
    from benchmark_support import load_proof
    # Import the pure-Python evidence guard without changing the archived proof.
    load_proof(ROOT/'data/sharegpt-reference', obj)
    print('READY: 100 historical inputs, 80 unique + 20 replays, 64 outputs/request.')
    print('MODEL EXPECTED:', model, '; no model loading or GPU work performed.')


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT/'experiments/sharegpt'))
    main()
