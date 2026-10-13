#!/usr/bin/env python3
"""Launch a separate reproducible experiment with shared source and run logging."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import zipfile

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / 'experiments/common'
sys.path.insert(0, str(COMMON))
from repro import snapshot_sources, verify_source

ENTRIES = {
    'dflash-graph': ('experiments/dflash_graph', 'benchmark.py'),
    'distributed': ('experiments/distributed', 'benchmark.py'),
    'sharegpt': ('experiments/sharegpt', 'bench_sharegpt.py'),
    'awq': ('experiments/awq', 'run.py'),
    'prefix': ('experiments/prefix', 'bench_prefix.py'),
    'chunked': ('experiments/chunked', 'bench_chunked.py'),
    'decode': ('benchmarks', 'benchmark_decoding.py'),
    'prefill': ('benchmarks', 'benchmark_prefilling.py'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=ENTRIES)
    p.add_argument('--work-root', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--draft', type=Path)
    p.add_argument('--awq-model', type=Path)
    p.add_argument('--workload', type=Path)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--dry-run', action='store_true', help='Print the command without importing CUDA')
    args, extra = p.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    verify_source()
    work = (args.work_root or Path(os.environ.get('WORK_ROOT', REPO / '.work'))).expanduser().resolve()
    folder, script = ENTRIES[args.experiment]
    package = REPO / folder
    env = os.environ.copy()
    env.update(WORK_ROOT=str(work), REPO_ROOT=str(REPO),
               MODEL_PATH=str((args.model or Path(env.get('MODEL_PATH', work / 'models/Qwen3-8B'))).expanduser().resolve()),
               DRAFT_MODEL_PATH=str((args.draft or Path(env.get('DRAFT_MODEL_PATH', work / 'models/Qwen3-8B-DFlash-b16'))).expanduser().resolve()),
               AWQ_MODEL_PATH=str((args.awq_model or Path(env.get('AWQ_MODEL_PATH', work / 'models/Qwen3-8B-AWQ'))).expanduser().resolve()),
               SHAREGPT_WORKLOAD=str((args.workload or Path(env.get('SHAREGPT_WORKLOAD', work / 'data/sharegpt-workload.json'))).expanduser().resolve()),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HOME=str(work / 'cache/huggingface'),
               HF_MODULES_CACHE=str(work / 'cache/huggingface/modules'),
               TRITON_CACHE_DIR=str(work / 'cache/triton'),
               TORCHINDUCTOR_CACHE_DIR=str(work / 'cache/torchinductor'))
    env['PYTHONPATH'] = os.pathsep.join(map(str, [package, COMMON, REPO / 'src', REPO])) + (
        os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if args.experiment != 'distributed':
        env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%SZ')
    if args.dry_run:
        run = args.run_dir or work / 'results' / (args.experiment + '-' + stamp + '-UNIQUE')
    elif args.run_dir:
        run = args.run_dir.expanduser().resolve()
        if run.exists() and any(run.iterdir()):
            p.error('Run directory is not empty; use a new directory')
        run.mkdir(parents=True, exist_ok=True)
    else:
        (work / 'results').mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix=args.experiment + '-' + stamp + '-', dir=work / 'results'))
    env['RUN_DIR'] = str(run)
    command = [sys.executable, '-u', str(package / script)]
    if args.experiment == 'distributed':
        command += ['--run-dir', str(run)]
    elif args.experiment == 'sharegpt':
        command += ['--root', str(work), '--run-dir', str(run), '--model', env['MODEL_PATH'],
                    '--proof', str(REPO / 'data/sharegpt-reference')]
    command += extra
    print('Experiment:', args.experiment, '\nResults:', run, flush=True)
    if args.dry_run:
        print(json.dumps({'command': command, 'work_root': str(work),
                          'pythonpath': env['PYTHONPATH']}, ensure_ascii=False, indent=2))
        return 0
    if args.experiment in ('sharegpt', 'distributed') and not Path(env['SHAREGPT_WORKLOAD']).is_file():
        p.error('Prepare the dataset first: python scripts/prepare_workload.py --work-root ' + str(work))
    if args.experiment == 'sharegpt':
        shutil.copy2(env['SHAREGPT_WORKLOAD'], run / 'workload.json')
    snapshot_sources(run / 'source')
    status = 1
    try:
        with (run / 'run.log').open('w') as log:
            child = subprocess.Popen(command, env=env, cwd=REPO, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     start_new_session=True)
            try:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                status = child.wait()
            except KeyboardInterrupt:
                # Let the controller's finally blocks release NCCL workers.
                # This group belongs exclusively to the experiment we started.
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                status = 130
        if status == 0 and args.experiment == 'sharegpt':
            result = subprocess.run([sys.executable, str(package / 'summarize.py'), str(run)], env=env)
            status = result.returncode
    finally:
        (run / 'exit-status.json').write_text(json.dumps({'exit_status': status}) + '\n')
        archive = run.with_name(run.name + '-records.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(run.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc':
                    z.write(f, Path(run.name) / f.relative_to(run))
        print('EXIT STATUS:', status, '\nRecords:', archive, flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
