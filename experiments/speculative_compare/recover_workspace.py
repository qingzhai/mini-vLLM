"""Restore archived source/records offline; never download models or run a benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(bundle, root):
    manifest = json.loads((bundle / 'RECOVERY.json').read_text())
    source_root = (bundle / 'payload').resolve()
    root = root.resolve()
    planned = []
    for relative, expected in manifest['payload_sha256'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe recovery path')
        source = source_root / path
        destination = root / path
        if source.is_symlink() or not source.is_file() or digest(source) != expected:
            raise ValueError(f'Invalid recovery source: {relative}')
        if not destination.resolve().is_relative_to(root):
            raise ValueError(f'Destination escapes workspace: {relative}')
        if destination.exists():
            if not destination.is_file() or digest(destination) != expected:
                raise FileExistsError(f'Existing content differs; retained untouched: {destination}')
        else:
            planned.append((source, destination))
    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    print('RESTORED FILES:', len(planned), flush=True)
    print('WORK_ROOT:', root, flush=True)
    source = root / 'third_party/DeepSpec'
    if not (source / '.git/index').exists():
        subprocess.run(['git', '-C', str(source), 'read-tree', 'HEAD'], check=True)
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != manifest['deepspec_commit']:
        raise ValueError('Restored commit differs')
    status = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if status.strip():
        raise ValueError(f'Restored source is not clean: {status}')
    for relative, expected in manifest['deepspec_source_sha256'].items():
        if digest(source / relative) != expected:
            raise ValueError(f'Restored official source differs: {relative}')
    print('OFFLINE DEEPSPEC READY:', commit, flush=True)
    print('Original record ZIPs:', root / 'recovered-records', flush=True)


def inspect_environment(root):
    probe = '''import importlib.metadata as m, json, sys
r = dict(python=sys.executable, version=list(sys.version_info[:3]), packages={})
for name in ('pip','torch','triton','transformers','huggingface-hub','numpy','safetensors','datasets'):
    try: r['packages'][name] = m.version(name)
    except m.PackageNotFoundError: r['packages'][name] = None
try:
    import torch
    r.update(cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(), torch_source=torch.__file__)
except Exception as e: r['torch_error'] = str(e)
print(json.dumps(r))
'''
    candidates = [shutil.which('python'), '/usr/local/bin/python', shutil.which('python3'), sys.executable]
    checked, reports = set(), []
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        real = str(Path(candidate).resolve())
        if real in checked:
            continue
        checked.add(real)
        try:
            process = subprocess.run([candidate, '-c', probe], text=True, capture_output=True,
                                     timeout=45, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '0'})
            if process.returncode:
                reports.append(dict(python=candidate, probe_error=process.stderr[-1500:]))
                continue
            report = json.loads(process.stdout.strip().splitlines()[-1])
        except (subprocess.TimeoutExpired, ValueError) as error:
            reports.append(dict(python=candidate, probe_error=str(error)))
            continue
        reports.append(report)
        packages = report['packages']
        if (report['version'][:2] == [3, 11] and (packages['torch'] or '').split('+')[0] == '2.9.1'
                and packages['triton'] == '3.5.1' and report.get('cuda_runtime') == '12.8'
                and report.get('cuda_available')):
            environment = root / 'venv-deepspec'
            if not environment.exists():
                subprocess.run([candidate, '-m', 'venv', '--without-pip', '--system-site-packages', str(environment)], check=True)
            print('REUSABLE BASE PYTHON:', candidate, flush=True)
            print('VENV CREATED WITHOUT DOWNLOADS:', environment, flush=True)
            break
    (root / 'recovery-environment.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
    print('ENVIRONMENT:', json.dumps(reports, ensure_ascii=False, indent=2), flush=True)
    print('RECOVERY STAGE 1 COMPLETE. No packages/models downloaded; no GPU benchmark started.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, default=Path('/hy-tmp/minivllm-work'))
    args = parser.parse_args()
    restore(Path(__file__).resolve().parent, args.work_root)
    inspect_environment(args.work_root)
