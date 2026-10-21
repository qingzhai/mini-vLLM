"""Collect existing evidence without loading models or changing the experiment."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile


EVIDENCE_DIRS = (
    'minivllm-speculative-compare-v1',
    'third_party/DeepSpec',
    'results/prepared-local-20261019-065730Z',
    'results/gpu-smoke-20261019-070027Z',
    'results/dspark-debug-20261019-073742Z',
)
EXCLUDED = {'.git', '__pycache__', '.cache', '.venv', 'node_modules'}


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return dict(argv=args, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(argv=args, error=str(error))


def collect(root, output, *, max_file_bytes=32 * 2**20, max_total_bytes=192 * 2**20,
            inventory=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    metadata = dict(schema=1, work_root=str(root), python=sys.executable,
                    collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    gpu_benchmark_run=False, model_weights_rehashed=False,
                    files={}, skipped=[], missing=[], commands=[])
    paths = set()
    for relative in EVIDENCE_DIRS:
        directory = root / relative
        if not directory.is_dir():
            metadata['missing'].append(relative)
            continue
        for path in directory.rglob('*'):
            if not any(part in EXCLUDED for part in path.relative_to(root).parts):
                if path.is_file() or path.is_symlink():
                    paths.add(path)
    # 可选回收历史 Token 负载，不重新分词或构造替代样本。
    candidates = {root / 'data/sharegpt-workload.json',
                  root / 'MinivLLM/data/sharegpt-workload.json'}
    results = root / 'results'
    if results.is_dir():
        candidates.update(results.rglob('workload.json'))
    metadata['sharegpt_workloads'] = []
    for path in sorted(candidates):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > max_file_bytes:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get('workload_label') == 'sharegpt-with-20pct-replayed-prompts':
            paths.add(path)
            metadata['sharegpt_workloads'].append(path.relative_to(root).as_posix())
    # 模型仅收集配置和文件尺寸；已有锁文件中的权重哈希原样归档。
    model_files = {}
    for name in ('Qwen3-8B', 'eagle3', 'dflash', 'dspark'):
        directory = root / 'models' / name
        if not directory.is_dir():
            metadata['missing'].append(str(directory.relative_to(root)))
            continue
        model_files[name] = {p.name: p.stat().st_size for p in directory.iterdir()
                             if p.is_file()}
        for filename in ('config.json', 'generation_config.json', 'tokenizer_config.json',
                         'model.safetensors.index.json'):
            path = directory / filename
            if path.is_file():
                paths.add(path)
    metadata['model_file_sizes'] = model_files
    if inventory:
        source = str(root / 'third_party/DeepSpec')
        for args in ([sys.executable, '-m', 'pip', 'freeze'],
                     ['git', '-C', source, 'rev-parse', 'HEAD'],
                     ['git', '-C', source, 'status', '--porcelain', '--untracked-files=all'],
                     ['git', '-C', source, 'diff', 'HEAD', '--'],
                     ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu',
                      '--format=csv']):
            metadata['commands'].append(command(args))
        cfg = Path(sys.prefix) / 'pyvenv.cfg'
        metadata['pyvenv_config'] = cfg.read_text() if cfg.is_file() else None
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    total = 0
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(paths):
                relative = path.relative_to(root).as_posix()
                if path.is_symlink() or not path.is_file():
                    metadata['skipped'].append(dict(path=relative, reason='symlink or special file'))
                    continue
                before = path.stat()
                if path.suffix in ('.safetensors', '.bin') or before.st_size > max_file_bytes:
                    metadata['skipped'].append(dict(path=relative, reason='weight or per-file limit',
                                                    size=before.st_size))
                    continue
                if total + before.st_size > max_total_bytes:
                    metadata['skipped'].append(dict(path=relative, reason='total size limit',
                                                    size=before.st_size))
                    continue
                data = path.read_bytes()
                after = path.stat()
                stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                archive.writestr('work/' + relative, data)
                metadata['files'][relative] = dict(size=len(data), stable=stable,
                    sha256=hashlib.sha256(data).hexdigest())
                total += len(data)
            metadata['uncompressed_bytes'] = total
            archive.writestr('AUDIT_MANIFEST.json', json.dumps(metadata, ensure_ascii=False, indent=2))
        os.rename(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print('ARCHIVE:', output, flush=True)
    print('Files:', len(metadata['files']), 'ZIP MiB:', round(output.stat().st_size / 2**20, 2), flush=True)
    print('Missing:', metadata['missing'], flush=True)
    print('Skipped:', metadata['skipped'], flush=True)
    print('Historical ShareGPT workloads:', metadata['sharegpt_workloads'], flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/hy-tmp/minivllm-work'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    info = collect(args.root, args.output)
    if info['missing'] or info['skipped'] or any(not item['stable'] for item in info['files'].values()):
        raise SystemExit('Archive saved, but coverage is incomplete; inspect AUDIT_MANIFEST.json.')


if __name__ == '__main__':
    main()
