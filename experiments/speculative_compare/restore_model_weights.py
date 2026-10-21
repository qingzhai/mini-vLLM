"""Restore only original checkpoint files from ModelScope; accept original SHA256 only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import quote


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def original_manifest(model):
    # 重建原下载脚本的本地记录，不冒充 ModelScope 自己的 commit。
    return json.dumps(dict(repo=model['repo_id'], hf_commit=model['revision'], source='modelscope'), indent=2).encode()


def model_tasks(root, lock):
    tasks = []
    for name, model in lock['models'].items():
        if name not in ('target', 'eagle3', 'dflash', 'dspark'):
            raise ValueError('Unexpected model name')
        folder = root / 'models' / ('Qwen3-8B' if name == 'target' else name)
        for filename, expected in model['sha256'].items():
            if Path(filename).name != filename or filename in ('.', '..'):
                raise ValueError('Unsafe checkpoint filename')
            url = f"https://www.modelscope.cn/models/{model['repo_id']}/resolve/master/{quote(filename, safe='')}"
            seed = root / 'recovery-metadata/models' / folder.name / filename
            content = original_manifest(model) if filename == 'download_manifest.json' else None
            if content is not None and hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Reconstructed manifest does not match the archived original')
            tasks.append(dict(destination=folder / filename, expected=expected, url=url, seed=seed, content=content))
    return tasks


def curl_download(url, partial, stopped):
    log_path = partial.with_suffix(partial.suffix + '.curl.log')
    initial = partial.stat().st_size if partial.exists() else 0
    started = last_update = time.monotonic()
    with log_path.open('a') as log:
        process = subprocess.Popen(['curl', '--fail', '--location', '--silent', '--show-error',
            '--continue-at', '-', '--connect-timeout', '20', '--speed-limit', '10240',
            '--speed-time', '60', '--retry', '3', '--retry-all-errors', '--retry-delay', '3',
            '--output', str(partial), url], stdout=subprocess.DEVNULL, stderr=log)
        try:
            while process.poll() is None:
                if stopped.wait(2):
                    raise RuntimeError('Download stopped after another file failed; partial data retained')
                now = time.monotonic()
                if now - last_update >= 20:
                    size = partial.stat().st_size if partial.exists() else initial
                    speed = max(0, size - initial) / max(now - started, 1) / 1024**2
                    print(f'PROGRESS {partial.name}: {size/1024**2:.1f} MiB, {speed:.2f} MiB/s', flush=True)
                    last_update = now
            if process.returncode:
                raise RuntimeError(f'curl failed ({process.returncode}); resume retained: {partial}; details: {log_path}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def restore_file(task, stopped, downloader=curl_download):
    destination = task['destination']
    if stopped.is_set():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256(destination) != task['expected']:
            raise ValueError(f'Existing file differs from archived SHA256; left untouched: {destination}')
        print('ALREADY VERIFIED:', destination, flush=True)
        return dict(path=str(destination), sha256=task['expected'], action='existing_original')
    partial = destination.with_name(destination.name + '.download')
    if task['content'] is not None:
        partial.write_bytes(task['content'])
        action = 'reconstructed_original_manifest'
    elif task['seed'].is_file() and sha256(task['seed']) == task['expected']:
        shutil.copyfile(task['seed'], partial)
        action = 'archived_original_metadata'
    else:
        print('MODELSCOPE DOWNLOAD:', destination, flush=True)
        downloader(task['url'], partial, stopped)
        action = 'modelscope_url_with_original_sha256'
    print('SHA256 CHECK:', destination.name, flush=True)
    if sha256(partial) != task['expected']:
        raise ValueError(f'ModelScope file differs from original experiment; NOT accepted: {partial}')
    partial.replace(destination)
    print('VERIFIED:', destination, flush=True)
    return dict(path=str(destination), sha256=task['expected'], action=action,
                source_url=task['url'] if action.startswith('modelscope') else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    root = args.root.resolve()
    lock = json.loads(args.lock.read_text())
    tasks = model_tasks(root, lock)
    print('Free GiB:', round(shutil.disk_usage(root).free / 1024**3, 2), flush=True)
    print('Original files:', len(tasks), '; concurrent downloads:', args.workers, flush=True)
    print('No Hub SDK cache, no extra formats; existing original files are reused.', flush=True)
    stopped = threading.Event()
    restored = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(restore_file, task, stopped) for task in tasks]
        try:
            for future in as_completed(futures):
                restored.append(future.result())
        except BaseException:
            stopped.set()
            for future in futures:
                future.cancel()
            raise
    provenance = root / 'recovery-environment/weights-restored.json'
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(dict(
        restored_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_lock_sha256=sha256(args.lock),
        identity_policy='Exact original file SHA256; ModelScope master is transport, not a claimed immutable revision',
        files=sorted(restored, key=lambda row: row['path'])), ensure_ascii=False, indent=2) + '\n')
    print('ALL FOUR MODELS RESTORED WITH ORIGINAL SHA256. No GPU benchmark started.', flush=True)


if __name__ == '__main__':
    main()
