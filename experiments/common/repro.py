"""Release source integrity and complete experiment snapshots, without CUDA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]


def source_before_annotations(path):
    """Recover exact historical bytes by removing only recorded comment lines."""
    root = REPO.resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError('Source is outside this repository: ' + str(path))
    raw = path.read_bytes()
    record_path = root / 'data/source-annotations.json'
    if not record_path.is_file():
        return raw
    records = json.loads(record_path.read_text())['files']
    record = records.get(path.relative_to(root).as_posix())
    if record is None:
        return raw
    lines = raw.splitlines(keepends=True)
    added = record['added_comment_lines']
    if (added != sorted(set(added)) or
            any(not isinstance(i, int) or not 1 <= i <= len(lines) for i in added)):
        raise RuntimeError('Invalid annotation line record: ' + str(path))
    if any(not lines[i-1].lstrip().startswith(b'# ') for i in added):
        raise RuntimeError('Recorded annotation is not a comment: ' + str(path))
    excluded = set(added)
    original = b''.join(line for i, line in enumerate(lines, 1) if i not in excluded)
    if hashlib.sha256(original).hexdigest() != record['original_sha256']:
        raise RuntimeError('Non-comment source changed: ' + str(path))
    return original


def verify_source():
    root = REPO.resolve()
    manifest = json.loads((root / 'source-manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError('Missing or invalid release source: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Release file changed: ' + name +
                               '; regenerate source-manifest.json after reviewing your changes')
    return manifest


def snapshot_sources(destination):
    destination = Path(destination)
    manifest = verify_source()
    for name in manifest['sha256']:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    shutil.copy2(REPO / 'source-manifest.json', destination / 'source-manifest.json')
    return manifest
