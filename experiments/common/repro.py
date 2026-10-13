"""Release source integrity and complete experiment snapshots, without CUDA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]


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
