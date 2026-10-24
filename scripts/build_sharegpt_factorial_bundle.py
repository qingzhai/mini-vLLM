#!/usr/bin/env python3
"""Package reviewed sources and small records, without weights, caches or venvs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/common'))
from repro import verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'minivllm-sharegpt-factorial-v1.zip')
    args = parser.parse_args()
    manifest = verify_source()
    names = set(manifest['sha256']) | {'source-manifest.json'}
    # Small raw records needed by the CPU evidence checks are outside the source manifest.
    for folder in ('experiments', 'results'):
        for path in (ROOT/folder).rglob('*'):
            if (path.is_file() and '__pycache__' not in path.parts
                    and path.suffix in ('.txt', '.jsonl') and path.name != 'nvidia-smi.txt'):
                names.add(path.relative_to(ROOT).as_posix())
    prefix = 'minivllm-sharegpt-factorial-v1'
    digests = {}
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            path = ROOT/name
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise ValueError('Unexpected external file: ' + name)
            content = path.read_bytes()
            archive.writestr(prefix+'/'+name, content)
            digests[name] = hashlib.sha256(content).hexdigest()
        archive.writestr(prefix+'/package-manifest.json', json.dumps(dict(
            protocol='sharegpt-chunk-prefix-2x2-v1', gpu_validation=False,
            sha256=digests), ensure_ascii=False, indent=2)+'\n')
    print('BUNDLE:', args.output.resolve())
    print('FILES:', len(digests), 'BYTES:', args.output.stat().st_size)
    print('SHA256:', hashlib.sha256(args.output.read_bytes()).hexdigest())
    print('Run instructions: docs/SHAREGPT_FACTORIAL.zh.md; no models/environment included.')


if __name__ == '__main__':
    main()
