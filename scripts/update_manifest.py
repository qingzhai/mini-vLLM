#!/usr/bin/env python3
"""Record reviewed release sources; this does not certify GPU correctness."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = {}
for base in ('src', 'experiments', 'scripts', 'tests', 'benchmarks', 'data', 'docs', 'results'):
    for p in sorted((root / base).rglob('*')):
        if (p.is_file() and '__pycache__' not in p.parts
                and p.relative_to(root).as_posix() != 'docs/NEXT_SESSION.zh.md'
                and p.suffix in ('.py', '.sh', '.json', '.md', '.csv')):
            files[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('main.py', 'pyproject.toml', 'requirements-gpu.txt', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'README.md', 'docs/REPRODUCE.zh.md'):
    p = root / name
    if p.exists():
        files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
output = {'schema': 1, 'purpose': 'Source integrity after consolidation; not a new GPU run', 'sha256': files}
(root / 'source-manifest.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print('Recorded source files:', len(files))
