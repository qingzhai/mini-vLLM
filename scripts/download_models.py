#!/usr/bin/env python3
"""Download external Qwen checkpoints and record resolved revisions and metadata."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-8B-AWQ', 'Qwen3-8B-DFlash-b16'])
    p.add_argument('--work-root', type=Path, default=Path(os.environ.get('WORK_ROOT', ROOT/'.work')))
    p.add_argument('--revision', help='Explicit base checkpoint commit; fixed AWQ/Draft revisions cannot be changed')
    args = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    pin = json.loads((ROOT/'data/models.json').read_text())['models'][args.model]
    if pin['revision'] and args.revision and pin['revision'] != args.revision:
        p.error('AWQ/Draft revision differs from the recorded experiment')
    work = args.work_root.expanduser().resolve()
    folder = work/'models'/args.model
    source_path = folder/'source.json'
    prior = json.loads(source_path.read_text()) if source_path.is_file() else {}
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    revision = pin['revision'] or args.revision or prior.get('revision')
    # Historical base checkpoint commits were not archived. Resolve and record
    # a concrete commit on the first new download; do not invent that history.
    if not revision:
        revision = HfApi(endpoint=endpoint).model_info(pin['repo_id']).sha
        print('Resolved a base-model revision for this download:', revision, flush=True)
    print('Downloading:', pin['repo_id'], 'revision:', revision, 'endpoint:', endpoint, flush=True)
    snapshot_download(repo_id=pin['repo_id'], revision=revision, endpoint=endpoint,
        local_dir=str(folder), allow_patterns=['*.json', '*.safetensors', '*.py', '*.jinja', '*.txt', '*.model'])
    checks = {}
    for name, expected in pin['metadata_sha256'].items():
        path = folder/name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        checks[name] = actual
        if actual != expected:
            raise ValueError('Downloaded metadata differs from the historical configuration: '+name)
    if pin.get('modeling_dflash_sha256'):
        digest = hashlib.sha256((folder/'modeling_dflash.py').read_bytes()).hexdigest()
        if digest != pin['modeling_dflash_sha256']:
            raise ValueError('Official Draft implementation hash differs')
    record = dict(repo_id=pin['repo_id'], revision=revision, endpoint=endpoint,
                  metadata_sha256=checks, historical_metadata_matches=True,
                  historical_checkpoint_revision_recorded=pin['revision'] is not None)
    source_path.write_text(json.dumps(record, indent=2)+'\n')
    print('Model ready:', folder, flush=True)


if __name__ == '__main__':
    main()
