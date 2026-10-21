"""Download immutable official checkpoints and freeze a shared token workload."""
import argparse
import json
import os
from pathlib import Path
import re

from protocol import (DATASETS, DEEPSPEC_COMMIT, MODEL_REPOS, json_hash,
                      model_fingerprint, select_rows, sha256, source_fingerprint, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=Path, help='Reuse existing local Qwen3-8B; hash all weights')
    parser.add_argument('--drafts-root', type=Path,
                        help='Use locally downloaded EAGLE-3, DFlash and DSpark')
    parser.add_argument('--per-dataset', type=int, default=4)
    parser.add_argument('--max-input-tokens', type=int, default=1536)
    parser.add_argument('--seed', type=int, default=20261019)
    args = parser.parse_args()
    if args.per_dataset < 1 or args.max_input_tokens < 1:
        parser.error('Sample count and input length must be positive')
    source = args.source.resolve()
    fingerprint = source_fingerprint(source)
    args.output.mkdir(parents=True, exist_ok=False)
    # 必须在导入 Hub 前设置 HF_ENDPOINT，镜像与官方均记录在锁文件。
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
    api = HfApi(endpoint=endpoint)
    models = {}
    for name, repo in MODEL_REPOS.items():
        if name != 'target' and args.drafts_root is not None:
            root = (args.drafts_root / name).resolve()
            manifest_path = root / 'download_manifest.json'
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            revision = manifest.get('hf_commit')
            if manifest.get('repo') != repo or not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision):
                raise ValueError(f'Invalid local model manifest: {name}')
            if not (root / 'config.json').is_file() or not list(root.glob('*.safetensors')):
                raise ValueError(f'Incomplete local model: {root}')
            print('LOCAL MODEL:', name, root, revision, flush=True)
        elif name == 'target' and args.target is not None:
            root, revision = args.target.resolve(), None
        else:
            revision = api.model_info(repo).sha
            if not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision):
                raise ValueError(f'Hub did not resolve an immutable commit: {repo}')
            print('DOWNLOAD:', repo, revision, 'endpoint:', endpoint, flush=True)
            root = Path(snapshot_download(repo, revision=revision, endpoint=endpoint,
                allow_patterns=['*.safetensors', '*.json', '*.model', '*.txt', '*.jinja']))
        print('HASHING:', name, root, flush=True)
        models[name] = dict(repo_id=repo, revision=revision, path=str(root),
                            sha256=model_fingerprint(root))
    target_path = Path(models['target']['path'])
    config = json.loads((target_path / 'config.json').read_text())
    if (config.get('model_type'), config.get('num_hidden_layers'), config.get('hidden_size')) != ('qwen3', 36, 4096):
        raise ValueError('Target must be Qwen3-8B (36 layers, hidden size 4096)')
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    selection = []
    dataset_hashes = {}
    for task in DATASETS:
        path = source / 'eval_datasets' / f'{task}.jsonl'
        dataset_hashes[task] = sha256(path)
        eligible = []
        for line, text in enumerate(path.read_text().splitlines(), start=1):
            if not text.strip():
                continue
            row = json.loads(text)
            prompt = row['turns'][0]
            rendered = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
            if 0 < len(ids) <= args.max_input_tokens:
                eligible.append(dict(case=f'{task}:{line}', task=task, line=line,
                    prompt_sha256=json_hash(prompt), input_ids=ids, input_tokens=len(ids),
                    input_ids_sha256=json_hash(ids)))
        selection.extend(select_rows(eligible, count=args.per_dataset, seed=args.seed))
    write_json(args.output / 'workload.json', selection)
    lock = dict(schema=1, backend='official-deepspec-reference', deepspec_commit=DEEPSPEC_COMMIT,
        source=str(source), source_sha256=fingerprint, models=models, endpoint=endpoint,
        seed=args.seed, per_dataset=args.per_dataset, max_input_tokens=args.max_input_tokens,
        dataset_sha256=dataset_hashes, workload_sha256=sha256(args.output / 'workload.json'),
        completed_gpu_benchmark=False)
    write_json(args.output / 'lock.json', lock)
    print('PREPARED:', args.output / 'lock.json', flush=True)
    print('Requests:', len(selection), 'No GPU benchmark has run.', flush=True)


if __name__ == '__main__':
    main()
