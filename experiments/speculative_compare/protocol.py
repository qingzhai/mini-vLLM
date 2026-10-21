"""CPU-only provenance, selection and reporting for the DeepSpec comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess


DEEPSPEC_COMMIT = '005e03b81cec38b7da6399833d609ee89a2587f2'
MODEL_REPOS = {
    'target': 'Qwen/Qwen3-8B',
    'eagle3': 'deepseek-ai/eagle3_qwen3_8b_ttt7',
    'dflash': 'deepseek-ai/dflash_qwen3_8b_block7',
    'dspark': 'deepseek-ai/dspark_qwen3_8b_block7',
}
DATASETS = ('gsm8k', 'humaneval', 'mt-bench')
MODES = {
    'eagle3': ('ar', 'eagle3'),
    'dflash': ('ar', 'dflash'),
    'dspark': ('ar', 'dspark_no_markov', 'dspark_markov_only',
               'dspark_fixed', 'dspark_conf_0.5', 'dspark_conf_0.8'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def source_fingerprint(root):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != DEEPSPEC_COMMIT:
        raise ValueError('DeepSpec commit differs from the reviewed version')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('DeepSpec checkout must be clean')
    tracked = git('ls-files').splitlines()
    # Git 忽略的 Python 文件也可能被导入，因此一并拒绝。
    extra_python = {p.relative_to(root).as_posix() for p in root.rglob('*.py')
                    if '.git' not in p.parts and '__pycache__' not in p.parts} - set(tracked)
    if extra_python:
        raise ValueError(f'Untracked Python in DeepSpec: {sorted(extra_python)}')
    return {name: sha256(root / name) for name in tracked}


def model_fingerprint(root):
    root = Path(root)
    files = sorted(p for p in root.iterdir() if p.is_file() and
                   (p.suffix in ('.json', '.safetensors', '.model', '.txt', '.jinja')))
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError(f'No safetensors weights: {root}')
    index = root / 'model.safetensors.index.json'
    if index.exists():
        shards = set(json.loads(index.read_text())['weight_map'].values())
        if not shards or any(not (root / shard).is_file() for shard in shards):
            raise ValueError(f'Missing checkpoint shards: {root}')
    elif not (root / 'model.safetensors').is_file():
        raise ValueError(f'Missing single-file checkpoint or shard index: {root}')
    return {p.name: sha256(p) for p in files}


def select_rows(rows, *, count, seed):
    if count < 1 or count > len(rows):
        raise ValueError('Insufficient eligible dataset rows')
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[i] for i in indices[:count]]


def first_mismatch(expected, actual):
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return index
    return None if len(expected) == len(actual) else min(len(expected), len(actual))


def summarize(records, *, cases, repeats):
    """Use actual output / total wall time; never average per-request speedups."""
    expected = {(r, case) for r in range(repeats) for case in cases}
    groups = {}
    for row in records:
        groups.setdefault((row['algorithm'], row['mode']), []).append(row)
    summary = []
    for algorithm, modes in MODES.items():
        baseline = groups.get((algorithm, 'ar'), [])
        reference = {(x['repeat'], x['case']): x['output_ids'] for x in baseline}
        baseline_rates = {}
        for repeat in range(repeats):
            rows = [x for x in baseline if x['repeat'] == repeat]
            if rows:
                baseline_rates[repeat] = sum(x['output_tokens'] for x in rows) / sum(x['total_s'] for x in rows)
        for mode in modes:
            rows = groups.get((algorithm, mode), [])
            keys = [(x['repeat'], x['case']) for x in rows]
            complete = len(keys) == len(expected) and set(keys) == expected
            failures = []
            for row in rows:
                key = row['repeat'], row['case']
                mismatch = first_mismatch(reference[key], row['output_ids']) if key in reference else 0
                if mismatch is not None:
                    failures.append(dict(repeat=key[0], case=key[1], first_mismatch=mismatch))
            repeat_metrics = []
            for repeat in range(repeats):
                subset = [x for x in rows if x['repeat'] == repeat]
                if not subset:
                    continue
                seconds = sum(x['total_s'] for x in subset)
                rate = sum(x['output_tokens'] for x in subset) / seconds
                repeat_metrics.append(dict(repeat=repeat, total_s=seconds, tokens_per_s=rate,
                    ratio_to_paired_ar=rate / baseline_rates[repeat] if repeat in baseline_rates else None))
            median = lambda name: statistics.median(x[name] for x in repeat_metrics) if repeat_metrics else None
            ratios = [x['ratio_to_paired_ar'] for x in repeat_metrics if x['ratio_to_paired_ar'] is not None]
            verified = sum(sum(x['proposal_lengths']) for x in rows)
            accepted = sum(sum(x['accepted_draft_lengths']) for x in rows)
            verifies = sum(x['verify_count'] for x in rows)
            summary.append(dict(algorithm=algorithm, mode=mode, complete=complete,
                paired_greedy_match=complete and not failures, mismatches=failures,
                median_total_s=median('total_s'), median_tokens_per_s=median('tokens_per_s'),
                median_paired_speedup=statistics.median(ratios) if ratios else None,
                peak_allocated_gib=max((x['peak_allocated_gib'] for x in rows), default=None),
                verified_candidate_tokens=verified, accepted_candidate_tokens=accepted,
                verified_acceptance_ratio=accepted / verified if verified else None,
                output_tokens_per_verify=sum(max(0, x['output_tokens'] - 1) for x in rows) / verifies if verifies else None,
                repeat_metrics=repeat_metrics))
    return summary
