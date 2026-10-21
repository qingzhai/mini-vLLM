"""Reuse a verified formal workload and lock; select three failures without downloads."""
import argparse
import copy
import json
from pathlib import Path

from protocol import json_hash, sha256, write_json


CASES = ('humaneval:19', 'humaneval:4', 'mt-bench:9')


def prepare(root, output, canonical_path, source_lock=None):
    canonical = json.loads(canonical_path.read_text())
    candidates = ([source_lock] if source_lock else
                  sorted((root / 'results').glob('speculative-formal-*/prepared/lock.json'), reverse=True) +
                  sorted((root / 'results').glob('speculative-formal-*/measured/lock.json'), reverse=True))
    selected = None
    for path in candidates:
        workload = path.parent / 'workload.json'
        if not workload.is_file() or json.loads(workload.read_text()) != canonical:
            continue
        lock = json.loads(path.read_text())
        if sha256(workload) != lock['workload_sha256']:
            continue
        selected = path
        break
    if selected is None:
        raise ValueError('No original twelve-prompt formal lock matches the archived workload; set SOURCE_LOCK explicitly')
    lock = copy.deepcopy(json.loads(selected.read_text()))
    rows = [next(row for row in canonical if row['case'] == name) for name in CASES]
    for row in rows:
        if row['input_ids_sha256'] != json_hash(row['input_ids']):
            raise ValueError('Archived token hash differs')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'workload.json', rows)
    lock['workload_sha256'] = sha256(output / 'workload.json')
    # 原模型路径、权重指纹、源码指纹和种子全部继承；只缩小请求集合。
    write_json(output / 'lock.json', lock)
    write_json(output / 'selection.json', dict(source_lock=str(selected), source_lock_sha256=sha256(selected),
               original_cases=len(canonical), cases=list(CASES), source_workload_sha256=sha256(canonical_path)))
    print('REUSED LOCK:', selected, flush=True)
    print('SELECTED REQUESTS:', [(x['case'], x['input_tokens']) for x in rows], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--canonical-workload', type=Path, required=True)
    parser.add_argument('--source-lock', type=Path)
    args = parser.parse_args()
    prepare(args.root, args.output, args.canonical_workload, args.source_lock)
