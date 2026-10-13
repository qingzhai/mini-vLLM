"""Synthetic CPU contract fixtures only; never a ShareGPT performance result."""
import copy
import hashlib
import json
from pathlib import Path


def synthetic_workload():
    root = Path(__file__).resolve().parents[2]
    obj = copy.deepcopy(json.loads((root/'data/sharegpt-selection.json').read_text()))
    by_id = {}
    for i, row in enumerate(obj['requests']):
        if row['repeated_from']:
            ids = by_id[row['repeated_from']]['prompt_token_ids'].copy()
        else:
            count = (288 if row['bucket'] == 'short' else 1152) + i % 17
            ids = [17] * count
            ids[0] = i + 100
        row['prompt_token_ids'] = ids
        row['prompt_tokens'] = len(ids)
        row['prompt_sha256'] = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        by_id[row['request_id']] = row
    obj['workload_sha256'] = hashlib.sha256(json.dumps(obj['requests'], sort_keys=True,
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    obj['fixture_is_synthetic'] = True
    return obj
