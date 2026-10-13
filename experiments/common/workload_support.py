"""Check regenerated ShareGPT requests against the recorded selection."""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_workload(path):
    obj = json.loads(Path(path).read_text())
    if obj.get('fixture_is_synthetic'):
        raise ValueError('CPU fixtures cannot be used as ShareGPT GPU workloads')
    spec = importlib.util.spec_from_file_location('pinned_sharegpt_data', REPO/'experiments/sharegpt/prepare_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_workload(obj)
    selected = json.loads((REPO/'data/sharegpt-selection.json').read_text())
    if obj['workload_sha256'] != selected['workload_sha256']:
        raise ValueError('Regenerated request tokens differ from the historical workload')
    for expected, actual in zip(selected['requests'], obj['requests']):
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError('Selection mismatch: ' + actual['request_id'] + ' / ' + key)
    for key in ('config_sha256', 'tokenizer_files_sha256', 'vocab_size'):
        if obj['model'][key] != selected['model'][key]:
            raise ValueError('Model/tokenizer metadata differs: ' + key)
    return obj
