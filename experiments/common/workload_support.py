"""Check regenerated ShareGPT requests against the recorded selection."""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def tokenizer_metadata_diff(obj, selected=None):
    if selected is None:
        selected = json.loads((REPO/'data/sharegpt-selection.json').read_text())
    expected = selected['model']['tokenizer_files_sha256']
    actual = obj['model']['tokenizer_files_sha256']
    return {name: dict(historical=expected.get(name), actual=actual.get(name))
            for name in sorted(set(expected) | set(actual))
            if expected.get(name) != actual.get(name)}


def validate_recorded_identity(obj, selected, *, require_tokenizer_file_identity=True):
    # Token identity is mandatory even when file-level tokenizer provenance differs.
    if obj['workload_sha256'] != selected['workload_sha256']:
        raise ValueError('Regenerated request tokens differ from the historical workload')
    for expected, actual in zip(selected['requests'], obj['requests']):
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError('Selection mismatch: ' + actual['request_id'] + ' / ' + key)
    for key in ('config_sha256', 'vocab_size'):
        if obj['model'][key] != selected['model'][key]:
            raise ValueError('Model metadata differs: ' + key)
    if require_tokenizer_file_identity and tokenizer_metadata_diff(obj, selected):
        raise ValueError('Model/tokenizer metadata differs: tokenizer_files_sha256')


def load_workload(path, *, require_tokenizer_file_identity=True):
    obj = json.loads(Path(path).read_text())
    if obj.get('fixture_is_synthetic'):
        raise ValueError('CPU fixtures cannot be used as ShareGPT GPU workloads')
    spec = importlib.util.spec_from_file_location('pinned_sharegpt_data', REPO/'experiments/sharegpt/prepare_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_workload(obj)
    selected = json.loads((REPO/'data/sharegpt-selection.json').read_text())
    validate_recorded_identity(obj, selected,
        require_tokenizer_file_identity=require_tokenizer_file_identity)
    return obj
