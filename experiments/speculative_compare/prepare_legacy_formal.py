"""Restore the original twelve prompts after a passed native Target regression."""
import argparse
import copy
import json
from pathlib import Path

from protocol import MODES, json_hash, sha256, summarize, write_json
from prepare_fp16_regression import CASES


HERE = Path(__file__).resolve().parent
COMPUTE_FILES = ('benchmark.py', 'protocol.py', 'precision.py', 'legacy_target.py',
                 'legacy-source-manifest.json')


def prepare(previous, output, harness=HERE):
    measured = previous / 'measured'
    report = json.loads((measured/'report.json').read_text())
    config = json.loads((measured/'run-config.json').read_text())
    required = dict(target_backend='legacy_minivllm', target_backbone='fp16',
        target_head='fp32', full_fp16_reduction=True, full_bf16_reduction=True,
        target_logit_diagnostics=True, max_new_tokens=256, warmup_tokens=32, repeats=1)
    if (not report.get('qualified_greedy_comparison') or not report.get('completed')
            or not report.get('gpu_execution') or report.get('cases') != 3
            or report.get('errors') or report.get('ar_divergences') or report.get('precision_errors')
            or any(config.get(key) != value for key, value in required.items())):
        raise ValueError('The native Target three-prompt regression did not pass with the required settings')
    for name in COMPUTE_FILES:
        if (harness/name).read_bytes() != (measured/'harness-source'/name).read_bytes():
            raise ValueError('Compute source differs from the passed regression: ' + name)
    canonical_path = harness/'formal-workload.json'
    canonical = json.loads(canonical_path.read_text())
    if canonical_path.read_bytes() != (measured/'harness-source/formal-workload.json').read_bytes():
        raise ValueError('Original twelve-prompt workload differs from the passed regression archive')
    if (len(canonical) != 12 or len({row['case'] for row in canonical}) != 12
            or any(sum(row['task'] == name for row in canonical) != 4
                   for name in ('gsm8k', 'humaneval', 'mt-bench'))):
        raise ValueError('Expected the original three datasets, four prompts each')
    for row in canonical:
        if (row['input_tokens'] != len(row['input_ids'])
                or row['input_ids_sha256'] != json_hash(row['input_ids'])
                or not 0 < row['input_tokens'] <= 1536):
            raise ValueError('Original prompt tokens are invalid')
    regression_workload = json.loads((measured/'workload.json').read_text())
    if regression_workload != [next(row for row in canonical if row['case'] == case) for case in CASES]:
        raise ValueError('Passed requests are not the three original regression prompts')
    lock = copy.deepcopy(json.loads((measured/'lock.json').read_text()))
    if sha256(measured/'workload.json') != lock['workload_sha256']:
        raise ValueError('Regression workload hash differs from its lock')
    records = [json.loads(line) for algorithm in MODES
               for line in (measured/f'{algorithm}-records.jsonl').read_text().splitlines() if line.strip()]
    summaries = summarize(records, cases=list(CASES), repeats=1)
    if (len(records) != 30 or not all(row['paired_greedy_match'] for row in summaries)
            or any(not row.get('native_cache_released') or row['output_tokens'] != 256
                   or len(row['output_ids']) != 256 or row.get('target_backend') != 'legacy_minivllm'
                   for row in records)):
        raise ValueError('Raw regression records are incomplete, divergent or have unreleased Target pages')
    for case in CASES:
        baselines = [row['output_ids'] for row in records if row['case'] == case and row['mode'] == 'ar']
        if len(baselines) != 3 or any(ids != baselines[0] for ids in baselines):
            raise ValueError('AR differs between Draft workers')
    output.mkdir(parents=True, exist_ok=False)
    (output/'workload.json').write_bytes(canonical_path.read_bytes())
    lock['workload_sha256'] = sha256(output/'workload.json')
    write_json(output/'lock.json', lock)
    write_json(output/'selection.json', dict(
        passed_regression=str(previous.resolve()), report_sha256=sha256(measured/'report.json'),
        config_sha256=sha256(measured/'run-config.json'),
        records_sha256={f'{a}-records.jsonl': sha256(measured/f'{a}-records.jsonl') for a in MODES},
        compute_sha256={name: sha256(harness/name) for name in COMPUTE_FILES},
        canonical_workload_sha256=sha256(canonical_path), cases=[row['case'] for row in canonical],
        changes=dict(cases='original twelve prompts', repeats=3, target_logit_diagnostics=False)))
    print('PASSED REGRESSION:', previous, flush=True)
    print('Formal workload: original twelve prompts; model/source fingerprints and seed unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.previous, args.output)
