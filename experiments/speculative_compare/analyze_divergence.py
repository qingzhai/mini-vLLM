"""Aggregate archived output IDs without loading models or changing measurements."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def verification_site(record, output_index):
    """Locate the Target prediction row; the initial Prefill emits index zero."""
    if output_index is None:
        return None
    if output_index == 0:
        return {'phase': 'prefill'}
    pending = 0
    for round_index, (proposed, accepted) in enumerate(zip(
            record['proposal_lengths'], record['accepted_draft_lengths']), 1):
        if not 0 <= accepted <= proposed:
            raise ValueError('Invalid accepted/proposed length')
        if pending < output_index <= pending + accepted + 1:
            row = output_index - pending - 1
            return dict(phase='verification', round=round_index,
                        pending_output_index=pending,
                        cache_length_before=record['input_tokens'] + pending,
                        query_length=proposed + 1, target_prediction_row=row,
                        output_origin='accepted_draft' if row < accepted else 'target_correction_or_bonus')
        pending += accepted + 1
    raise ValueError('Mismatch position is not covered by the recorded trajectory')


def prediction_evidence(record, output_index, site=None):
    """Read observed Top-2/argmax; never infer missing logits from output IDs."""
    trace = record.get('target_prediction_trace')
    if output_index is None or not trace:
        return None
    if record['mode'] == 'ar':
        call, row = output_index, -1
    elif output_index == 0:
        call, row = 0, -1
    else:
        call, row = site['round'], site['target_prediction_row']
    forward = trace[call]
    expected_start = (0 if call == 0 else record['input_tokens'] + output_index - 1)
    if record['mode'] != 'ar' and call:
        expected_start = site['cache_length_before']
    if forward['cache_length_before'] != expected_start:
        raise ValueError('Recorded Target trace does not match the acceptance trajectory')
    logits = forward['top_logits'][row]
    if len(logits) != 2 or logits[0] < logits[1]:
        raise ValueError('Invalid recorded Top-2 logits')
    # Top-k 在并列时的顺序不一定等于 argmax，必须分别报告。
    return dict(target_call=call, prediction_row=row,
                cache_length_before=forward['cache_length_before'],
                query_ids=forward['query_ids'], argmax=forward['argmax'][row],
                top_ids=forward['top_ids'][row], top_logits=logits,
                top2_margin=logits[0] - logits[1])


def analyze(records):
    indexed = {}
    for row in records:
        key = (row['algorithm'], row['mode'], row['case'], row['repeat'])
        if key in indexed:
            raise ValueError(f'Duplicate record: {key}')
        if len(row['output_ids']) != row['output_tokens']:
            raise ValueError(f'Output length mismatch: {key}')
        indexed[key] = row
    modes = sorted({(r['algorithm'], r['mode']) for r in records if r['mode'] != 'ar'})
    cases = sorted({r['case'] for r in records})
    repeats = sorted({r['repeat'] for r in records})
    details, groups, summaries = [], [], []
    for algorithm, mode in modes:
        mode_groups = []
        for case in cases:
            comparisons = []
            for repeat in repeats:
                ar = indexed[(algorithm, 'ar', case, repeat)]
                spec = indexed[(algorithm, mode, case, repeat)]
                if ar['input_tokens'] != spec['input_tokens']:
                    raise ValueError('Paired input length differs')
                pos = first_difference(ar['output_ids'], spec['output_ids'])
                detail = dict(algorithm=algorithm, mode=mode, case=case, repeat=repeat,
                    first_mismatch=pos, common_prefix_tokens=len(ar['output_ids']) if pos is None else pos,
                    ar_token=ar['output_ids'][pos] if pos is not None and pos < len(ar['output_ids']) else None,
                    speculative_token=spec['output_ids'][pos] if pos is not None and pos < len(spec['output_ids']) else None,
                    verification_site=verification_site(spec, pos),
                    target_top2_margin=None, margin_status='not_recorded_in_formal_jsonl')
                ar_prediction = prediction_evidence(ar, pos)
                spec_prediction = prediction_evidence(spec, pos, detail['verification_site'])
                if ar_prediction is not None or spec_prediction is not None:
                    detail.update(ar_prediction=ar_prediction, speculative_prediction=spec_prediction,
                                  target_top2_margin=spec_prediction['top2_margin'] if spec_prediction else None,
                                  margin_status='observed_top2' if ar_prediction and spec_prediction else 'partial_trace')
                details.append(detail)
                comparisons.append(detail)
            ar_outputs = [indexed[(algorithm, 'ar', case, r)]['output_ids'] for r in repeats]
            spec_outputs = [indexed[(algorithm, mode, case, r)]['output_ids'] for r in repeats]
            group = dict(algorithm=algorithm, mode=mode, case=case,
                repeat_indices=repeats,
                first_mismatch_by_repeat=[r['first_mismatch'] for r in comparisons],
                common_prefix_by_repeat=[r['common_prefix_tokens'] for r in comparisons],
                matching_repeats=sum(r['first_mismatch'] is None for r in comparisons),
                ar_output_stable=all(x == ar_outputs[0] for x in ar_outputs),
                speculative_output_stable=all(x == spec_outputs[0] for x in spec_outputs))
            groups.append(group)
            mode_groups.append(group)
        prefix = [n for g in mode_groups for n in g['common_prefix_by_repeat']]
        failing_prefix = [n for g in mode_groups for n, pos in zip(
            g['common_prefix_by_repeat'], g['first_mismatch_by_repeat']) if pos is not None]
        summaries.append(dict(algorithm=algorithm, mode=mode, unique_requests=len(cases),
            repeats=len(repeats), matching_executions=sum(g['matching_repeats'] for g in mode_groups),
            matching_unique_all_repeats=sum(g['matching_repeats'] == len(repeats) for g in mode_groups),
            all_outputs_stable=all(g['ar_output_stable'] and g['speculative_output_stable'] for g in mode_groups),
            common_prefix_min=min(prefix), common_prefix_mean=statistics.mean(prefix),
            common_prefix_median=statistics.median(prefix),
            failed_common_prefix_mean=statistics.mean(failing_prefix) if failing_prefix else None,
            failed_common_prefix_median=statistics.median(failing_prefix) if failing_prefix else None))
    pairwise = []
    for (aa, am), (ba, bm) in itertools.combinations(modes, 2):
        equal_outputs = common_failures = same_fork = 0
        for case, repeat in itertools.product(cases, repeats):
            left = indexed[(aa, am, case, repeat)]['output_ids']
            right = indexed[(ba, bm, case, repeat)]['output_ids']
            equal_outputs += left == right
            a = first_difference(indexed[(aa, 'ar', case, repeat)]['output_ids'], left)
            b = first_difference(indexed[(ba, 'ar', case, repeat)]['output_ids'], right)
            common_failures += a is not None and b is not None
            same_fork += a is not None and a == b
        pairwise.append(dict(left=f'{aa}/{am}', right=f'{ba}/{bm}', executions=len(cases)*len(repeats),
            identical_outputs=equal_outputs, both_fail_ar=common_failures, same_first_fork=same_fork))
    ar_cross_process = all(
        indexed[(a, 'ar', c, r)]['output_ids'] == indexed[(modes[0][0], 'ar', c, r)]['output_ids']
        for a in {a for a, _ in modes} for c, r in itertools.product(cases, repeats))
    traced = sum(bool(r.get('target_prediction_trace')) for r in records)
    return dict(index_base=0,
                margin_evidence=f'Target Top-2 traces in {traced}/{len(records)} executions; missing logits are not inferred',
                ar_cross_process_identical=ar_cross_process,
                summaries=summaries, by_unique_request=groups, comparisons=details, pairwise=pairwise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.records_dir.glob('*-records.jsonl'))
    if not paths:
        raise ValueError('No archived JSONL files found')
    records = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
    report = analyze(records)
    report['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / 'divergence-analysis.json'
    with out.open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    rows = report['by_unique_request']
    with (args.output_dir / 'by-unique-request.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    for row in report['summaries']:
        print(row['mode'], f"{row['matching_unique_all_repeats']}/{row['unique_requests']} unique match;",
              f"mean prefix={row['common_prefix_mean']:.2f}; median={row['common_prefix_median']}")
    print('Analysis saved:', out)


if __name__ == '__main__':
    main()
