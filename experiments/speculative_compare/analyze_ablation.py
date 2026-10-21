"""Recompute method/DSpark ablation tables from archived records; no GPU work."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from protocol import MODES, json_hash, sha256, summarize


def analyze(folder):
    workload = json.loads((folder/'workload.json').read_text())
    config = json.loads((folder/'run-config.json').read_text())
    report = json.loads((folder/'report.json').read_text())
    paths = [folder/f'{algorithm}-records.jsonl' for algorithm in MODES]
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        if (len(row['output_ids']) != row['output_tokens']
                or row['output_tokens'] != config['max_new_tokens']
                or json_hash(row['output_ids']) != row['output_sha256']):
            raise ValueError('Output tokens differ from their length/hash record')
        if row['mode'] != 'ar' and (
                not len(row['proposal_lengths']) == len(row['accepted_draft_lengths']) == len(row['acceptance_lengths']) == row['verify_count']
                or any(a + 1 != b for a, b in zip(row['accepted_draft_lengths'], row['acceptance_lengths']))):
            raise ValueError('Acceptance trajectory is inconsistent')
    summaries = summarize(rows, cases=[row['case'] for row in workload], repeats=config['repeats'])
    if summaries != report['summary']:
        raise ValueError('Recomputed statistics differ from archived report')
    ablations = []
    dspark = {row['mode']: row for row in summaries if row['algorithm'] == 'dspark'}
    for baseline, candidate in (
            ('dspark_no_markov', 'dspark_markov_only'), ('dspark_markov_only', 'dspark_fixed'),
            ('dspark_fixed', 'dspark_conf_0.5'), ('dspark_fixed', 'dspark_conf_0.8')):
        left = {row['repeat']: row for row in dspark[baseline]['repeat_metrics']}
        right = {row['repeat']: row for row in dspark[candidate]['repeat_metrics']}
        ratios = [right[r]['tokens_per_s']/left[r]['tokens_per_s'] for r in left]
        ratio = statistics.median(ratios)
        ablations.append(dict(baseline=baseline, candidate=candidate,
            median_paired_throughput_ratio=ratio, throughput_change_percent=(ratio-1)*100,
            paired_throughput_ratios=ratios))
    trajectories = []
    for item in summaries:
        if item['mode'] == 'ar':
            continue
        subset = [row for row in rows if (row['algorithm'], row['mode']) == (item['algorithm'], item['mode'])]
        verifies = sum(row['verify_count'] for row in subset)
        proposed = sum(sum(row['proposal_lengths']) for row in subset)
        accepted = sum(sum(row['accepted_draft_lengths']) for row in subset)
        trajectories.append(dict(algorithm=item['algorithm'], mode=item['mode'], executions=len(subset),
            verification_calls=verifies, average_proposal_length=proposed/verifies,
            average_accepted_draft_length=accepted/verifies,
            average_computed_advance=(accepted+verifies)/verifies,
            verified_acceptance_ratio=accepted/proposed if proposed else None,
            delivered_output_tokens_per_verify=sum(row['output_tokens']-1 for row in subset)/verifies))
    tasks = {row['case']: row['task'] for row in workload}
    by_task = []
    for task in sorted(set(tasks.values())):
        subset = [row for row in rows if tasks[row['case']] == task]
        task_cases = [case for case in tasks if tasks[case] == task]
        for item in summarize(subset, cases=task_cases, repeats=config['repeats']):
            by_task.append(dict(task=task, algorithm=item['algorithm'], mode=item['mode'],
                median_total_s=item['median_total_s'], median_tokens_per_s=item['median_tokens_per_s'],
                median_paired_speedup=item['median_paired_speedup'], paired_greedy_match=item['paired_greedy_match']))
    markov = {(row['case'], row['repeat']): row for row in rows if row['mode'] == 'dspark_markov_only'}
    fixed = [row for row in rows if row['mode'] == 'dspark_fixed']
    equivalent = all(all(row[key] == markov[row['case'], row['repeat']][key]
        for key in ('output_ids', 'proposal_lengths', 'accepted_draft_lengths', 'acceptance_lengths', 'verify_count'))
        for row in fixed)
    return dict(source_sha256={path.name: sha256(path) for path in
                [*paths, folder/'report.json', folder/'run-config.json', folder/'workload.json', folder/'lock.json']},
        qualified_greedy_comparison=report['qualified_greedy_comparison'],
        executions=len(rows), generated_output_tokens=sum(row['output_tokens'] for row in rows),
        total_measured_generation_s=sum(row['total_s'] for row in rows),
        summary=summaries, dspark_ablations=ablations, trajectories=trajectories,
        by_task=by_task, markov_only_and_fixed_identical_trajectories=equivalent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = analyze(args.records_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'ablation-analysis.json').open('x') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    for key in ('by_task', 'trajectories', 'dspark_ablations'):
        with (args.output_dir/(key+'.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[key][0]))
            writer.writeheader()
            writer.writerows(data[key])
    print('Executions:', data['executions'], 'output tokens:', data['generated_output_tokens'])
    for row in data['dspark_ablations']:
        print(row['baseline'], '->', row['candidate'], f"{row['median_paired_throughput_ratio']:.6f}x")
    print('Analysis saved:', args.output_dir)


if __name__ == '__main__':
    main()
