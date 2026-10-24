#!/usr/bin/env python3
"""Audit a completed records ZIP offline and publish compact, traceable metrics."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments/sharegpt'), str(ROOT/'experiments/common')]
from benchmark_support import load_proof
from factorial_protocol import MODES, PROTOCOL, paired_effects, settings, trial_order
from metrics import compare_sequences, request_metrics, summarize
from prepare_data import validate_workload
from summarize import METRICS
from workload_support import tokenizer_metadata_diff, validate_recorded_identity


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def audit(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, 'Archive CRC failed')
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate archive entries')
        reports = [n for n in names if n.count('/') == 1 and n.endswith('/report.json')]
        require(len(reports) == 1, 'Expected one top-level run report')
        prefix = reports[0].removesuffix('report.json')
        used = {}

        def raw(name):
            content = archive.read(prefix+name)
            used[prefix+name] = digest(content)
            return content

        def read(name):
            return json.loads(raw(name))

        require(read('exit-status.json') == {'exit_status': 0}, 'Experiment did not exit successfully')
        report = read('report.json')
        require(report.get('protocol') == PROTOCOL and report.get('repeats') == 3,
                'Expected the four-mode, three-repeat protocol')
        for key in ('completed', 'passed', 'functional_checks_passed', 'all_exact_greedy',
                    'all_timings_clean', 'matches_historical_reference_outputs', 'all_replays_exact'):
            require(report.get(key) is True, 'Report check failed: '+key)
        source_manifest = read('source/source-manifest.json')
        for name, expected in source_manifest['sha256'].items():
            require(digest(raw('source/'+name)) == expected, 'Archived source mismatch: '+name)
        workload = read('workload.json')
        validate_workload(workload)
        selection = json.loads((ROOT/'data/sharegpt-selection.json').read_text())
        validate_recorded_identity(workload, selection, require_tokenizer_file_identity=False)
        _, reference = load_proof(ROOT/'data/sharegpt-reference', workload)
        expected_order = [(r, m) for r in range(1, 4) for m in trial_order(r)]
        require([(t['repeat'], t['summary']['mode']) for t in report['measured_trials']] == expected_order,
                'Missing/duplicated or reordered measured trials')
        token_rows, trials = {}, []
        expected_hits = sum(((r['prompt_tokens']-1)//256)*256 for r in workload['requests'][80:])
        for trial in report['measured_trials']:
            repeat, summary = trial['repeat'], trial['summary']
            mode = summary['mode']
            require(trial['file'] == f'round{repeat}-{mode}.json', 'Unexpected trial filename')
            obj = read(trial['file'])
            require(obj['summary'] == summary and summary['mode_config'] == settings(mode), 'Trial metadata differs')
            require(obj['graph_diagnostics_added'] == [], 'Diagnostic work inside measured trial')
            rows = obj['requests']
            require(len(rows) == 100, 'Incomplete request population')
            require(summary['output_tokens'] == 6400 and summary['peak_active_requests'] <= 8,
                    'Output or residency constraint failed')
            for key in ('all_pages_released', 'finite_logits', 'timing_clean'):
                require(summary[key] is True, 'Trial state failed: '+key)
            require(summary['cache_hit_tokens'] == (expected_hits if settings(mode)['prefix_cache'] else 0),
                    'Unexpected prefix-cache reuse')
            require((summary['partial_prefill_chunks'] > 0) == settings(mode)['chunk_enabled'],
                    'Chunk switch does not match actual partial Prefill')
            by_id = {}
            for expected, row in zip(workload['requests'], rows):
                for key in ('request_id', 'source_id', 'bucket', 'repeated_from', 'prompt_tokens'):
                    require(row[key] == expected[key], 'Request selection/order differs')
                require(row['output_tokens'] == len(row['output_token_ids']) == 64, 'Incomplete generation')
                require(row['arrival_s'] == 0, 'Expected simultaneous arrivals')
                reconstructed = request_metrics(expected, row['output_token_ids'], row['token_times_s'], row['arrival_s'])
                for key in ('ttft_ms', 'tpot_ms', 'e2e_ms', 'max_inter_token_ms'):
                    require(math.isclose(row[key], reconstructed[key], rel_tol=1e-9, abs_tol=1e-6),
                            'Request timing mismatch: '+key)
                if row['repeated_from']:
                    require(row['output_token_ids'] == by_id[row['repeated_from']]['output_token_ids'],
                            'Replay output differs')
                by_id[row['request_id']] = row
            for key, value in summarize(rows, summary['total_s']).items():
                require(math.isclose(value, summary[key], rel_tol=1e-9, abs_tol=1e-6), 'Metric mismatch: '+key)
            ref_mode = 'full_prefill' if mode == 'prefix_only' else mode
            require(compare_sequences(reference[ref_mode], rows)['all_exact'], 'Historical output mismatch')
            token_rows[repeat, mode] = rows
            trials.append(dict(repeat=repeat, file=trial['file'], summary=summary,
                request_output_sha256={r['request_id']: digest(json.dumps(r['output_token_ids'],
                    separators=(',', ':')).encode()) for r in rows}))
        for rows in token_rows.values():
            require(compare_sequences(token_rows[1, 'full_prefill'], rows)['all_exact'],
                    'Cross-mode or cross-round output mismatch')
        effects = read('factorial-effects.json')
        require(effects['qualified_comparison'] is True, 'Unqualified effect report')
        require(len(effects['effects']) == 4 and effects['protocol'] == PROTOCOL, 'Incomplete effect report')
        for actual, expected in zip(effects['effects'], paired_effects(report['measured_trials'], 3)):
            require(all(actual[k] == v for k, v in expected.items()), 'Paired effect mismatch')
        csv_rows = list(csv.DictReader(io.StringIO(raw('summary.csv').decode())))
        require([row['mode'] for row in csv_rows] == list(MODES), 'Summary modes differ')
        for row in csv_rows:
            summaries = [t['summary'] for t in trials if t['summary']['mode'] == row['mode']]
            for key in METRICS:
                require(math.isclose(float(row[key]), median(s[key] for s in summaries), abs_tol=1e-6),
                        'Median CSV mismatch: '+key)
            require(row['strict_greedy_equivalence'] == row['performance_comparison_valid'] == 'True',
                    'Summary contains an unqualified comparison')
        exports = {name: raw(name) for name in ('report.json', 'summary.csv', 'factorial-effects.json',
                    'factorial-effects.csv', 'installed-packages.json')}
        for name in exports:
            if name.endswith('.csv'):
                exports[name] = exports[name].replace(b'\r\n', b'\n')
        audit_result = dict(offline_audit_passed=True, gpu_rerun=False, measured_trials=12,
            request_executions=1200, generated_tokens=76800, output_tokens_per_request=64,
            historical_and_cross_mode_outputs_exact=True, all_replays_exact=True,
            per_request_and_trial_metrics_recomputed=True, archived_source_manifest_verified=True,
            tokenizer_file_differences=tokenizer_metadata_diff(workload),
            note='Checks archived observations; does not replace hardware profiling or general model-quality evaluation.')
        sources = dict(archive_filename=archive_path.name, archive_sha256=digest(archive_path.read_bytes()),
            run_id=prefix.rstrip('/'), record_sha256=used,
            published_transforms={'summary.csv': 'CRLF normalized to LF; values unchanged',
                                  'factorial-effects.csv': 'Line endings normalized to LF; values unchanged'},
            note='Original ZIP retained locally. Published records omit ShareGPT prompt text, input token arrays, '
                 'complete output arrays and step traces; request output hashes and complete trial summaries remain.')
        return exports, trials, audit_result, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, help='Publish compact records; otherwise audit only')
    args = parser.parse_args()
    exports, trials, result, sources = audit(args.archive)
    if args.output:
        for name, obj in (('trials.json', trials), ('AUDIT.json', result), ('SOURCES.json', sources)):
            exports[name] = (json.dumps(obj, ensure_ascii=False, indent=2)+'\n').encode()
        # Refuse different existing records instead of replacing historical evidence.
        for name, content in exports.items():
            target = args.output/name
            require(not target.exists() or target.read_bytes() == content, 'Existing record differs: '+str(target))
        args.output.mkdir(parents=True, exist_ok=True)
        for name, content in exports.items():
            (args.output/name).write_bytes(content)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
