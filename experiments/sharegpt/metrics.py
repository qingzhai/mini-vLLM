"""Pure-Python timing/sequence accounting shared by benchmark and local tests."""
from statistics import mean


def percentile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError('Empty metric population')
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def request_metrics(request, output_ids, token_times_s, arrival_s=0.0):
    if len(output_ids) != len(token_times_s) or len(output_ids) < 2:
        raise ValueError('Need one timestamp per output token and at least two outputs')
    if token_times_s[0] < arrival_s or any(b < a for a, b in zip(token_times_s, token_times_s[1:])):
        raise ValueError('Invalid monotonic timestamps')
    gaps = [b - a for a, b in zip(token_times_s, token_times_s[1:])]
    return dict(request_id=request['request_id'], source_id=request['source_id'],
                bucket=request['bucket'], repeated_from=request.get('repeated_from'),
                prompt_tokens=request['prompt_tokens'], output_tokens=len(output_ids),
                arrival_s=arrival_s, ttft_ms=(token_times_s[0]-arrival_s)*1000,
                tpot_ms=mean(gaps)*1000, e2e_ms=(token_times_s[-1]-arrival_s)*1000,
                max_inter_token_ms=max(gaps)*1000, output_token_ids=list(output_ids),
                token_times_s=list(token_times_s))


def summarize(rows, total_s):
    if not rows or total_s <= 0:
        raise ValueError('Invalid workload duration or request count')
    result = dict(requests=len(rows), output_tokens=sum(x['output_tokens'] for x in rows),
                  total_s=total_s, requests_per_s=len(rows)/total_s,
                  output_tokens_per_s=sum(x['output_tokens'] for x in rows)/total_s,
                  avg_prompt_tokens=mean(x['prompt_tokens'] for x in rows))
    for key in ('ttft_ms', 'tpot_ms', 'e2e_ms'):
        values = [x[key] for x in rows]
        result['avg_' + key] = mean(values)
        result['p95_' + key] = percentile(values, .95)
    for bucket in ('short', 'long'):
        subset = [x for x in rows if x['bucket'] == bucket]
        if subset:
            for key in ('ttft_ms', 'tpot_ms', 'e2e_ms'):
                result[bucket + '_avg_' + key] = mean(x[key] for x in subset)
    return result


def compare_sequences(baseline, candidate):
    old = {r['request_id']: r for r in baseline}
    new = {r['request_id']: r for r in candidate}
    if len(old) != len(baseline) or len(new) != len(candidate) or old.keys() != new.keys():
        raise ValueError('Different or duplicated request IDs in comparison')
    rows = []
    for key, a in old.items():
        b = new[key]
        x, y = a['output_token_ids'], b['output_token_ids']
        index = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
        if index is None and len(x) != len(y):
            index = min(len(x), len(y))
        rows.append(dict(request_id=key, exact_match=x == y, first_mismatch=index,
                         baseline_token=x[index] if index is not None and index < len(x) else None,
                         candidate_token=y[index] if index is not None and index < len(y) else None))
    return dict(requests=len(rows), exact_matches=sum(x['exact_match'] for x in rows),
                all_exact=all(x['exact_match'] for x in rows),
                mismatches=[x for x in rows if not x['exact_match']])
