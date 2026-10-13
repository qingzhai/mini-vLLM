"""CPU-only workload, page ownership and metric contracts."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def digest_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class PagePool:
    def __init__(self, pages, reverse=False):
        self.total = pages
        self.free = list(range(pages))
        if reverse:
            self.free.reverse()
        self.owners = {}
        self.peak = 0

    def allocate(self, rid, count):
        if rid in self.owners or count <= 0 or count > len(self.free):
            raise RuntimeError(f'Invalid page admission: {rid}, count={count}, free={len(self.free)}')
        result, self.free = self.free[:count], self.free[count:]
        self.owners[rid] = result
        self.peak = max(self.peak, self.total - len(self.free))
        return result.copy()

    def release(self, rid):
        if rid not in self.owners:
            raise RuntimeError(f'Double/unknown release: {rid}')
        self.free.extend(self.owners.pop(rid))
        assert len(set(self.free)) == len(self.free)

    def clean(self):
        return not self.owners and len(self.free) == self.total


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] * (hi-index) + ordered[hi] * (index-lo) if lo != hi else ordered[lo]


def metrics(requests, total_s, gpu_count):
    if not requests or total_s <= 0:
        raise ValueError('No completed workload')
    ttft, e2e, tpot, intervals = [], [], [], []
    for request in requests:
        times = request['token_times_s']
        arrival = request['arrival_s']
        assert len(times) == request['max_new_tokens'] == len(request['output_token_ids'])
        assert times and times[0] >= arrival and all(b >= a for a, b in zip(times, times[1:]))
        ttft.append(times[0] - arrival)
        e2e.append(times[-1] - arrival)
        if len(times) > 1:
            tpot.append((times[-1] - times[0]) / (len(times)-1))
            intervals.extend(b-a for a, b in zip(times, times[1:]))
    tokens = sum(len(r['output_token_ids']) for r in requests)
    return dict(requests=len(requests), output_tokens=tokens, total_s=total_s,
                output_tokens_per_s=tokens/total_s,
                output_tokens_per_gpu_s=tokens/total_s/gpu_count,
                requests_per_s=len(requests)/total_s,
                avg_ttft_s=mean(ttft), p95_ttft_s=percentile(ttft, 95),
                avg_e2e_s=mean(e2e), avg_tpot_ms=mean(tpot)*1000 if tpot else None,
                p95_itl_ms=percentile(intervals, 95)*1000 if intervals else None,
                p99_itl_ms=percentile(intervals, 99)*1000 if intervals else None,
                max_itl_ms=max(intervals)*1000 if intervals else None,
                measured_itl_samples=len(intervals))


def compare_tokens(reference, candidate):
    by_id = {r['request_id']: r for r in reference}
    assert set(by_id) == {r['request_id'] for r in candidate}
    result = []
    for row in candidate:
        base = by_id[row['request_id']]
        assert base['prompt_token_ids'] == row['prompt_token_ids']
        assert base['max_new_tokens'] == row['max_new_tokens']
        a, b = base['output_token_ids'], row['output_token_ids']
        mismatch = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if mismatch is None and len(a) != len(b):
            mismatch = min(len(a), len(b))
        result.append(dict(request_id=row['request_id'], exact=a == b, first_mismatch=mismatch))
    return dict(exact_sequences=sum(r['exact'] for r in result),
                sequences=len(result), all_exact=all(r['exact'] for r in result), details=result)


def build_cases(workload, burst_count=32, stream_count=16, outputs=64, gap=0.35):
    source = workload['requests']
    assert 4 <= burst_count <= len(source) and 4 <= stream_count <= 78
    assert 4 <= outputs <= 256 and gap > 0
    def row(original, index, arrival):
        tokens = list(original['prompt_token_ids'])
        assert 0 < len(tokens) and len(tokens) + outputs <= 2048
        return dict(request_id=f'r{index:03d}', source_id=original['source_id'],
                    source_row=original['source_row'], source_request_id=original['request_id'],
                    bucket=original['bucket'], prompt_token_ids=tokens,
                    max_new_tokens=outputs, arrival_s=arrival)
    burst = [row(r, i, 0.0) for i, r in enumerate(source[:burst_count])]
    shorts = iter(r for r in source[:80] if r['bucket'] == 'short')
    longs = iter(r for r in source[:80] if r['bucket'] == 'long')
    stream = [row(next(shorts), 0, 0.0), row(next(shorts), 1, 0.0)]
    for i in range(2, stream_count):
        stream.append(row(next(longs if i % 2 == 0 else shorts), i, (i-1)*gap))
    return [dict(case='burst', requests=burst), dict(case='interference_stream', requests=stream)]


def warmup_case(workload):
    # 257 prompt tokens exercise the partially filled second page. Truncation
    # is diagnostic only; formal requests retain their original token IDs.
    short = next(r for r in workload['requests'] if r['bucket'] == 'short')
    long = next(r for r in workload['requests'] if r['bucket'] == 'long')
    return dict(case='warmup', requests=[
        dict(request_id='warm-a', prompt_token_ids=short['prompt_token_ids'][:31],
             max_new_tokens=4, arrival_s=0.0),
        dict(request_id='warm-b', prompt_token_ids=long['prompt_token_ids'][:257],
             max_new_tokens=4, arrival_s=0.0)])
