"""Independent Chunk/Prefix switches; historical three-mode results stay separate."""
from statistics import median

PROTOCOL = 'sharegpt-chunk-prefix-2x2-v1'
MODES = ('full_prefill', 'chunked', 'prefix_only', 'chunked_prefix')
PAIRS = (
    ('full_prefill', 'prefix_only', 'prefix_without_chunk'),
    ('chunked', 'chunked_prefix', 'prefix_with_chunk'),
    ('full_prefill', 'chunked', 'chunk_without_prefix'),
    ('prefix_only', 'chunked_prefix', 'chunk_with_prefix'),
)


def settings(mode):
    if mode not in MODES:
        raise ValueError('Unknown factorial mode: ' + str(mode))
    chunk = mode in ('chunked', 'chunked_prefix')
    return dict(chunk_enabled=chunk, chunk_size=256 if chunk else 2048,
                prefix_cache=mode in ('prefix_only', 'chunked_prefix'),
                decode_first=True, atomic_full_prefill=not chunk)


def trial_order(repeat):
    if repeat < 1:
        raise ValueError('Repeat numbering starts at one')
    offset = (repeat - 1) % len(MODES)
    return MODES[offset:] + MODES[:offset]


def paired_effects(trials, repeats):
    by_key = {}
    for trial in trials:
        key = (trial['repeat'], trial['summary']['mode'])
        if key in by_key:
            raise ValueError('Duplicate measured trial: ' + str(key))
        by_key[key] = trial['summary']
    expected = {(r, m) for r in range(1, repeats + 1) for m in MODES}
    if set(by_key) != expected:
        raise ValueError('Missing/extra factorial rounds or modes')
    effects = []
    for baseline, candidate, name in PAIRS:
        ratios = []
        for repeat in range(1, repeats + 1):
            a, b = by_key[repeat, baseline], by_key[repeat, candidate]
            if a['output_tokens'] != 6400 or b['output_tokens'] != 6400:
                raise ValueError('Factorial trials must generate the same 6400 output tokens')
            if a['output_tokens_per_s'] <= 0 or b['output_tokens_per_s'] <= 0:
                raise ValueError('Throughput must be positive')
            ratios.append(b['output_tokens_per_s'] / a['output_tokens_per_s'])
        ratio = median(ratios)
        effects.append(dict(effect=name, baseline=baseline, candidate=candidate,
                            paired_throughput_ratios=ratios,
                            median_paired_throughput_ratio=ratio,
                            throughput_change_percent=(ratio - 1) * 100))
    return effects
