"""Summarize measured trials without upgrading performance runs to accuracy passes."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

MODES = ('full_prefill', 'chunked', 'chunked_prefix')
METRICS = ('avg_ttft_ms','p95_ttft_ms','avg_tpot_ms','p95_tpot_ms',
           'avg_e2e_ms','p95_e2e_ms','output_tokens_per_s','requests_per_s',
           'total_s','peak_allocated_gib','cache_hit_tokens','cache_hit_prompt_token_fraction',
           'partial_prefill_chunks','mixed_steps','peak_active_requests')


def write_summary(run_dir):
    run_dir = Path(run_dir)
    report = json.loads((run_dir/'report.json').read_text())
    if not report.get('completed'):
        raise ValueError('The experiment is incomplete; partial raw reports remain available')
    trials = report['measured_trials']
    repeats = report['repeats']
    if len(trials) != repeats*3:
        raise ValueError('Missing measured trials')
    rows = []
    for mode in MODES:
        entries = [x for x in trials if x['summary']['mode'] == mode]
        if len(entries) != repeats or len({x['repeat'] for x in entries}) != repeats:
            raise ValueError(f'Missing/duplicate rounds for {mode}')
        data = [x['summary'] for x in entries]
        if any(x['requests'] != 100 or x['output_tokens'] != 6400 or
               not x['all_pages_released'] or not x['finite_logits'] for x in data):
            raise ValueError('Invalid measured trial')
        row = dict(mode=mode, repeats=repeats, requests=100, output_tokens_per_request=64,
                   **{k:median(x[k] for x in data) for k in METRICS})
        row['tokens_per_s_min'] = min(x['output_tokens_per_s'] for x in data)
        row['tokens_per_s_max'] = max(x['output_tokens_per_s'] for x in data)
        row['timing_clean'] = all(x['timing_clean'] for x in data)
        comparisons = [x for x in report['greedy_comparisons'] if x['mode'] == mode]
        if mode == 'full_prefill':
            row['exact_sequences_vs_baseline'] = 100
            row['strict_greedy_equivalence'] = True
        else:
            if len(comparisons) != repeats:
                raise ValueError('Missing output comparisons')
            row['exact_sequences_vs_baseline'] = min(x['exact_matches'] for x in comparisons)
            row['strict_greedy_equivalence'] = all(x['all_exact'] for x in comparisons)
        rows.append(row)
    base = rows[0]
    for row in rows:
        row['throughput_ratio_vs_full'] = row['output_tokens_per_s']/base['output_tokens_per_s']
        row['ttft_ratio_full_over_mode'] = base['avg_ttft_ms']/row['avg_ttft_ms']
        row['e2e_ratio_full_over_mode'] = base['avg_e2e_ms']/row['avg_e2e_ms']
        row['performance_comparison_valid'] = row['timing_clean'] and base['timing_clean']
    with (run_dir/'summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:round(v,6) if isinstance(v,float) else v for k,v in row.items()})
    table = ['| 配置 | 平均 TTFT ms | 平均 TPOT ms | 平均 E2E ms | 输出 tokens/s | 吞吐比值 | 基线序列一致 | 缓存命中 tokens |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append(f"| {row['mode']} | {row['avg_ttft_ms']:.2f} | {row['avg_tpot_ms']:.2f} | "
                     f"{row['avg_e2e_ms']:.2f} | {row['output_tokens_per_s']:.2f} | "
                     f"{row['throughput_ratio_vs_full']:.3f} | {row['exact_sequences_vs_baseline']}/100 | "
                     f"{row['cache_hit_tokens']:.0f} |")
    note = '' if report['all_exact_greedy'] else (
        '\n存在跨配置生成差异：已保存逐请求完整 token 和首个分叉位置。'
        '这些性能数字用于观察负载表现，尚不能作为严格输出等价优化的验收结果。\n')
    timing_note = '' if report['all_timings_clean'] else (
        '\n正式测量期间出现新的 CUDA Graph 诊断开销；相应比值不作为有效性能结论。\n')
    markdown = '\n'.join([
        '# Qwen3-8B ShareGPT 固定计算配置压测结果','',
        '单GPU；BF16 主干 + FP32 LM Head；并发上限8；每请求固定64个输出token，忽略EOS。',
        '固定尺寸 Linear/RMS/SiLU；所有阶段使用 per-query PagedAttention；纯 Decode 启用 CUDA Graph。'
        '此处 Prefill 未使用原 FlashAttention 内核，数据不能并入此前 Flash 配置的性能表。',
        '100个请求同时到达引擎等待队列。TTFT/E2E包含排队时间；不含HTTP、网络、分词和模型加载。',
        'ShareGPT派生受控负载：50条短输入256..512 tokens，50条长输入1024..1536 tokens；'
        '80个不同问题加20条完整prompt重放。20%重复请求与实际缓存命中率是不同指标。',
        (f'每种配置正式运行{repeats}轮，表中为每轮指标的中位数；预热轮已排除。' if repeats>1 else
         '每种配置正式运行1轮，表中为该次实测值；已排除预热和 Graph/eager 诊断开销，尚未测量运行间波动。'),'',
        *table,'',note,timing_note,
        '吞吐比值 = 本配置输出tokens/s ÷ 完整Prefill基线输出tokens/s；大于1才表示吞吐提升。',
        '三种配置都启用已有CUDA Graph解码；本实验对照Chunk和Prefix，不单独衡量Graph收益。',
        '这次测量范围为普通Qwen；AWQ、DFlash、TP已有各自独立实验记录。',
        '数值/模型质量与严格序列等价需要分别评估；功能状态检查不能代替模型质量评测。',
        '详细记录：workload.json、report.json、round*.json；数据集版本、原始样本ID、源码哈希均已记录。',''])
    (run_dir/'RESULTS.md').write_text(markdown)
    print('\n'+markdown,flush=True)
    return rows


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    write_summary(parser.parse_args().run_dir)
