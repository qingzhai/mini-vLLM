"""Single-GPU AR/DFlash × eager/target-graph experiment; no downloads."""
from __future__ import annotations
import csv
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback

PACKAGE = Path(__file__).resolve().parent
MODES = ('ar_eager', 'ar_graph', 'dflash_eager', 'dflash_graph')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def compare_generation(reference, actual, compare_trace=False):
    assert len(reference['requests']) == len(actual['requests'])
    checks = []
    for a, b in zip(reference['requests'], actual['requests']):
        assert a['request'] == b['request']
        left, right = a['token_ids'], b['token_ids']
        first = next((i for i, (x, y) in enumerate(zip(left, right)) if x != y), None)
        same = first is None and len(left) == len(right)
        check = dict(request=a['request'], tokens_match=same, first_mismatch=first,
                     expected_length=len(left), actual_length=len(right))
        if compare_trace:
            check['acceptance_trace_match'] = a['trace'] == b['trace']
        checks.append(check)
    summary_keys = ('accepted_draft_tokens', 'proposed_draft_tokens',
        'target_forward_calls', 'verification_batch_calls', 'max_verification_batch',
        'refill_while_active', 'rollback_pages', 'all_pages_released')
    summary_matches = {k: reference['summary'][k] == actual['summary'][k]
                       for k in summary_keys} if compare_trace else {}
    passed = all(c['tokens_match'] and c.get('acceptance_trace_match', True) for c in checks)
    return dict(passed=passed and all(summary_matches.values()), checks=checks,
                summary_matches=summary_matches)


def summarize(report):
    rows = []
    for case in report['cases']:
        for mode in MODES:
            runs = case['runs'][mode]
            if not runs:
                continue
            summaries = [r['summary'] for r in runs]
            med = lambda key: statistics.median(s[key] for s in summaries)
            row = dict(case=case['case'], mode=mode, repetitions=len(runs),
                requests=len(case['input_lengths']), output_tokens=summaries[0]['output_tokens'],
                median_total_s=med('total_s'), median_output_tokens_per_s=med('output_tokens_per_s'),
                median_avg_ttft_s=med('avg_ttft_s'), median_avg_e2e_s=med('avg_e2e_s'),
                median_avg_tpot_ms=med('avg_tpot_ms'),
                max_peak_allocated_gib=max(s['peak_allocated_gib'] for s in summaries),
                all_pages_released=all(s['all_pages_released'] for s in summaries),
                all_correct=all(r.get('correctness', {}).get('passed', False) for r in runs),
                graph_replays=[s['target_execution']['total_replays'] for s in summaries],
                total_s_min=min(s['total_s'] for s in summaries),
                total_s_max=max(s['total_s'] for s in summaries))
            reference = case['runs']['ar_eager']
            graph_ref = case['runs']['ar_graph']
            draft_ref = case['runs']['dflash_eager']
            if reference:
                row['throughput_ratio_vs_ar_eager'] = row['median_output_tokens_per_s']/statistics.median(
                    r['summary']['output_tokens_per_s'] for r in reference)
            if graph_ref:
                row['throughput_ratio_vs_ar_graph'] = row['median_output_tokens_per_s']/statistics.median(
                    r['summary']['output_tokens_per_s'] for r in graph_ref)
            if mode == 'dflash_graph' and draft_ref:
                row['throughput_ratio_vs_dflash_eager'] = row['median_output_tokens_per_s']/statistics.median(
                    r['summary']['output_tokens_per_s'] for r in draft_ref)
            rows.append(row)
    return rows


def save_report(run_dir, report):
    write_json(run_dir/'details.json', report)
    rows = summarize(report)
    write_json(run_dir/'summary.json', rows)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (run_dir/'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    text = ['# DFlash + CUDA Graph 四项对照', '',
        '单卡 A100，Qwen3-8B FP16 主干 + FP32 LM Head，BF16 DFlash Draft。',
        '所有阶段使用固定算术和 per-query PagedAttention；初始 Prefill、Draft、接受判断及回滚在图外。',
        '四模式使用同一 target executor，均收集五层特征；所有模式都保留 Draft 和四张 Graph，峰值显存不是独立部署显存。',
        '计时含模型前向、CPU 元数据、GPU 输出复制/读取、Draft、采样、队列与回滚；不含加载、捕获、预热及诊断。', '',
        '| Case | 模式 | 轮数 | 中位总耗时 s | 中位 tokens/s | 对 AR Graph 吞吐比 | 正确性 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        ratio = r.get('throughput_ratio_vs_ar_graph')
        text.append(f"| {r['case']} | {r['mode']} | {r['repetitions']} | {r['median_total_s']:.3f} | "
                    f"{r['median_output_tokens_per_s']:.3f} | {ratio:.3f} | {r['all_correct']} |" if ratio is not None else
                    f"| {r['case']} | {r['mode']} | {r['repetitions']} | {r['median_total_s']:.3f} | {r['median_output_tokens_per_s']:.3f} | 待测 | {r['all_correct']} |")
    text += ['', '只有相应吞吐比大于 1 才表示本轮测到了提升。此处 graph 指 target 前向图，不包括 Draft Graph。',
        '两个负载是固定的代码/解释/页边界提示词，不是 ShareGPT 压测；输出均为 greedy、忽略 EOS。',
        '本轮没有同时启用 TP、DP、PD 分离、AWQ、Prefix Cache 或 Chunked Prefill。',
        f"完成状态：{report['completed']}。"]
    (run_dir/'REPORT.md').write_text('\n'.join(text)+'\n')


def graph_diagnostics(target, executor, runtime, seed_tokens):
    """Eight same-history comparisons; values/page offsets change per graph."""
    import torch
    from state import PagePool, Request, Window
    checks = []
    for batch in (1, 2):
        for start in (250, 505):
            target.cache.zero_()
            pool = PagePool(target.pages, 256)
            reqs = []
            for key in range(batch):
                n = start + key
                tokens = (seed_tokens*((n+len(seed_tokens)-1)//len(seed_tokens)))[:n]
                reqs.append(Request(key, list(tokens), n))
            prefills = runtime.target_forward(reqs, [r.tokens for r in reqs], pool)
            for req, (picks, features) in zip(reqs, prefills):
                req.computed = len(req.tokens)
                req.tokens.append(picks[-1])
                req.features = features
            for query in (1, 16):
                blocks = [[req.tokens[-1]] + [seed_tokens[i % len(seed_tokens)] for i in range(query-1)]
                          for req in reqs]
                views = [Window(req, block, pool.reserve(req.key, req.computed+query))
                         for req, block in zip(reqs, blocks)]
                eager = executor.run(views, use_graph=False, diagnostic=True)
                graph = executor.run(views, use_graph=True, diagnostic=True)
                row = dict(batch=batch, query=query, starts=[r.computed for r in reqs],
                    pages=[v.block_table for v in views], tensors={})
                for key in ('logits', 'features', 'hidden', 'picks'):
                    a, b = eager[key], graph[key]
                    exact = torch.equal(a, b)
                    finite = bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
                    delta = (a.float()-b.float()).abs()
                    row['tensors'][key] = dict(exact=exact, finite=finite, max_abs=delta.max().item(),
                                               shape=list(a.shape))
                row['passed'] = all(v['exact'] and v['finite'] for v in row['tensors'].values())
                checks.append(row)
                print('GRAPH SAME-HISTORY CHECK:', row, flush=True)
            for req in reqs:
                pool.release(req.key)
            pool.check()
    target.cache.zero_()
    torch.cuda.synchronize()
    return dict(passed=all(r['passed'] for r in checks), checks=checks,
                rule='Exact tensor equality; no tolerance relaxation; diagnostics outside timing')


def main():
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from target import Target
    from graph_executor import TargetGraphExecutor
    from graph_runtime import GraphDFlashRuntime

    root = Path(os.environ['WORK_ROOT'])
    run_dir = Path(os.environ['RUN_DIR'])
    run_dir.mkdir(parents=True, exist_ok=True)
    from repro import REPO, verify_source, snapshot_sources
    manifest = verify_source()
    snapshot_sources(run_dir/'source')
    model_path = Path(os.environ.get('MODEL_PATH', root/'models/Qwen3-8B'))
    draft_path = Path(os.environ.get('DRAFT_MODEL_PATH', root/'models/Qwen3-8B-DFlash-b16'))
    count = int(os.environ.get('OUTPUT_TOKENS', '256'))
    rounds = int(os.environ.get('REPETITIONS', '3'))
    assert 64 <= count <= 512 and 1 <= rounds <= 5
    report = dict(completed=False, actual_gpu_execution=False, config=dict(
        modes=list(MODES), output_tokens=count, repetitions=rounds, max_batch=2, kv_pages=64,
        core_dtype='float16', head_dtype='float32', draft_dtype='bfloat16',
        target_graph_only=True, draft_graph=False, sampling='greedy', ignore_eos=True,
        initial_prefill_graph=False, target_attention='per-query PagedAttention for every query',
        fixed_linear_rows=32, fixed_rms_rows={'head_dim': 1024, 'hidden_dim': 32},
        target_features_in_all_modes=True, all_four_graphs_and_draft_resident_in_all_modes=True,
        formal_finite_logit_diagnostics=False, model_path=str(model_path), draft_path=str(draft_path),
        source_manifest_sha256=hashlib.sha256((REPO/'source-manifest.json').read_bytes()).hexdigest()), cases=[])
    write_json(run_dir/'config.json', report['config'])
    save = lambda: save_report(run_dir, report)
    save()
    executor = runtime = target = None
    try:
        assert torch.cuda.is_available(), 'CUDA unavailable'
        assert torch.cuda.device_count() == 1, 'Set CUDA_VISIBLE_DEVICES=0 for this single-GPU experiment'
        assert (model_path/'config.json').is_file() and (draft_path/'modeling_dflash.py').is_file(), 'Original local models are missing'
        cfg = json.loads((draft_path/'config.json').read_text())
        assert cfg['block_size'] == 16 and cfg['dflash_config']['target_layer_ids'] == [1,9,17,25,33]
        assert cfg['dflash_config']['mask_token_id'] == 151669
        torch.cuda.set_device(0)
        torch.set_default_device('cpu')
        torch.set_default_dtype(torch.float16)
        torch.manual_seed(42)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        report['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), device_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
        print('Loading existing Qwen3-8B FP16 target; strict loader; 64 KV pages...', flush=True)
        dist.init_process_group('nccl', init_method='file://'+str((run_dir/'nccl-init').resolve()),
            rank=0, world_size=1, timeout=timedelta(seconds=180), device_id=torch.device('cuda', 0))
        target = Target(model_path, 0, dist.group.WORLD, 64)
        report['actual_gpu_execution'] = True
        report['loading'], report['arithmetic'] = target.loading, target.arithmetic
        print('Loading existing BF16 DFlash Draft; no downloads...', flush=True)
        cls = get_class_from_dynamic_module('modeling_dflash.DFlashDraftModel', str(draft_path), local_files_only=True)
        draft = cls.from_pretrained(draft_path, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval()
        assert all(p.dtype == torch.bfloat16 for p in draft.parameters())
        source_record = root/'results/dflash-reference-source.json'
        if source_record.is_file():
            report['previous_draft_download_record'] = json.loads(source_record.read_text())
        report['draft_code_sha256'] = hashlib.sha256((draft_path/'modeling_dflash.py').read_bytes()).hexdigest()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        def encode(text):
            return tokenizer.apply_chat_template([dict(role='user', content=text)],
                add_generation_prompt=True, enable_thinking=False, tokenize=True)
        code = encode('Write a Python function implementing binary search. Include an explanation, edge cases, and several examples.')
        explain = encode('Explain how paged KV caching and continuous batching work together in an LLM inference engine. Give a detailed example.')
        boundary = encode('Background: '+'KV pages store attention keys and values. '*28+
                          'Explain how rejected speculative tokens should be rolled back.')
        cases = [('single-code', [code], 1), ('batch2-requests3', [code, explain, boundary], 2)]
        write_json(run_dir/'prompts.json', {name: inputs for name, inputs, _ in cases})
        executor = TargetGraphExecutor(target.model, [1,9,17,25,33])
        runtime = GraphDFlashRuntime(target, draft, executor)
        began = time.perf_counter()
        executor.capture_all()
        report['capture_s'] = time.perf_counter()-began
        report['capture'] = executor.stats()
        save()
        target.cache.zero_()
        report['graph_diagnostics'] = graph_diagnostics(target, executor, runtime, code)
        save()
        assert report['graph_diagnostics']['passed'], 'Eager/Graph exact forward parity failed; diagnostics saved'
        print('GRAPH FORWARD CHECKS PASS. Warmup all four modes, 32 output tokens...', flush=True)
        for mode in MODES:
            target.cache.zero_()
            runtime.graph_enabled = mode.endswith('_graph')
            runtime.run_label = 'warmup/'+mode
            runtime.generate([code, explain, boundary], max_new_tokens=32,
                max_batch_size=2, speculative=mode.startswith('dflash'))
        runtime.validate_tensors = False
        for name, inputs, batch in cases:
            row = dict(case=name, input_lengths=list(map(len, inputs)), max_batch=batch,
                runs={mode: [] for mode in MODES})
            report['cases'].append(row)
            for repetition in range(rounds):
                # Rotate order after the first round; all runs compare to the
                # first independent AR output, never feed reference tokens in.
                order = list(MODES[repetition % 4:]+MODES[:repetition % 4])
                for mode in order:
                    print(f'RUN: {name} {mode} repetition={repetition+1}/{rounds}', flush=True)
                    target.cache.zero_()
                    torch.cuda.synchronize()
                    torch.manual_seed(42)
                    runtime.graph_enabled = mode.endswith('_graph')
                    runtime.run_label = f'{name}/{mode}/r{repetition+1}'
                    actual = runtime.generate(inputs, max_new_tokens=count, max_batch_size=batch,
                                              speculative=mode.startswith('dflash'))
                    actual['repetition'] = repetition+1
                    summary = actual['summary']
                    summary['avg_ttft_s'] = statistics.mean(r['first_token_s'] for r in actual['requests'])
                    summary['avg_e2e_s'] = statistics.mean(r['finish_s'] for r in actual['requests'])
                    summary['avg_tpot_ms'] = statistics.mean((r['finish_s']-r['first_token_s'])/(count-1)*1000
                                                            for r in actual['requests'])
                    row['runs'][mode].append(actual)
                    reference = row['runs']['ar_eager'][0]
                    actual['correctness'] = compare_generation(reference, actual)
                    if mode.startswith('dflash') and row['runs']['dflash_eager']:
                        actual['same_speculative_trace'] = compare_generation(row['runs']['dflash_eager'][0], actual, True)
                        actual['correctness']['passed'] &= actual['same_speculative_trace']['passed']
                    save()
                    print('RESULT:', name, mode, summary, flush=True)
                    assert actual['correctness']['passed'], 'Greedy tokens or speculative state diverged; full details saved'
                    assert summary['all_pages_released'] and summary['output_tokens'] == len(inputs)*count
                    expected_replays = summary['verification_batch_calls'] if runtime.graph_enabled else 0
                    assert summary['target_execution']['total_replays'] == expected_replays, 'Some verification steps bypassed Graph'
                    assert summary['target_execution']['fallbacks'] == 0, 'Unexpected verification Graph fallback'
                    if mode.startswith('dflash'):
                        assert summary['accepted_draft_tokens'] > 0, 'No proposals accepted; not an effective speculative run'
                    save()
            print('CASE COMPLETE:', name, flush=True)
        report['completed'] = True
        save()
        print((run_dir/'REPORT.md').read_text(), flush=True)
        print('DFLASH + TARGET CUDA GRAPH FOUR-WAY CHECKS PASS', flush=True)
        print('Details:', run_dir/'details.json', flush=True)
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        save()
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if executor is not None:
            executor.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        main()
