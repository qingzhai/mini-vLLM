"""Single-GPU canonical arithmetic benchmark, separate from the Flash results."""
import argparse
import atexit
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from metrics import request_metrics, summarize, compare_sequences
from benchmark_support import load_proof, warmup_requests, install_kv_budget

MODES = ('full_prefill', 'chunked', 'chunked_prefix')


def save_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temp.replace(path)


def check_source(repo):
    expected = json.loads((Path(__file__).parent / 'server-source-sha256.json').read_text())
    actual = {}
    for name, digest in expected.items():
        path = repo / name
        got = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if got != digest:
            raise RuntimeError(f'Server source changed: {name}; preserve the changes and send the new source')
        actual[name] = got
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--proof', required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        raise ValueError('repeats must be between 1 and 5')
    repo = Path(__file__).resolve().parents[2]
    run_dir = args.run_dir
    source_hashes = check_source(repo)
    workload = json.loads((run_dir / 'workload.json').read_text())
    from prepare_data import validate_workload
    validate_workload(workload)
    proof, prior_rows = load_proof(args.proof, workload)
    requests = workload['requests']
    if len(requests) != 100:
        raise ValueError('This workload must contain 100 requests')
    model_path = args.model or args.root / 'models/Qwen3-8B'
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'src'))
    import torch
    from transformers import AutoConfig
    from main import config as base_config
    from myvllm.engine.llm_engine import LLMEngine
    from myvllm.engine.sequence import Sequence
    from myvllm.engine.block_manager import BlockManager
    from myvllm.sampling_parameters import SamplingParams
    from canonical_runtime import install_runner_hooks

    class Greedy(torch.nn.Module):
        def reset(self):
            self.finite = torch.ones((), device='cuda', dtype=torch.bool)

        def forward(self, logits, temperatures):
            assert logits.dtype == torch.float32
            self.finite.logical_and_(torch.isfinite(logits).all())
            return logits.argmax(-1)

    if hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest() != workload['model']['config_sha256']:
        raise ValueError('Model config differs from the tokenizer workload manifest')
    hc = AutoConfig.from_pretrained(model_path, local_files_only=True)
    assert hc.model_type == 'qwen3' and model_path.name == 'Qwen3-8B'
    for req in requests:
        ids = req['prompt_token_ids']
        assert len(ids) == req['prompt_tokens'] and 256 <= len(ids) <= 1536
        assert all(isinstance(t, int) and 0 <= t < hc.vocab_size for t in ids)
    config = dict(base_config)
    config.update(model_name_or_path=str(model_path), world_size=1, enforce_eager=False,
                  vocab_size=hc.vocab_size, hidden_size=hc.hidden_size,
                  num_heads=hc.num_attention_heads, num_kv_heads=hc.num_key_value_heads,
                  head_dim=hc.head_dim, intermediate_size=hc.intermediate_size,
                  num_layers=hc.num_hidden_layers, tie_word_embeddings=hc.tie_word_embeddings,
                  rms_norm_epsilon=hc.rms_norm_eps, qkv_bias=hc.attention_bias,
                  ffn_bias=False, base=hc.rope_theta, scale=1.0, eos=hc.eos_token_id,
                  block_size=256, max_num_sequences=8, max_num_seqs=8,
                  max_num_batched_tokens=2048, max_num_batch_tokens=2048,
                  max_model_length=2048, max_position=2048, gpu_memory_utilization=.5,
                  enable_chunked_prefill=True, prefill_chunk_size=256,
                  chunk_decode_first=True, enable_prefix_caching=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_default_dtype(torch.bfloat16)
    install_runner_hooks()
    install_kv_budget(proof['kv_pages'])
    report = dict(schema_version=1, completed=False, measured_trials=[], warmups=[],
                  model=str(model_path), config=config, source_sha256=source_hashes,
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                   transformers=importlib.metadata.version('transformers'),
                                   triton=importlib.metadata.version('triton')),
                  workload_label=workload['workload_label'], repeats=args.repeats,
                  scope='Single-GPU canonical Qwen3 BF16 + fixed-row FP32 head; all-phase per-query PagedAttention; Graph decode',
                  proof_run=str(args.proof), performance_benchmark=True, diagnostic_top2_collection=False,
                  canonical_source_sha256=proof['canonical_source_sha256'],
                  limitation='Separate arithmetic reference configuration; does not validate old Flash Prefill performance',
                  metric_scope='In-process engine execution with 100 simultaneous arrivals; excludes HTTP/network/tokenization/model loading',
                  precision_note='Cross-mode exact greedy parity is reported separately; functional completion alone does not establish numerical equivalence',
                  sampling=dict(method='greedy', output_tokens=64, ignore_eos=True),
                  modes={m:dict(chunk_size=2048 if m=='full_prefill' else 256,
                                decode_first=m!='full_prefill', prefix_cache=m=='chunked_prefix') for m in MODES})
    report_path = run_dir / 'report.json'
    save_json(report_path, report)
    engine = None
    try:
        print('Initializing Qwen3-8B: BF16 core, FP32 LM head, batch limit 8, GPU 0', flush=True)
        engine = LLMEngine(config)
        torch.set_default_device('cpu')
        runner, scheduler = engine.model_runner, engine.scheduler
        runner.model.eval()
        assert all(p.dtype == torch.bfloat16 for p in runner.model.parameters() if p.is_floating_point())
        caches = [m for m in runner.model.modules() if hasattr(m, 'k_cache') and m.k_cache.numel()]
        assert caches and all(m.k_cache.dtype == m.v_cache.dtype == torch.bfloat16 for m in caches)
        report['canonical_runtime'] = runner._canonical_config
        report['lm_head'] = runner._canonical_config['head']
        total_pages = len(scheduler.block_manager.blocks)
        assert total_pages == proof['kv_pages'], 'KV budget differs from passed configuration'
        # Conservative admission capacity: 8 live requests with up to 1600 tokens each.
        assert total_pages >= 8 * math.ceil((1536 + 64) / 256), 'KV capacity insufficient for this workload'
        sampler = Greedy()
        sampler.reset()
        runner.sampler = sampler
        report['kv_pages'] = total_pages
        print('CONFIG READY:', dict(kv_pages=total_pages, **report['lm_head']), flush=True)
        params = SamplingParams(temperature=1.0, max_tokens=64, ignore_eos=True, max_model_length=2048)
        monitor = {}
        original_schedule = scheduler.schedule

        def schedule():
            result = original_schedule()
            monitor['peak_active'] = max(monitor.get('peak_active', 0), len(scheduler.running))
            monitor['peak_kv_pages'] = max(monitor.get('peak_kv_pages', 0), len(scheduler.block_manager.used_block_ids))
            return result
        scheduler.schedule = schedule

        def trial(mode, label, measured):
            assert scheduler.is_finished()
            scheduler.block_manager = BlockManager(total_pages, 256)
            bm = scheduler.block_manager
            bm.enable_prefix_caching = mode == 'chunked_prefix'
            scheduler.chunk_size = 2048 if mode == 'full_prefill' else 256
            scheduler.decode_first = mode != 'full_prefill'
            scheduler.trace.clear()
            scheduler.last_schedule = None
            monitor.clear()
            sampler.reset()
            seqs = [Sequence(r['prompt_token_ids'], 256, params) for r in requests]
            seq_by_id = {s.seq_id:s for s in seqs}
            req_by_id = {s.seq_id:r for s,r in zip(seqs, requests)}
            times = {s.seq_id:[] for s in seqs}
            # All 100 arrivals are at t=0; waiting-queue delay contributes to TTFT/E2E.
            for seq in seqs:
                scheduler.add_sequence(seq)
            checked_before = set(runner._graph_checked)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            print(f'{"MEASURE" if measured else "WARMUP"}: {label} mode={mode}', flush=True)
            started = time.perf_counter()
            next_progress = started + 15
            finished = 0
            try:
                while not scheduler.is_finished():
                    outputs, _, _ = engine.step()
                    # chunked_step already copies sampled IDs to CPU; synchronize makes timing explicit.
                    torch.cuda.synchronize()
                    now = time.perf_counter()
                    event = scheduler.last_schedule
                    for seq_id in event['sampled_seq_ids']:
                        observed = times[seq_id]
                        assert seq_by_id[seq_id].num_completion_tokens == len(observed)+1
                        observed.append(now-started)
                    finished += len(outputs)
                    if now >= next_progress:
                        print(f'PROGRESS: {label} {finished}/{len(requests)} requests, {now-started:.1f}s', flush=True)
                        next_progress = now + 15
                total = time.perf_counter()-started
                if not sampler.finite.item():
                    raise RuntimeError('Non-finite logits detected; timings are invalid')
                rows = [request_metrics(req, seq.completion_token_ids, times[seq.seq_id])
                        for req, seq in zip(requests, seqs)]
                assert all(r['output_tokens'] == 64 for r in rows)
                assert monitor['peak_active'] <= 8
                assert not bm.used_block_ids and len(bm.free_block_ids) == total_pages
                partial, mixed, processed, decode_steps, prefix_steps = 0, 0, 0, 0, 0
                for event in scheduler.trace:
                    assert event['tokens'] <= 2048
                    mixed += int(event['mixed'])
                    processed += event['tokens']
                    decode_steps += int(event['execution'] == 'existing_decode_path')
                    prefix_steps += int(event['execution'] == 'prefill_varlen')
                    for item in event['items']:
                        if item['phase'] == 'prefill' and item['end'] < item['prompt_tokens']:
                            partial += 1
                            assert item['completion_tokens_before'] == 0
                            assert item['seq_id'] not in event['sampled_seq_ids']
                        if mode != 'full_prefill' and item['phase'] == 'prefill':
                            assert item['end'] - item['start'] <= 256
                        item['request_id'] = req_by_id[item['seq_id']]['request_id']
                assert (partial == 0 and mixed == 0) if mode == 'full_prefill' else (partial > 0 and mixed > 0)
                if mode == 'chunked_prefix':
                    assert bm.stats['cache_hit_tokens'] > 0, 'No prefix reuse occurred'
                else:
                    assert bm.stats['cache_hit_tokens'] == 0
                if measured:
                    expected_hits = sum(((r['prompt_tokens']-1)//256)*256 for r in requests[80:])
                    assert bm.stats['cache_hit_tokens'] == (expected_hits if mode=='chunked_prefix' else 0)
                by_id = {r['request_id']:r for r in rows}
                replay_rows = [r for r in requests if r.get('repeated_from')]
                replay_matches = sum(by_id[r['request_id']]['output_token_ids'] ==
                                     by_id[r['repeated_from']]['output_token_ids'] for r in replay_rows)
                new_checks = sorted(set(runner._graph_checked)-checked_before)
                summary = summarize(rows, total)
                summary.update(mode=mode, label=label, peak_active_requests=monitor['peak_active'],
                               peak_kv_pages=monitor['peak_kv_pages'], kv_pages=total_pages,
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                               partial_prefill_chunks=partial, mixed_steps=mixed,
                               decode_steps=decode_steps, prefill_steps=prefix_steps,
                               processed_tokens=processed, cache_hit_tokens=bm.stats['cache_hit_tokens'],
                               cache_hit_prompt_token_fraction=bm.stats['cache_hit_tokens']/sum(r['prompt_tokens'] for r in requests),
                               cache_evictions=bm.stats['evictions'], all_pages_released=True,
                               finite_logits=True, new_graph_diagnostic_checks=len(new_checks),
                               timing_clean=len(new_checks)==0, replay_sequence_matches=replay_matches,
                               replay_requests=len(replay_rows))
                result = dict(summary=summary, requests=rows, graph_diagnostics_added=new_checks,
                              trace=scheduler.trace)
                save_json(run_dir/f'{label}.json', result)
                print('RESULT:', json.dumps(summary, ensure_ascii=False), flush=True)
                return dict(file=f'{label}.json', summary=summary)
            except Exception:
                save_json(run_dir/f'{label}-failure.json', dict(
                    mode=mode, error=traceback.format_exc(), trace=scheduler.trace,
                    partial_requests=[dict(request_id=req_by_id[s.seq_id]['request_id'],
                                           output_token_ids=s.completion_token_ids,
                                           token_times_s=times[s.seq_id]) for s in seqs]))
                raise

        # Canonical operators have fixed arithmetic sizes and table width 8.
        # Exercise every Graph batch key outside measured timing. Each synthetic
        # one-token sequence uses its own scratch page; subsequent cold-cache
        # trials overwrite valid KV before reading it.
        print('Warmup: Graph/eager keys batch=1..8, then 16 real requests per mode', flush=True)
        for bs in range(1,9):
            scratch = [Sequence([requests[0]['prompt_token_ids'][0]],256,params) for _ in range(bs)]
            for i,seq in enumerate(scratch):
                seq.block_table = [i]
            runner.run(scratch,False).cpu()
        assert runner._graph_checked == {(bs,8) for bs in range(1,9)}
        report['graph_keys_warmed'] = sorted(runner._graph_checked)
        full_requests = requests
        requests = warmup_requests(full_requests)
        for mode in MODES:
            entry = trial(mode, f'warmup-{mode}', False)
            report['warmups'].append(entry)
            save_json(report_path, report)
        requests = full_requests
        for repeat in range(1, args.repeats+1):
            order = MODES if repeat % 2 else tuple(reversed(MODES))
            for mode in order:
                entry = trial(mode, f'round{repeat}-{mode}', True)
                entry['repeat'] = repeat
                report['measured_trials'].append(entry)
                save_json(report_path, report)
        comparisons = []
        for repeat in range(1, args.repeats+1):
            baseline = json.loads((run_dir/f'round{repeat}-full_prefill.json').read_text())['requests']
            for mode in MODES[1:]:
                candidate = json.loads((run_dir/f'round{repeat}-{mode}.json').read_text())['requests']
                comparisons.append(dict(repeat=repeat, mode=mode, **compare_sequences(baseline,candidate)))
        report['greedy_comparisons'] = comparisons
        baseline_first = json.loads((run_dir/'round1-full_prefill.json').read_text())['requests']
        report['baseline_repeat_comparisons'] = [dict(repeat=i, **compare_sequences(baseline_first,
            json.loads((run_dir/f'round{i}-full_prefill.json').read_text())['requests']))
            for i in range(2,args.repeats+1)]
        report['all_exact_greedy'] = all(x['all_exact'] for x in comparisons)
        report['all_timings_clean'] = all(x['summary']['timing_clean'] for x in report['measured_trials'])
        report['proof_comparisons'] = []
        for trial_entry in report['measured_trials']:
            mode=trial_entry['summary']['mode']
            rows=json.loads((run_dir/trial_entry['file']).read_text())['requests']
            report['proof_comparisons'].append(dict(repeat=trial_entry['repeat'],mode=mode,
                                                   **compare_sequences(prior_rows[mode],rows)))
        report['matches_passed_configuration'] = all(x['all_exact'] for x in report['proof_comparisons'])
        report['all_replays_exact'] = all(x['summary']['replay_sequence_matches'] ==
                                         x['summary']['replay_requests'] == 20
                                         for x in report['measured_trials'])
        report['completed'] = True
        report['functional_checks_passed'] = True
        report['passed'] = (report['all_exact_greedy'] and report['all_timings_clean'] and
                            report['matches_passed_configuration'] and report['all_replays_exact'])
        save_json(report_path, report)
        from summarize import write_summary
        write_summary(run_dir)
        print('SHAREGPT FINAL MEASUREMENTS COMPLETE', flush=True)
        print('Greedy equality:', [(x['repeat'],x['mode'],x['exact_matches'],x['requests']) for x in comparisons], flush=True)
        if not report['all_exact_greedy']:
            print('OUTPUT DIFFERENCES RECORDED: benchmark completed; strict greedy equivalence is NOT established. See report.json.', flush=True)
        print('Results:', run_dir, flush=True)
        assert report['passed'], 'Final measurement failed a parity/timing guard; see saved report'
        print('SHAREGPT FINAL CHECKS PASS: strict greedy equality, prior-run equality, clean timing and KV release', flush=True)
    except Exception:
        report['completed'] = False
        report['error'] = traceback.format_exc()
        save_json(report_path, report)
        raise
    finally:
        if engine is not None:
            atexit.unregister(engine.exit)
            try:
                engine.exit()
            except Exception as error:
                print('Cleanup:', str(error), flush=True)


if __name__ == '__main__':
    main()
