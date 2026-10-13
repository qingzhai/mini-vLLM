#!/usr/bin/env python3
"""Paired single/TP2/DP2 and two-GPU 1P1D experiments, no downloads."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
from common import build_cases, warmup_case, compare_tokens, write_json, digest_json
from runtime import Session, run_load

PACKAGE = Path(__file__).resolve().parent


def verify_manifest():
    from repro import verify_source
    return verify_source()


def summarize(run_dir):
    import csv
    modes = ['single', 'tp2', 'dp2', 'pd']
    reports = {m: json.loads((run_dir/f'{m}-details.json').read_text())
               for m in modes if (run_dir/f'{m}-details.json').is_file()}
    rows = []
    for mode in modes:
        for case in reports.get(mode, {}).get('cases', []):
            match = case.get('comparison', {})
            row = dict(case=case['case'], mode=mode, gpus=1 if mode == 'single' else 2,
                       functional_pass=reports[mode]['functional_pass'],
                       **case['metrics'], peak_active_requests=case['peak_active_requests'],
                       peak_rank0_gib=case['ranks'][0]['peak_allocated_gib'],
                       peak_rank1_gib=case['ranks'][1]['peak_allocated_gib'] if len(case['ranks']) > 1 else None,
                       exact_sequences=match.get('exact_sequences'), sequences=match.get('sequences'),
                       all_pages_released=case['all_pages_released'])
            transfer = case.get('transfers', [])
            row.update(kv_transfer_bytes=sum(t['bytes'] for t in transfer),
                       kv_handoff_total_s=sum(t['total_s'] for t in transfer),
                       kv_handoff_avg_ms=sum(t['total_s'] for t in transfer)*1000/len(transfer) if transfer else None)
            rows.append(row)
    by_key = {(r['mode'], r['case']): r for r in rows}
    for row in rows:
        base = by_key.get(('single', row['case']))
        dp = by_key.get(('dp2', row['case']))
        row['throughput_ratio_vs_single'] = row['output_tokens_per_s']/base['output_tokens_per_s'] if base else None
        row['throughput_ratio_vs_dp2'] = row['output_tokens_per_s']/dp['output_tokens_per_s'] if dp else None
    if rows:
        with (run_dir/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
    write_json(run_dir/'summary.json', rows)
    text = ['# Single / TP2 / DP2 / 1P1D 实验结果', '',
            '同一 FP16 Qwen3-8B + FP32 LM Head；固定计算路径，所有阶段使用 per-query PagedAttention。',
            '本轮关闭 Flash Prefill、CUDA Graph、AWQ、DFlash、Prefix、Chunk。每种负载一次正式测量。',
            '含 CPU 调度、worker RPC、排队、采样与 PD KV 传输；不含模型加载、分词、预热及诊断。', '',
            '| 负载 | 模式 | 总耗时 s | tokens/s | tokens/(GPU·s) | 平均 TTFT s | P99 ITL ms | 序列一致 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        exact = '基线' if r['mode'] == 'single' else f"{r['exact_sequences']}/{r['sequences']}"
        text.append(f"| {r['case']} | {r['mode']} | {r['total_s']:.3f} | {r['output_tokens_per_s']:.3f} | "
                    f"{r['output_tokens_per_gpu_s']:.3f} | {r['avg_ttft_s']:.3f} | {r['p99_itl_ms']:.3f} | {exact} |")
    text += ['', '## 验收范围', '',
             '- DP2、1P1D 必须与单卡生成全部 token 一致，页全部回收；未通过的报告保留但不能宣称性能通过。',
             '- TP2 使用相同历史下的 logits 容差检查；自由生成完整序列一致率单列。',
             '- 1P1D 预热包含跨物理页重映射、KV 哈希一致性；正式计时关闭哈希采集。D 卡不执行 Prefill。',
             '- PD 的第一个 token 在 P 卡算出后立即交给控制器；后续 token 的间隔包含 KV 迁移。',
             '- PD 与 DP2 都使用两张 GPU；不要将相对单卡的资源扩展当成每卡效率提升。',
             '- KV handoff 总耗时包含打包、两端等待、NCCL 传输和写入。发送与接收等待相互重叠，不能相加。',
             '- 各模式最大驻留请求总数相同；DP 每副本上限为总上限的一半，PD 上限包含等待交接的请求。',
             '- 合计逻辑 KV 容量相同；TP2 按 KV heads 分片，DP/PD 在两个完整副本间分配 KV 页。',
             '- 流式负载为真实 ShareGPT prompt 加受控到达时间，不能称为自然线上流量。P99 仅描述本次有限样本。',
             '- 这些是独立实验入口，没有验证与 AWQ/DFlash/Graph/Prefix/Chunk 同时启用。', '']
    (run_dir/'REPORT.md').write_text('\n'.join(text))
    print('\n'.join(text[:8+len(rows)]), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--modes', default='single,tp2,dp2,pd')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = verify_manifest()
    root = Path(os.environ.get('WORK_ROOT', PACKAGE.parents[1]/'.work'))
    from workload_support import load_workload
    workload = load_workload(Path(os.environ.get('SHAREGPT_WORKLOAD', root/'data/sharegpt-workload.json')))
    settings = dict(global_batch=int(os.environ.get('GLOBAL_BATCH', '4')),
                    burst_requests=int(os.environ.get('BURST_REQUESTS', '32')),
                    stream_requests=int(os.environ.get('STREAM_REQUESTS', '16')),
                    max_new_tokens=int(os.environ.get('OUTPUT_TOKENS', '64')),
                    stream_gap_s=float(os.environ.get('STREAM_GAP_S', '0.35')))
    assert settings['global_batch'] in (2, 4, 6, 8)
    settings['logical_kv_pages'] = 16*settings['global_batch']
    cases = build_cases(workload, settings['burst_requests'], settings['stream_requests'],
                        settings['max_new_tokens'], settings['stream_gap_s'])
    model_path = Path(os.environ.get('MODEL_PATH', root/'models/Qwen3-8B'))
    config = dict(**settings, model_path=str(model_path), core_dtype='float16', lm_head_dtype='float32',
                  repetitions=1, sampling='greedy', ignore_eos=True, block_size=256,
                  max_model_len=2048, source_manifest_sha256=digest_json(manifest),
                  original_sharegpt_dataset=workload['dataset'], workload_sha256=digest_json(cases),
                  prefill_batch_size=1, router='least resident requests, deterministic tie break',
                  cublas_workspace_config=':4096:8',
                  arrival_clock='controller monotonic clock; CPU/RPC included')
    if args.dry_run:
        print(json.dumps(dict(config=config, cases=[dict(case=c['case'], requests=len(c['requests']),
             input_tokens=sum(len(r['prompt_token_ids']) for r in c['requests'])) for c in cases]), indent=2))
        return
    assert (model_path/'config.json').is_file() and list(model_path.glob('*.safetensors')), 'Existing local Qwen3-8B weights not found'
    model_config = json.loads((model_path/'config.json').read_text())
    assert model_config['model_type'] == 'qwen3' and model_config['num_hidden_layers'] == 36
    assert model_config['vocab_size'] == workload['model']['vocab_size']
    configuration = run_dir/'config.json'
    if configuration.exists():
        assert json.loads(configuration.read_text()) == config, 'Resume configuration differs; use the original settings'
    else:
        write_json(configuration, config)
        write_json(run_dir/'workload.json', cases)
        source = run_dir/'source'
        from repro import snapshot_sources
        snapshot_sources(source)
    modes = args.modes.split(',')
    assert len(set(modes)) == len(modes) and all(m in ('single', 'tp2', 'dp2', 'pd') for m in modes)
    for mode in modes:
        destination = run_dir/f'{mode}-details.json'
        assert not destination.exists(), f'Refusing to overwrite {destination}; use a new run directory'
        reference_file = run_dir/'single-details.json'
        if mode != 'single':
            assert reference_file.is_file(), 'Run single baseline first in this run directory'
            baseline = json.loads(reference_file.read_text())
            assert baseline['completed'] and baseline['workload_sha256'] == config['workload_sha256']
        else:
            baseline = None
        report = dict(mode=mode, completed=False, functional_pass=False,
                      workload_sha256=config['workload_sha256'], config=config, cases=[])
        write_json(destination, report)
        try:
            with Session(mode, configuration, run_dir) as session:
                if mode in ('single', 'tp2'):
                    prompts = [r['prompt_token_ids'] for r in cases[0]['requests'][:2]]
                    command = dict(op='probe', prompts=prompts, output=str(run_dir/f'{mode}-probe.npz'))
                    if mode == 'tp2':
                        command['reference'] = str(run_dir/'single-probe.npz')
                    check = session.synchronous({0: command})[0]['numerical_check']
                    report['numerical_check'] = check
                    write_json(destination, report)
                    print(mode, 'SAME-HISTORY NUMERICAL CHECK:', check, flush=True)
                    assert check['passed'], 'Same-history FP16 tolerance failed; results retained'
                print(mode, 'WARMUP: Prefill, Decode, 257-token page boundary and PD handoff', flush=True)
                session.reset()
                warm = run_load(session, warmup_case(workload), config, diagnostic=(mode == 'pd'),
                                partial_path=run_dir/f'{mode}-warmup-partial.json')
                report['warmup'] = warm
                if baseline:
                    warm['comparison'] = compare_tokens(baseline['warmup']['requests'], warm['requests'])
                    if mode in ('dp2', 'pd'):
                        assert warm['comparison']['all_exact'], 'Warmup greedy parity failed; no formal timing run'
                write_json(destination, report)
                for index, case in enumerate(cases):
                    session.reset()
                    print('FORMAL:', mode, case['case'], len(case['requests']), 'requests', flush=True)
                    result = run_load(session, case, config, diagnostic=False,
                                      partial_path=run_dir/f'{mode}-{case["case"]}-partial.json')
                    if baseline:
                        result['comparison'] = compare_tokens(baseline['cases'][index]['requests'], result['requests'])
                    report['cases'].append(result)
                    write_json(destination, report)
                    print('RESULT:', mode, case['case'], result['metrics'], flush=True)
                    if mode in ('dp2', 'pd'):
                        assert result['comparison']['all_exact'], 'Full greedy parity failed; inspect saved request token IDs'
                report['completed'] = report['functional_pass'] = True
                write_json(destination, report)
            summarize(run_dir)
            print('MODE COMPLETE:', mode, flush=True)
        except Exception:
            report.update(error=traceback.format_exc(), completed=False, functional_pass=False)
            write_json(destination, report)
            summarize(run_dir)
            raise
    print('REQUESTED STAGES COMPLETE. Results:', run_dir, flush=True)


if __name__ == '__main__':
    main()
