"""Paired single-request Eager comparison using unchanged official DeepSpec loops."""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

from protocol import (MODES, first_mismatch, json_hash, model_fingerprint,
                      sha256, source_fingerprint, summarize, write_json)

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_inputs(lock_path):
    lock = read_json(lock_path)
    if source_fingerprint(lock['source']) != lock['source_sha256']:
        raise ValueError('DeepSpec source changed since preparation')
    workload_path = lock_path.parent / 'workload.json'
    if sha256(workload_path) != lock['workload_sha256']:
        raise ValueError('Workload changed since preparation')
    for name, model in lock['models'].items():
        print('VERIFY WEIGHTS:', name, flush=True)
        if model_fingerprint(model['path']) != model['sha256']:
            raise ValueError(f'Checkpoint files changed: {name}')
    workload = read_json(workload_path)
    if len({row['case'] for row in workload}) != len(workload) or not workload:
        raise ValueError('Cases must be nonempty and unique')
    for row in workload:
        if len(row['input_ids']) != row['input_tokens'] or json_hash(row['input_ids']) != row['input_ids_sha256']:
            raise ValueError(f'Invalid input token record: {row["case"]}')
    return lock, workload


def worker(config, algorithm):
    if config.get('target_backend', 'hf') == 'legacy_minivllm':
        sys.path.insert(0, read_json(config['lock'])['source'])
        import torch.distributed as dist
        from deepspec.eval import base_evaluator
        from legacy_target import legacy_arithmetic_context
        original_factory = base_evaluator.DynamicCache
        already_initialized = dist.is_initialized()
        try:
            with legacy_arithmetic_context():
                return _worker(config, algorithm)
        finally:
            base_evaluator.DynamicCache = original_factory
            if not already_initialized and dist.is_initialized():
                dist.destroy_process_group()
    return _worker(config, algorithm)


def _worker(config, algorithm):
    # HF 参考与归档 Target 适配后端分别标记，不混用其性能结论。
    lock = read_json(config['lock'])
    sys.path.insert(0, lock['source'])
    import torch
    from transformers import DynamicCache
    from deepspec.eval.dspark import Qwen3DSparkEvaluator
    from deepspec.eval.eagle3 import Qwen3Eagle3Evaluator
    from deepspec.utils import seed_all
    from deepspec.utils.sampling import logits_to_probs, sample_from_probs
    from types import SimpleNamespace
    from precision import configure_target_backbone, configure_target_head, install_draft_feature_bridge

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Expose exactly one BF16-capable NVIDIA GPU')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    if config.get('full_bf16_reduction', False):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if config.get('full_fp16_reduction', False):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    cls = Qwen3Eagle3Evaluator if algorithm == 'eagle3' else Qwen3DSparkEvaluator
    # 使用官方 build_models/generate_one_sample；跳过仅用于多 GPU 分发评测样本的 NCCL 初始化。
    evaluator = cls.__new__(cls)
    evaluator.args = SimpleNamespace(target_name_or_path=lock['models']['target']['path'],
        draft_name_or_path=lock['models'][algorithm]['path'], max_new_tokens=config['max_new_tokens'],
        temperature=0.0, confidence_threshold=0.0, tensorboard_dir=None, step=None)
    evaluator.device = torch.device('cuda:0')
    native = config.get('target_backend', 'hf') == 'legacy_minivllm'
    if native:
        from transformers import AutoTokenizer
        from deepspec.eval import base_evaluator
        from legacy_target import LegacyTarget
        evaluator.draft_model = cls.draft_model_cls.from_pretrained(lock['models'][algorithm]['path'],
            dtype=torch.bfloat16, attn_implementation=cls.EVAL_ATTN_IMPLEMENTATION,
            local_files_only=True).to(evaluator.device).eval()
        evaluator.target_model = LegacyTarget(lock['models']['target']['path'],
            evaluator.draft_model.target_layer_ids, config['output'], algorithm,
            logit_diagnostics=config.get('target_logit_diagnostics', False))
        evaluator.tokenizer = AutoTokenizer.from_pretrained(lock['models']['target']['path'], local_files_only=True)
        base_evaluator.assert_no_final_target_layer(evaluator.target_model, evaluator.draft_model.target_layer_ids)
        # 只替换官方生成函数内的 Target cache factory；Draft 模块自己的 DynamicCache 不变。
        base_evaluator.DynamicCache = evaluator.target_model.new_cache
        backbone_precision = dict(requested='fp16', backbone_dtype='torch.float16',
            loader='Archived strict loader into FP16 Target', checkpoint_files_modified=False)
        target_precision = evaluator.target_model.precision_record()
    else:
        evaluator.target_model, evaluator.draft_model, evaluator.tokenizer = evaluator.build_models()
        backbone_precision = configure_target_backbone(evaluator.target_model, config.get('target_backbone', 'checkpoint'))
        target_precision = configure_target_head(evaluator.target_model, config.get('target_head', 'checkpoint'))
    feature_bridge = (install_draft_feature_bridge(evaluator.draft_model)
                      if config.get('target_backbone') == 'fp16' else None)
    print('TARGET PRECISION:', target_precision, flush=True)
    print('TARGET BACKBONE:', backbone_precision, 'DRAFT FEATURE BRIDGE:', feature_bridge, flush=True)
    evaluator.confidence_head_recorder = None
    model = evaluator.draft_model
    if evaluator.max_proposal_tokens != 7:
        raise ValueError('Reviewed comparison requires seven proposed tokens, excluding pending')
    if algorithm == 'eagle3' and int(model.config.draft_num_hidden_layers) != 1:
        raise ValueError('Official EAGLE3 cache update requires a single draft layer')
    if algorithm == 'dflash' and (model.markov_head is not None or model.confidence_head is not None):
        raise ValueError('DFlash checkpoint unexpectedly enables DSpark heads')
    if algorithm == 'dspark' and (model.markov_head is None or model.confidence_head is None):
        raise ValueError('DSpark checkpoint must have Markov and confidence heads')
    # 消融期间仍持有全部权重，防止把不同驻留量混入计算开销对照。
    original_markov = getattr(model, 'markov_head', None)
    original_confidence = getattr(model, 'confidence_head', None)
    folder = Path(config['output'])
    workload = read_json(Path(config['lock']).parent / 'workload.json')
    props = torch.cuda.get_device_properties(0)
    write_json(folder / f'{algorithm}-environment.json', dict(
        gpu=props.name, gpu_total_gib=props.total_memory / 2**30,
        torch=torch.__version__, cuda=torch.version.cuda,
        python=sys.executable, python_base_prefix=sys.base_prefix, torch_source=torch.__file__,
        pyvenv_config=(Path(sys.prefix) / 'pyvenv.cfg').read_text() if (Path(sys.prefix) / 'pyvenv.cfg').exists() else None,
        packages={name: importlib.metadata.version(name) for name in
                  ('transformers', 'huggingface-hub', 'safetensors', 'triton', 'numpy')},
        target_dtype=str(next(evaluator.target_model.parameters()).dtype),
        draft_dtype=str(next(model.parameters()).dtype),
        target_precision=target_precision,
        backbone_precision=backbone_precision, draft_feature_bridge=feature_bridge,
        target_backend=config.get('target_backend', 'hf'),
        target_arithmetic=evaluator.target_model.arithmetic if native else None,
        legacy_source_identity=evaluator.target_model.source_identity if native else None,
        target_logit_diagnostics=config.get('target_logit_diagnostics', False),
        attention='Archived per-query PagedAttention' if native else evaluator.EVAL_ATTN_IMPLEMENTATION,
        draft_attention='SDPA MATH' if native else evaluator.EVAL_ATTN_IMPLEMENTATION,
        graph=False, batch_size=1,
        max_proposal_tokens=evaluator.max_proposal_tokens,
        target_layer_ids=list(map(int, model.target_layer_ids)),
        draft_config=model.config.to_dict(),
        target_parameters=sum(p.numel() for p in evaluator.target_model.parameters()),
        draft_parameters=sum(p.numel() for p in model.parameters()),
        target_parameter_bytes=sum(p.numel() * p.element_size() for p in evaluator.target_model.parameters()),
        extra_target_weight_bytes=evaluator.target_model.arithmetic['head_extra_bytes'] if native else 0,
        draft_parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        baseline_resident_draft=True, confidence_recorder=False))

    @torch.inference_mode()
    def generate_ar(ids, count):
        cache = evaluator.target_model.new_cache() if native else DynamicCache()
        generated = []
        current = ids
        for _ in range(count):
            output = evaluator.target_model(input_ids=current, past_key_values=cache,
                use_cache=True, output_hidden_states=True, logits_to_keep=1)
            current = sample_from_probs(logits_to_probs(output.logits, 0.0))
            generated.append(current)
        return SimpleNamespace(output_ids=torch.cat([ids, *generated], dim=1),
            num_output_tokens=count, verify_count=count - 1,
            proposal_lengths=[], accepted_draft_lengths=[], acceptance_lengths=[])

    @torch.inference_mode()
    def generate(ids, mode, count):
        if native:
            evaluator.target_model.trace = []
            evaluator.target_model.executor.reset_counters()
        evaluator.args.max_new_tokens = count
        evaluator.args.confidence_threshold = float(mode.rsplit('_', 1)[-1]) if mode.startswith('dspark_conf_') else 0.0
        try:
            if mode == 'ar':
                return generate_ar(ids, count)
            if algorithm == 'dspark':
                model.markov_head = None if mode == 'dspark_no_markov' else original_markov
                model.confidence_head = None if mode in ('dspark_no_markov', 'dspark_markov_only') else original_confidence
            return evaluator.generate_one_sample(input_ids=ids, stop_token_ids=None)
        finally:
            if algorithm == 'dspark':
                model.markov_head, model.confidence_head = original_markov, original_confidence
            if native:
                evaluator.target_model.release_cache()

    modes = MODES[algorithm]
    for mode in modes:
        print('WARMUP:', algorithm, mode, flush=True)
        ids = torch.tensor([workload[0]['input_ids']], device='cuda', dtype=torch.long)
        seed_all(lock['seed'])
        response = generate(ids, mode, config['warmup_tokens'])
        torch.cuda.synchronize()
        del response, ids
    records_path = folder / f'{algorithm}-records.jsonl'
    with records_path.open('x') as handle:
        for repeat in range(config['repeats']):
            order = list(workload)
            random.Random(lock['seed'] + repeat).shuffle(order)
            for index, case in enumerate(order):
                ids = torch.tensor([case['input_ids']], device='cuda', dtype=torch.long)
                offset = (repeat + index) % len(modes)
                for mode in (*modes[offset:], *modes[:offset]):
                    seed_all(lock['seed'] + repeat * 10000 + index)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    response = generate(ids, mode, config['max_new_tokens'])
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    # Token 导出、磁盘写入和哈希均在计时之外。
                    output_ids = response.output_ids[0, case['input_tokens']:].tolist()
                    if len(output_ids) != config['max_new_tokens']:
                        raise ValueError('Fixed output benchmark must return the exact token budget')
                    proposed = list(map(int, response.proposal_lengths))
                    accepted = list(map(int, response.accepted_draft_lengths))
                    advances = list(map(int, response.acceptance_lengths))
                    if mode != 'ar' and (not len(proposed) == len(accepted) == len(advances) == response.verify_count
                            or any(not 0 <= a <= p <= 7 or advance != a + 1
                                   for p, a, advance in zip(proposed, accepted, advances))):
                        raise ValueError('Unexpected official verification trajectory')
                    row = dict(algorithm=algorithm, mode=mode, repeat=repeat, case=case['case'],
                        target_head=config.get('target_head', 'checkpoint'),
                        target_backbone=config.get('target_backbone', 'checkpoint'),
                        target_backend=config.get('target_backend', 'hf'),
                        input_tokens=case['input_tokens'], output_tokens=len(output_ids), output_ids=output_ids,
                        output_sha256=json_hash(output_ids), total_s=seconds,
                        output_tokens_per_s=len(output_ids) / seconds, peak_allocated_gib=peak,
                        verify_count=int(response.verify_count),
                        target_forward_calls=1 + int(response.verify_count),
                        proposal_lengths=proposed, accepted_draft_lengths=accepted, acceptance_lengths=advances,
                        final_round_excess_tokens=max(0, 1 + sum(advances) - len(output_ids)) if mode != 'ar' else 0)
                    if native:
                        row['native_cache_released'] = evaluator.target_model.last_cache_released
                        row['target_execution'] = evaluator.target_model.executor.stats()
                        if config.get('target_logit_diagnostics'):
                            row['target_prediction_trace'] = evaluator.target_model.trace
                    handle.write(json.dumps(row) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                    print('RESULT:', algorithm, mode, repeat, case['case'],
                          f'{seconds:.3f}s {len(output_ids)/seconds:.2f} tokens/s', flush=True)
                    del response
                del ids
    write_json(folder / f'{algorithm}-completed.json', dict(completed=True))
    if native:
        evaluator.target_model.close()


def collect_report(folder, workload, repeats, errors):
    records = []
    for path in sorted(folder.glob('*-records.jsonl')):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    summary = summarize(records, cases=[x['case'] for x in workload], repeats=repeats)
    canonical = {}
    ar_divergences = []
    for row in records:
        if row['mode'] != 'ar':
            continue
        key = row['case']
        canonical.setdefault(key, row['output_ids'])
        mismatch = first_mismatch(canonical[key], row['output_ids'])
        if mismatch is not None:
            ar_divergences.append(dict(algorithm=row['algorithm'], repeat=row['repeat'],
                                      case=key, first_mismatch=mismatch))
    run_config = read_json(folder / 'run-config.json') if (folder / 'run-config.json').exists() else {}
    target_head = run_config.get('target_head', 'checkpoint')
    target_backbone = run_config.get('target_backbone', 'checkpoint')
    target_backend = run_config.get('target_backend', 'hf')
    precision_errors = []
    signatures = []
    for algorithm in MODES:
        path = folder / f'{algorithm}-environment.json'
        if not path.exists():
            if target_head != 'checkpoint':
                precision_errors.append(f'Missing precision environment: {algorithm}')
            continue
        environment_record = read_json(path)
        actual = environment_record.get('target_precision')
        if target_backend == 'legacy_minivllm' and (
                environment_record.get('target_backend') != target_backend
                or not environment_record.get('legacy_source_identity')
                or environment_record.get('target_arithmetic', {}).get('modules') !=
                   dict(linear=144, rms=145, activation=36, attention=36)):
            precision_errors.append(f'Archived Target source/arithmetic was not confirmed: {algorithm}')
        if actual is None:
            if target_head != 'checkpoint':
                precision_errors.append(f'Missing precision configuration: {algorithm}')
            continue
        if actual['requested'] != target_head or (target_head == 'fp32' and any(
                actual[key] != 'torch.float32' for key in
                ('head_weight_dtype', 'head_compute_dtype', 'head_output_dtype'))):
            precision_errors.append(f'Target precision differs from protocol: {algorithm}')
        if actual['tf32'] or actual['bf16_reduced_precision_reduction'] != (not run_config.get('full_bf16_reduction', False)):
            precision_errors.append(f'Target reduction settings differ from protocol: {algorithm}')
        if target_backbone == 'fp16' and actual['backbone_dtype'] != 'torch.float16':
            precision_errors.append(f'Target backbone is not FP16: {algorithm}')
        if target_backbone == 'fp16' and (environment_record.get('draft_dtype') != 'torch.bfloat16'
                                         or not environment_record.get('draft_feature_bridge')):
            precision_errors.append(f'BF16 Draft / FP16 feature bridge is missing: {algorithm}')
        if run_config.get('full_fp16_reduction') and actual.get('fp16_reduced_precision_reduction') is not False:
            precision_errors.append(f'FP16 reduced reduction was not disabled: {algorithm}')
        signatures.append((actual['head_weight_dtype'], actual['head_compute_dtype'],
                           actual['backbone_dtype'], actual['tf32'],
                           actual['bf16_reduced_precision_reduction'],
                           actual.get('fp16_reduced_precision_reduction')))
    if len(set(signatures)) > 1:
        precision_errors.append('Target precision differs across workers')
    if any(row.get('target_head', 'checkpoint') != target_head for row in records):
        precision_errors.append('Record precision differs from protocol')
    if any(row.get('target_backbone', 'checkpoint') != target_backbone for row in records):
        precision_errors.append('Record backbone differs from protocol')
    if any(row.get('target_backend', 'hf') != target_backend for row in records):
        precision_errors.append('Record Target backend differs from protocol')
    if target_backend == 'legacy_minivllm' and any(not row.get('native_cache_released') for row in records):
        precision_errors.append('Native Target cache was not released')
    qualified = not errors and not ar_divergences and not precision_errors and all(x['paired_greedy_match'] for x in summary)
    report = dict(completed=not errors, qualified_greedy_comparison=qualified,
        gpu_execution=bool(records), backend='legacy-target-official-draft-hybrid' if target_backend == 'legacy_minivllm' else 'official-deepspec-reference',
        measurement_repeats=repeats, cases=len(workload),
        scope=f'Single-request {target_backend} Target / {target_backbone} backbone / {target_head} head Eager; no Graph or concurrent serving',
        target_backend=target_backend, target_logit_diagnostics=run_config.get('target_logit_diagnostics', False),
        target_backbone=target_backbone, full_fp16_reduction=run_config.get('full_fp16_reduction', False),
        target_head=target_head, full_bf16_reduction=run_config.get('full_bf16_reduction', False),
        precision_errors=precision_errors,
        errors=errors, ar_divergences=ar_divergences, summary=summary)
    write_json(folder / 'report.json', report)
    fields = ['algorithm', 'mode', 'complete', 'paired_greedy_match', 'median_total_s',
              'median_tokens_per_s', 'median_paired_speedup', 'peak_allocated_gib',
              'verified_acceptance_ratio', 'output_tokens_per_verify']
    with (folder / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--warmup-tokens', type=int, default=32)
    parser.add_argument('--target-head', choices=('checkpoint', 'fp32'), default='checkpoint')
    parser.add_argument('--target-backbone', choices=('checkpoint', 'fp16'), default='checkpoint')
    parser.add_argument('--target-backend', choices=('hf', 'legacy_minivllm'), default='hf')
    parser.add_argument('--target-logit-diagnostics', action='store_true')
    parser.add_argument('--full-fp16-reduction', action='store_true')
    parser.add_argument('--full-bf16-reduction', action='store_true')
    parser.add_argument('--worker', choices=tuple(MODES))
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(read_json(args.config), args.worker)
        return
    if args.lock is None or args.output is None:
        parser.error('--lock and --output are required')
    if min(args.repeats, args.max_new_tokens, args.warmup_tokens) < 1:
        parser.error('Counts must be positive')
    if args.target_backend == 'legacy_minivllm' and not (
            args.target_backbone == 'fp16' and args.target_head == 'fp32'
            and args.full_fp16_reduction and args.full_bf16_reduction):
        parser.error('Legacy Target requires FP16 backbone, FP32 head, and both full reduction flags')
    lock, workload = validate_inputs(args.lock.resolve())
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    config = dict(lock=str(args.lock.resolve()), output=str(folder), repeats=args.repeats,
        max_new_tokens=args.max_new_tokens, warmup_tokens=args.warmup_tokens,
        target_head=args.target_head, full_bf16_reduction=args.full_bf16_reduction,
        target_backbone=args.target_backbone, full_fp16_reduction=args.full_fp16_reduction,
        target_backend=args.target_backend, target_logit_diagnostics=args.target_logit_diagnostics,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(folder / 'run-config.json', config)
    shutil.copy2(args.lock, folder / 'lock.json')
    shutil.copy2(args.lock.parent / 'workload.json', folder / 'workload.json')
    shutil.copytree(HERE, folder / 'harness-source', ignore=shutil.ignore_patterns('__pycache__'))
    # 保存实际外部源码及许可；权重与虚拟环境不打包。
    source_copy = folder / 'deepspec-source'
    source_copy.mkdir()
    for name in lock['source_sha256']:
        destination = source_copy / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(lock['source']) / name, destination)
    (folder / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    inventory = subprocess.run(['nvidia-smi', '-q'], text=True, capture_output=True)
    (folder / 'nvidia-smi.txt').write_text(inventory.stdout + inventory.stderr)
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       PYTHONDONTWRITEBYTECODE='1')
    errors = []
    try:
        for algorithm in MODES:
            with (folder / f'{algorithm}.log').open('w') as log:
                process = subprocess.Popen([sys.executable, '-u', str(HERE / 'benchmark.py'),
                    '--worker', algorithm, '--config', str(folder / 'run-config.json')],
                    env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                if code:
                    errors.append(dict(algorithm=algorithm, exit_status=code))
    except BaseException:
        errors.append(dict(orchestrator_error=traceback.format_exc()))
        raise
    finally:
        report = collect_report(folder, workload, args.repeats, errors)
        archive = shutil.make_archive(str(folder) + '-records', 'zip', folder)
        print('ARCHIVE:', archive, flush=True)
        print('QUALIFIED GREEDY COMPARISON:', report['qualified_greedy_comparison'], flush=True)
    if not report['qualified_greedy_comparison']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
