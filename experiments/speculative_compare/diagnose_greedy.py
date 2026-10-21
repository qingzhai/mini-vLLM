"""Offline, untimed DSpark greedy reproduction with independent cache controls."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import traceback
from types import SimpleNamespace

from benchmark import validate_inputs
from protocol import MODES, first_mismatch, sha256, write_json


def commit_history(history, query, accepted):
    if not query or not 0 <= accepted < len(query):
        raise ValueError('Invalid accepted candidate prefix')
    return history + query[:accepted + 1]


def cache_geometry(cache):
    result = []
    for index, layer in enumerate(cache.layers):
        if layer.keys is None or layer.values is None:
            raise ValueError(f'Uninitialized cache layer {index}')
        result.append(dict(layer=index, length=int(layer.get_seq_length()),
            key_shape=list(layer.keys.shape), value_shape=list(layer.values.shape),
            key_stride=list(layer.keys.stride()), value_stride=list(layer.values.stride()),
            key_offset=layer.keys.storage_offset(), value_offset=layer.values.storage_offset()))
    return result


def require_cache_length(cache, expected, layers):
    geometry = cache_geometry(cache)
    if len(geometry) != layers or any(
            row['length'] != expected or row['key_shape'] != row['value_shape'] or
            len(row['key_shape']) != 4 or row['key_shape'][-2] != expected
            for row in geometry):
        raise ValueError(f'Cache length invariant failed: expected {layers} layers of length {expected}; {geometry}')
    return geometry


def tensor_difference(left, right):
    import torch
    if left.shape != right.shape:
        return dict(shape_match=False, left_shape=list(left.shape), right_shape=list(right.shape))
    difference = (left.float() - right.float()).abs()
    return dict(shape_match=True, exact=torch.equal(left, right),
        finite=bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
        max_abs=float(difference.max()), mean_abs=float(difference.mean()))


def compare_caches(left, right):
    if len(left.layers) != len(right.layers):
        raise ValueError('Cache layer counts differ')
    return [dict(layer=index, keys=tensor_difference(a.keys, b.keys),
                 values=tensor_difference(a.values, b.values))
            for index, (a, b) in enumerate(zip(left.layers, right.layers))]


def logit_details(logits, watch=(5980, 429)):
    import torch
    # Top-k 的并列顺序不能替代真正的 argmax。
    values = logits.detach().float().reshape(-1)
    best = int(torch.argmax(values))
    top = torch.topk(values, min(5, values.numel()))
    return dict(dtype=str(logits.dtype), argmax=best,
        max_value=float(values[best]),
        tied_max_ids=torch.nonzero(values == values[best], as_tuple=False).flatten().tolist(),
        topk_ids=top.indices.tolist(), topk_values=top.values.tolist(),
        watched={str(i): float(values[i]) for i in watch if i < values.numel()})


def response_record(response, prompt_length):
    return dict(output_ids=response.output_ids[0, prompt_length:].tolist(),
        proposal_lengths=list(map(int, response.proposal_lengths)),
        accepted_draft_lengths=list(map(int, response.accepted_draft_lengths)))


def run(args, report):
    # 在任何模型导入前启用离线模式；不调用 prepare/setup/run_gpu。
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    lock, workload = validate_inputs(args.lock.resolve())
    write_json(args.output / 'lock.json', lock)
    write_json(args.output / 'workload.json', workload)
    sys.path.insert(0, lock['source'])
    import torch
    import torch.nn.functional as F
    from transformers import DynamicCache
    from deepspec.eval import base_evaluator as base
    from deepspec.eval.dspark import Qwen3DSparkEvaluator
    from deepspec.utils import seed_all
    from deepspec.utils.sampling import logits_to_probs, sample_from_probs

    if torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.full_bf16_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    evaluator = Qwen3DSparkEvaluator.__new__(Qwen3DSparkEvaluator)
    evaluator.args = SimpleNamespace(target_name_or_path=lock['models']['target']['path'],
        draft_name_or_path=lock['models']['dspark']['path'], max_new_tokens=args.tokens,
        temperature=0.0, confidence_threshold=0.0, tensorboard_dir=None, step=None)
    evaluator.device = torch.device('cuda:0')
    evaluator.target_model, evaluator.draft_model, evaluator.tokenizer = evaluator.build_models()
    evaluator.confidence_head_recorder = None
    target, draft = evaluator.target_model, evaluator.draft_model
    if evaluator.max_proposal_tokens != 7 or draft.markov_head is None or draft.confidence_head is None:
        raise ValueError('Expected the reviewed DSpark block7 checkpoint')
    markov, confidence = draft.markov_head, draft.confidence_head

    report['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
        transformers=importlib.metadata.version('transformers'),
        gpu=torch.cuda.get_device_name(0), tf32=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        core_dtype=str(next(target.model.parameters()).dtype),
        head_dtype=str(target.lm_head.weight.dtype), seed=lock['seed'],
        model_paths={key: value['path'] for key, value in lock['models'].items()})

    @torch.inference_mode()
    def generate(ids, mode):
        if mode == 'ar':
            cache, current, generated = DynamicCache(), ids, []
            for _ in range(args.tokens):
                out = target(input_ids=current, past_key_values=cache, use_cache=True,
                             output_hidden_states=True, logits_to_keep=1)
                current = sample_from_probs(logits_to_probs(out.logits, 0.0))
                generated.append(current)
            return SimpleNamespace(output_ids=torch.cat([ids, *generated], dim=1),
                                   proposal_lengths=[], accepted_draft_lengths=[])
        evaluator.args.confidence_threshold = float(mode.rsplit('_', 1)[-1]) if mode.startswith('dspark_conf_') else 0.0
        draft.markov_head = None if mode == 'dspark_no_markov' else markov
        draft.confidence_head = None if mode in ('dspark_no_markov', 'dspark_markov_only') else confidence
        try:
            return evaluator.generate_one_sample(input_ids=ids, stop_token_ids=None)
        finally:
            draft.markov_head, draft.confidence_head = markov, confidence

    evidence = [json.loads(line) for line in args.evidence.read_text().splitlines() if line.strip()]
    historical = {row['mode']: row for row in evidence if row['case'] == args.case and row['repeat'] == 0}
    if set(historical) != set(MODES['dspark']):
        raise ValueError('Expected all six DSpark modes for the locked failure case')
    expected_ar = historical['ar']['output_ids'][:args.tokens]
    cases = list(workload)
    random.Random(lock['seed']).shuffle(cases)
    selected = next((row for row in cases if row['case'] == args.case), None)
    if selected is None:
        raise ValueError('Failure case absent from the locked workload')
    report['case'] = selected
    report['historical_evidence_sha256'] = sha256(args.evidence)
    modes = MODES['dspark']
    # 保留原 Smoke 的预热及失败样本之前的调用顺序，减小复现条件变化。
    warm_ids = torch.tensor([workload[0]['input_ids']], device='cuda')
    warm_count = evaluator.args.max_new_tokens
    evaluator.args.max_new_tokens = 32
    saved_tokens = args.tokens
    args.tokens = 32
    for mode in modes:
        seed_all(lock['seed'])
        generate(warm_ids, mode)
    args.tokens = saved_tokens
    evaluator.args.max_new_tokens = warm_count
    report['untraced'] = {}
    for index, case in enumerate(cases):
        ids = torch.tensor([case['input_ids']], dtype=torch.long, device='cuda')
        offset = index % len(modes)
        for mode in (*modes[offset:], *modes[:offset]):
            seed_all(lock['seed'] + index)
            response = generate(ids, mode)
            if case['case'] == args.case:
                row = response_record(response, case['input_tokens'])
                row['matches_historical'] = row['output_ids'] == historical[mode]['output_ids'][:args.tokens]
                row['first_mismatch_to_ar'] = first_mismatch(expected_ar, row['output_ids'])
                report['untraced'][mode] = row
        if case['case'] == args.case:
            case_seed = lock['seed'] + index
            break
    report['reproduced_historical_failure'] = (
        report['untraced']['ar']['output_ids'] == expected_ar and
        report['untraced']['dspark_conf_0.8']['output_ids'] == historical['dspark_conf_0.8']['output_ids'][:args.tokens] and
        report['untraced']['dspark_conf_0.8']['first_mismatch_to_ar'] == 26)
    write_json(args.output / 'report.json', report)

    trace, focus = [], {}
    history = list(selected['input_ids'])
    original_verify, original_propose = base.verify_draft_tokens, evaluator._propose
    latest = {}

    def traced_propose(**kwargs):
        context, start = kwargs['context'], kwargs['start']
        draft_length = context.past_key_values_draft.get_seq_length()
        feature_length = context.target_hidden_states.shape[1]
        if draft_length + feature_length != start:
            raise ValueError('Draft cache plus new target features must reach start')
        latest.clear()
        latest.update(draft_cache_before=int(draft_length), target_feature_length=int(feature_length))
        proposal = original_propose(**kwargs)
        latest['draft_cache_after_propose'] = require_cache_length(
            context.past_key_values_draft, start, int(draft.config.num_hidden_layers))
        return proposal

    def traced_verify(**kwargs):
        step = len(trace) + 1
        cache, start = kwargs['past_key_values_target'], kwargs['start']
        query = kwargs['proposal'].verify_input_ids[0].tolist()
        geometry = require_cache_length(cache, start, int(target.config.num_hidden_layers))
        if start != len(history) or query[:1] != kwargs['current_token_ids'][0].tolist():
            raise ValueError('Target history/pending invariant failed')
        row = dict(round=step, start=start, query=query,
                   positions=kwargs['position_ids'][0, start:start + len(query)].tolist(),
                   target_cache_before=geometry, **copy.deepcopy(latest))
        trace.append(row)
        if step == args.focus_round:
            focus.update(cache=copy.deepcopy(cache), history=list(history), query=list(query))

        def observed_target(**call):
            output = target(**call)
            row['first_prediction'] = logit_details(output.logits[0, 0])
            if step == args.focus_round:
                focus['logits'] = output.logits.detach().clone()
                focus['hidden_states'] = tuple(x.detach().clone() for x in output.hidden_states)
            return output

        result = original_verify(**dict(kwargs, target_model=observed_target))
        row['target_probability_argmax'] = int(result.target_probs[0, 0].argmax())
        row['accepted'] = int(result.accepted_draft_tokens)
        row['new_pending'] = int(result.next_token[0])
        if row['first_prediction']['argmax'] != row['target_probability_argmax']:
            raise ValueError('Logged logits and official greedy probabilities disagree')
        require_cache_length(cache, start + len(query), int(target.config.num_hidden_layers))
        history[:] = commit_history(history, query, row['accepted'])
        row['expected_crop_length'] = len(history)
        return result

    try:
        base.verify_draft_tokens, evaluator._propose = traced_verify, traced_propose
        seed_all(case_seed)
        ids = torch.tensor([selected['input_ids']], dtype=torch.long, device='cuda')
        response = generate(ids, 'dspark_conf_0.8')
        report['traced'] = response_record(response, selected['input_tokens'])
        report['trace_matches_untraced'] = report['traced'] == {
            key: report['untraced']['dspark_conf_0.8'][key] for key in report['traced']}
        report['trace'] = trace
    finally:
        base.verify_draft_tokens, evaluator._propose = original_verify, original_propose
        report['trace'] = trace
        write_json(args.output / 'report.json', report)
    if not focus:
        raise ValueError('Requested focus round was not reached')

    @torch.inference_mode()
    def call(tokens, cache, *, only_last=False):
        length = cache.get_seq_length()
        return target(input_ids=torch.tensor([tokens], dtype=torch.long, device='cuda'),
            position_ids=torch.arange(length, length + len(tokens), device='cuda').unsqueeze(0),
            past_key_values=cache, use_cache=True, output_hidden_states=True,
            **({'logits_to_keep': 1} if only_last else {}))

    with torch.inference_mode():
        prompt = selected['input_ids']
        focus_history, query = focus['history'], focus['query']
        focus_round = trace[args.focus_round - 1]
        report['focus_history_matches_ar'] = (focus_history + query[:1] ==
            prompt + expected_ar[:len(focus_history) - len(prompt) + 1])
        # 从同一 Prompt Prefill 开始，逐 Token 强制执行同一历史；不采样新历史。
        ar_cache = DynamicCache()
        call(prompt, ar_cache, only_last=True)
        for token in focus_history[len(prompt):]:
            call([token], ar_cache, only_last=True)
        replay_cache = DynamicCache()
        call(prompt, replay_cache, only_last=True)
        for row in trace[:args.focus_round - 1]:
            if replay_cache.get_seq_length() != row['start']:
                raise ValueError('Replay cache start differs from recorded start')
            call(row['query'], replay_cache)
            replay_cache.crop(row['expected_crop_length'])
        require_cache_length(ar_cache, len(focus_history), int(target.config.num_hidden_layers))
        require_cache_length(replay_cache, len(focus_history), int(target.config.num_hidden_layers))
        report['cache_comparisons'] = dict(
            live_vs_forced_ar=compare_caches(focus['cache'], ar_cache),
            live_vs_speculative_replay=compare_caches(focus['cache'], replay_cache))
        report['cache_geometries'] = dict(live=cache_geometry(focus['cache']),
            forced_ar=cache_geometry(ar_cache), speculative_replay=cache_geometry(replay_cache))
        controls = {}
        for name, cache, tokens in (
            ('same_cache_q_full', focus['cache'], query),
            ('same_cache_q1', focus['cache'], query[:1]),
            ('forced_ar_cache_q_full', ar_cache, query),
            ('forced_ar_cache_q1', ar_cache, query[:1]),
            ('speculative_replay_q_full', replay_cache, query)):
            output = call(tokens, copy.deepcopy(cache))
            controls[name] = dict(logits=output.logits.detach().clone(),
                                 hidden=tuple(x.detach().clone() for x in output.hidden_states))
        head32 = target.lm_head.weight.detach().float()
        report['controls'] = {}
        for name, output in controls.items():
            hidden = output['hidden'][-1]
            fp32_logits = F.linear(hidden.float(), head32)
            report['controls'][name] = dict(
                first_prediction=logit_details(output['logits'][0, 0]),
                difference_to_live_first_logits=tensor_difference(output['logits'][:, :1], focus['logits'][:, :1]),
                hidden_differences_by_layer=[tensor_difference(a[:, :1], b[:, :1])
                    for a, b in zip(output['hidden'], focus['hidden_states'])],
                same_hidden_fp32_head=logit_details(fp32_logits[0, 0]))
        real_fp32 = F.linear(focus['hidden_states'][-1].float(), head32)
        report['live_same_hidden_fp32_head'] = logit_details(real_fp32[0, 0])
        report['live_bf16_head_recompute'] = tensor_difference(
            F.linear(focus['hidden_states'][-1], target.lm_head.weight), focus['logits'])
        # 只保存小型观察张量；不把完整模型或几十 MB 的 Cache 打包。
        torch.save(dict(history=focus_history, query=query,
            live_logits=focus['logits'].cpu(), final_hidden=focus['hidden_states'][-1].cpu(),
            control_first_logits={name: row['logits'][:, :1].cpu() for name, row in controls.items()}),
            args.output / 'focus-tensors.pt')
        report['focus_round'] = focus_round
        report['diagnostic_complete'] = True
        report['performance_measurement'] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', default='humaneval:19')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--focus-round', type=int, default=5)
    parser.add_argument('--full-bf16-reduction', action='store_true')
    args = parser.parse_args()
    if args.tokens != 32 or args.focus_round < 1:
        parser.error('This reproducer freezes the original 32-token smoke; focus round must be positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, diagnostic_complete=False, performance_measurement=False,
                  args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    write_json(args.output / 'run-config.json', report['args'])
    for name in ('diagnose_greedy.py', 'benchmark.py', 'protocol.py'):
        source = Path(__file__).resolve().parent / name
        shutil.copy2(source, args.output / name)
    code = 0
    try:
        run(args, report)
    except BaseException:
        report['error'] = traceback.format_exc()
        print(report['error'], file=sys.stderr, flush=True)
        code = 1
    finally:
        write_json(args.output / 'report.json', report)
        archive = shutil.make_archive(str(args.output) + '-records', 'zip', root_dir=args.output)
        print('ARCHIVE:', archive, flush=True)
        print('REPRODUCED HISTORICAL FAILURE:', report.get('reproduced_historical_failure'), flush=True)
        print('TRACE MATCHES UNTRACED:', report.get('trace_matches_untraced'), flush=True)
        if report.get('focus_round'):
            print('FOCUS:', json.dumps(report['focus_round']['first_prediction']), flush=True)
        print('DIAGNOSTIC COMPLETE:', report['diagnostic_complete'], flush=True)
        print('This is a diagnostic; no benchmark or correctness pass is claimed.', flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
