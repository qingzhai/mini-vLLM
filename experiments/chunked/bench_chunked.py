"""Offline engine test: late long prompts while short requests are decoding."""
import csv
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for location in (REPO, REPO / "src", REPO / "experiments/common", HERE):
    if str(location) in sys.path:
        sys.path.remove(str(location))
    sys.path.insert(0, str(location))
import torch
from transformers import AutoConfig, AutoTokenizer
from main import config as base_config
from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams
from fp32_head import install_fp32_head


class Greedy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ready = []
        self.records = {}
        self.force_template = None
        self.forced = {}
        self.capture_full_logits = False
        self.audit_logits = {}
        self.diagnostics = os.environ.get("CHUNK_LOGITS_DIAGNOSTICS", "1") == "1"

    def forward(self, logits, temperatures):
        assert logits.dtype == torch.float32, "Expected actual FP32 LM Head output"
        assert torch.isfinite(logits).all().item(), "Non-finite model logits"
        chosen = logits.argmax(-1)
        if self.diagnostics:
            assert len(self.ready) == logits.shape[0]
            values, ids = logits.float().topk(5, dim=-1)
            values, ids, chosen_ids = values.cpu().tolist(), ids.cpu().tolist(), chosen.cpu().tolist()
            for seq, scores, tokens, token in zip(self.ready, values, ids, chosen_ids):
                self.records.setdefault(seq.seq_id, []).append(dict(
                    output_position=seq.num_completion_tokens + 1,
                    chosen_token=token, top5_ids=tokens, top5_logits=scores,
                    top2_margin=scores[0] - scores[1]))
        if self.capture_full_logits:
            assert len(self.ready) == logits.shape[0]
            values = logits.detach().float().cpu().clone()
            for seq, row in zip(self.ready, values):
                self.audit_logits.setdefault(seq.seq_id, []).append(row)
        if self.forced:
            assert len(self.ready) == logits.shape[0]
            return torch.tensor([self.forced[seq.seq_id][seq.num_completion_tokens]
                                 for seq in self.ready], device=logits.device, dtype=torch.long)
        return chosen


def compare_fixed_histories(full, chunk, template):
    """Diagnostic metrics only; no tolerance is selected from observed results."""
    rows, mismatches = [], []
    for label, expected in template.items():
        a_rows, b_rows = full["audit_logits"][label], chunk["audit_logits"][label]
        assert len(a_rows) == len(b_rows) == len(expected) == 32
        assert next(x for x in full["requests"] if x["request"] == label)["token_ids"] == expected
        assert next(x for x in chunk["requests"] if x["request"] == label)["token_ids"] == expected
        for index, (a, b) in enumerate(zip(a_rows, b_rows)):
            assert a.dtype == b.dtype == torch.float32 and a.shape == b.shape
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            delta = a - b
            ta, tb = int(a.argmax().item()), int(b.argmax().item())
            va, ia = a.topk(2)
            vb, ib = b.topk(2)
            denom = max(a.norm().item(), 1e-30)
            row = dict(request=label, output_position=index+1,
                       max_abs=delta.abs().max().item(), mean_abs=delta.abs().mean().item(),
                       relative_l2=delta.norm().item()/denom,
                       cosine=torch.nn.functional.cosine_similarity(a[None], b[None]).item(),
                       argmax_match=ta == tb, full_top2_ids=ia.tolist(), chunk_top2_ids=ib.tolist(),
                       full_top2_logits=va.tolist(), chunk_top2_logits=vb.tolist(),
                       full_top2_margin=(va[0]-va[1]).item(), chunk_top2_margin=(vb[0]-vb[1]).item(),
                       emitted_reference_token=expected[index])
            rows.append(row)
            if ta != tb:
                mismatches.append(row)
    summary = dict(positions=len(rows), argmax_matches=sum(x["argmax_match"] for x in rows),
                   max_abs=max(x["max_abs"] for x in rows),
                   max_relative_l2=max(x["relative_l2"] for x in rows),
                   min_cosine=min(x["cosine"] for x in rows),
                   acceptance_threshold_applied=False,
                   free_running_greedy_pass=False)
    return dict(summary=summary, argmax_mismatches=mismatches, positions=rows)


@torch.inference_mode()
def main():
    root = Path(os.environ["WORK_ROOT"])
    model_path = Path(os.environ.get("MODEL_PATH", str(root / "models/Qwen3-8B")))
    kernel_checks = dict(previous_dispatch_check="passed for H=16 and H=32 in prior GPU run")
    dtype_name = os.environ.get("CORE_DTYPE", "bfloat16")
    assert dtype_name == "bfloat16", "This entry preserves the verified BF16 backbone + FP32 head + Graph configuration"
    core_dtype = getattr(torch, dtype_name)
    eager = dtype_name == "float16"
    tag = f"{dtype_name}-fp32head-{'eager' if eager else 'graph'}"
    for relative, marker in (
        ("src/myvllm/engine/model_runner.py", "# mixed_decode_metadata_v1"),
        ("src/myvllm/engine/chunk_runtime.py", "# chunk_lm_head_rows_v2"),
        ("src/myvllm/layers/layernorm.py", "# rms_fp32_v1"),
    ):
        assert marker in (REPO / relative).read_text(), f"Missing prerequisite: {marker}"
    hc = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    assert hc.model_type == "qwen3"
    config = dict(base_config)
    config.update(
        model_name_or_path=str(model_path), world_size=1, enforce_eager=eager,
        vocab_size=hc.vocab_size, hidden_size=hc.hidden_size,
        num_heads=hc.num_attention_heads, num_kv_heads=hc.num_key_value_heads,
        head_dim=hc.head_dim, intermediate_size=hc.intermediate_size,
        num_layers=hc.num_hidden_layers, tie_word_embeddings=hc.tie_word_embeddings,
        rms_norm_epsilon=hc.rms_norm_eps, qkv_bias=hc.attention_bias,
        ffn_bias=False, base=hc.rope_theta, scale=1.0, eos=hc.eos_token_id,
        block_size=256, max_num_sequences=4, max_num_seqs=4,
        max_num_batched_tokens=2048, max_num_batch_tokens=2048,
        max_model_length=2048, max_position=2048, gpu_memory_utilization=0.5,
        enable_chunked_prefill=True, prefill_chunk_size=256,
        chunk_decode_first=True, enable_prefix_caching=False,
    )
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(core_dtype)
    print("MODEL:", model_path, flush=True)
    engine = LLMEngine(config)
    torch.set_default_device("cpu")
    engine.model_runner.model.eval()
    model = engine.model_runner.model
    assert all(p.dtype == core_dtype for p in model.parameters() if p.is_floating_point()), (
        "Actual model parameter dtype differs from the requested precision")
    caches = [m for m in model.modules() if hasattr(m, "k_cache") and m.k_cache.numel()]
    assert caches and all(m.k_cache.dtype == m.v_cache.dtype == core_dtype for m in caches)
    head_info = install_fp32_head(model)
    print("HEAD PRECISION CONFIG:", dict(core_dtype=str(core_dtype), eager=eager,
                                        tf32=False, **head_info), flush=True)
    sampler = Greedy()
    engine.model_runner.sampler = sampler
    original_prepare_sample = engine.model_runner.prepare_sample
    def prepare_sample(seqs):
        sampler.ready = seqs
        return original_prepare_sample(seqs)
    engine.model_runner.prepare_sample = prepare_sample
    norms = [m for m in engine.model_runner.model.modules() if m.__class__.__name__ == "LayerNorm"]
    assert len(norms) == 4 * hc.num_hidden_layers + 1
    assert all(m.eps == hc.rms_norm_eps for m in norms)
    print("PRECISION CONFIG:", dict(rms_accumulation="fp32", rms_epsilon=hc.rms_norm_eps,
                                    norm_layers=len(norms), bf16_reduced_reduction=False,
                                    logits_diagnostics=sampler.diagnostics), flush=True)
    if sampler.diagnostics:
        print("TIMING NOTE: candidate-logit collection is enabled; this is a diagnostic run", flush=True)
    scheduler = engine.scheduler
    manager = scheduler.block_manager
    params = SamplingParams(temperature=1.0, max_tokens=32,
                            ignore_eos=True, max_model_length=2048)

    system = "You are a helpful assistant. "
    sentence = "A language model reuses cached attention keys and values during decoding. "
    while len(tokenizer.encode(system)) < 1400:
        system += sentence

    def prompt(index, long):
        messages = [dict(role="system", content=system if long else "You are a helpful assistant."),
                    dict(role="user", content=f"Request {index}: Explain KV caching in one short sentence.")]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text)
        assert 0 < len(ids) < 1800
        return ids

    inputs = [prompt(0, False), prompt(1, False), prompt(2, True), prompt(3, True)]
    labels = ["short0", "short1", "long0", "long1"]
    result_dir = Path(os.environ["RUN_DIR"])
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / f"{model_path.name}-{tag}-details.json"
    assert not report_path.exists(), f"Refusing to overwrite {report_path}; use a new RUN_DIR"
    report = dict(model=str(model_path), experiment=tag, kernel_checks=kernel_checks,
                  actual_precision=dict(core_dtype=str(core_dtype), eager=eager, tf32=False, **head_info),
                  workload="2 short prompts first; 2 long prompts arrive after their first tokens",
                  output_tokens=32, max_active_requests=4, token_budget=2048,
                  modes=dict(full_prefill=dict(chunk_size=2048, decode_first=False, prefix_cache=False),
                             chunked_mixed=dict(chunk_size=256, decode_first=True, prefix_cache=False)),
                  diagnostic_timing=sampler.diagnostics,
                  precision=dict(rms_accumulation="fp32", rms_epsilon=hc.rms_norm_eps,
                                 bf16_reduced_reduction=False),
                  numerical_caveat="Prior prefix model-logits discrepancy remains unresolved; this run tests greedy sequences and state invariants")

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    def set_mode(chunk, decode_first, caching=False):
        assert scheduler.is_finished()
        manager.clear_cache()
        manager.enable_prefix_caching = caching
        scheduler.chunk_size = chunk
        scheduler.decode_first = decode_first
        scheduler.trace.clear()
        scheduler.last_schedule = None
        sampler.records.clear()
        sampler.forced.clear()
        sampler.audit_logits.clear()

    def workload(mode, warmup=False):
        set_mode(2048 if mode == "full_prefill" else 256,
                 decode_first=mode != "full_prefill")
        seqs = [Sequence(token_ids=ids, block_size=256, sampling_params=params) for ids in inputs]
        if sampler.force_template is not None:
            sampler.forced = {seq.seq_id: sampler.force_template[label]
                              for seq, label in zip(seqs, labels)}
        arrivals, token_times = {}, {s.seq_id: [] for s in seqs}
        torch.cuda.synchronize()
        started = time.perf_counter()

        def add(index):
            arrivals[seqs[index].seq_id] = time.perf_counter() - started
            scheduler.add_sequence(seqs[index])

        add(0)
        add(1)
        injected = False
        peak_active = 0
        while not scheduler.is_finished():
            engine.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak_active = max(peak_active, len(scheduler.running))
            for seq in seqs:
                observed = token_times[seq.seq_id]
                count = seq.num_completion_tokens
                assert count in (len(observed), len(observed) + 1)
                if count > len(observed):
                    observed.append(elapsed)
            if not injected and all(token_times[seqs[i].seq_id] for i in (0, 1)):
                add(2)
                add(3)
                injected = True
        total = time.perf_counter() - started
        assert injected and peak_active <= 4
        rows = []
        id_to_label = {s.seq_id: label for s, label in zip(seqs, labels)}
        trace = json.loads(json.dumps(scheduler.trace))
        for event in trace:
            for item in event["items"]:
                item["request"] = id_to_label[item["seq_id"]]
                if item["phase"] == "prefill" and item["end"] < item["prompt_tokens"]:
                    assert item["completion_tokens_before"] == 0
                    assert item["seq_id"] not in event["sampled_seq_ids"]
        for label, seq in zip(labels, seqs):
            times = token_times[seq.seq_id]
            assert len(times) == 32
            gaps = [b - a for a, b in zip(times, times[1:])]
            rows.append(dict(request=label, prompt_tokens=seq.num_prompt_tokens,
                             ttft_ms=round((times[0] - arrivals[seq.seq_id]) * 1000, 3),
                             e2e_ms=round((times[-1] - arrivals[seq.seq_id]) * 1000, 3),
                             tpot_ms=round(mean(gaps) * 1000, 3),
                             max_inter_token_ms=round(max(gaps) * 1000, 3),
                             token_ids=seq.completion_token_ids,
                             sampler_top5=sampler.records.get(seq.seq_id, [])))
        partial = sum(x["phase"] == "prefill" and x["end"] < x["prompt_tokens"]
                      for e in trace for x in e["items"])
        mixed = sum(e["mixed"] for e in trace)
        if mode == "chunked_mixed":
            assert partial > 0 and mixed > 0
            assert all(x["end"] - x["start"] <= 256
                       for e in trace for x in e["items"] if x["phase"] == "prefill")
        else:
            assert partial == 0 and mixed == 0
        assert not manager.used_block_ids
        assert len(manager.free_block_ids) == len(manager.blocks)
        summary = dict(mode=mode, requests=4,
                       short_avg_ttft_ms=round(mean(x["ttft_ms"] for x in rows[:2]), 3),
                       short_avg_tpot_ms=round(mean(x["tpot_ms"] for x in rows[:2]), 3),
                       short_max_inter_token_ms=max(x["max_inter_token_ms"] for x in rows[:2]),
                       long_avg_ttft_ms=round(mean(x["ttft_ms"] for x in rows[2:]), 3),
                       output_tokens_per_s=round(128 / total, 3),
                       total_s=round(total, 4), partial_prefill_chunks=partial,
                       mixed_steps=mixed, peak_active_requests=peak_active)
        if not warmup:
            print("RESULT:", summary, flush=True)
        result = dict(summary=summary, requests=rows, trace=trace)
        if sampler.capture_full_logits:
            result["audit_logits"] = {label: sampler.audit_logits[seq.seq_id].copy()
                                      for label, seq in zip(labels, seqs)}
        return result

    print("Warmup: full and chunked workload", flush=True)
    workload("full_prefill", warmup=True)
    workload("chunked_mixed", warmup=True)
    report["full_prefill"] = workload("full_prefill")
    save()
    report["chunked_mixed"] = workload("chunked_mixed")
    comparisons = []
    for old, new in zip(report["full_prefill"]["requests"], report["chunked_mixed"]["requests"]):
        check = dict(request=old["request"], greedy_tokens_match=old["token_ids"] == new["token_ids"],
                     matching_positions=sum(a == b for a, b in zip(old["token_ids"], new["token_ids"])),
                     output_tokens=32)
        if not check["greedy_tokens_match"]:
            index = next(i for i, (a, b) in enumerate(zip(old["token_ids"], new["token_ids"])) if a != b)
            check["first_difference"] = dict(
                output_position=index + 1,
                full_token=old["token_ids"][index], chunk_token=new["token_ids"][index],
                full_text=tokenizer.decode([old["token_ids"][index]]),
                chunk_text=tokenizer.decode([new["token_ids"][index]]))
            if sampler.diagnostics:
                check["first_difference"]["full_top5"] = old["sampler_top5"][index]
                check["first_difference"]["chunk_top5"] = new["sampler_top5"][index]
            for mode in ("full_prefill", "chunked_mixed"):
                event = next(e for e in report[mode]["trace"] if any(
                    x["request"] == old["request"] and x["completion_tokens_before"] == index
                    and x["seq_id"] in e["sampled_seq_ids"] for x in e["items"]))
                check["first_difference"][mode + "_execution"] = event["execution"]
        comparisons.append(check)
        print("GREEDY CHECK:", check, flush=True)
    report["greedy_comparisons"] = comparisons
    save()
    summary = [report[mode]["summary"] for mode in ("full_prefill", "chunked_mixed")]
    csv_path = result_dir / f"{model_path.name}-{tag}-summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    greedy_pass = all(x["greedy_tokens_match"] for x in comparisons)
    if not greedy_pass:
        print("TEACHER-FORCED AUDIT: compare logits under identical emitted histories; "
              "forced token equality is NOT a greedy correctness pass", flush=True)
        sampler.force_template = {row["request"]: row["token_ids"]
                                  for row in report["full_prefill"]["requests"]}
        sampler.capture_full_logits = True
        try:
            fixed_full = workload("full_prefill", warmup=True)
            fixed_chunk = workload("chunked_mixed", warmup=True)
            audit = compare_fixed_histories(fixed_full, fixed_chunk, sampler.force_template)
            report["teacher_forced_audit"] = audit
            save()
            print("TEACHER-FORCED SUMMARY:", audit["summary"], flush=True)
            print("TEACHER-FORCED MISMATCHES (first 8; all saved in JSON):",
                  audit["argmax_mismatches"][:8], flush=True)
        finally:
            sampler.force_template = None
            sampler.capture_full_logits = False
            sampler.forced.clear()
            sampler.audit_logits.clear()
    assert greedy_pass, "Full/chunk greedy outputs differ; teacher-forced diagnostics saved, do not claim correctness passed"

    print("Combined prefix-cache + chunked-prefill check", flush=True)
    set_mode(64, True, caching=True)

    def single(ids):
        assert scheduler.is_finished()
        seq = Sequence(token_ids=ids, block_size=256, sampling_params=params)
        scheduler.add_sequence(seq)
        while not scheduler.is_finished():
            engine.step()
        return seq.completion_token_ids.copy()

    seed = single(inputs[2])
    scheduler.trace.clear()
    before = manager.stats["cache_hit_tokens"]
    repeated = single(inputs[2])
    cached = manager.stats["cache_hit_tokens"] - before
    trace = json.loads(json.dumps(scheduler.trace))
    prefills = [x for e in trace for x in e["items"] if x["phase"] == "prefill"]
    combined = dict(cached_tokens=cached, prefill_chunks=len(prefills),
                    first_chunk_start=prefills[0]["start"],
                    greedy_tokens_match=seed == repeated,
                    free_pages=len(manager.free_block_ids), total_pages=len(manager.blocks),
                    trace=trace)
    report["combined"] = combined
    save()
    assert cached >= 256 and combined["first_chunk_start"] == cached
    assert len(prefills) >= 2 and seed == repeated
    assert not manager.used_block_ids and combined["free_pages"] == combined["total_pages"]
    report["validation"] = dict(previous_bf16_greedy_discrepancy_resolved=False,
                                 tested_configuration=tag,
                                 no_intermediate_prefill_sampling=True,
                                 mixed_batches_observed=True,
                                 four_greedy_sequences_match=True,
                                 combined_prefix_chunk_check_pass=True,
                                 all_kv_pages_released=True,
                                 prior_logits_numerical_issue_resolved=False)
    save()
    print("COMBINED CHECK:", {k: v for k, v in combined.items() if k != "trace"}, flush=True)
    print("\n" + csv_path.read_text(), flush=True)
    print("PRECISION CONFIG FUNCTIONAL CHECKS PASS:", tag, flush=True)
    print("Details:", report_path, flush=True)


if __name__ == "__main__":
    main()
