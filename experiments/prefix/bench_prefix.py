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
from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams
from verify_prefix_kernel import verify_prefix_kernel


class Greedy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_logits = None

    def forward(self, logits, temperatures):
        if self.first_logits is None:
            self.first_logits = logits.detach().float().cpu().clone()
        return logits.argmax(-1)


@torch.inference_mode()
def main():
    kernel_checks = verify_prefix_kernel()
    root = Path(os.environ["WORK_ROOT"])
    model_path = Path(os.environ.get("MODEL_PATH", str(root / "models/Qwen3-8B")))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    hc = AutoConfig.from_pretrained(model_path, local_files_only=True)
    assert hc.model_type == "qwen3"
    torch.manual_seed(42)
    torch.set_default_dtype(torch.bfloat16)
    config = dict(base_config)
    config.update(
        model_name_or_path=str(model_path), world_size=1, enforce_eager=False,
        vocab_size=hc.vocab_size, hidden_size=hc.hidden_size,
        num_heads=hc.num_attention_heads, num_kv_heads=hc.num_key_value_heads,
        head_dim=hc.head_dim, intermediate_size=hc.intermediate_size,
        num_layers=hc.num_hidden_layers, tie_word_embeddings=hc.tie_word_embeddings,
        rms_norm_epsilon=hc.rms_norm_eps, qkv_bias=hc.attention_bias,
        ffn_bias=False, base=hc.rope_theta, scale=1.0, eos=hc.eos_token_id,
        block_size=256, max_num_sequences=4,
        max_num_batched_tokens=2048, max_num_batch_tokens=2048,
        max_model_length=1024, max_position=1024, gpu_memory_utilization=0.5,
    )
    print("MODEL:", model_path, flush=True)
    engine = LLMEngine(config)
    torch.set_default_device("cpu")
    engine.model_runner.model.eval()
    sampler = Greedy()
    engine.model_runner.sampler = sampler
    # Physically valid subset of the GPU pool; makes LRU pressure easy to reach.
    manager = BlockManager(min(16, config["max_cached_blocks"]), 256)
    assert len(manager.blocks) >= 8
    engine.scheduler.block_manager = manager
    params = SamplingParams(temperature=1.0, max_tokens=16,
                            ignore_eos=True, max_model_length=1024)

    system = "You are a helpful assistant. "
    sentence = "A language model reuses cached attention keys and values during decoding. "
    while len(tokenizer.encode(system)) < 620:
        system += sentence

    def prompt(index, eviction=False):
        system_text = (f"Eviction namespace {index}. " if eviction else "") + system
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_text},
             {"role": "user", "content": f"Request {index}: Explain KV caching in one short sentence."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer.encode(text)
        assert 512 < len(ids) < 1000
        return ids

    inputs = [prompt(i) for i in range(5)]

    def run(ids):
        assert engine.scheduler.is_finished()
        seq = Sequence(token_ids=ids, block_size=256, sampling_params=params)
        sampler.first_logits = None
        before = manager.stats.copy()
        first = None
        previous = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        engine.scheduler.add_sequence(seq)
        while not engine.scheduler.is_finished():
            engine.step()
            torch.cuda.synchronize()
            now = time.perf_counter() - start
            if seq.num_completion_tokens > previous:
                if first is None:
                    first = now
                last = now
                previous = seq.num_completion_tokens
        assert previous == 16
        row = dict(
            prompt_tokens=len(ids), output_tokens=previous,
            cached_tokens=manager.stats["cache_hit_tokens"] - before["cache_hit_tokens"],
            ttft_ms=round(first * 1000, 3), e2e_ms=round(last * 1000, 3),
            tpot_ms=round((last - first) / 15 * 1000, 3),
        )
        return row, seq.completion_token_ids.copy(), sampler.first_logits.clone()

    print("Warmup: uncached and cached prefill", flush=True)
    manager.enable_prefix_caching = False
    for ids in inputs:
        run(ids)
    manager.enable_prefix_caching = True
    manager.clear_cache()
    for ids in inputs:
        run(ids)

    manager.enable_prefix_caching = False
    manager.clear_cache()
    baseline = [run(ids) for ids in inputs[1:]]
    manager.enable_prefix_caching = True
    manager.clear_cache()
    run(inputs[0])  # Seed shared system prefix, outside measurement.
    optimized = [run(ids) for ids in inputs[1:]]

    result_dir = Path(os.environ["RUN_DIR"])
    result_dir.mkdir(parents=True, exist_ok=True)
    assert not (result_dir / f"{model_path.name}-prefix-v2-details.json").exists(), "Use a new RUN_DIR"
    report = dict(kernel_checks=kernel_checks, model=str(model_path), output_tokens=16, greedy=True,
                  measured_requests_per_mode=4, baseline=[], prefix_cache=[])
    for i, ((old, old_tokens, old_logits), (new, new_tokens, new_logits)) in enumerate(zip(baseline, optimized)):
        assert torch.isfinite(old_logits).all().item()
        assert torch.isfinite(new_logits).all().item()
        delta = (old_logits - new_logits).abs()
        close = torch.isclose(new_logits, old_logits, rtol=0.02, atol=0.15)
        relative_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(old_logits).clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(old_logits.flatten(), new_logits.flatten(), dim=0)
        check = dict(max_abs=delta.max().item(), mean_abs=delta.mean().item(),
                     cosine=cosine.item(), relative_l2=relative_l2.item(),
                     elementwise_pass=bool(close.all().item()),
                     elementwise_mismatches=int((~close).sum().item()),
                     diagnostic_rtol=0.02, diagnostic_atol=0.15,
                     top1_match=bool((old_logits.argmax(-1) == new_logits.argmax(-1)).all().item()),
                     greedy_tokens_match=old_tokens == new_tokens)
        report["baseline"].append(dict(old, token_ids=old_tokens))
        report["prefix_cache"].append(dict(new, token_ids=new_tokens, comparison=check))
        (result_dir / f"{model_path.name}-prefix-v2-details.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2))
        print("REQUEST", i, "BASELINE", old, "PREFIX", new, "CHECK", check, flush=True)
        assert old["cached_tokens"] == 0 and new["cached_tokens"] >= 512
        # Model-level logits are a diagnostic, separate from the frozen-QKV test.
        # Keep the ORIGINAL threshold and expose any failure in both files and logs.
        if not check["elementwise_pass"]:
            print("LOGITS ELEMENTWISE WARNING: original tolerance failed; numerical equivalence is unresolved", flush=True)
        assert old_tokens == new_tokens, "Greedy output differs; see saved token IDs"

    print("LRU pressure: 12 distinct long prefixes", flush=True)
    manager.clear_cache()
    evictions_before = manager.stats["evictions"]
    last_result = None
    for i in range(12):
        last_result = run(prompt(i, eviction=True))
    repeat = run(prompt(11, eviction=True))
    evictions = manager.stats["evictions"] - evictions_before
    assert evictions > 0
    assert repeat[0]["cached_tokens"] >= 512
    assert repeat[1] == last_result[1]
    assert not manager.used_block_ids and len(manager.free_block_ids) == len(manager.blocks)
    report["lru"] = dict(evictions=evictions,
                         repeat_cached_tokens=repeat[0]["cached_tokens"],
                         free_pages=len(manager.free_block_ids),
                         total_pages=len(manager.blocks))
    summary = []
    for label, records in (("cache_off", baseline), ("cache_on", optimized)):
        summary.append(dict(
            mode=label, requests=len(records),
            avg_prompt_tokens=mean(x[0]["prompt_tokens"] for x in records),
            avg_cached_tokens=mean(x[0]["cached_tokens"] for x in records),
            avg_ttft_ms=round(mean(x[0]["ttft_ms"] for x in records), 3),
            avg_e2e_ms=round(mean(x[0]["e2e_ms"] for x in records), 3),
            avg_tpot_ms=round(mean(x[0]["tpot_ms"] for x in records), 3),
        ))
    report["summary"] = summary
    report["validation"] = dict(
        kernel_fp32_reference_pass=True,
        tested_greedy_sequences_match=True,
        cache_hits_and_lru_pass=True,
        all_logits_elementwise_pass=all(x["comparison"]["elementwise_pass"] for x in report["prefix_cache"]),
        scope="4 prompts x 16 generated tokens; limited functional evidence, not general numerical equivalence",
    )
    (result_dir / f"{model_path.name}-prefix-v2-details.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    csv_path = result_dir / f"{model_path.name}-prefix-v2-summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("\n" + csv_path.read_text(), flush=True)
    print("LRU:", report["lru"], flush=True)
    print("VALIDATION:", report["validation"], flush=True)
    print("PREFIX FUNCTIONAL CHECKS PASS", flush=True)
    if not report["validation"]["all_logits_elementwise_pass"]:
        print("NUMERICAL STATUS: WARNING - elementwise logits equivalence remains unresolved", flush=True)
    else:
        print("NUMERICAL STATUS: tested logits passed the original elementwise tolerance", flush=True)


if __name__ == "__main__":
    main()
