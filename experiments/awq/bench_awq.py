"""AWQ integration and paired offline-engine benchmark; one mode per process."""
import csv
import json
import os
from pathlib import Path
from statistics import mean, median
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
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from awq_linear import AWQLinear, dequantize_reference
from awq_loader import load_awq_model
from fp32_head import install_fp32_head


class Greedy(torch.nn.Module):
    def forward(self, logits, temperatures):
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all().item(), "Non-finite logits"
        return logits.argmax(-1)


def weight_storage(model):
    dense = sum(p.numel() * p.element_size() for p in model.parameters())
    packed = sum(t.numel() * t.element_size() for m in model.modules()
                 if isinstance(m, AWQLinear) for t in m.buffers())
    return dict(dense_parameter_bytes=dense, packed_quantization_bytes=packed,
                total_weight_bytes=dense + packed)


@torch.inference_mode()
def check_operators(model):
    checks = []
    first = model.model.layers[0]
    for name, layer in (("merged_qkv", first.self_attn.qkv_projection),
                        ("down_proj", first.mlp.down_proj)):
        assert isinstance(layer, AWQLinear)
        weight = dequantize_reference(layer).float()
        for rows in (1, 4, 33):
            torch.manual_seed(731)
            x = torch.randn(rows, layer.in_features, device="cuda", dtype=torch.float16) * 0.25
            expected = x.float() @ weight
            actual = layer(x).float()
            assert torch.isfinite(actual).all().item()
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.005)
            delta = (actual - expected).abs()
            row = dict(module=name, rows=rows, max_abs=delta.max().item(),
                       mean_abs=delta.mean().item(), atol=0.02, rtol=0.005, passed=True)
            checks.append(row)
            print("AWQ OPERATOR PASS:", row, flush=True)
        del weight
    return checks


def cap_kv_pages(pages=64):
    original = ModelRunner.allocate_kv_cache
    def allocate(self):
        # Existing allocator derives pages from free memory, peak scratch and
        # utilization. Choose utilization to obtain the same fixed page count.
        free, _ = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        scratch = stats["allocated_bytes.all.peak"] - stats["allocated_bytes.all.current"]
        block_bytes = (self.block_size * 2 * self.config["num_layers"] *
                       self.config["num_kv_heads"] * self.config["head_dim"] *
                       self.default_dtype.itemsize)
        fraction = ((pages + 0.125) * block_bytes + scratch) / free
        assert 0 < fraction < 0.95, "Insufficient free memory for the common KV budget"
        prior = self.config["gpu_memory_utilization"]
        self.config["gpu_memory_utilization"] = fraction
        try:
            original(self)
        finally:
            self.config["gpu_memory_utilization"] = prior
        assert self.config["max_cached_blocks"] == pages, (
            "Allocator contract changed", self.config["max_cached_blocks"], pages)
    ModelRunner.allocate_kv_cache = allocate


def fp32_rope_initialization():
    from myvllm.layers.rotary_embedding import RotaryEmbedding
    original = RotaryEmbedding.__init__
    def initialize(self, *args, **kwargs):
        # Qwen3 rope_theta=1e6; constructing powers in FP16 can overflow
        # before taking their reciprocal. Only RoPE caches use FP32 here.
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            original(self, *args, **kwargs)
        finally:
            torch.set_default_dtype(previous)
    RotaryEmbedding.__init__ = initialize


@torch.inference_mode()
def main():
    root = Path(os.environ["WORK_ROOT"])
    mode = os.environ["QUANT_MODE"]
    assert mode in ("baseline", "awq")
    path = Path(os.environ.get("MODEL_PATH", str(root / "models/Qwen3-8B")))
    awq_path = Path(os.environ.get("AWQ_MODEL_PATH", str(root / "models/Qwen3-8B-AWQ")))
    hc = AutoConfig.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    config = dict(base_config)
    config.update(model_name_or_path=str(path), world_size=1, enforce_eager=True,
                  vocab_size=hc.vocab_size, hidden_size=hc.hidden_size,
                  num_heads=hc.num_attention_heads, num_kv_heads=hc.num_key_value_heads,
                  head_dim=hc.head_dim, intermediate_size=hc.intermediate_size,
                  num_layers=hc.num_hidden_layers, tie_word_embeddings=hc.tie_word_embeddings,
                  rms_norm_epsilon=hc.rms_norm_eps, qkv_bias=hc.attention_bias,
                  ffn_bias=False, base=hc.rope_theta, scale=1.0, eos=hc.eos_token_id,
                  block_size=256, max_num_sequences=4, max_num_seqs=4,
                  max_num_batched_tokens=2048, max_num_batch_tokens=2048,
                  max_model_length=2048, max_position=2048, gpu_memory_utilization=0.5,
                  enable_chunked_prefill=False, enable_prefix_caching=False)
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float16)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    cap_kv_pages()
    fp32_rope_initialization()
    if mode == "awq":
        import myvllm.utils.loader as loader_module
        import myvllm.engine.model_runner as runner_module
        def load(model, original_path):
            assert Path(original_path).resolve() == path.resolve()
            load_awq_model(model, awq_path)
        loader_module.load_weights_from_checkpoint = load
        runner_module.load_weights_from_checkpoint = load
    print("INITIALIZING:", mode, "FP16 activations; eager; 64 KV pages", flush=True)
    engine = LLMEngine(config)
    torch.set_default_device("cpu")
    model = engine.model_runner.model
    model.eval()
    assert all(p.dtype == torch.float16 for p in model.parameters())
    ropes = [m for m in model.modules() if hasattr(m, "cos_sin_cache")]
    assert len(ropes) == hc.num_hidden_layers
    assert all(m.cos_sin_cache.dtype == torch.float32 and m.inv_freq.dtype == torch.float32
               and torch.isfinite(m.cos_sin_cache).all().item() and (m.inv_freq > 0).all().item()
               for m in ropes)
    quantized = [m for m in model.modules() if isinstance(m, AWQLinear)]
    assert len(quantized) == (4 * hc.num_hidden_layers if mode == "awq" else 0)
    storage = weight_storage(model)
    head_info = install_fp32_head(model)
    storage["fp32_head_extra_bytes"] = head_info["extra_weight_bytes"]
    storage["total_weight_and_head_bytes"] = storage["total_weight_bytes"] + head_info["extra_weight_bytes"]
    engine.model_runner.sampler = Greedy()
    checks = check_operators(model) if mode == "awq" else []
    torch.cuda.empty_cache()
    scheduler = engine.scheduler
    manager = scheduler.block_manager
    params = SamplingParams(temperature=1.0, max_tokens=32, ignore_eos=True, max_model_length=2048)

    def prompt(index, long):
        system = "You are a helpful assistant."
        if long:
            system += " Cached keys and values reduce repeated attention computations." * 55
        text = tokenizer.apply_chat_template(
            [dict(role="system", content=system),
             dict(role="user", content=f"Request {index}: Explain KV caching in one short sentence.")],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text)
        assert len(ids) + 32 < 2048
        return ids

    cases = [("short-b1", [prompt(0, False)]),
             ("short-b4", [prompt(i, False) for i in range(4)]),
             ("long-b1", [prompt(0, True)]),
             ("long-b4", [prompt(i, True) for i in range(4)])]

    def run(inputs):
        assert scheduler.is_finished()
        manager.clear_cache()
        seqs = [Sequence(token_ids=ids.copy(), block_size=256, sampling_params=params) for ids in inputs]
        seen = {s.seq_id: [] for s in seqs}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for seq in seqs:
            scheduler.add_sequence(seq)
        while not scheduler.is_finished():
            engine.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            for seq in seqs:
                times = seen[seq.seq_id]
                assert seq.num_completion_tokens in (len(times), len(times) + 1)
                if seq.num_completion_tokens > len(times):
                    times.append(elapsed)
        total = time.perf_counter() - started
        times = [seen[s.seq_id] for s in seqs]
        assert all(len(t) == 32 for t in times)
        assert not manager.used_block_ids and len(manager.free_block_ids) == len(manager.blocks)
        result = dict(avg_ttft_ms=mean(t[0] for t in times) * 1000,
                      avg_tpot_ms=mean((t[-1]-t[0])/31 for t in times) * 1000,
                      avg_e2e_ms=mean(t[-1] for t in times) * 1000,
                      output_tokens_per_s=len(seqs)*32/total,
                      total_s=total,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        return result, [s.completion_token_ids.copy() for s in seqs]

    expected_revision = "4da05a8edb55c6046cce958586c33b61da07bb79"
    source_path = Path(os.environ.get("AWQ_SOURCE_PATH", str(awq_path / "source.json")))
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    if source:
        assert source.get("repo_id") == "Qwen/Qwen3-8B-AWQ"
        assert source.get("revision") == expected_revision, "Unexpected AWQ checkpoint revision"
    report = dict(mode=mode, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  awq_source_revision=source.get("revision"),
                  awq_expected_revision=expected_revision,
                  awq_source_metadata_verified=bool(source),
                  model_path=str(path), awq_model_path=str(awq_path),
                  rope_cache_dtype="float32",
                  core_dtype="float16", lm_head="float32", enforce_eager=True,
                  kv_pages=64, kv_bytes=64*256*2*hc.num_hidden_layers*hc.num_key_value_heads*hc.head_dim*2,
                  prefix_cache=False, chunked_prefill=False, storage=storage,
                  awq_loader=getattr(model, "awq_load_report", None), operator_checks=checks,
                  correctness_scope="Packed-linear reference checks, finite generation, page release; not a model quality benchmark",
                  cases=[], all_kv_pages_released=True)
    result_dir = Path(os.environ["RUN_DIR"])
    result_dir.mkdir(parents=True, exist_ok=True)
    destination = result_dir / f"awq-{mode}-details.json"
    assert not destination.exists(), f"Refusing to overwrite {destination}; use a new RUN_DIR"
    for name, inputs in cases:
        print("CASE:", name, "prompt lengths:", [len(x) for x in inputs], "warmup", flush=True)
        run(inputs)
        repetitions, outputs = [], []
        for _ in range(3):
            metrics, tokens = run(inputs)
            repetitions.append(metrics)
            outputs.append(tokens)
        assert outputs[0] == outputs[1] == outputs[2], "Repeated identical greedy workload produced different outputs"
        summary = dict(case=name, batch=len(inputs), output_tokens=32, repetitions=3,
                       avg_prompt_tokens=mean(len(x) for x in inputs),
                       **{key: round(median(row[key] for row in repetitions), 4) for key in repetitions[0]})
        report["cases"].append(dict(summary=summary, measurements=repetitions, input_ids=inputs,
                                    token_ids=outputs[0], texts=[tokenizer.decode(x) for x in outputs[0]]))
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print("RESULT:", summary, flush=True)
        print("SAMPLE:", report["cases"][-1]["texts"][0], flush=True)
    report["completed"] = True
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("MODE FUNCTIONAL CHECKS PASS:", mode, flush=True)
    print("Details:", destination, flush=True)


if __name__ == "__main__":
    main()
