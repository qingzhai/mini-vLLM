import csv
import json
import os
from pathlib import Path


def main():
    root = Path(os.environ["RUN_DIR"])
    base = json.loads((root / "awq-baseline-details.json").read_text())
    quant = json.loads((root / "awq-awq-details.json").read_text())
    assert base["completed"] and quant["completed"]
    for key in ("gpu", "torch", "core_dtype", "lm_head", "enforce_eager", "kv_pages", "kv_bytes", "awq_source_revision", "rope_cache_dtype",
                "prefix_cache", "chunked_prefill"):
        assert base[key] == quant[key], f"Incomparable configuration: {key}"
    assert base["all_kv_pages_released"] and quant["all_kv_pages_released"]
    assert len(quant["operator_checks"]) == 6 and all(x["passed"] for x in quant["operator_checks"])
    assert len(base["cases"]) == len(quant["cases"]) == 4
    rows = []
    for a, b in zip(base["cases"], quant["cases"]):
        assert a["input_ids"] == b["input_ids"]
        x, y = a["summary"], b["summary"]
        assert x["case"] == y["case"]
        rows.append(dict(case=x["case"], fp16_ttft_ms=x["avg_ttft_ms"], awq_ttft_ms=y["avg_ttft_ms"],
                         fp16_tpot_ms=x["avg_tpot_ms"], awq_tpot_ms=y["avg_tpot_ms"],
                         fp16_tokens_per_s=x["output_tokens_per_s"], awq_tokens_per_s=y["output_tokens_per_s"],
                         throughput_ratio=round(y["output_tokens_per_s"]/x["output_tokens_per_s"], 4)))
    storage = dict(
        fp16_weight_bytes=base["storage"]["total_weight_bytes"],
        awq_weight_bytes=quant["storage"]["total_weight_bytes"],
        weight_reduction_pct=round(100*(1-quant["storage"]["total_weight_bytes"]/base["storage"]["total_weight_bytes"]), 3),
        fp16_weights_plus_head_bytes=base["storage"]["total_weight_and_head_bytes"],
        awq_weights_plus_head_bytes=quant["storage"]["total_weight_and_head_bytes"],
        weights_plus_head_reduction_pct=round(100*(1-quant["storage"]["total_weight_and_head_bytes"]/base["storage"]["total_weight_and_head_bytes"]), 3),
        common_kv_bytes=base["kv_bytes"], quantization_quality_evaluated=False)
    with (root / "awq-comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "awq-comparison.json").write_text(json.dumps(dict(storage=storage, cases=rows), indent=2))
    print("WEIGHT STORAGE:", storage)
    print((root / "awq-comparison.csv").read_text())
    print("AWQ W4A16 INTEGRATION CHECKS PASS")
    print("Small offline workload only; use measured ratios, do not assume a speedup or quality parity.")


if __name__ == "__main__":
    main()
