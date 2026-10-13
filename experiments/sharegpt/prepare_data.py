#!/usr/bin/env python3
"""Prepare a pinned, tokenized ShareGPT workload without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

DATASET_REPO = "heka-ai/sharegpt-english-10k-vllm-serving-benchmark"
DATASET_REVISION = "854a6eb3c310f07f9e8b269a735bf6fe547e543d"
DATASET_FILENAME = "train.json"
DATASET_SHA256 = "bc915b8dc677edc8820659cb37ba039928af24f6df4cdd58c89afcca87cd2ca5"
WORKLOAD_LABEL = "sharegpt-with-20pct-replayed-prompts"
BUCKETS = {"short": (256, 512), "long": (1024, 1536)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def request_sha256(requests: list[dict]) -> str:
    return hashlib.sha256(json.dumps(requests, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def first_human(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    conversation = record.get("conversations")
    if not isinstance(conversation, list):
        return None
    for item in conversation:
        if isinstance(item, dict) and item.get("from", item.get("user", item.get("role"))) in ("human", "user"):
            text = item.get("value")
            if isinstance(text, str) and text.strip():
                return text
    return None


def build_requests(records: list, tokenizer, vocab_size: int, seed: int = 42):
    """Shuffle source indices once; select unique prompts, then controlled replays."""
    if not isinstance(records, list):
        raise ValueError("Expected train.json to contain a JSON list of conversations")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("A positive model vocabulary size is required")
    rng = random.Random(seed)
    row_order = list(range(len(records)))
    rng.shuffle(row_order)
    selected = {"short": [], "long": []}
    seen_texts: set[str] = set()
    seen_tokens: set[str] = set()
    scanned = 0
    for row in row_order:
        scanned += 1
        record = records[row]
        prompt = first_human(record)
        if prompt is None or prompt in seen_texts:
            continue
        seen_texts.add(prompt)
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        if not isinstance(tokens, list) or not tokens or any(
                not isinstance(t, int) or isinstance(t, bool) or t < 0 or t >= vocab_size
                for t in tokens):
            raise ValueError(f"Invalid token IDs from source row {row}")
        bucket = next((name for name, (lo, hi) in BUCKETS.items()
                       if lo <= len(tokens) <= hi), None)
        if bucket is None or len(selected[bucket]) >= 40:
            continue
        digest = token_sha256(tokens)
        if digest in seen_tokens:
            continue
        seen_tokens.add(digest)
        source_id = record.get("id", f"row-{row}")
        if not isinstance(source_id, (str, int)) or isinstance(source_id, bool):
            source_id = f"row-{row}"
        selected[bucket].append({
            "source_id": str(source_id), "source_row": row,
            "bucket": bucket, "prompt_token_ids": tokens,
            "prompt_tokens": len(tokens), "prompt_sha256": digest,
            "repeated_from": None,
        })
        if all(len(rows) == 40 for rows in selected.values()):
            break
    if any(len(rows) != 40 for rows in selected.values()):
        counts = {name: len(rows) for name, rows in selected.items()}
        raise ValueError(f"Insufficient unique ShareGPT candidates: {counts}; need "
                         "40 short (256..512) and 40 long (1024..1536) after the "
                         "Qwen chat template. No synthetic replacement was generated.")
    for rows in selected.values():
        rng.shuffle(rows)
    requests = []
    for pair in zip(selected["short"], selected["long"]):
        for record in pair:
            record["request_id"] = f"sg-{len(requests):03d}"
            requests.append(record)
    replay_rows = {name: rng.sample(rows, 10) for name, rows in selected.items()}
    for pair in zip(replay_rows["short"], replay_rows["long"]):
        for original in pair:
            replay = dict(original)
            replay["prompt_token_ids"] = list(original["prompt_token_ids"])
            replay["repeated_from"] = original["request_id"]
            replay["request_id"] = f"sg-{len(requests):03d}"
            requests.append(replay)
    metadata = {
        "seed": seed, "source_rows": len(records), "scanned_rows": scanned,
        "selection": "seeded source-row shuffle; first 40 token-unique eligible prompts per bucket",
        "bucket_token_ranges_inclusive": {k: list(v) for k, v in BUCKETS.items()},
        "bucket_counts": {"short": 50, "long": 50},
        "unique_prompts": 80, "replayed_requests": 20, "total_requests": 100,
        "order": "80 unique prompts interleaved short/long, then 20 exact replays interleaved short/long",
        "prefix_reuse": "controlled exact prompt replay; not a natural prefix-frequency estimate",
        "cache_note": "20% of requests are replayed; measured cache hit tokens and hit rate are reported separately",
        "prompt_extraction": "first nonempty human message only; assistant answers are unused",
        "chat_template": {"add_generation_prompt": True, "enable_thinking": False},
    }
    return requests, metadata


def validate_workload(workload: dict) -> None:
    """Fail before GPU work if provenance or request invariants are inconsistent."""
    if workload.get("schema_version") != 1 or workload.get("workload_label") != WORKLOAD_LABEL:
        raise ValueError("Unknown workload schema or label")
    dataset = workload.get("dataset", {})
    for key, expected in (("repo_id", DATASET_REPO), ("revision", DATASET_REVISION),
                          ("filename", DATASET_FILENAME), ("sha256", DATASET_SHA256)):
        if dataset.get(key) != expected:
            raise ValueError(f"Dataset provenance mismatch: {key}")
    generation = workload.get("generation", {})
    if generation != {"max_new_tokens": 64, "ignore_eos": True, "max_model_len": 2048,
                      "temperature": 0.0}:
        raise ValueError("Unexpected generation settings")
    rows = workload.get("requests")
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("Exactly 100 requests are required")
    vocab_size = workload["model"]["vocab_size"]
    by_id = {}
    counts = {name: 0 for name in BUCKETS}
    originals = set()
    replay_sources = set()
    for index, row in enumerate(rows):
        rid = row["request_id"]
        if rid in by_id or rid != f"sg-{index:03d}":
            raise ValueError("Duplicate or unordered request ID")
        bucket = row["bucket"]
        if bucket not in BUCKETS:
            raise ValueError("Unknown length bucket")
        counts[bucket] += 1
        tokens = row["prompt_token_ids"]
        if not isinstance(tokens, list) or any(not isinstance(x, int) or isinstance(x, bool)
                                              or not 0 <= x < vocab_size for x in tokens):
            raise ValueError(f"Invalid token IDs in {rid}")
        lo, hi = BUCKETS[bucket]
        if len(tokens) != row["prompt_tokens"] or not lo <= len(tokens) <= hi:
            raise ValueError(f"Invalid prompt length in {rid}")
        if len(tokens) + generation["max_new_tokens"] > generation["max_model_len"]:
            raise ValueError("Context limit exceeded")
        digest = token_sha256(tokens)
        if digest != row["prompt_sha256"]:
            raise ValueError("Prompt token hash mismatch")
        source = row.get("repeated_from")
        if index < 80:
            if source is not None or digest in originals:
                raise ValueError("First 80 requests must be distinct original prompts")
            originals.add(digest)
        else:
            if source not in by_id or source in replay_sources:
                raise ValueError("Invalid or reused replay source")
            original = by_id[source]
            if original["repeated_from"] is not None or any(
                row[key] != original[key] for key in
                ("source_id", "source_row", "bucket", "prompt_token_ids", "prompt_tokens")):
                raise ValueError("Replay differs from its original prompt")
            replay_sources.add(source)
        if bucket != ("short" if index % 2 == 0 else "long"):
            raise ValueError("Short/long request interleaving changed")
        by_id[rid] = row
    if counts != {"short": 50, "long": 50} or len(originals) != 80 or len(replay_sources) != 20:
        raise ValueError("Unexpected workload composition")
    if request_sha256(rows) != workload.get("workload_sha256"):
        raise ValueError("Workload checksum mismatch")


def get_dataset(root: Path, override: Path | None, endpoint: str) -> Path:
    if override is not None:
        path = override.expanduser().resolve()
        actual = file_sha256(path)
        if actual != DATASET_SHA256:
            raise ValueError(f"--dataset-file SHA256 mismatch: expected {DATASET_SHA256}, got {actual}")
        print(f"Using verified offline ShareGPT file: {path}", flush=True)
        return path
    directory = root / "datasets" / "sharegpt" / DATASET_REVISION
    path = directory / DATASET_FILENAME
    if path.is_file() and file_sha256(path) == DATASET_SHA256:
        print(f"Using verified cached ShareGPT file: {path}", flush=True)
        return path
    from huggingface_hub import hf_hub_download
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ShareGPT: {DATASET_REPO} revision={DATASET_REVISION} endpoint={endpoint}", flush=True)
    downloaded = Path(hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset",
        filename=DATASET_FILENAME, revision=DATASET_REVISION, endpoint=endpoint,
        local_dir=str(directory), force_download=path.exists()))
    actual = file_sha256(downloaded)
    if actual != DATASET_SHA256:
        raise ValueError(f"Downloaded ShareGPT SHA256 mismatch: expected {DATASET_SHA256}, got {actual}")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("WORK_ROOT", str(Path(__file__).resolve().parents[2]/'.work')))
    parser.add_argument("--model", help="Existing local Qwen3-8B directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-file", type=Path, help="Offline copy of the exact pinned train.json")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    model = Path(args.model).expanduser().resolve() if args.model else root / "models" / "Qwen3-8B"
    config_path = model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Local model config not found: {config_path}")
    config = json.loads(config_path.read_text())
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    data_path = get_dataset(root, args.dataset_file, endpoint)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    print("Tokenizing ShareGPT candidates with the existing local Qwen tokenizer...", flush=True)
    with data_path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    requests, sampling = build_requests(records, tokenizer, config["vocab_size"])
    tokenizer_files = {p.name: file_sha256(p) for p in sorted(model.iterdir())
                       if p.is_file() and (p.name.startswith("tokenizer") or
                          p.name in ("vocab.json", "merges.txt", "special_tokens_map.json",
                                     "added_tokens.json", "chat_template.jinja"))}
    workload = {
        "schema_version": 1, "workload_label": WORKLOAD_LABEL,
        "dataset": {"repo_id": DATASET_REPO, "revision": DATASET_REVISION,
                    "filename": DATASET_FILENAME, "sha256": DATASET_SHA256,
                    "local_path": str(data_path), "endpoint_requested": endpoint},
        "model": {"path": str(model), "vocab_size": config["vocab_size"],
                  "config_sha256": file_sha256(config_path), "tokenizer_files_sha256": tokenizer_files},
        "sampling": sampling,
        "generation": {"max_new_tokens": 64, "ignore_eos": True,
                       "max_model_len": 2048, "temperature": 0.0},
        "requests": requests, "workload_sha256": request_sha256(requests),
    }
    validate_workload(workload)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = json.loads(output.read_text())
        validate_workload(previous)
        if previous != workload:
            raise FileExistsError(f"Refusing to replace a different workload: {output}")
        print(f"Existing identical workload preserved: {output}", flush=True)
    else:
        temp = output.with_name(output.name + f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(workload, ensure_ascii=False, indent=2) + "\n")
        temp.replace(output)
    print(f"WORKLOAD READY: {output}", flush=True)
    print(f"100 requests, 80 unique prompts, 20 controlled exact replays; sha256={workload['workload_sha256']}", flush=True)
    print("Prompt token lengths: " + json.dumps({bucket: {
        "min": min(x["prompt_tokens"] for x in requests if x["bucket"] == bucket),
        "max": max(x["prompt_tokens"] for x in requests if x["bucket"] == bucket),
        "mean": round(sum(x["prompt_tokens"] for x in requests if x["bucket"] == bucket) / 50, 2)
    } for bucket in BUCKETS}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        sys.exit(f"DATA PREPARATION FAILED: {error}")
