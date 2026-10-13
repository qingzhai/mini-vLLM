"""Run a mixed varlen batch once; sample only completed prefills and decodes."""
import torch
from myvllm.utils import reset_context


@torch.inference_mode()
def run_chunked(runner, seqs):
    assert runner.world_size == 1
    try:
        input_ids = runner.prepare_prefill(seqs)
        hidden = runner.model(input_ids)
        indices, ready, output_indices = [], [], []
        cursor = 0
        for i, seq in enumerate(seqs):
            cursor += seq._scheduled_end - seq._scheduled_start
            if not seq._scheduled_prefill or seq._scheduled_end == len(seq):
                indices.append(cursor - 1)
                ready.append(seq)
                output_indices.append(i)
        output = torch.full((len(seqs),), -1, dtype=torch.long,
                            device=input_ids.device)
        if ready:
            index = torch.tensor(indices, dtype=torch.long, device=hidden.device)
            selected_hidden = hidden.index_select(0, index)
            # chunk_lm_head_rows_v2: rows already selected; disable Prefill gathering.
            reset_context()
            logits = runner.model.compute_logits(selected_hidden)
            assert logits.ndim == 2 and logits.shape[0] == len(ready), (
                "Chunk LM Head must return one logit row per sampled sequence")
            sampled = runner.sampler(logits, runner.prepare_sample(ready))
            positions = torch.tensor(output_indices, dtype=torch.long, device=output.device)
            output.index_copy_(0, positions, sampled.to(torch.long).reshape(-1))
        return output
    finally:
        reset_context()


def chunked_step(engine):
    seqs, is_prefill = engine.scheduler.schedule()
    if not seqs:
        return [], 0, is_prefill
    # Pure decode keeps the existing CUDA Graph path. A mixed step is ONE
    # model forward through the paged varlen prefill path, including Q=1 decodes.
    method = "run_chunked" if is_prefill else "run"
    args = (seqs,) if is_prefill else (seqs, False)
    result = engine.model_runner.call(method, *args)
    tokens = result.cpu().tolist()
    assert len(tokens) == len(seqs)
    completed, sampled = [], []
    processed = 0
    for seq, token in zip(seqs, tokens):
        end = seq._scheduled_end
        processed += end - seq._scheduled_start
        seq.num_computed_tokens = end
        # Partial prompts must publish ONLY pages whose KV is already written.
        engine.scheduler.block_manager.mark_computed(seq, computed_tokens=end)
        if seq._scheduled_prefill and end < len(seq):
            assert token == -1 and seq.num_completion_tokens == 0
        else:
            assert token >= 0
            completed.append(seq)
            sampled.append(token)
    engine.scheduler.last_schedule["sampled_seq_ids"] = [s.seq_id for s in completed]
    engine.scheduler.last_schedule["execution"] = (
        "mixed_varlen" if engine.scheduler.last_schedule["mixed"]
        else "prefill_varlen" if is_prefill else "existing_decode_path")
    engine.scheduler.postprocess(completed, sampled)
    outputs = [(s.seq_id, s.completion_token_ids) for s in completed if s.is_finished]
    return outputs, processed, is_prefill
