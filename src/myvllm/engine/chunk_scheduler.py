"""Single-GPU chunk scheduling; decode-first batches may contain both phases."""
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import SequenceStatus


class ChunkScheduler(Scheduler):
    def __init__(self, original, chunk_size=256, decode_first=True):
        assert not original.waiting and not original.running
        self.__dict__.update(original.__dict__)
        self.chunk_size = int(chunk_size)
        self.decode_first = bool(decode_first)
        assert self.chunk_size >= 1
        assert self.max_num_sequences >= 1 and self.max_num_batched_tokens >= 1
        self.last_schedule = None
        self.trace = []

    def add_sequence(self, seq):
        if len(seq) == 0 or seq.max_tokens < 1:
            raise ValueError("A nonempty prompt and max_tokens >= 1 are required")
        if seq.max_model_length is not None and len(seq) >= seq.max_model_length:
            raise ValueError("Prompt must leave room for at least one output token")
        super().add_sequence(seq)

    def schedule(self):
        bm = self.block_manager
        # max_num_sequences is also a true resident-request bound in this path.
        while self.waiting and len(self.running) < self.max_num_sequences:
            seq = self.waiting[0]
            if not bm.can_allocate(seq):
                break
            self.waiting.popleft()
            bm.allocate(seq)
            seq.num_computed_tokens = seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
        prefills = [s for s in self.running if s.num_completion_tokens == 0]
        decodes = [s for s in self.running if s.num_completion_tokens > 0]
        selected = []
        budget = self.max_num_batched_tokens

        def add_decode(seq):
            nonlocal budget
            if budget == 0 or len(selected) >= self.max_num_sequences:
                return
            if not bm.can_append(seq):
                raise RuntimeError("Chunk scheduler exhausted KV pages; increase KV pool or lower concurrency")
            bm.append(seq)
            assert seq.num_computed_tokens == len(seq) - 1
            seq._scheduled_start = len(seq) - 1
            seq._scheduled_end = len(seq)
            seq._scheduled_prefill = False
            selected.append(seq)
            budget -= 1

        def add_prefill(seq):
            nonlocal budget
            if budget == 0 or len(selected) >= self.max_num_sequences:
                return
            start = seq.num_computed_tokens
            remaining = len(seq) - start
            assert remaining > 0
            # The full-prefill control must not silently split a prompt to fit.
            if not self.decode_first and remaining > budget:
                if not selected:
                    raise ValueError("Full-prefill control requires token budget >= uncached prompt length")
                return
            count = min(remaining, self.chunk_size, budget)
            seq._scheduled_start = start
            seq._scheduled_end = start + count
            seq._scheduled_prefill = True
            selected.append(seq)
            budget -= count

        if self.decode_first:
            for seq in decodes:
                add_decode(seq)
            for seq in prefills:
                add_prefill(seq)
        elif prefills:
            for seq in prefills:
                add_prefill(seq)
        else:
            for seq in decodes:
                add_decode(seq)

        if not selected:
            if self.waiting or self.running:
                raise RuntimeError("Chunk scheduler made no progress; check KV capacity and token budget")
            return [], False
        for seq in selected:
            self.running.remove(seq)
            self.running.append(seq)
        items = [dict(seq_id=s.seq_id,
                      phase="prefill" if s._scheduled_prefill else "decode",
                      start=s._scheduled_start, end=s._scheduled_end,
                      prompt_tokens=s.num_prompt_tokens,
                      completion_tokens_before=s.num_completion_tokens)
                 for s in selected]
        phases = {x["phase"] for x in items}
        self.last_schedule = dict(step=len(self.trace),
                                  tokens=sum(x["end"] - x["start"] for x in items),
                                  mixed=len(phases) == 2, items=items)
        assert self.last_schedule["tokens"] <= self.max_num_batched_tokens
        self.trace.append(self.last_schedule)
        return selected, "prefill" in phases
