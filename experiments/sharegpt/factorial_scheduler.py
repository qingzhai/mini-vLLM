"""Decode-first in all four modes; unchunked prefills are indivisible windows."""
from myvllm.engine.chunk_scheduler import ChunkScheduler
from myvllm.engine.sequence import SequenceStatus


class FactorialScheduler(ChunkScheduler):
    def __init__(self, original):
        super().__init__(original, chunk_size=2048, decode_first=True)
        self.chunk_enabled = False

    def schedule(self):
        if not self.decode_first:
            raise ValueError('The factorial protocol requires decode-first in every mode')
        bm = self.block_manager
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
        budget = self.max_num_batched_tokens
        selected = []
        deferred = 0
        for seq in decodes:
            if not budget or len(selected) == self.max_num_sequences:
                break
            if not bm.can_append(seq):
                raise RuntimeError('Factorial scheduler exhausted KV pages')
            bm.append(seq)
            assert seq.num_computed_tokens == len(seq) - 1
            seq._scheduled_start, seq._scheduled_end = len(seq) - 1, len(seq)
            seq._scheduled_prefill = False
            selected.append(seq)
            budget -= 1

        for seq in prefills:
            if not budget or len(selected) == self.max_num_sequences:
                break
            start = seq.num_computed_tokens
            remaining = len(seq) - start
            assert remaining > 0
            if not self.chunk_enabled:
                # Budget must accommodate a full prompt even when seven peers decode.
                if remaining > self.max_num_batched_tokens - (self.max_num_sequences - 1):
                    raise ValueError('Token budget is too small for atomic Prefill plus Decode peers')
                if remaining > budget:
                    # Defer this prompt; never silently split it to consume spare budget.
                    deferred += 1
                    continue
                count = remaining
            else:
                count = min(remaining, self.chunk_size, budget)
            seq._scheduled_start, seq._scheduled_end = start, start + count
            seq._scheduled_prefill = True
            selected.append(seq)
            budget -= count

        if not selected:
            if self.waiting or self.running:
                raise RuntimeError('Factorial scheduler made no progress')
            return [], False
        for seq in selected:
            self.running.remove(seq)
            self.running.append(seq)
        items = [dict(seq_id=s.seq_id, phase='prefill' if s._scheduled_prefill else 'decode',
                      start=s._scheduled_start, end=s._scheduled_end,
                      prompt_tokens=s.num_prompt_tokens,
                      completion_tokens_before=s.num_completion_tokens) for s in selected]
        phases = {item['phase'] for item in items}
        self.last_schedule = dict(step=len(self.trace),
                                  tokens=sum(x['end'] - x['start'] for x in items),
                                  mixed=len(phases) == 2, items=items,
                                  deferred_atomic_prefills=deferred)
        assert self.last_schedule['tokens'] <= self.max_num_batched_tokens
        self.trace.append(self.last_schedule)
        return selected, 'prefill' in phases
