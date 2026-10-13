"""Pure Python state accounting for experimental paged speculative decoding."""
from collections import deque
from dataclasses import dataclass, field

class PagePool:
    def __init__(self, pages, block_size):
        self.pages, self.block_size = pages, block_size
        self.free = deque(list(range(0,pages,2))+list(range(1,pages,2)))
        self.tables = {}
        self.peak = 0
        self.rollback_pages = 0
    def reserve(self, key, tokens):
        table = self.tables.setdefault(key, [])
        needed = (tokens+self.block_size-1)//self.block_size
        assert len(self.free) >= max(needed-len(table),0), 'Speculative KV pool exhausted'
        while len(table) < needed:
            table.append(self.free.popleft())
        self.peak = max(self.peak, self.pages-len(self.free))
        return table
    def trim(self, key, tokens):
        table = self.tables[key]
        needed = (tokens+self.block_size-1)//self.block_size
        while len(table) > needed:
            self.free.append(table.pop())
            self.rollback_pages += 1
    def release(self, key):
        self.free.extend(self.tables.pop(key))
    def check(self):
        values = list(self.free)+[p for t in self.tables.values() for p in t]
        assert len(values)==self.pages and set(values)==set(range(self.pages))

@dataclass
class Request:
    key: int
    tokens: list
    prompt_length: int
    computed: int = 0
    features: object = None
    draft_cache: object = None
    first_token_s: float = 0.0
    finish_s: float = 0.0
    trace: list = field(default_factory=list)
    prediction_trace: list = field(default_factory=list)
    @property
    def outputs(self):
        return self.tokens[self.prompt_length:]

class Window:
    def __init__(self, req, query, table):
        assert len(req.tokens) == req.computed+1 or req.computed == 0
        self.token_ids = req.tokens[:req.computed]+list(query)
        self.block_table = table.copy()
        self.num_cached_tokens = 0
        self._scheduled_start = req.computed
        self._scheduled_end = req.computed+len(query)
        # Verification uses causal paged varlen attention, including Q=1 baseline.
        self._scheduled_prefill = True
    def __len__(self):
        return len(self.token_ids)

def accept_prefix(block, posterior):
    assert len(block)==len(posterior) and block
    accepted = 0
    for draft, target in zip(block[1:], posterior[:-1]):
        if draft != target:
            break
        accepted += 1
    return accepted

def commit(req, block, posterior, limit):
    """Maintain one pending token; reject all KV beyond the committed prefix."""
    assert len(req.tokens)==req.computed+1
    assert req.tokens[-1]==block[0]
    accepted = accept_prefix(block, posterior)
    remaining = limit-len(req.outputs)
    assert remaining > 0
    emitted = (list(block[1:accepted+1])+[posterior[accepted]])[:remaining]
    advance = len(emitted)
    req.tokens.extend(emitted)
    req.computed += advance
    assert req.computed == len(req.tokens)-1
    return dict(accepted_draft=min(accepted, advance), matched_draft_prefix=accepted, proposed_draft=len(block)-1,
                advance=advance, rejected_draft=len(block)-1-accepted)
