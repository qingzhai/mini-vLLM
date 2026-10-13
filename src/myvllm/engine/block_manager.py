"""Paged KV ownership, ready-only prefix reuse, and LRU of unreferenced pages."""
from collections import deque
from hashlib import blake2b
import struct


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = -1
        self.ref_count = 0
        self.token_ids = []
        self.ready = False

    def update(self, h, token_ids):
        self.hash = h
        self.token_ids = list(token_ids)

    def reset(self):
        self.hash = -1
        self.ref_count = 1
        self.token_ids = []
        self.ready = False


class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()
        self.enable_prefix_caching = True
        self.stats = dict(cache_hit_tokens=0, cache_hit_blocks=0,
                          allocations=0, evictions=0)

    def compute_hash(self, token_ids, prefix_hash_value):
        digest = blake2b(digest_size=8)
        if prefix_hash_value != -1:
            digest.update(prefix_hash_value.to_bytes(8, "little"))
        digest.update(struct.pack(f"<{len(token_ids)}I", *token_ids))
        return int.from_bytes(digest.digest(), "little")

    def _cached_prefix(self, seq):
        if not self.enable_prefix_caching:
            return []
        hits, h = [], -1
        # Leave at least one uncached token; the last writable page stays private.
        for i in range((len(seq) - 1) // self.block_size):
            tokens = seq.block(i)
            h = self.compute_hash(tokens, h)
            block_id = self.hash_to_block_id.get(h)
            if block_id is None:
                break
            block = self.blocks[block_id]
            if not block.ready or block.hash != h or block.token_ids != tokens:
                break
            hits.append(block_id)
        return hits

    def can_allocate(self, seq):
        hits = self._cached_prefix(seq)
        reserved_free = sum(self.blocks[i].ref_count == 0 for i in hits)
        return seq.num_blocks - len(hits) <= len(self.free_block_ids) - reserved_free

    def _allocate_block(self, block_id):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
            self.stats["evictions"] += 1
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        block.reset()
        self.stats["allocations"] += 1
        return block

    def _deallocate_block(self, block_id):
        block = self.blocks[block_id]
        assert block.ref_count == 0 and block_id in self.used_block_ids
        self.used_block_ids.remove(block_id)
        # Keep computed full pages resident until their physical page is reused.
        if not block.ready:
            block.token_ids = []
            block.hash = -1
        self.free_block_ids.append(block_id)

    def allocate(self, seq):
        assert not seq.block_table and seq.num_cached_tokens == 0
        assert self.can_allocate(seq), "Insufficient free KV pages"
        hits = self._cached_prefix(seq)
        # Pin all hit pages before allocating misses, so they cannot be evicted.
        for block_id in hits:
            block = self.blocks[block_id]
            if block.ref_count == 0:
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            block.ref_count += 1
            seq.block_table.append(block_id)
        seq.num_cached_tokens = len(hits) * self.block_size
        self.stats["cache_hit_blocks"] += len(hits)
        self.stats["cache_hit_tokens"] += seq.num_cached_tokens
        while len(seq.block_table) < seq.num_blocks:
            block = self._allocate_block(self.free_block_ids[0])
            seq.block_table.append(block.block_id)

    def mark_computed(self, seq, computed_tokens=None):
        # Called AFTER model execution and BEFORE appending the sampled token.
        # Only pages whose entire KV has actually been written become reusable.
        if not self.enable_prefix_caching:
            return
        h = -1
        # chunk_publish_limit_v1
        limit = len(seq) if computed_tokens is None else computed_tokens
        assert 0 <= limit <= len(seq)
        for i in range(limit // self.block_size):
            tokens = seq.block(i)
            h = self.compute_hash(tokens, h)
            block = self.blocks[seq.block_table[i]]
            if block.ready:
                assert block.hash == h and block.token_ids == tokens
            else:
                block.update(h, tokens)
                block.ready = True
            self.hash_to_block_id[h] = block.block_id

    def deallocate(self, seq):
        # Returning prefix pages last makes them recent in the free-page LRU.
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            assert block.ref_count > 0
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.block_table = []
        seq.num_cached_tokens = 0

    def can_append(self, seq):
        return seq.num_blocks <= len(seq.block_table) or bool(self.free_block_ids)

    def append(self, seq):
        while len(seq.block_table) < seq.num_blocks:
            block = self._allocate_block(self.free_block_ids[0])
            seq.block_table.append(block.block_id)

    def clear_cache(self):
        assert not self.used_block_ids, "Cannot clear a cache with live requests"
        self.hash_to_block_id.clear()
        for block in self.blocks:
            block.ready = False
            block.hash = -1
            block.token_ids = []
