"""Shared model forward, phase-specific attention kernels for mixed batches."""
from myvllm.layers.prefix_attention import paged_prefix_prefill


def mixed_paged_prefill(q, k_cache, v_cache, context, scale,
                       num_heads, num_kv_heads, head_dim, block_size):
    output = paged_prefix_prefill(q, k_cache, v_cache, context, scale,
                                 num_heads, num_kv_heads, head_dim, block_size)
    indices = getattr(context, "chunk_decode_query_indices", None)
    if indices is not None:
        # Import after Attention has finished loading to avoid a circular import.
        from myvllm.layers.attention import paged_attention_decode
        decoded = paged_attention_decode(
            q.index_select(0, indices), k_cache, v_cache,
            context.chunk_decode_block_tables, context.chunk_decode_context_lens,
            scale, num_heads, num_kv_heads, head_dim, block_size)
        output.index_copy_(0, indices, decoded)
    return output
