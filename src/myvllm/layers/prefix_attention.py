"""Triton causal prefill over paged KV, with Q starting after a cached prefix."""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_prefix_prefill(
    Q, KC, VC, O, CUQ, CUK, BT,
    scale: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
    D: tl.constexpr, PAGE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.program_id(2)
    qstart = tl.load(CUQ + seq)
    qlen = tl.load(CUQ + seq + 1) - qstart
    klen = tl.load(CUK + seq + 1) - tl.load(CUK + seq)
    prefix = klen - qlen
    if tile * BM >= qlen:
        return
    kh = head // (H // HK)
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    q = tl.load(Q + (qstart + rows[:, None]) * H * D
                + head * D + dims[None, :],
                mask=rows[:, None] < qlen, other=0)
    m = tl.full((BM,), float("-inf"), tl.float32)
    normalizer = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for tile_n in range(tl.cdiv(klen, BN)):
        cols = tile_n * BN + tl.arange(0, BN)
        valid = (cols < klen) & (cols // PAGE < MAX_BLOCKS)
        physical = tl.load(BT + seq * MAX_BLOCKS + cols // PAGE,
                           mask=valid, other=-1)
        valid = valid & (physical >= 0)
        physical = tl.where(valid, physical, 0).to(tl.int64)
        base = physical * PAGE * HK * D + (cols % PAGE) * HK * D + kh * D
        k = tl.load(KC + base[None, :] + dims[:, None],
                    mask=valid[None, :], other=0)
        score = tl.dot(q, k) * scale
        causal = cols[None, :] <= prefix + rows[:, None]
        score = tl.where(causal & valid[None, :] & (rows[:, None] < qlen),
                         score, -1.0e10)
        new_m = tl.maximum(m, tl.max(score, 1))
        alpha = tl.exp(m - new_m)
        probability = tl.exp(score - new_m[:, None])
        v = tl.load(VC + base[:, None] + dims[None, :],
                    mask=valid[:, None], other=0)
        acc = acc * alpha[:, None] + tl.dot(probability.to(v.dtype), v)
        normalizer = normalizer * alpha + tl.sum(probability, 1)
        m = new_m
    result = acc / normalizer[:, None]
    tl.store(O + (qstart + rows[:, None]) * H * D + head * D + dims[None, :],
             result, mask=rows[:, None] < qlen)


def paged_prefix_prefill(q, k_cache, v_cache, context, scale,
                         num_heads, num_kv_heads, head_dim, block_size):
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert head_dim in (32, 64, 128, 256)
    q = q.contiguous()
    output = torch.empty_like(q)
    # BM/BN match the existing dense prefill at head_dim=128.
    bm = bn = 32 if head_dim <= 128 else 16
    grid = (triton.cdiv(context.max_seqlen_q, bm), num_heads,
            context.cu_seqlens_q.numel() - 1)
    _paged_prefix_prefill[grid](
        q, k_cache, v_cache, output,
        context.cu_seqlens_q, context.cu_seqlens_k, context.block_tables,
        scale=scale, H=num_heads, HK=num_kv_heads, D=head_dim,
        PAGE=block_size, MAX_BLOCKS=context.block_tables.shape[1],
        BM=bm, BN=bn, num_warps=4,
    )
    return output
