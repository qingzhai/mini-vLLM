"""Frozen-QKV checks: paged prefix attention against explicit FP32 attention."""
import math
from types import SimpleNamespace

import torch
from myvllm.layers.prefix_attention import paged_prefix_prefill


@torch.inference_mode()
def verify_prefix_kernel():
    page, dim, kv_heads = 256, 128, 8
    # Includes a cache miss, one remaining Q token, partial final tiles,
    # and mixed lengths within one batch. Pages are deliberately permuted.
    specs = [(0, 33), (256, 1), (512, 148), (256, 19)]
    page_counts = [math.ceil((p + n) / page) for p, n in specs]
    width, pages = max(page_counts), sum(page_counts) + 3
    generator = torch.Generator(device="cpu").manual_seed(731)
    order = torch.randperm(pages, generator=generator, device="cpu").tolist()
    tables, cursor = [], 0
    for count in page_counts:
        tables.append(order[cursor:cursor + count] + [-1] * (width - count))
        cursor += count
    cuq, cuk = [0], [0]
    for prefix, length in specs:
        cuq.append(cuq[-1] + length)
        cuk.append(cuk[-1] + prefix + length)
    context = SimpleNamespace(
        max_seqlen_q=max(n for _, n in specs),
        cu_seqlens_q=torch.tensor(cuq, dtype=torch.int32, device="cuda"),
        cu_seqlens_k=torch.tensor(cuk, dtype=torch.int32, device="cuda"),
        block_tables=torch.tensor(tables, dtype=torch.int32, device="cuda"),
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    reports = []
    try:
        for dtype in (torch.bfloat16, torch.float16):
            for heads in (16, 32):
                torch.cuda.manual_seed_all(731)
                q = torch.randn(cuq[-1], heads, dim, device="cuda", dtype=dtype)
                kc = torch.full((pages, page, kv_heads, dim), float("nan"),
                                device="cuda", dtype=dtype)
                vc = torch.full_like(kc, float("nan"))
                references = []
                for seq, (prefix, length) in enumerate(specs):
                    total = prefix + length
                    k = torch.randn(total, kv_heads, dim, device="cuda", dtype=dtype)
                    v = torch.randn_like(k)
                    for logical in range(page_counts[seq]):
                        start = logical * page
                        count = min(page, total - start)
                        physical = tables[seq][logical]
                        kc[physical, :count].copy_(k[start:start + count])
                        vc[physical, :count].copy_(v[start:start + count])
                    qf = q[cuq[seq]:cuq[seq + 1]].float().transpose(0, 1)
                    kf = k.float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
                    vf = v.float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
                    scores = torch.bmm(qf, kf.transpose(1, 2)) / math.sqrt(dim)
                    rows = prefix + torch.arange(length, device="cuda")
                    cols = torch.arange(total, device="cuda")
                    scores.masked_fill_(cols[None, :] > rows[:, None], float("-inf"))
                    references.append(torch.bmm(scores.softmax(-1), vf).transpose(0, 1))
                expected = torch.cat(references, dim=0)
                actual = paged_prefix_prefill(q, kc, vc, context, 1 / math.sqrt(dim),
                                              heads, kv_heads, dim, page).float()
                assert torch.isfinite(actual).all().item(), "Paged kernel returned non-finite values"
                # Fixed input-level bounds, independent of observed model logits.
                atol, rtol = (0.01, 0.01) if dtype == torch.bfloat16 else (0.002, 0.002)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                delta = (actual - expected).abs()
                row = dict(dtype=str(dtype), heads=heads, kv_heads=kv_heads,
                           max_abs=delta.max().item(), mean_abs=delta.mean().item(),
                           atol=atol, rtol=rtol, passed=True)
                reports.append(row)
                print("PREFIX KERNEL CHECK PASS:", row, flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return reports
