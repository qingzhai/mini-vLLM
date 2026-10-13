"""Single-GPU AutoAWQ GEMM-format W4A16; independently implemented Triton GEMM.

Format reference: casper-hansen/AutoAWQ/awq/utils/packing_utils.py.
Eight output columns per int32, reverse nibble order [0,4,1,5,2,6,3,7].
Weights are unpacked inside each GEMM tile, never cached as a dense matrix.
"""
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _gemm(A, W, Z, S, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    lane = cols % 8
    shift = ((lane % 2) * 4 + lane // 2) * 4
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + inner
        a = tl.load(A + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        packed = tl.load(W + kk[:, None] * (N // 8) + cols[None, :] // 8,
                         (kk[:, None] < K) & (cols[None, :] < N), other=0)
        zero = tl.load(Z + (kk[:, None] // G) * (N // 8) + cols[None, :] // 8,
                       (kk[:, None] < K) & (cols[None, :] < N), other=0)
        scale = tl.load(S + (kk[:, None] // G) * N + cols[None, :],
                        (kk[:, None] < K) & (cols[None, :] < N), other=0)
        quant = (packed >> shift[None, :]) & 15
        zero = (zero >> shift[None, :]) & 15
        dense = ((quant - zero).to(tl.float32) * scale.to(tl.float32)).to(tl.float16)
        acc = tl.dot(a, dense, acc)
    tl.store(O + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


class AWQLinear(nn.Module):
    def __init__(self, qweight, qzeros, scales, group_size=128):
        super().__init__()
        self.in_features = qweight.shape[0]
        self.out_features = qweight.shape[1] * 8
        self.group_size = group_size
        assert qweight.dtype == qzeros.dtype == torch.int32
        assert scales.dtype in (torch.float16, torch.bfloat16)
        assert self.in_features % group_size == 0
        assert qzeros.shape == (self.in_features // group_size, self.out_features // 8)
        assert scales.shape == (self.in_features // group_size, self.out_features)
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("qzeros", qzeros.contiguous())
        self.register_buffer("scales", scales.contiguous())

    def forward(self, x):
        assert x.dtype == torch.float16 and x.is_cuda
        assert x.shape[-1] == self.in_features
        shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, self.in_features).contiguous()
        out = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
        if x.shape[0]:
            grid = (triton.cdiv(x.shape[0], 16), triton.cdiv(self.out_features, 64))
            _gemm[grid](x, self.qweight, self.qzeros, self.scales, out,
                        x.shape[0], self.out_features, self.in_features,
                        self.group_size, 16, 64, 32, num_warps=4)
        return out.reshape(shape)


def dequantize_reference(layer):
    """Independent PyTorch unpacking; only used for the small operator check."""
    device = layer.qweight.device
    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device=device, dtype=torch.int32)
    q = ((layer.qweight[:, :, None] >> shifts) & 15).reshape(layer.in_features, -1)
    z = ((layer.qzeros[:, :, None] >> shifts) & 15).reshape(layer.in_features // layer.group_size, -1)
    z = z.repeat_interleave(layer.group_size, dim=0)
    scale = layer.scales.float().repeat_interleave(layer.group_size, dim=0)
    return ((q - z).float() * scale).half()
