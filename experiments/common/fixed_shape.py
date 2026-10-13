"""Process-local single-GPU linear wrappers with a fixed GEMM row dimension.

Only linear inputs are padded. Returned rows are trimmed before norms,
attention, KV writes, hidden-state hooks, or sampling can see padding.
This reduces shape-dependent numerics; GPU parity must still be measured.
"""
import torch

ROWS = 32

def fixed_rows_forward(forward, x, rows=ROWS):
    assert rows > 0
    leading = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] == 0:
        return forward(x)
    parts = []
    for start in range(0, flat.shape[0], rows):
        real = min(rows, flat.shape[0]-start)
        tile = flat.new_zeros((rows, flat.shape[-1]))
        tile[:real] = flat[start:start+real]
        y = forward(tile)
        assert isinstance(y, torch.Tensor) and y.ndim == 2 and y.shape[0] == rows
        parts.append(y[:real])
    result = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    return result.reshape(*leading, result.shape[-1])

def fixed_linear(x, weight, bias=None, rows=ROWS):
    return fixed_rows_forward(lambda tile: torch.nn.functional.linear(tile, weight, bias), x, rows)

def install_fixed_shape_linears(model):
    from myvllm.layers.linear import LinearBase
    count = 0
    for module in model.model.modules():
        if isinstance(module, LinearBase):
            assert module.tp_size == 1, 'Fixed-shape experiment is single-GPU only'
            assert not hasattr(module, '_dflash_original_forward'), 'Already installed'
            original = module.forward
            module._dflash_original_forward = original
            def forward(x, original=original):
                return fixed_rows_forward(original, x)
            module.forward = forward
            count += 1
    expected = 4 * len(model.model.layers)
    assert count == expected, (count, expected)
    return dict(rows=ROWS, backbone_linear_modules=count, lm_head_rows=ROWS,
                padding_scope='Linear only; never visible to attention/KV/sampling',
                source_modified=False)
