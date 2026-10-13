"""Process-local FP32 LM Head for the single-GPU Qwen test runner."""
import torch
from myvllm.utils import get_context


def install_fp32_head(model):
    head = model.lm_head
    assert head.tp_size == 1, "This diagnostic only supports single-GPU inference"
    assert head.weight.ndim == 2
    weight = head.weight.detach().float().contiguous()
    assert weight.dtype == torch.float32
    assert weight.shape[0] == head.num_embeddings

    def forward(x):
        context = get_context()
        if context.is_prefill:
            # Match ParallelLMHead's ordinary Prefill gather exactly. The chunk
            # runtime already reset the context after selecting its ready rows.
            x = x[context.cu_seqlens_q[1:] - 1].contiguous()
        assert x.shape[-1] == weight.shape[1]
        # Compute the matrix product in FP32; converting already rounded BF16
        # logits to FP32 would retain the original tie and is NOT this change.
        logits = torch.nn.functional.linear(x.float(), weight)
        assert logits.dtype == torch.float32
        return logits

    head.forward = forward
    return dict(original_weight_dtype=str(head.weight.dtype),
                compute_dtype="torch.float32", output_dtype="torch.float32",
                extra_weight_bytes=weight.numel() * weight.element_size(),
                repository_source_modified=False)
