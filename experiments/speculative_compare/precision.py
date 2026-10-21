"""Explicit Target precision controls, shared by AR and speculative modes."""


def configure_target_backbone(target, requested='checkpoint'):
    import torch

    if requested not in ('checkpoint', 'fp16'):
        raise ValueError(f'Unsupported Target backbone precision: {requested}')
    original_dtype = str(next(target.model.parameters()).dtype)
    if requested == 'fp16':
        if getattr(target.config, 'tie_word_embeddings', False):
            raise ValueError('Reviewed FP16 conversion requires an untied Target head')
        # 只转换主干参数，保留 RoPE 等 FP32 Buffer 及尚未配置的 LM Head。
        with torch.no_grad():
            for parameter in target.model.parameters():
                if parameter.is_floating_point():
                    parameter.data = parameter.data.to(dtype=torch.float16)
        if any(p.dtype != torch.float16 for p in target.model.parameters() if p.is_floating_point()):
            raise ValueError('Target backbone conversion is incomplete')
    return dict(requested=requested, original_backbone_dtype=original_dtype,
                backbone_dtype=str(next(target.model.parameters()).dtype),
                buffers_converted=False, checkpoint_files_modified=False)


def install_draft_feature_bridge(draft):
    import torch
    from torch import nn

    if not isinstance(draft.fc, nn.Linear) or draft.fc.weight.dtype != torch.bfloat16:
        raise ValueError('Expected the reviewed BF16 Draft context projection')

    def cast_features(module, args, kwargs):
        if args:
            return (args[0].to(dtype=module.weight.dtype), *args[1:]), kwargs
        return args, {**kwargs, 'input': kwargs['input'].to(dtype=module.weight.dtype)}

    # 三种官方 Draft 均经 fc 投影 Target 特征；在边界转换，不改变 Target KV。
    draft.fc.register_forward_pre_hook(cast_features, with_kwargs=True)
    return dict(boundary='draft.fc input', source_dtype='torch.float16',
                projection_dtype=str(draft.fc.weight.dtype), target_kv_modified=False,
                official_source_modified=False)


def configure_target_head(target, requested='checkpoint'):
    import torch
    from torch import nn
    import torch.nn.functional as F

    if requested not in ('checkpoint', 'fp32'):
        raise ValueError(f'Unsupported Target head precision: {requested}')
    original = target.lm_head
    original_dtype = str(original.weight.dtype)
    original_bytes = sum(p.numel() * p.element_size() for p in target.parameters())
    if requested == 'fp32':
        if not isinstance(original, nn.Linear) or getattr(target.config, 'tie_word_embeddings', False):
            raise ValueError('FP32 control requires the reviewed untied Linear Qwen3 Target head')

        class FP32Head(nn.Module):
            def __init__(self, head):
                super().__init__()
                self.in_features, self.out_features = head.in_features, head.out_features
                # 转换已加载的 checkpoint 值，不改变磁盘上的权重。
                self.weight = nn.Parameter(head.weight.detach().float(), requires_grad=False)
                self.bias = nn.Parameter(head.bias.detach().float(), requires_grad=False) if head.bias is not None else None

            def forward(self, hidden):
                # 真正执行 FP32 GEMM；不是把已经舍入的 BF16 Logits 转成 float。
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    return F.linear(hidden.float(), self.weight, self.bias)

        target.lm_head = FP32Head(original).train(original.training)
    return dict(requested=requested, original_head_dtype=original_dtype,
        head_weight_dtype=str(target.lm_head.weight.dtype),
        head_compute_dtype='torch.float32' if requested == 'fp32' else original_dtype,
        head_output_dtype='torch.float32' if requested == 'fp32' else original_dtype,
        backbone_dtype=str(next(target.model.parameters()).dtype),
        added_parameter_bytes=sum(p.numel() * p.element_size() for p in target.parameters()) - original_bytes,
        tf32=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        checkpoint_files_modified=False)
