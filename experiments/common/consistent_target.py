"""Use the existing one-query PagedAttention for every target query.

No new CUDA/Triton kernel. Metadata expands a packed multi-query window into
independent query rows. Context length is absolute position + 1 for each row.
Only this experimental runtime opts in; ordinary engine behavior is untouched.
"""
import torch
from fixed_shape import fixed_rows_forward
from myvllm.utils import get_context


def attach_query_metadata(views, table_width=None):
    context = get_context()
    needed = max(len(v.block_table) for v in views)
    width = needed if table_width is None else table_width
    assert width >= needed and width > 0
    tables, lengths = [], []
    for view in views:
        table = view.block_table + [-1]*(width-len(view.block_table))
        for position in range(view._scheduled_start,view._scheduled_end):
            tables.append(table)
            lengths.append(position+1)
    device = context.slot_mapping.device
    context.spec_query_tables = torch.tensor(tables,dtype=torch.int32,device=device)
    context.spec_query_lengths = torch.tensor(lengths,dtype=torch.int32,device=device)
    assert len(lengths)==context.slot_mapping.numel()


def install_consistent_target(model, canonical_rms=True):
    from myvllm.layers.attention import Attention,store_kvcache,paged_attention_decode
    counts = dict(attention=0,rms=0,activation=0)
    for module in model.model.modules():
        if isinstance(module,Attention):
            assert not hasattr(module,'_spec_original_attention'), 'Already installed'
            original=module.forward
            module._spec_original_attention=original
            def forward(q,k,v,module=module,original=original):
                context=get_context()
                if not hasattr(context,'spec_query_tables'):
                    return original(q,k,v)
                assert q.ndim==k.ndim==v.ndim==3
                assert q.dtype==k.dtype==v.dtype==module.k_cache.dtype==module.v_cache.dtype
                assert q.dtype==torch.float16
                assert context.spec_query_lengths.numel()==q.shape[0]
                store_kvcache(k.contiguous(),v.contiguous(),module.k_cache,module.v_cache,
                              context.slot_mapping,module.block_size)
                out=paged_attention_decode(q,module.k_cache,module.v_cache,
                    context.spec_query_tables,context.spec_query_lengths,
                    module.scale/(module.head_dim**0.5),module.num_heads,module.num_kv_heads,
                    module.head_dim,module.block_size)
                return out.reshape(q.shape[0],module.num_heads*module.head_dim)
            module.forward=forward
            counts['attention']+=1
        elif canonical_rms and type(module).__name__=='LayerNorm':
            original=module.rms_forward
            assert not hasattr(original,'_torchdynamo_orig_callable'), 'Unwrap compilation first'
            # Q/K norm (D=128) can contain hundreds of head rows per query block;
            # a fixed 1024-row tile avoids many small Python/GPU launches.
            rows=1024 if module.weight.numel()<=128 else 32
            def rms(x,original=original,rows=rows):
                return fixed_rows_forward(original,x,rows)
            module.rms_forward=rms
            counts['rms']+=1
        elif canonical_rms and type(module).__name__=='SiluAndMul':
            original=module.forward
            assert not hasattr(original,'_torchdynamo_orig_callable'), 'Unwrap compilation first'
            def activation(x,original=original):
                return fixed_rows_forward(original,x,32)
            module.forward=activation
            counts['activation']+=1
    assert counts['attention']==len(model.model.layers)==36,counts
    if canonical_rms: assert counts['rms']==145 and counts['activation']==36,counts
    return dict(**counts,attention_kernel='Existing paged_attention_decode for every query, including initial Prefill',
                attention_reduction='Per-query FP32 scores/softmax/value reduction',
                rms_rows={'D128':1024,'D4096':32},activation_rows=32,
                source_modified=False,new_cuda_kernels=False)
