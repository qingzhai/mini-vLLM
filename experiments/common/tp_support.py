"""Strict streaming checkpoint loader and FP32 sharded head; no source edits."""
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn import functional as F


def load_tensors(model,items):
    parameters=dict(model.named_parameters())
    loaded=set();sources=set();sharded=0
    for name,weight in items:
        assert name not in sources, 'Duplicate checkpoint tensor: '+name
        sources.add(name);target=name;part=None
        for suffix,label in [('q_proj','q'),('k_proj','k'),('v_proj','v')]:
            if f'.self_attn.{suffix}.' in name:
                target=name.replace(f'.self_attn.{suffix}.','.self_attn.qkv_projection.')
                part=label;break
        for suffix,label in [('gate_proj',0),('up_proj',1)]:
            if f'.mlp.{suffix}.' in name:
                target=name.replace(f'.mlp.{suffix}.','.mlp.gate_up.')
                part=label;break
        if name.endswith('.rotary_emb.inv_freq'):
            continue
        assert target in parameters, 'Unmapped checkpoint tensor: '+name
        param=parameters[target]
        module_name,attribute=target.rsplit('.',1)
        module=model.get_submodule(module_name)
        loader=getattr(module,'weight_loader',None)
        if part is not None:
            assert callable(loader), 'Missing merged shard loader: '+target
            loader(param,weight,part)
        elif callable(loader):
            loader(param,weight)
        else:
            assert tuple(param.shape)==tuple(weight.shape), (target,param.shape,weight.shape)
            param.data.copy_(weight)
        sharded+=int(getattr(module,'tp_size',1)>1)
        loaded.add(name)
    missing=[]
    for name in parameters:
        if '.self_attn.qkv_projection.' in name:
            expected=[name.replace('.self_attn.qkv_projection.',f'.self_attn.{label}_proj.') for label in ('q','k','v')]
        elif '.mlp.gate_up.' in name:
            expected=[name.replace('.mlp.gate_up.',f'.mlp.{label}_proj.') for label in ('gate','up')]
        else: expected=[name]
        missing.extend(x for x in expected if x not in loaded)
    assert not missing, 'Checkpoint missing required tensors: '+str(missing[:10])
    return dict(checkpoint_tensors=len(loaded),model_parameters=len(parameters),shard_loader_calls=sharded,strict=True)


def load_checkpoint(model,path):
    from safetensors import safe_open
    files=sorted(Path(path).glob('*.safetensors'))
    assert files, 'No local safetensors checkpoint: '+str(path)
    def items():
        for p in files:
            with safe_open(str(p),framework='pt',device='cpu') as f:
                for name in f.keys(): yield name,f.get_tensor(name)
    return load_tensors(model,items())


def install_sharded_fp32_head(model):
    from myvllm.utils import get_context
    head=model.lm_head
    weight=head.weight.detach().float().contiguous()
    def forward(x):
        context=get_context()
        if context.is_prefill:
            x=x[context.cu_seqlens_q[1:]-1].contiguous()
        local=F.linear(x.float(),weight)
        if head.tp_size>1:
            pieces=[torch.empty_like(local) for _ in range(head.tp_size)]
            dist.all_gather(pieces,local.contiguous())
            local=torch.cat(pieces,dim=-1)[...,:head.num_embeddings]
        return local
    head.forward=forward
    return dict(local_rows=weight.shape[0],compute_dtype='float32',extra_bytes=weight.numel()*4,collective='all_gather')


def encode_sequence(seq):
    # Full metadata is required by patched prepare_prefill and block properties.
    # Stock Sequence.__getstate__ omits block_size; do not pickle Sequence itself.
    return vars(seq).copy()


def decode_sequence(values,cls):
    obj=cls.__new__(cls);obj.__dict__.update(values)
    return obj
