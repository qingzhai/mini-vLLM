"""torchrun worker. Controller RPC carries metadata/tokens, never KV tensors."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import traceback
from datetime import timedelta
from multiprocessing.connection import Client
import numpy as np
import torch
import torch.distributed as dist
from target import Target, tensor_sha


def probe(target, command, rank):
    """Same-history TP check outside measured generation, eight full-logit rows."""
    source = command.get('reference')
    reference = np.load(source, allow_pickle=False) if source else None
    prompts = command['prompts']
    seqs = [dict(token_ids=list(p), block_table=list(range(i*8, (i+1)*8)))
            for i, p in enumerate(prompts)]
    saved, picks = [], []
    for step in range(4):
        logits = target.forward(seqs, step == 0)
        saved.append(logits.cpu().numpy())
        tokens = (reference['tokens'][step].tolist() if reference is not None
                  else logits.argmax(-1).cpu().tolist())
        picks.append(tokens)
        for seq, token in zip(seqs, tokens):
            seq['token_ids'].append(int(token))
    actual = np.stack(saved)
    result = dict(rows=actual.shape[0]*actual.shape[1], teacher_forced=True)
    if reference is not None:
        expected = reference['logits']
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        relative = float(np.sqrt(np.mean(delta**2)) / max(np.sqrt(np.mean(expected.astype(np.float64)**2)), 1e-8))
        result.update(max_abs=float(np.abs(delta).max()), relative_rmse=relative,
                      greedy_agreement=float(np.mean(actual.argmax(-1) == expected.argmax(-1))),
                      max_abs_limit=0.35, relative_rmse_limit=0.01)
        result['passed'] = bool(np.isfinite(actual).all() and result['max_abs'] <= 0.35 and relative <= 0.01)
    else:
        result['passed'] = bool(np.isfinite(actual).all())
    if rank == 0:
        np.savez_compressed(command['output'], logits=actual, tokens=np.array(picks, dtype=np.int64))
    return result


@torch.inference_mode()
def main():
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    mode = os.environ['BENCH_MODE']
    config = json.loads(Path(os.environ['BENCH_CONFIG']).read_text())
    torch.cuda.set_device(local)
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float16)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print(f'[{mode} rank {rank}] initializing NCCL, world={world}', flush=True)
    dist.init_process_group('nccl', init_method='env://', timeout=timedelta(seconds=180),
                            device_id=torch.device('cuda', local))
    model_group = dist.group.WORLD
    if mode in ('dp2', 'pd'):
        groups = [dist.new_group([i], backend='nccl', timeout=timedelta(seconds=180)) for i in range(world)]
        model_group = groups[rank]
    print(f'[{mode} rank {rank}] loading existing Qwen3-8B, model TP={dist.get_world_size(model_group)}', flush=True)
    pages = config['logical_kv_pages'] if mode in ('single', 'tp2') else config['logical_kv_pages']//2
    target = Target(config['model_path'], local, model_group, pages)
    assert target.tp == (2 if mode == 'tp2' else 1), 'Replica group unexpectedly shards full-model DP/PD weights'
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local])
    conn = None
    if mode != 'tp2' or rank == 0:
        host, port = os.environ['BENCH_CONTROL_ADDRESS'].split(':')
        conn = Client((host, int(port)), authkey=bytes.fromhex(os.environ['BENCH_CONTROL_KEY']))
        conn.send(dict(type='ready', rank=rank, pages=pages, loading=target.loading,
                       arithmetic=target.arithmetic, stats=target.stats(rank)))
    print(f'[{mode} rank {rank}] ready, KV pages={pages}', flush=True)
    try:
        while True:
            if mode == 'tp2':
                command = [conn.recv() if rank == 0 else None]
                dist.broadcast_object_list(command, src=0, device=torch.device('cuda', local))
                command = command[0]
            else:
                command = conn.recv()
            op = command['op']
            began = time.monotonic()
            result = {}
            if op == 'close':
                if conn is not None:
                    conn.send(dict(type='closed', rank=rank))
                break
            elif op == 'reset':
                target.cache.zero_()
                target.reset_counters()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            elif op == 'step':
                logits = target.forward(command['seqs'], command['prefill'])
                if rank == 0 or mode != 'tp2':
                    result['tokens'] = logits.argmax(-1).cpu().tolist()
                del logits
            elif op == 'probe':
                result['numerical_check'] = probe(target, command, rank)
            elif op == 'send_kv':
                assert mode == 'pd' and rank == 0
                before_pack = time.monotonic()
                packed = target.pack(command['block_table'], command['length'])
                torch.cuda.synchronize()
                pack_s = time.monotonic() - before_pack
                if command.get('diagnostic'):
                    result['sha256'] = tensor_sha(packed)
                before_send = time.monotonic()
                dist.send(packed, dst=1)
                torch.cuda.synchronize()
                result.update(bytes=packed.numel()*packed.element_size(), pack_s=pack_s,
                              transport_wait_s=time.monotonic()-before_send)
                target.counters['transfer_send_bytes'] += result['bytes']
                del packed
            elif op == 'recv_kv':
                assert mode == 'pd' and rank == 1
                shape = (2, target.hc.num_hidden_layers, command['length'],
                         target.hc.num_key_value_heads, target.hc.head_dim)
                packed = torch.empty(shape, dtype=torch.float16, device=target.device)
                before_recv = time.monotonic()
                dist.recv(packed, src=0)
                torch.cuda.synchronize()
                receive_s = time.monotonic()-before_recv
                before_copy = time.monotonic()
                target.unpack(packed, command['block_table'], command['length'])
                torch.cuda.synchronize()
                unpack_s = time.monotonic()-before_copy
                if command.get('diagnostic'):
                    result['sha256'] = tensor_sha(packed)
                    result['mapped_sha256'] = tensor_sha(target.pack(command['block_table'], command['length']))
                result.update(bytes=packed.numel()*packed.element_size(),
                              transport_wait_s=receive_s, unpack_s=unpack_s)
                target.counters['transfer_recv_bytes'] += result['bytes']
                del packed
            elif op == 'stats':
                result['ranks'] = [target.stats(rank)]
                if mode == 'tp2':
                    all_stats = [None]*world
                    dist.all_gather_object(all_stats, result['ranks'][0])
                    result['ranks'] = all_stats
            else:
                raise ValueError(f'Unknown worker operation: {op}')
            torch.cuda.synchronize()
            if conn is not None:
                ended = time.monotonic()
                conn.send(dict(type='result', rank=rank, op=op, worker_s=ended-began,
                               worker_begin_monotonic_s=began, worker_end_monotonic_s=ended, **result))
    except Exception:
        detail = traceback.format_exc()
        print(detail, flush=True)
        if conn is not None:
            try:
                conn.send(dict(type='error', rank=rank, traceback=detail))
            except Exception:
                pass
        raise
    finally:
        if conn is not None:
            conn.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
