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


# 将该调用设为无梯度推理
@torch.inference_mode()
# 初始化 worker 并处理控制器命令
def main():
    # 设置 worker 编号
    rank = int(os.environ['RANK'])
    # 设置本机 GPU 编号
    local = int(os.environ['LOCAL_RANK'])
    # 设置进程总数
    world = int(os.environ['WORLD_SIZE'])
    # 设置部署模式
    mode = os.environ['BENCH_MODE']
    # 设置实验配置
    config = json.loads(Path(os.environ['BENCH_CONFIG']).read_text())
    # 将当前进程绑定到自己的 GPU
    torch.cuda.set_device(local)
    # 普通新张量默认留在 CPU，按需显式迁到 GPU
    torch.set_default_device('cpu')
    # 当前受控 Target 使用 FP16 参数
    torch.set_default_dtype(torch.float16)
    # 固定随机种子，减少初始化差异
    torch.manual_seed(42)
    # FP32 矩阵乘采用最高精度设置
    torch.set_float32_matmul_precision('highest')
    # 设置 torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    # 禁止 FP16 GEMM 使用低精度中间归约
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    # 输出本轮进度或诊断信息
    print(f'[{mode} rank {rank}] initializing NCCL, world={world}', flush=True)
    # 建立两端都可参与的 NCCL WORLD 通信组
    dist.init_process_group('nccl', init_method='env://', timeout=timedelta(seconds=180),
                            # 向 NCCL 指明本进程绑定的 GPU
                            device_id=torch.device('cuda', local))
    # 设置模型计算通信组
    model_group = dist.group.WORLD
    # DP 与 PD 两卡分别持有完整模型，需要单 rank 模型组
    if mode in ('dp2', 'pd'):
        # 设置量化模块或通信分组
        groups = [dist.new_group([i], backend='nccl', timeout=timedelta(seconds=180)) for i in range(world)]
        # 完整副本只在自身模型组计算，避免被 WORLD 错误分片
        model_group = groups[rank]
    # 输出本轮进度或诊断信息
    print(f'[{mode} rank {rank}] loading existing Qwen3-8B, model TP={dist.get_world_size(model_group)}', flush=True)
    # 设置 KV 页数量或页表
    pages = config['logical_kv_pages'] if mode in ('single', 'tp2') else config['logical_kv_pages']//2
    # 设置受控 Target 执行器
    target = Target(config['model_path'], local, model_group, pages)
    # 只有 TP2 才允许将模型分片到两卡
    assert target.tp == (2 if mode == 'tp2' else 1), 'Replica group unexpectedly shards full-model DP/PD weights'
    # 等待 GPU 工作结束，使计时和完成响应准确
    torch.cuda.synchronize()
    # 等所有卡完成模型加载后再处理控制器命令
    dist.barrier(device_ids=[local])
    # 先将控制器连接设为空
    conn = None
    # 按部署模式或命令类型选择分支
    if mode != 'tp2' or rank == 0:
        # 设置控制器地址、控制器端口
        host, port = os.environ['BENCH_CONTROL_ADDRESS'].split(':')
        # 设置控制器连接
        conn = Client((host, int(port)), authkey=bytes.fromhex(os.environ['BENCH_CONTROL_KEY']))
        # 向控制器报告加载完成及页池容量
        conn.send(dict(type='ready', rank=rank, pages=pages, loading=target.loading,
                       # 附上真实算术配置与当前显存统计
                       arithmetic=target.arithmetic, stats=target.stats(rank)))
    # 输出本轮进度或诊断信息
    print(f'[{mode} rank {rank}] ready, KV pages={pages}', flush=True)
    # 执行操作并确保后续清理
    try:
        # 条件成立时继续初始化 worker 并处理控制器命令
        while True:
            # 按部署模式或命令类型选择分支
            if mode == 'tp2':
                # 设置控制器命令
                command = [conn.recv() if rank == 0 else None]
                # TP 两卡必须收到同一个前向或控制命令
                dist.broadcast_object_list(command, src=0, device=torch.device('cuda', local))
                # 设置控制器命令
                command = command[0]
            # 处理上述条件不成立的情况
            else:
                # 设置控制器命令
                command = conn.recv()
            # 设置操作类型
            op = command['op']
            # 设置计时起点
            began = time.monotonic()
            # 初始化本轮执行结果
            result = {}
            # 按部署模式或命令类型选择分支
            if op == 'close':
                # 控制器连接已建立时执行此分支
                if conn is not None:
                    # 报告当前 worker 已收到关闭命令
                    conn.send(dict(type='closed', rank=rank))
                # 停止当前循环，保留已有进度
                break
            # 按部署模式或命令类型选择分支
            elif op == 'reset':
                # 清空本 worker 的 KV，隔离不同实验轮次
                target.cache.zero_()
                # 清零执行计数
                target.reset_counters()
                # 等待 GPU 工作结束，使计时和完成响应准确
                torch.cuda.synchronize()
                # 从本轮开始重新统计显存峰值
                torch.cuda.reset_peak_memory_stats()
            # 按部署模式或命令类型选择分支
            elif op == 'step':
                # 设置词表分数
                logits = target.forward(command['seqs'], command['prefill'])
                # 按部署模式或命令类型选择分支
                if rank == 0 or mode != 'tp2':
                    # 设置 Token 序列
                    result['tokens'] = logits.argmax(-1).cpu().tolist()
                # 移除不再需要的词表分数
                del logits
            # 按部署模式或命令类型选择分支
            elif op == 'probe':
                # 保存同历史 TP 数值探针的结果
                result['numerical_check'] = probe(target, command, rank)
            # 按部署模式或命令类型选择分支
            elif op == 'send_kv':
                # 只有 P 卡允许发送 Prompt KV
                assert mode == 'pd' and rank == 0
                # 设置打包计时起点
                before_pack = time.monotonic()
                # 设置按逻辑位置连续排列的全层 KV
                packed = target.pack(command['block_table'], command['length'])
                # 等待 GPU 工作结束，使计时和完成响应准确
                torch.cuda.synchronize()
                # 设置 KV 打包耗时
                pack_s = time.monotonic() - before_pack
                # 诊断开启时再做额外检查
                if command.get('diagnostic'):
                    # 设置 sha256
                    result['sha256'] = tensor_sha(packed)
                # 设置发送计时起点
                before_send = time.monotonic()
                # 通过 WORLD 的 NCCL 点对点发送全层 KV
                dist.send(packed, dst=1)
                # 等待 GPU 工作结束，使计时和完成响应准确
                torch.cuda.synchronize()
                # 返回发送端实际字节与打包耗时
                result.update(bytes=packed.numel()*packed.element_size(), pack_s=pack_s,
                              # 指定通信及等待耗时
                              transport_wait_s=time.monotonic()-before_send)
                # 更新发送 KV 字节数
                target.counters['transfer_send_bytes'] += result['bytes']
                # 移除不再需要的按逻辑位置连续排列的全层 KV
                del packed
            # 按部署模式或命令类型选择分支
            elif op == 'recv_kv':
                # 只有 D 卡允许接收 Prompt KV
                assert mode == 'pd' and rank == 1
                # 设置输出或迁移张量形状
                shape = (2, target.hc.num_hidden_layers, command['length'],
                         # 补全全层 KV 张量的 Head 维度
                         target.hc.num_key_value_heads, target.hc.head_dim)
                # 设置按逻辑位置连续排列的全层 KV
                packed = torch.empty(shape, dtype=torch.float16, device=target.device)
                # 设置接收计时起点
                before_recv = time.monotonic()
                # 接收 P 卡已打包的全层 KV，不重新 Prefill
                dist.recv(packed, src=0)
                # 等待 GPU 工作结束，使计时和完成响应准确
                torch.cuda.synchronize()
                # 设置接收等待耗时
                receive_s = time.monotonic()-before_recv
                # 设置重映射计时起点
                before_copy = time.monotonic()
                # 根据 D 卡物理页表写入迁移数据
                target.unpack(packed, command['block_table'], command['length'])
                # 等待 GPU 工作结束，使计时和完成响应准确
                torch.cuda.synchronize()
                # 设置写入目的页的耗时
                unpack_s = time.monotonic()-before_copy
                # 诊断开启时再做额外检查
                if command.get('diagnostic'):
                    # 设置 sha256
                    result['sha256'] = tensor_sha(packed)
                    # 设置 mapped_sha256
                    result['mapped_sha256'] = tensor_sha(target.pack(command['block_table'], command['length']))
                # 返回接收等待与目的页写入耗时
                result.update(bytes=packed.numel()*packed.element_size(),
                              # 指定通信及等待耗时
                              transport_wait_s=receive_s, unpack_s=unpack_s)
                # 更新接收 KV 字节数
                target.counters['transfer_recv_bytes'] += result['bytes']
                # 移除不再需要的按逻辑位置连续排列的全层 KV
                del packed
            # 按部署模式或命令类型选择分支
            elif op == 'stats':
                # 设置各卡统计
                result['ranks'] = [target.stats(rank)]
                # 按部署模式或命令类型选择分支
                if mode == 'tp2':
                    # 为每个 TP rank 准备统计结果位置
                    all_stats = [None]*world
                    # 收齐 TP 各卡统计，仅用于报告
                    dist.all_gather_object(all_stats, result['ranks'][0])
                    # 设置各卡统计
                    result['ranks'] = all_stats
            # 处理上述条件不成立的情况
            else:
                # 状态不满足要求时立即报错
                raise ValueError(f'Unknown worker operation: {op}')
            # 等待 GPU 工作结束，使计时和完成响应准确
            torch.cuda.synchronize()
            # 控制器连接已建立时执行此分支
            if conn is not None:
                # 设置操作结束时刻
                ended = time.monotonic()
                # 向控制器返回完成响应，KV 本身不经过此连接
                conn.send(dict(type='result', rank=rank, op=op, worker_s=ended-began,
                               # 返回操作时间区间以分析两卡前向重叠
                               worker_begin_monotonic_s=began, worker_end_monotonic_s=ended, **result))
    # 捕获异常并保留失败现场
    except Exception:
        # 设置异常堆栈
        detail = traceback.format_exc()
        # 输出本轮进度或诊断信息
        print(detail, flush=True)
        # 控制器连接已建立时执行此分支
        if conn is not None:
            # 执行操作并确保后续清理
            try:
                # 尽可能将失败堆栈通知控制器
                conn.send(dict(type='error', rank=rank, traceback=detail))
            # 捕获异常并保留失败现场
            except Exception:
                # 忽略异常报告通道自身的失败
                pass
        # 继续抛出原始 worker 异常
        raise
    # 无论成功或异常都执行清理
    finally:
        # 控制器连接已建立时执行此分支
        if conn is not None:
            # 释放本执行器持有的资源
            conn.close()
        # 只有进程组已创建才执行 NCCL 清理
        if dist.is_initialized():
            # 关闭进程通信组，避免资源残留
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
