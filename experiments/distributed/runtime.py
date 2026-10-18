"""CPU controller, independent replica scheduling and explicit KV handoff."""
from __future__ import annotations
from collections import deque
import json
import math
from multiprocessing.connection import Listener, wait
import os
from pathlib import Path
import queue
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
from common import PagePool, metrics, write_json


class Session:
    def __init__(self, mode, config_path, run_dir):
        self.mode, self.config_path, self.run_dir = mode, Path(config_path), Path(run_dir)
        self.config = json.loads(self.config_path.read_text())
        self.clients, self.pending, self.ready = {}, {}, {}
        self.process = None
        self.listener = None
        self.pump_thread = None

    def __enter__(self):
        expected = 2 if self.mode in ('dp2', 'pd') else 1
        auth = secrets.token_bytes(32)
        self.listener = Listener(('127.0.0.1', 0), authkey=auth)
        accepted = queue.Queue()
        def accept_clients():
            try:
                for _ in range(expected):
                    conn = self.listener.accept()
                    message = conn.recv()
                    accepted.put((conn, message))
            except Exception as exc:
                accepted.put(exc)
        threading.Thread(target=accept_clients, daemon=True).start()
        env = os.environ.copy()
        env.update(BENCH_MODE=self.mode, BENCH_CONFIG=str(self.config_path),
                   BENCH_CONTROL_ADDRESS=f'127.0.0.1:{self.listener.address[1]}',
                   BENCH_CONTROL_KEY=auth.hex(), OMP_NUM_THREADS='2',
                   PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   CUDA_VISIBLE_DEVICES='0' if self.mode == 'single' else '0,1')
        env.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
        package = Path(__file__).resolve().parent
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                   '--nnodes=1', '--nproc_per_node='+('1' if self.mode == 'single' else '2'),
                   str(package/'worker.py')]
        print('START MODE:', self.mode, flush=True)
        self.process = subprocess.Popen(command, env=env, cwd=package, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                                        start_new_session=True)
        def pump():
            with (self.run_dir/f'{self.mode}-worker.log').open('w') as stream:
                for line in self.process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line.rstrip(), flush=True)
        self.pump_thread = threading.Thread(target=pump, daemon=True)
        self.pump_thread.start()
        began, last_message = time.monotonic(), time.monotonic()
        try:
            while len(self.clients) < expected:
                self.check_alive()
                if time.monotonic()-began > 360:
                    raise TimeoutError('Worker initialization exceeded 360 seconds; see worker log')
                try:
                    item = accepted.get(timeout=0.5)
                except queue.Empty:
                    if time.monotonic()-last_message > 20:
                        print(f'{self.mode}: waiting for model/NCCL initialization, {time.monotonic()-began:.0f}s', flush=True)
                        last_message = time.monotonic()
                    continue
                if isinstance(item, Exception):
                    raise item
                conn, ready = item
                assert ready['type'] == 'ready', ready
                rank = ready['rank']
                assert rank not in self.clients
                self.clients[rank], self.ready[rank] = conn, ready
            write_json(self.run_dir/f'{self.mode}-initialization.json', self.ready)
            return self
        except BaseException:
            self.shutdown()
            raise

    def check_alive(self):
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f'{self.mode} worker exited with {self.process.returncode}; see {self.mode}-worker.log')

    def submit(self, rank, command):
        assert rank in self.clients and rank not in self.pending, (rank, self.pending)
        self.check_alive()
        self.clients[rank].send(command)
        self.pending[rank] = (command['op'], time.monotonic())

    def collect(self, timeout=0.05):
        self.check_alive()
        for rank, (op, began) in self.pending.items():
            if time.monotonic()-began > 240:
                raise TimeoutError(f'{self.mode} rank {rank} operation {op} exceeded 240 seconds')
        reverse = {self.clients[r]: r for r in self.pending}
        result = []
        for conn in wait(list(reverse), timeout=timeout) if reverse else []:
            rank = reverse[conn]
            message = conn.recv()
            op, _ = self.pending.pop(rank)
            if message['type'] == 'error':
                raise RuntimeError(message['traceback'])
            assert message['type'] == 'result' and message['rank'] == rank and message['op'] == op, message
            result.append((rank, message))
        return result

    def synchronous(self, commands):
        assert not self.pending
        for rank, command in commands.items():
            self.submit(rank, command)
        result = {}
        while self.pending:
            for rank, message in self.collect():
                result[rank] = message
        return result

    def reset(self):
        return self.synchronous({rank: dict(op='reset') for rank in self.clients})

    def stats(self):
        messages = self.synchronous({rank: dict(op='stats') for rank in self.clients})
        return [row for rank in sorted(messages) for row in messages[rank]['ranks']]

    def shutdown(self):
        if self.process is not None and self.process.poll() is None:
            if not self.pending:
                try:
                    for conn in self.clients.values():
                        conn.send(dict(op='close'))
                    for conn in self.clients.values():
                        if conn.poll(5):
                            conn.recv()
                    self.process.wait(timeout=15)
                except Exception:
                    pass
            if self.process.poll() is None:
                # The process group was created by this Session; unrelated jobs
                # and other users' Python processes are never targeted.
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
        for conn in self.clients.values():
            conn.close()
        if self.listener is not None:
            self.listener.close()
        if self.pump_thread is not None:
            self.pump_thread.join(timeout=2)
            if not self.pump_thread.is_alive() and self.process.stdout is not None:
                self.process.stdout.close()

    def __exit__(self, *exc):
        self.shutdown()


# 用同一控制器执行 Single、TP2、DP2 或 1P1D
def run_load(session, case, config, diagnostic=False, partial_path=None):
    """One full prefill per step; batched decode; same global residency cap.

    DP routes each admission to the replica with the fewest resident requests.
    PD reserves destination pages before prefill and blocks new admission when
    the global cap is reached. KV transfer is synchronous and explicitly timed.
    """
    # 设置部署模式
    mode = session.mode
    # 设置全局驻留上限
    cap = config['global_batch']
    # 全局驻留上限必须是支持范围内的偶数
    assert 2 <= cap <= 8 and cap % 2 == 0
    # 设置每个副本的页数
    replica_pages = config['logical_kv_pages']//2 if mode in ('dp2', 'pd') else config['logical_kv_pages']
    # 设置可调度 worker
    engines = list(session.clients)
    # 设置各 worker 独立页池
    pools = {rank: PagePool(replica_pages, reverse=(mode == 'pd' and rank == 1)) for rank in engines}
    # 设置各 worker 驻留请求状态
    active = {rank: {} for rank in engines}
    # 设置各卡等待 Prefill 的请求
    pending_prefill = {rank: deque() for rank in engines}
    # 设置按请求编号保存的状态
    records = {r['request_id']: dict(r, output_token_ids=[], token_times_s=[]) for r in case['requests']}
    # 设置按到达时间排列的请求
    arrivals = deque(sorted(records, key=lambda rid: records[rid]['arrival_s']))
    # 设置等待队列
    waiting = deque()
    # 设置在途 worker 操作、执行轨迹、迁移统计
    busy, trace, transfers = {}, [], []
    # 设置已完成请求
    completed = set()
    # 先将等待迁移的请求设为空
    pending_transfer = None
    # 先将当前同步迁移状态设为空
    transfer = None
    # 设置驻留请求峰值
    peak_active = 0
    # 设置计时起点
    began = time.monotonic()
    # 设置上次进度输出时间
    last_progress = began

    # 读取控制器相对时间
    def elapsed():
        # 返回 time.monotonic() - began
        return time.monotonic()-began

    # 按 Prompt 与输出预算计算页需求
    def pages_needed(rid, prefill_only=False):
        # 取得当前请求的输入长度与输出预算
        r = records[rid]
        # 返回本次计算结果
        return math.ceil((len(r['prompt_token_ids']) + (0 if prefill_only else r['max_new_tokens']))/256)

    # 构造发给 worker 的请求视图
    def seq(rid, rank):
        # 取得当前请求的 Prompt 和已生成输出
        r = records[rid]
        # 将 Prompt 和已生成输出组成当前逻辑历史
        return dict(request_id=rid, token_ids=r['prompt_token_ids']+r['output_token_ids'],
                    # 指定逻辑到物理页表
                    block_table=active[rank][rid]['pages'])

    # 按逻辑请求统计全局驻留数
    def count_active():
        # PD 两端同一请求只计一次驻留，DP 则合计两副本
        return len(active[1]) if mode == 'pd' else sum(map(len, active.values()))

    # 在驻留与容量限制内接纳请求
    def admit():
        # 共享外层的驻留请求峰值
        nonlocal peak_active
        # 全局驻留仍有空位时尝试接纳等待请求
        while waiting and count_active() < cap:
            # 设置请求编号
            rid = waiting[0]
            # 按部署模式或命令类型选择分支
            if mode == 'pd':
                # P 卡尚未交接完当前请求，暂不接入下一条
                if active[0]:
                    # 停止当前循环，保留已有进度
                    break
                # 将已分配两端页的 PD 请求移出等待队列
                waiting.popleft()
                # 设置各 worker 驻留请求状态
                active[0][rid] = dict(pages=pools[0].allocate(rid, pages_needed(rid, True)), ready=False)
                # 设置各 worker 驻留请求状态
                active[1][rid] = dict(pages=pools[1].allocate(rid, pages_needed(rid)), ready=False)
                # 向各卡等待 Prefill 的请求追加本次内容
                pending_prefill[0].append(rid)
                # 标记此请求由 P0 交接给 D1
                records[rid]['replica'] = 'P0-D1'
            # 处理上述条件不成立的情况
            else:
                # 设置可接纳请求的副本
                candidates = [r for r in engines if len(active[r]) < (cap//2 if mode == 'dp2' else cap)]
                # 可接纳请求的副本未就绪或为空时执行此分支
                if not candidates:
                    # 停止当前循环，保留已有进度
                    break
                # DP 优先选择驻留请求较少的卡，同数按卡号选
                rank = min(candidates, key=lambda r: (len(active[r]), r))
                # 将路由成功的请求移出等待队列
                waiting.popleft()
                # 设置各 worker 驻留请求状态
                active[rank][rid] = dict(pages=pools[rank].allocate(rid, pages_needed(rid)), ready=False)
                # 向各卡等待 Prefill 的请求追加本次内容
                pending_prefill[rank].append(rid)
                # 记录此请求归属的完整副本
                records[rid]['replica'] = rank
            # 保存真正接纳时刻，仍保留最初到达时刻
            records[rid]['admitted_s'] = elapsed()
            # 设置驻留请求峰值
            peak_active = max(peak_active, count_active())
            # 接纳后再次验证全局驻留上限
            assert count_active() <= cap

    # 提交 worker 命令并登记在途状态
    def launch(rank, command, ids):
        # 通过控制连接提交命令，KV 不经过此通道
        session.submit(rank, command)
        # 设置在途 worker 操作
        busy[rank] = dict(op=command['op'], ids=ids, began_s=elapsed(),
                          # 指定本步是否计算完整 Prompt
                          prefill=command.get('prefill'))

    # 选择本轮请求与计算窗口
    def schedule():
        # 共享外层的当前同步迁移状态、等待迁移的请求
        nonlocal transfer, pending_transfer
        # 同步迁移必须等发送和接收两卡都空闲
        if mode == 'pd' and pending_transfer is not None and 0 not in busy and 1 not in busy:
            # 设置请求编号
            rid = pending_transfer
            # 先将等待迁移的请求设为空
            pending_transfer = None
            # 取得当前待迁移请求的 Prompt 信息
            r = records[rid]
            # 设置当前同步迁移状态
            transfer = dict(request_id=rid, begin_s=elapsed(),
                            # 指定输入 Token 数
                            prompt_tokens=len(r['prompt_token_ids']), diagnostic=diagnostic,
                            # 指定P 卡源页表
                            source_pages=active[0][rid]['pages'].copy(),
                            # 指定D 卡目的页表
                            destination_pages=active[1][rid]['pages'].copy(), responses={})
            # Post both endpoints before waiting; a blocking send cannot prevent
            # the receiver from being issued by the independent controller.
            # 提交 worker 命令并登记在途状态
            launch(1, dict(op='recv_kv', block_table=active[1][rid]['pages'],
                           # 接收端按 Prompt 长度准备全层缓存
                           length=len(r['prompt_token_ids']), diagnostic=diagnostic), [rid])
            # 提交 worker 命令并登记在途状态
            launch(0, dict(op='send_kv', block_table=active[0][rid]['pages'],
                           # 发送端按相同 Prompt 长度打包全层缓存
                           length=len(r['prompt_token_ids']), diagnostic=diagnostic), [rid])
            # 结束当前操作
            return
        # 逐个处理worker 编号
        for rank in engines:
            # 当前 worker 在途操作未完成，暂不提交新工作
            if rank in busy:
                # 跳过本项，继续下一次循环
                continue
            # P 已计算完 Prompt 时等待迁移，不能提前复用源页
            if mode == 'pd' and rank == 0 and active[0] and not pending_prefill[0]:
                # 跳过本项，继续下一次循环
                continue  # Prefill pages must remain owned until handoff ACK.
            # 同卡仍有新 Prompt 时优先执行完整 Prefill
            if pending_prefill[rank]:
                # 设置请求编号
                rid = pending_prefill[rank].popleft()
                # 提交 worker 命令并登记在途状态
                launch(rank, dict(op='step', prefill=True, seqs=[seq(rid, rank)]), [rid])
            # 处理上述条件不成立的情况
            else:
                # 只对已完成 Prefill 或 KV 迁移的请求安排 Decode
                ids = [rid for rid, state in active[rank].items() if state['ready']]
                # 有历史 KV 已就绪的请求才启动 Decode
                if ids:
                    # 提交 worker 命令并登记在途状态
                    launch(rank, dict(op='step', prefill=False, seqs=[seq(rid, rank) for rid in ids]), ids)

    # 处理采样或迁移完成响应
    def receive(rank, message):
        # 共享外层的当前同步迁移状态、等待迁移的请求
        nonlocal transfer, pending_transfer
        # 设置刚完成的 worker 操作
        task = busy.pop(rank)
        # 设置当前计时
        now = elapsed()
        # 向执行轨迹追加本次内容
        trace.append(dict(rank=rank, op=task['op'], prefill=task['prefill'],
                          # 指定参与操作的请求编号
                          request_ids=task['ids'], begin_s=task['began_s'], end_s=now,
                          # 指定worker 内部耗时
                          worker_s=message['worker_s'],
                          # 指定worker 相对开始时刻
                          worker_begin_s=message['worker_begin_monotonic_s']-began,
                          # 指定worker 相对完成时刻
                          worker_end_s=message['worker_end_monotonic_s']-began))
        # 检查操作类型是否满足本分支条件
        if task['op'] in ('send_kv', 'recv_kv'):
            # 收到传输响应时必须存在对应的迁移状态
            assert transfer is not None
            # 设置已完成迁移端的响应
            transfer['responses'][rank] = message
            # 等两端 worker 都确认完成，再转移页所有权
            if len(transfer['responses']) == 2:
                # 设置发送端完成响应、接收端完成响应
                sent, received = transfer['responses'][0], transfer['responses'][1]
                # 设置全层 Prompt KV 字节数
                expected_bytes = 2*36*transfer['prompt_tokens']*8*128*2
                # 两端传输字节数必须覆盖全部 36 层 Prompt KV
                assert sent['bytes'] == received['bytes'] == expected_bytes
                # 额外哈希与页表检查仅在诊断阶段执行
                if diagnostic:
                    # 发送、接收和页重映射后的逻辑 KV 必须完全一致
                    assert sent['sha256'] == received['sha256'] == received['mapped_sha256'], 'KV handoff changed logical data'
                    # 诊断必须使用不同物理页号，防止漏测重映射
                    assert transfer['source_pages'] != transfer['destination_pages'], 'Diagnostic must exercise remapping'
                # 设置请求编号
                rid = transfer['request_id']
                # 两端均完成后，P 卡才能归还源页
                pools[0].release(rid)
                # 移除不再需要的active[0][rid]
                del active[0][rid]
                # 目的 KV 就绪后才允许 D 卡 Decode
                active[1][rid]['ready'] = True
                # 汇总含打包、等待、传输和写入的 Handoff 时间
                transfer.update(end_s=now, total_s=now-transfer['begin_s'], bytes=expected_bytes,
                                # 指定KV 打包耗时
                                pack_s=sent['pack_s'], send_transport_wait_s=sent['transport_wait_s'],
                                # 指定接收等待耗时
                                recv_transport_wait_s=received['transport_wait_s'], unpack_s=received['unpack_s'])
                # 向迁移统计追加本次内容
                transfers.append(transfer)
                # 先将当前同步迁移状态设为空
                transfer = None
            # 结束当前操作
            return
        # 非迁移响应必须来自模型前向
        assert task['op'] == 'step'
        # 每个参与前向的请求必须收到一个采样结果
        assert len(message['tokens']) == len(task['ids'])
        # 逐个处理请求编号、当前输出 Token
        for rid, token in zip(task['ids'], message['tokens']):
            # 设置当前请求状态
            record = records[rid]
            # 向已生成的真实 Token追加本次内容
            record['output_token_ids'].append(int(token))
            # 向每个输出的控制器观测时刻追加本次内容
            record['token_times_s'].append(now)
            # 固定输出长度结束，忽略自然 EOS
            finished = len(record['output_token_ids']) == record['max_new_tokens']
            # 不得把超出固定配额的 Token 计入吞吐
            assert len(record['output_token_ids']) <= record['max_new_tokens']
            # 按部署模式或命令类型选择分支
            if mode == 'pd' and rank == 0:
                # P 只产生首 Token，后续必须交给 D
                assert task['prefill'] and len(record['output_token_ids']) == 1 and not finished
                # 同时最多保留一个等待交接的 P 请求
                assert pending_transfer is None
                # 首 Token 已由 P 产生，安排 Prompt KV 迁移
                pending_transfer = rid
            # 完成请求退出其副本并归还页
            elif finished:
                # 把当前元素加入集合
                completed.add(rid)
                # 归还请求的全部页所有权
                pools[rank].release(rid)
                # 移除不再需要的active[rank][rid]
                del active[rank][rid]
            # 处理上述条件不成立的情况
            else:
                # 该请求的历史 KV 已可用于下一轮 Decode
                active[rank][rid]['ready'] = True

    # 执行操作并确保后续清理
    try:
        # 等待所有请求完成固定长度输出
        while len(completed) < len(records):
            # 设置当前计时
            now = elapsed()
            # 将到达时刻不晚于当前时间的请求移入等待队列
            while arrivals and records[arrivals[0]]['arrival_s'] <= now:
                # 向等待队列追加本次内容
                waiting.append(arrivals.popleft())
            # 在驻留与容量限制内接纳请求
            admit()
            # 选择本轮请求与计算窗口
            schedule()
            # 设置距离下一请求到达的时间
            next_arrival = max(0.0, records[arrivals[0]]['arrival_s']-elapsed()) if arrivals else 0.05
            # 设置本次轮询等待上限
            timeout = min(0.05, next_arrival)
            # 有在途任务时等待 worker 完成响应
            if busy:
                # 逐个处理worker 编号、worker 完成响应
                for rank, message in session.collect(timeout=timeout):
                    # 处理采样或迁移完成响应
                    receive(rank, message)
            # 暂无可执行请求时等到下一次到达
            elif arrivals:
                # 短暂等待下一请求到达，避免空转
                time.sleep(min(0.05, next_arrival))
            # 无在途任务却仍无法接纳等待请求时报告停滞
            elif waiting:
                # 状态不满足要求时立即报错
                raise RuntimeError('No progress: queued requests cannot be admitted')
            # 处理上述条件不成立的情况
            else:
                # 状态不满足要求时立即报错
                raise RuntimeError('No progress with unfinished requests')
            # 检查time.monotonic() - last_progress是否满足本分支条件
            if time.monotonic()-last_progress >= 20:
                # 输出本轮进度或诊断信息
                print(f'{mode}/{case["case"]}: {len(completed)}/{len(records)} complete, active={count_active()}, elapsed={elapsed():.1f}s', flush=True)
                # 设置上次进度输出时间
                last_progress = time.monotonic()
        # 记录所有请求完成时的整组耗时
        total = elapsed()
        # 测量结束不能仍有在途前向或迁移操作
        assert not busy and not session.pending and transfer is None and pending_transfer is None
        # 检查页数量、容量或所有权约束
        assert all(pool.clean() for pool in pools.values())
        # 设置原请求顺序的结果
        ordered = [records[r['request_id']] for r in case['requests']]
        # 设置各卡统计
        ranks = session.stats()
        # 设置本轮执行结果
        result = dict(case=case['case'], mode=mode, requests=ordered, trace=trace,
                      # 指定迁移统计
                      transfers=transfers, ranks=ranks, peak_active_requests=peak_active,
                      # 保存各 worker 的页使用峰值
                      peak_pages={rank: pool.peak for rank, pool in pools.items()},
                      # 报告总逻辑页预算与全部归还状态
                      logical_kv_pages=config['logical_kv_pages'], all_pages_released=True,
                      # 同时报总吞吐和按 GPU 数折算的效率
                      metrics=metrics(ordered, total, 1 if mode == 'single' else 2))
        # 设置短请求子集
        shorts = [r for r in ordered if r.get('bucket') == 'short']
        # 有短请求时单独报告该子集时延
        if shorts:
            # 设置短请求指标
            short_metrics = metrics(shorts, total, 1 if mode == 'single' else 2)
            # 逐个处理请求编号或图形状
            for key in ('avg_ttft_s', 'p95_itl_ms', 'p99_itl_ms', 'max_itl_ms'):
                # 单独补充短请求的时延分布
                result['metrics']['short_'+key] = short_metrics[key]
        # 按部署模式或命令类型选择分支
        if mode == 'pd':
            # 设置 Decode 卡统计
            d = next(r for r in ranks if r['rank'] == 1)
            # 设置 Prefill 卡统计
            p = next(r for r in ranks if r['rank'] == 0)
            # 证明 D 卡没有重新计算 Prompt
            assert d['counters']['prefill_calls'] == 0, 'Decode worker recomputed Prefill'
            # 证明 P 卡没有负责后续 Decode
            assert p['counters']['decode_calls'] == 0, 'Prefill worker performed Decode'
            # 每条请求必须恰好完成一次 KV 迁移
            assert len(transfers) == len(records)
            # 控制器、发送端、接收端三种字节统计必须一致
            assert sum(t['bytes'] for t in transfers) == p['counters']['transfer_send_bytes'] == d['counters']['transfer_recv_bytes']
            # 保存 P/D 职责隔离的实际执行证据
            result['pd_contract'] = dict(decode_prefill_calls=0, prefill_decode_calls=0,
                                        # 记录迁移请求数与迁移总字节数
                                        transferred_requests=len(transfers), transfer_bytes=sum(t['bytes'] for t in transfers),
                                        # 首 Token 来自 P 的 Prefill，早于迁移完成
                                        first_token_emitted_by='Prefill worker before handoff',
                                        # 明确 KV 使用同步 NCCL GPU 传输
                                        transport='NCCL GPU tensor send/recv; synchronous packed logical KV')
            # 设置 P 卡 Prefill 操作
            p_steps = [t for t in trace if t['rank'] == 0 and t['prefill'] is True]
            # 设置 D 卡 Decode 操作
            d_steps = [t for t in trace if t['rank'] == 1 and t['prefill'] is False]
            # 设置 Prefill 与其他请求 Decode 的重叠计数
            overlaps = sum(any(max(p['worker_begin_s'], d['worker_begin_s']) <
                              # 统计两卡前向时间区间是否相交
                              min(p['worker_end_s'], d['worker_end_s']) for d in d_steps) for p in p_steps)
            # 报告 P 前向与其他请求 D 前向的区间重叠
            result['pd_contract']['prefill_operations_overlapping_decode_operations'] = overlaps
        # 返回本轮执行结果
        return result
    # 捕获异常并保留失败现场
    except Exception:
        # 配置了失败输出路径时保存现场
        if partial_path:
            # 出错时保存当前请求、轨迹和页所有权
            write_json(partial_path, dict(completed=False, traceback=traceback.format_exc(),
                # 保存失败负载及尚未完成的请求状态
                case=case['case'], mode=mode, requests=list(records.values()), trace=trace,
                # 指定迁移统计
                transfers=transfers, current_transfer=transfer, busy=busy,
                # 保存各卡页的拥有者以检查泄漏
                owners={rank: pool.owners for rank, pool in pools.items()}))
        # 将原异常继续抛出给上层
        raise
