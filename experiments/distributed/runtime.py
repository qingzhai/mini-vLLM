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


def run_load(session, case, config, diagnostic=False, partial_path=None):
    """One full prefill per step; batched decode; same global residency cap.

    DP routes each admission to the replica with the fewest resident requests.
    PD reserves destination pages before prefill and blocks new admission when
    the global cap is reached. KV transfer is synchronous and explicitly timed.
    """
    mode = session.mode
    cap = config['global_batch']
    assert 2 <= cap <= 8 and cap % 2 == 0
    replica_pages = config['logical_kv_pages']//2 if mode in ('dp2', 'pd') else config['logical_kv_pages']
    engines = list(session.clients)
    pools = {rank: PagePool(replica_pages, reverse=(mode == 'pd' and rank == 1)) for rank in engines}
    active = {rank: {} for rank in engines}
    pending_prefill = {rank: deque() for rank in engines}
    records = {r['request_id']: dict(r, output_token_ids=[], token_times_s=[]) for r in case['requests']}
    arrivals = deque(sorted(records, key=lambda rid: records[rid]['arrival_s']))
    waiting = deque()
    busy, trace, transfers = {}, [], []
    completed = set()
    pending_transfer = None
    transfer = None
    peak_active = 0
    began = time.monotonic()
    last_progress = began

    def elapsed():
        return time.monotonic()-began

    def pages_needed(rid, prefill_only=False):
        r = records[rid]
        return math.ceil((len(r['prompt_token_ids']) + (0 if prefill_only else r['max_new_tokens']))/256)

    def seq(rid, rank):
        r = records[rid]
        return dict(request_id=rid, token_ids=r['prompt_token_ids']+r['output_token_ids'],
                    block_table=active[rank][rid]['pages'])

    def count_active():
        return len(active[1]) if mode == 'pd' else sum(map(len, active.values()))

    def admit():
        nonlocal peak_active
        while waiting and count_active() < cap:
            rid = waiting[0]
            if mode == 'pd':
                if active[0]:
                    break
                waiting.popleft()
                active[0][rid] = dict(pages=pools[0].allocate(rid, pages_needed(rid, True)), ready=False)
                active[1][rid] = dict(pages=pools[1].allocate(rid, pages_needed(rid)), ready=False)
                pending_prefill[0].append(rid)
                records[rid]['replica'] = 'P0-D1'
            else:
                candidates = [r for r in engines if len(active[r]) < (cap//2 if mode == 'dp2' else cap)]
                if not candidates:
                    break
                rank = min(candidates, key=lambda r: (len(active[r]), r))
                waiting.popleft()
                active[rank][rid] = dict(pages=pools[rank].allocate(rid, pages_needed(rid)), ready=False)
                pending_prefill[rank].append(rid)
                records[rid]['replica'] = rank
            records[rid]['admitted_s'] = elapsed()
            peak_active = max(peak_active, count_active())
            assert count_active() <= cap

    def launch(rank, command, ids):
        session.submit(rank, command)
        busy[rank] = dict(op=command['op'], ids=ids, began_s=elapsed(),
                          prefill=command.get('prefill'))

    def schedule():
        nonlocal transfer, pending_transfer
        if mode == 'pd' and pending_transfer is not None and 0 not in busy and 1 not in busy:
            rid = pending_transfer
            pending_transfer = None
            r = records[rid]
            transfer = dict(request_id=rid, begin_s=elapsed(),
                            prompt_tokens=len(r['prompt_token_ids']), diagnostic=diagnostic,
                            source_pages=active[0][rid]['pages'].copy(),
                            destination_pages=active[1][rid]['pages'].copy(), responses={})
            # Post both endpoints before waiting; a blocking send cannot prevent
            # the receiver from being issued by the independent controller.
            launch(1, dict(op='recv_kv', block_table=active[1][rid]['pages'],
                           length=len(r['prompt_token_ids']), diagnostic=diagnostic), [rid])
            launch(0, dict(op='send_kv', block_table=active[0][rid]['pages'],
                           length=len(r['prompt_token_ids']), diagnostic=diagnostic), [rid])
            return
        for rank in engines:
            if rank in busy:
                continue
            if mode == 'pd' and rank == 0 and active[0] and not pending_prefill[0]:
                continue  # Prefill pages must remain owned until handoff ACK.
            if pending_prefill[rank]:
                rid = pending_prefill[rank].popleft()
                launch(rank, dict(op='step', prefill=True, seqs=[seq(rid, rank)]), [rid])
            else:
                ids = [rid for rid, state in active[rank].items() if state['ready']]
                if ids:
                    launch(rank, dict(op='step', prefill=False, seqs=[seq(rid, rank) for rid in ids]), ids)

    def receive(rank, message):
        nonlocal transfer, pending_transfer
        task = busy.pop(rank)
        now = elapsed()
        trace.append(dict(rank=rank, op=task['op'], prefill=task['prefill'],
                          request_ids=task['ids'], begin_s=task['began_s'], end_s=now,
                          worker_s=message['worker_s'],
                          worker_begin_s=message['worker_begin_monotonic_s']-began,
                          worker_end_s=message['worker_end_monotonic_s']-began))
        if task['op'] in ('send_kv', 'recv_kv'):
            assert transfer is not None
            transfer['responses'][rank] = message
            if len(transfer['responses']) == 2:
                sent, received = transfer['responses'][0], transfer['responses'][1]
                expected_bytes = 2*36*transfer['prompt_tokens']*8*128*2
                assert sent['bytes'] == received['bytes'] == expected_bytes
                if diagnostic:
                    assert sent['sha256'] == received['sha256'] == received['mapped_sha256'], 'KV handoff changed logical data'
                    assert transfer['source_pages'] != transfer['destination_pages'], 'Diagnostic must exercise remapping'
                rid = transfer['request_id']
                pools[0].release(rid)
                del active[0][rid]
                active[1][rid]['ready'] = True
                transfer.update(end_s=now, total_s=now-transfer['begin_s'], bytes=expected_bytes,
                                pack_s=sent['pack_s'], send_transport_wait_s=sent['transport_wait_s'],
                                recv_transport_wait_s=received['transport_wait_s'], unpack_s=received['unpack_s'])
                transfers.append(transfer)
                transfer = None
            return
        assert task['op'] == 'step'
        assert len(message['tokens']) == len(task['ids'])
        for rid, token in zip(task['ids'], message['tokens']):
            record = records[rid]
            record['output_token_ids'].append(int(token))
            record['token_times_s'].append(now)
            finished = len(record['output_token_ids']) == record['max_new_tokens']
            assert len(record['output_token_ids']) <= record['max_new_tokens']
            if mode == 'pd' and rank == 0:
                assert task['prefill'] and len(record['output_token_ids']) == 1 and not finished
                assert pending_transfer is None
                pending_transfer = rid
            elif finished:
                completed.add(rid)
                pools[rank].release(rid)
                del active[rank][rid]
            else:
                active[rank][rid]['ready'] = True

    try:
        while len(completed) < len(records):
            now = elapsed()
            while arrivals and records[arrivals[0]]['arrival_s'] <= now:
                waiting.append(arrivals.popleft())
            admit()
            schedule()
            next_arrival = max(0.0, records[arrivals[0]]['arrival_s']-elapsed()) if arrivals else 0.05
            timeout = min(0.05, next_arrival)
            if busy:
                for rank, message in session.collect(timeout=timeout):
                    receive(rank, message)
            elif arrivals:
                time.sleep(min(0.05, next_arrival))
            elif waiting:
                raise RuntimeError('No progress: queued requests cannot be admitted')
            else:
                raise RuntimeError('No progress with unfinished requests')
            if time.monotonic()-last_progress >= 20:
                print(f'{mode}/{case["case"]}: {len(completed)}/{len(records)} complete, active={count_active()}, elapsed={elapsed():.1f}s', flush=True)
                last_progress = time.monotonic()
        total = elapsed()
        assert not busy and not session.pending and transfer is None and pending_transfer is None
        assert all(pool.clean() for pool in pools.values())
        ordered = [records[r['request_id']] for r in case['requests']]
        ranks = session.stats()
        result = dict(case=case['case'], mode=mode, requests=ordered, trace=trace,
                      transfers=transfers, ranks=ranks, peak_active_requests=peak_active,
                      peak_pages={rank: pool.peak for rank, pool in pools.items()},
                      logical_kv_pages=config['logical_kv_pages'], all_pages_released=True,
                      metrics=metrics(ordered, total, 1 if mode == 'single' else 2))
        shorts = [r for r in ordered if r.get('bucket') == 'short']
        if shorts:
            short_metrics = metrics(shorts, total, 1 if mode == 'single' else 2)
            for key in ('avg_ttft_s', 'p95_itl_ms', 'p99_itl_ms', 'max_itl_ms'):
                result['metrics']['short_'+key] = short_metrics[key]
        if mode == 'pd':
            d = next(r for r in ranks if r['rank'] == 1)
            p = next(r for r in ranks if r['rank'] == 0)
            assert d['counters']['prefill_calls'] == 0, 'Decode worker recomputed Prefill'
            assert p['counters']['decode_calls'] == 0, 'Prefill worker performed Decode'
            assert len(transfers) == len(records)
            assert sum(t['bytes'] for t in transfers) == p['counters']['transfer_send_bytes'] == d['counters']['transfer_recv_bytes']
            result['pd_contract'] = dict(decode_prefill_calls=0, prefill_decode_calls=0,
                                        transferred_requests=len(transfers), transfer_bytes=sum(t['bytes'] for t in transfers),
                                        first_token_emitted_by='Prefill worker before handoff',
                                        transport='NCCL GPU tensor send/recv; synchronous packed logical KV')
            p_steps = [t for t in trace if t['rank'] == 0 and t['prefill'] is True]
            d_steps = [t for t in trace if t['rank'] == 1 and t['prefill'] is False]
            overlaps = sum(any(max(p['worker_begin_s'], d['worker_begin_s']) <
                              min(p['worker_end_s'], d['worker_end_s']) for d in d_steps) for p in p_steps)
            result['pd_contract']['prefill_operations_overlapping_decode_operations'] = overlaps
        return result
    except Exception:
        if partial_path:
            write_json(partial_path, dict(completed=False, traceback=traceback.format_exc(),
                case=case['case'], mode=mode, requests=list(records.values()), trace=trace,
                transfers=transfers, current_transfer=transfer, busy=busy,
                owners={rank: pool.owners for rank, pool in pools.items()}))
        raise
