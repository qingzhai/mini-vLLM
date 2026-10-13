"""CPU checks for the smaller warmup, proof guards, and single-round reporting."""
import argparse
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
from benchmark_support import load_proof, warmup_requests
from prepare_data import validate_workload
from summarize import write_summary


def load_actual(repo):
    for name,path in [('myvllm',repo/'src/myvllm'),('myvllm.engine',repo/'src/myvllm/engine')]:
        module=types.ModuleType(name)
        module.__path__=[str(path)]
        sys.modules[name]=module
    for name in ('sampling_parameters','engine.sequence','engine.block_manager','engine.scheduler','engine.chunk_scheduler'):
        full='myvllm.'+name
        path=repo/'src/myvllm'/Path(*name.split('.')).with_suffix('.py')
        spec=importlib.util.spec_from_file_location(full,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[full]=module
        spec.loader.exec_module(module)
    return (sys.modules['myvllm.engine.scheduler'].Scheduler,
            sys.modules['myvllm.engine.chunk_scheduler'].ChunkScheduler,
            sys.modules['myvllm.engine.sequence'].Sequence,
            sys.modules['myvllm.sampling_parameters'].SamplingParams)


def main(repo,workload_file):
    workload=json.loads(workload_file.read_text())
    validate_workload(workload)
    warm=warmup_requests(workload['requests'])
    assert len(warm)==16 and sum(r['bucket']=='short' for r in warm)==8
    assert all(r['repeated_from'] is None for r in warm[:8])
    assert all(r['repeated_from']==original['request_id'] for r,original in zip(warm[8:],warm[:8]))
    Scheduler,Chunk,Sequence,Params=load_actual(repo)
    for mode in ('full_prefill','chunked','chunked_prefix'):
        scheduler=Chunk(Scheduler(8,2048,330,256,999999),
                        chunk_size=2048 if mode=='full_prefill' else 256,
                        decode_first=mode!='full_prefill')
        bm=scheduler.block_manager
        bm.enable_prefix_caching=mode=='chunked_prefix'
        seqs=[Sequence(r['prompt_token_ids'],256,Params(max_tokens=64,ignore_eos=True,max_model_length=2048)) for r in warm]
        for seq in seqs:scheduler.add_sequence(seq)
        partial=mixed=steps=0
        while not scheduler.is_finished():
            selected,_=scheduler.schedule()
            steps+=1
            assert steps<10000 and len(scheduler.running)<=8
            assert scheduler.last_schedule['tokens']<=2048
            mixed+=bool(scheduler.last_schedule['mixed'])
            ready=[]
            for seq in selected:
                end=seq._scheduled_end
                seq.num_computed_tokens=end
                bm.mark_computed(seq,computed_tokens=end)
                if seq._scheduled_prefill and end<len(seq):partial+=1
                else:ready.append(seq)
            scheduler.postprocess(ready,[17]*len(ready))
        assert all(s.num_completion_tokens==64 for s in seqs)
        assert not bm.used_block_ids and len(bm.free_block_ids)==330
        assert (partial==mixed==0) if mode=='full_prefill' else (partial>0 and mixed>0)
        assert (bm.stats['cache_hit_tokens']>0)==(mode=='chunked_prefix')
        print('16-request actual scheduler warmup:',mode,'steps',steps,'hits',bm.stats['cache_hit_tokens'])
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp)
        source=folder/'source'
        source.mkdir()
        for name in ('canonical_runtime.py','fixed_shape.py','fp32_head.py','server-source-sha256.json'):
            current = Path(__file__).parent.parent/'common'/name if name == 'fixed_shape.py' else Path(__file__).parent/name
            shutil.copy2(current,source/name)
        proof=dict(completed=True,passed=True,all_exact_greedy=True,all_replays_exact=True,
                   teacher_forcing=False,workload_sha256=workload['workload_sha256'],kv_pages=330,
                   config=dict(world_size=1),canonical_runtime=dict(cuda_graph=True,block_table_width=8,
                       prefill_flash_kernel_used=False,torch_compile=False))
        (folder/'report.json').write_text(json.dumps(proof))
        for mode in ('full_prefill','chunked','chunked_prefix'):
            (folder/(mode+'.json')).write_text(json.dumps(dict(summary=dict(finite_logits=True,
                all_pages_released=True,replay_sequence_matches=20),requests=[dict(request_id=r['request_id'],
                output_token_ids=[17]*64) for r in workload['requests']])))
        got,rows=load_proof(folder,workload)
        assert len(rows)==3 and len(got['canonical_source_sha256'])==4
        proof['passed']=False
        (folder/'report.json').write_text(json.dumps(proof))
        try:load_proof(folder,workload)
        except AssertionError:pass
        else:raise AssertionError('Failed proof incorrectly accepted')
        proof['passed']=True
        (folder/'report.json').write_text(json.dumps(proof))
        with (source/'canonical_runtime.py').open('a') as stream:stream.write('\n# changed\n')
        try:load_proof(folder,workload)
        except AssertionError:pass
        else:raise AssertionError('Changed runtime incorrectly accepted')
        # Synthetic metric fixture only: tests report construction, never GPU parity.
        trials=[]
        for mode,rate in [('full_prefill',100),('chunked',80),('chunked_prefix',120)]:
            summary=dict(mode=mode,requests=100,output_tokens=6400,output_tokens_per_s=rate,
                avg_ttft_ms=1000,p95_ttft_ms=2000,avg_tpot_ms=30,p95_tpot_ms=40,avg_e2e_ms=3000,
                p95_e2e_ms=5000,requests_per_s=rate/64,total_s=6400/rate,peak_allocated_gib=30,
                cache_hit_tokens=0,cache_hit_prompt_token_fraction=0,partial_prefill_chunks=0,
                mixed_steps=0,peak_active_requests=8,all_pages_released=True,finite_logits=True,timing_clean=True)
            trials.append(dict(repeat=1,summary=summary))
        report=dict(completed=True,repeats=1,measured_trials=trials,all_exact_greedy=True,
                    all_timings_clean=True,greedy_comparisons=[dict(repeat=1,mode=m,exact_matches=100,
                    all_exact=True) for m in ('chunked','chunked_prefix')])
        (folder/'report.json').write_text(json.dumps(report))
        with redirect_stdout(io.StringIO()):data=write_summary(folder)
        assert data[1]['throughput_ratio_vs_full']==.8 and data[2]['throughput_ratio_vs_full']==1.2
        markdown=(folder/'RESULTS.md').read_text()
        assert '单轮' not in markdown or '波动' in markdown
        assert '正式运行1轮' in markdown and '并入此前 Flash' in markdown
        assert '严格输出等价优化' not in markdown
    print('Proof/source guards and one-round metric report PASS; no GPU claims made')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('repo',type=Path)
    parser.add_argument('workload',type=Path)
    args=parser.parse_args()
    main(args.repo,args.workload)
