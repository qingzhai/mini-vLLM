"""Evidence guards and a smaller real-data warmup for the fixed-shape runner."""
import hashlib
import json
from pathlib import Path

MODES = ('full_prefill','chunked','chunked_prefix')


def load_proof(folder, workload):
    folder = Path(folder)
    proof = json.loads((folder/'report.json').read_text())
    assert proof.get('completed') and proof.get('passed') and proof.get('all_exact_greedy')
    assert proof.get('all_replays_exact') and proof.get('teacher_forcing') is False
    assert proof['workload_sha256'] == workload['workload_sha256']
    assert proof['kv_pages'] >= 56 and proof['config']['world_size'] == 1
    canonical = proof['canonical_runtime']
    assert canonical['cuda_graph'] and canonical['block_table_width'] == 8
    assert not canonical['prefill_flash_kernel_used'] and not canonical['torch_compile']
    package = Path(__file__).parent
    source_digests = {}
    for name in ('canonical_runtime.py','fixed_shape.py','fp32_head.py','server-source-sha256.json'):
        current_path = package.parent/'common'/name if name == 'fixed_shape.py' else package/name
        current = current_path.read_bytes()
        digest = hashlib.sha256(current).hexdigest()
        guard = folder/'runtime-sha256.json'
        expected = json.loads(guard.read_text())[name] if guard.is_file() else hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest()
        assert digest == expected, 'Canonical runtime differs from passed run: '+name
        source_digests[name] = digest
    proof = dict(proof, canonical_source_sha256=source_digests)
    rows = {}
    for mode in MODES:
        obj = json.loads((folder/(mode+'.json')).read_text())
        assert obj['summary']['finite_logits'] and obj['summary']['all_pages_released']
        assert obj['summary']['replay_sequence_matches'] == 20
        records = obj['requests']
        assert len(records) == 100 and len({r['request_id'] for r in records}) == 100
        assert [r['request_id'] for r in records] == [r['request_id'] for r in workload['requests']]
        assert all(len(r['output_token_ids']) == 64 for r in records)
        rows[mode] = records
    return proof, rows


def warmup_requests(requests):
    by_id = {r['request_id']:r for r in requests}
    repeats = requests[80:88]
    originals = [by_id[r['repeated_from']] for r in repeats]
    assert len(originals) == len(repeats) == 8
    assert len({r['request_id'] for r in originals+repeats}) == 16
    return originals+repeats


def install_kv_budget(pages):
    # Same fixed-capacity allocator wrapper used in the earlier DFlash checks.
    import torch
    from myvllm.engine.model_runner import ModelRunner
    original = ModelRunner.allocate_kv_cache
    def allocate(self):
        free,_ = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        scratch = stats['allocated_bytes.all.peak']-stats['allocated_bytes.all.current']
        block_bytes = (self.block_size*2*self.config['num_layers']*self.config['num_kv_heads']*
                       self.config['head_dim']*self.default_dtype.itemsize)
        fraction = ((pages+.125)*block_bytes+scratch)/free
        assert 0 < fraction < .95, 'Insufficient GPU memory for the passed KV budget'
        previous = self.config['gpu_memory_utilization']
        self.config['gpu_memory_utilization'] = fraction
        try:
            original(self)
        finally:
            self.config['gpu_memory_utilization'] = previous
        assert self.config['max_cached_blocks'] == pages
    ModelRunner.allocate_kv_cache = allocate
