"""V4 speculative state machine with a shared eager/graph target executor."""
from types import SimpleNamespace
import torch
from dflash_runtime import DFlashRuntime
from state import Window


class GraphDFlashRuntime(DFlashRuntime):
    def __init__(self, target, draft, executor):
        # The original experimental generate() requires an idle-engine contract.
        # This adapter owns the entire cache; ordinary engine.step is not used.
        runner = SimpleNamespace(model=target.model, world_size=1,
            enforce_eager=True, block_size=256,
            config={'max_model_length': 2048, 'max_cached_blocks': target.pages})
        engine = SimpleNamespace(model_runner=runner,
            scheduler=SimpleNamespace(is_finished=lambda: True,
                block_manager=SimpleNamespace(used_block_ids=set())))
        super().__init__(engine, draft)
        # Executor owns its own hooks and graph-resident feature outputs.
        super().close()
        self.executor = executor
        self.graph_enabled = False
        self.validate_tensors = True
        self.logit_diagnostics = False
        self.run_label = ''

    def target_forward(self, reqs, queries, pool):
        views = [Window(r, q, pool.reserve(r.key, r.computed+len(q)))
                 for r, q in zip(reqs, queries)]
        # Initial prompt stays eager even if its length happens to equal 1/16.
        result = self.executor.run(views,
            use_graph=self.graph_enabled and all(r.computed > 0 for r in reqs),
            diagnostic=self.validate_tensors)
        features = result['features']
        if self.validate_tensors:
            for key in ('features', 'hidden', 'logits'):
                assert torch.isfinite(result[key]).all().item(), 'Non-finite '+key
        picks = result['picks'].cpu().tolist()
        assert len(picks) == sum(map(len, queries))
        outputs, offset = [], 0
        for query in queries:
            end = offset+len(query)
            outputs.append((picks[offset:end], features[offset:end]))
            offset = end
        self.last_predictions = {}
        return outputs

    def close(self):
        super().close()

    @torch.inference_mode()
    def generate(self, *args, **kwargs):
        self.executor.reset_counters()
        result = super().generate(*args, **kwargs)
        result['summary']['target_execution'] = self.executor.stats()
        result['summary']['target_cuda_graph'] = self.graph_enabled
        result['summary']['draft_cuda_graph'] = False
        return result
