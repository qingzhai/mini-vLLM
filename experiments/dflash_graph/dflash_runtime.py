"""Experimental single-GPU MinivLLM target verification with DFlash drafting.

The target verifies all active requests in ONE packed forward per round.
Draft calls use independent request caches and run sequentially in this version.
This base API owns an exclusive page pool while running; ordinary engine.step
must not run concurrently. graph_runtime overrides target_forward to enable
target graphs; prefix sharing and mixed P/D scheduling remain disabled.
"""
from collections import deque
import time
import torch
from transformers import DynamicCache
from myvllm.utils import reset_context
from state import PagePool, Request, Window, commit
from consistent_target import attach_query_metadata

class DFlashRuntime:
    def __init__(self, engine, draft):
        self.engine, self.runner = engine, engine.model_runner
        self.model, self.draft = self.runner.model, draft
        assert self.runner.world_size == 1 and self.runner.enforce_eager
        assert draft.block_size == 16 and draft.target_layer_ids == [1,9,17,25,33]
        assert next(draft.parameters()).dtype == torch.bfloat16
        self.mask = draft.config.dflash_config['mask_token_id']
        self.captured, self.handles = {}, []
        for index in draft.target_layer_ids:
            def capture(module, args, output, index=index):
                assert isinstance(output, tuple) and len(output)==2
                x, residual = output
                # MinivLLM carries the residual separately; HF hidden_states
                # store the complete post-layer state used to train DFlash.
                self.captured[index] = x+residual
            self.handles.append(self.model.model.layers[index].register_forward_hook(capture))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def target_forward(self, reqs, queries, pool):
        views = [Window(r,q,pool.reserve(r.key,r.computed+len(q))) for r,q in zip(reqs,queries)]
        self.captured.clear()
        try:
            ids = self.runner.prepare_prefill(views)
            if getattr(self,'consistent_queries',False):
                width=(self.runner.config['max_model_length']+self.runner.block_size-1)//self.runner.block_size
                attach_query_metadata(views,table_width=width)
            hidden = self.model(ids)
            assert set(self.captured)==set(self.draft.target_layer_ids)
            features = torch.cat([self.captured[i] for i in self.draft.target_layer_ids], dim=-1)
            assert torch.isfinite(features).all().item(), 'Non-finite target features'
            reset_context()  # compute logits for EVERY verification row, no Prefill gather.
            logits = self.model.compute_logits(hidden)
            assert logits.shape[0] == sum(map(len,queries))
            assert logits.dtype == torch.float32 and torch.isfinite(logits).all().item()
            observer = getattr(self, 'observer', None)
            if observer is not None:
                observer(reqs, queries, logits, features, hidden)
            picks = logits.argmax(-1).cpu().tolist()
            self.last_predictions = {}
            if getattr(self, 'logit_diagnostics', False):
                vals, inds = torch.topk(logits, 2, dim=-1)
                vals, inds = vals.cpu().tolist(), inds.cpu().tolist()
                offset = 0
                for req, query in zip(reqs, queries):
                    self.last_predictions[req.key] = [
                        dict(token=picks[i], top_ids=inds[i], top_logits=vals[i],
                             top2_margin=vals[i][0]-vals[i][1])
                        for i in range(offset, offset+len(query))]
                    offset += len(query)
            result, cursor = [], 0
            for query in queries:
                end = cursor+len(query)
                result.append((picks[cursor:end],features[cursor:end].detach()))
                cursor = end
            return result
        finally:
            self.captured.clear()
            reset_context()

    def propose(self, req):
        assert req.features.shape[0] == req.computed-req.draft_cache.get_seq_length()
        block = torch.full((16,),self.mask,device='cuda',dtype=torch.long)
        block[0] = req.tokens[-1]
        embedding = self.model.model.embed_tokens(block).to(torch.bfloat16).unsqueeze(0)
        features = req.features.to(torch.bfloat16).unsqueeze(0)
        positions = torch.arange(req.draft_cache.get_seq_length(),req.computed+16,
                                  device='cuda',dtype=torch.long).unsqueeze(0)
        if getattr(self, 'validate_tensors', True):
            assert torch.isfinite(embedding).all().item() and torch.isfinite(features).all().item()
        hidden = self.draft(target_hidden=features, noise_embedding=embedding,
            position_ids=positions, past_key_values=req.draft_cache,
            use_cache=True, is_causal=False)
        if getattr(self, 'validate_tensors', True):
            assert torch.isfinite(hidden).all().item(), 'Non-finite BF16 draft hidden states'
        req.draft_cache.crop(req.computed)
        logits = self.model.compute_logits(hidden[:,1:,:])
        assert logits.shape[:2] == (1,15)
        if getattr(self, 'validate_tensors', True):
            assert torch.isfinite(logits).all().item(), 'Non-finite draft logits'
        proposals = logits.argmax(-1)[0].cpu().tolist()
        assert self.mask not in proposals, 'Draft selected mask token; inspect draft numerics'
        return [req.tokens[-1]]+proposals

    @torch.inference_mode()
    def generate(self, prompts, max_new_tokens=256, max_batch_size=2, speculative=True):
        assert prompts and max_new_tokens >= 1 and max_batch_size >= 1
        assert self.engine.scheduler.is_finished(), 'Regular scheduler must be idle'
        assert not self.engine.scheduler.block_manager.used_block_ids
        assert all(len(p)+max_new_tokens+16 <= self.runner.config['max_model_length'] for p in prompts)
        pool = PagePool(self.runner.config['max_cached_blocks'],self.runner.block_size)
        waiting = deque(Request(i,list(p),len(p)) for i,p in enumerate(prompts))
        active, completed, target_calls, verify_calls = [], {}, 0, 0
        peak_active, max_verify_batch, refill_rounds = 0, 0, 0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        last_progress = started
        try:
            while waiting or active:
                new = []
                while waiting and len(active)+len(new) < max_batch_size:
                    new.append(waiting.popleft())
                if new:
                    refill_rounds += bool(active)
                    outputs = self.target_forward(new,[r.tokens for r in new],pool)
                    target_calls += 1
                    for req,(picks,features) in zip(new,outputs):
                        req.computed = len(req.tokens)
                        req.tokens.append(picks[-1])
                        req.features = features
                        req.draft_cache = DynamicCache()
                        if getattr(self, 'logit_diagnostics', False):
                            req.prediction_trace.append(self.last_predictions[req.key][-1])
                        req.first_token_s = time.perf_counter()-started
                    active.extend(new)
                peak_active = max(peak_active,len(active))
                done = [r for r in active if len(r.outputs)==max_new_tokens]
                for req in done:
                    req.finish_s = time.perf_counter()-started
                    completed[req.key] = req
                    pool.release(req.key)
                    active.remove(req)
                if not active:
                    continue
                blocks = [self.propose(r) if speculative else [r.tokens[-1]] for r in active]
                outputs = self.target_forward(active,blocks,pool)
                target_calls += 1
                verify_calls += 1
                max_verify_batch = max(max_verify_batch,len(active))
                for req,block,(posterior,features) in zip(active,blocks,outputs):
                    row = commit(req,block,posterior,max_new_tokens)
                    req.features = features[:row['advance']]
                    if getattr(self, 'logit_diagnostics', False):
                        req.prediction_trace.extend(self.last_predictions[req.key][:row['advance']])
                    pool.trim(req.key,req.computed)
                    req.trace.append(dict(**row,computed_tokens=req.computed))
                for req in list(active):
                    if len(req.outputs)==max_new_tokens:
                        req.finish_s = time.perf_counter()-started
                        completed[req.key] = req
                        pool.release(req.key)
                        active.remove(req)
                pool.check()
                now = time.perf_counter()
                if now-last_progress >= 20:
                    made = sum(len(r.outputs) for r in completed.values()) + sum(len(r.outputs) for r in active)
                    print(f'{getattr(self, "run_label", "generation")}: {len(completed)}/{len(prompts)} requests complete, {made} tokens, elapsed={now-started:.1f}s', flush=True)
                    last_progress = now
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            assert len(completed)==len(prompts)
            assert not pool.tables and len(pool.free)==pool.pages
            requests = [completed[i] for i in range(len(prompts))]
            trace = [x for r in requests for x in r.trace]
            accepted = sum(x['accepted_draft'] for x in trace)
            proposed = sum(x['proposed_draft'] for x in trace)
            return dict(summary=dict(requests=len(prompts),max_batch_size=max_batch_size,
                output_tokens=len(prompts)*max_new_tokens,total_s=seconds,
                output_tokens_per_s=len(prompts)*max_new_tokens/seconds,
                target_forward_calls=target_calls,verification_batch_calls=verify_calls,
                max_verification_batch=max_verify_batch,peak_active_requests=peak_active,
                refill_while_active=refill_rounds,accepted_draft_tokens=accepted,
                proposed_draft_tokens=proposed,draft_acceptance_ratio=accepted/proposed if proposed else None,
                average_verified_advance=sum(x['advance'] for x in trace)/len(trace) if trace else None,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_kv_pages=pool.peak,rollback_pages=pool.rollback_pages,
                all_pages_released=True),requests=[dict(request=r.key,token_ids=r.outputs,
                first_token_s=r.first_token_s,finish_s=r.finish_s,trace=r.trace,
                prediction_trace=r.prediction_trace) for r in requests])
        finally:
            for key in list(pool.tables):
                pool.release(key)
            pool.check()
            reset_context()
