"""Experimental single-GPU MinivLLM target verification with DFlash drafting.

The target verifies all active requests in ONE packed forward per round.
Draft calls use independent request caches and run sequentially in this version.
This base API owns an exclusive page pool while running; ordinary engine.step
must not run concurrently. graph_runtime overrides target_forward to enable
target graphs; prefix sharing and mixed P/D scheduling remain disabled.
"""
# 导入 deque 所需接口
from collections import deque
# 导入 time 所需接口
import time
# 导入 torch 所需接口
import torch
# 导入 DynamicCache 所需接口
from transformers import DynamicCache
# 导入 reset_context 所需接口
from myvllm.utils import reset_context
# 导入 PagePool, Request, Window, commit 所需接口
from state import PagePool, Request, Window, commit
# 导入 attach_query_metadata 所需接口
from consistent_target import attach_query_metadata

# 管理 Draft 提议与 Target 批量验证
class DFlashRuntime:
    # 初始化当前对象的状态
    def __init__(self, engine, draft):
        # 设置引擎适配对象、模型执行器
        self.engine, self.runner = engine, engine.model_runner
        # 设置 Target 模型、官方 Draft
        self.model, self.draft = self.runner.model, draft
        # 确认当前并行配置在支持范围内
        assert self.runner.world_size == 1 and self.runner.enforce_eager
        # 只接受已验证的 16 位块与五层特征配置
        assert draft.block_size == 16 and draft.target_layer_ids == [1,9,17,25,33]
        # 官方 Draft 保持 BF16
        assert next(draft.parameters()).dtype == torch.bfloat16
        # 设置 Draft MASK ID
        self.mask = draft.config.dflash_config['mask_token_id']
        # 设置本次捕获的层特征、特征 Hook 句柄
        self.captured, self.handles = {}, []
        # 为官方指定的五个 Target 层逐个安装 Hook
        for index in draft.target_layer_ids:
            # 采集包含残差的 Target 层输出
            def capture(module, args, output, index=index):
                # MinivLLM 层必须返回特征与独立残差
                assert isinstance(output, tuple) and len(output)==2
                # 设置当前特征、残差分支
                x, residual = output
                # MinivLLM carries the residual separately; HF hidden_states
                # store the complete post-layer state used to train DFlash.
                # 合并残差得到与官方 Draft 兼容的完整层特征
                self.captured[index] = x+residual
            # 向特征 Hook 句柄追加本次内容
            self.handles.append(self.model.model.layers[index].register_forward_hook(capture))

    # 释放本执行器持有的资源
    def close(self):
        # 逐个处理层特征 Hook
        for handle in self.handles:
            # 从层特征 Hook移除此项
            handle.remove()
        # 清空特征 Hook 句柄
        self.handles.clear()

    # 将各请求 Query 合并成一次 Target 前向
    def target_forward(self, reqs, queries, pool):
        # 设置每请求的验证窗口
        views = [Window(r,q,pool.reserve(r.key,r.computed+len(q))) for r,q in zip(reqs,queries)]
        # 清空本次捕获的层特征
        self.captured.clear()
        # 执行操作并确保后续清理
        try:
            # 设置展平输入
            ids = self.runner.prepare_prefill(views)
            # 可选安装统一逐 Query 的参考元数据
            if getattr(self,'consistent_queries',False):
                # 设置页表固定宽度
                width=(self.runner.config['max_model_length']+self.runner.block_size-1)//self.runner.block_size
                # 让不同 Query 数使用相同的分页读取语义
                attach_query_metadata(views,table_width=width)
            # 设置模型输出特征
            hidden = self.model(ids)
            # 必须收到官方 Draft 所要求的全部五层特征
            assert set(self.captured)==set(self.draft.target_layer_ids)
            # 按官方层序拼接五层完整特征
            features = torch.cat([self.captured[i] for i in self.draft.target_layer_ids], dim=-1)
            # 检查张量没有 NaN 或无穷值
            assert torch.isfinite(features).all().item(), 'Non-finite target features'
            # 清除全局前向元数据
            reset_context()  # compute logits for EVERY verification row, no Prefill gather.
            # 设置词表分数
            logits = self.model.compute_logits(hidden)
            # 检查张量形状与本轮数据一致
            assert logits.shape[0] == sum(map(len,queries))
            # 检查张量没有 NaN 或无穷值
            assert logits.dtype == torch.float32 and torch.isfinite(logits).all().item()
            # 设置可选诊断回调
            observer = getattr(self, 'observer', None)
            # 可选诊断回调已建立时执行此分支
            if observer is not None:
                # 将本轮数值交给可选的诊断收集器
                observer(reqs, queries, logits, features, hidden)
            # 设置 Target 逐行后继预测
            picks = logits.argmax(-1).cpu().tolist()
            # 初始化逐请求 Top-2 诊断
            self.last_predictions = {}
            # 诊断开启时再做额外检查
            if getattr(self, 'logit_diagnostics', False):
                # 设置前两名分数、前两名 Token
                vals, inds = torch.topk(logits, 2, dim=-1)
                # 设置前两名分数、前两名 Token
                vals, inds = vals.cpu().tolist(), inds.cpu().tolist()
                # 设置展平预测行的起点
                offset = 0
                # 逐个处理投机请求、本请求的 Query 块
                for req, query in zip(reqs, queries):
                    # 设置逐请求 Top-2 诊断
                    self.last_predictions[req.key] = [
                        # 为每个位置保存 Token、Top-2 ID 和分数
                        dict(token=picks[i], top_ids=inds[i], top_logits=vals[i],
                             # 指定Top-1 与 Top-2 分差
                             top2_margin=vals[i][0]-vals[i][1])
                        # 将前述构造应用到每个元素
                        for i in range(offset, offset+len(query))]
                    # 更新展平预测行的起点
                    offset += len(query)
            # 设置本轮执行结果、展平行游标
            result, cursor = [], 0
            # 逐个处理本请求的 Query 块
            for query in queries:
                # 设置本请求展平行终点
                end = cursor+len(query)
                # 向本轮执行结果追加本次内容
                result.append((picks[cursor:end],features[cursor:end].detach()))
                # 设置展平行游标
                cursor = end
            # 返回本轮执行结果
            return result
        # 无论成功或异常都执行清理
        finally:
            # 清空本次捕获的层特征
            self.captured.clear()
            # 清除本次全局前向元数据，避免影响后续调用
            reset_context()

    # 利用已确认特征并行预测 15 个候选
    def propose(self, req):
        # 新特征必须恰好填补 Draft 尚未缓存的已确认位置
        assert req.features.shape[0] == req.computed-req.draft_cache.get_seq_length()
        # 先用 MASK 初始化待预测的 16 位块
        block = torch.full((16,),self.mask,device='cuda',dtype=torch.long)
        # 块首是已知 Pending，后 15 位才需要 Draft 预测
        block[0] = req.tokens[-1]
        # 使用 Target Embedding，并转为 Draft 所需 BF16
        embedding = self.model.model.embed_tokens(block).to(torch.bfloat16).unsqueeze(0)
        # 设置新确认的 Target 五层特征
        features = req.features.to(torch.bfloat16).unsqueeze(0)
        # 设置绝对位置或结果索引
        positions = torch.arange(req.draft_cache.get_seq_length(),req.computed+16,
                                  # 指定目标设备
                                  device='cuda',dtype=torch.long).unsqueeze(0)
        # 仅在诊断开启时同步检查有限值
        if getattr(self, 'validate_tensors', True):
            # 检查张量没有 NaN 或无穷值
            assert torch.isfinite(embedding).all().item() and torch.isfinite(features).all().item()
        # 以确认上下文和 Pending/MASK 块执行一次 Draft
        hidden = self.draft(target_hidden=features, noise_embedding=embedding,
            # 指定Draft 上下文与噪声块的绝对位置
            position_ids=positions, past_key_values=req.draft_cache,
            # 保留上下文缓存，允许噪声块内部双向 Attention
            use_cache=True, is_causal=False)
        # 仅在诊断开启时同步检查有限值
        if getattr(self, 'validate_tensors', True):
            # 检查张量没有 NaN 或无穷值
            assert torch.isfinite(hidden).all().item(), 'Non-finite BF16 draft hidden states'
        # 在 Target 验证前裁掉 Draft 临时噪声块的 KV
        req.draft_cache.crop(req.computed)
        # 借 Target Head 给后 15 个预测位置计算词表分数
        logits = self.model.compute_logits(hidden[:,1:,:])
        # 检查张量形状与本轮数据一致
        assert logits.shape[:2] == (1,15)
        # 仅在诊断开启时同步检查有限值
        if getattr(self, 'validate_tensors', True):
            # 检查张量没有 NaN 或无穷值
            assert torch.isfinite(logits).all().item(), 'Non-finite draft logits'
        # 设置 Draft 预测的候选 ID
        proposals = logits.argmax(-1)[0].cpu().tolist()
        # MASK 不能作为最终候选 Token 提交
        assert self.mask not in proposals, 'Draft selected mask token; inspect draft numerics'
        # 返回 Pending 加候选，交给 Target 因果验证
        return [req.tokens[-1]]+proposals

    # 将该调用设为无梯度推理
    @torch.inference_mode()
    # 运行请求直到达到固定输出配额
    def generate(self, prompts, max_new_tokens=256, max_batch_size=2, speculative=True):
        # 输入非空且输出与批大小均为正
        assert prompts and max_new_tokens >= 1 and max_batch_size >= 1
        # 此运行时独占 KV，普通引擎必须空闲
        assert self.engine.scheduler.is_finished(), 'Regular scheduler must be idle'
        # 普通引擎不能持有此独占运行时将要使用的 KV 页
        assert not self.engine.scheduler.block_manager.used_block_ids
        # 预留完整输出和最后一个 16 位验证窗口的容量
        assert all(len(p)+max_new_tokens+16 <= self.runner.config['max_model_length'] for p in prompts)
        # 设置独占 KV 页池
        pool = PagePool(self.runner.config['max_cached_blocks'],self.runner.block_size)
        # 设置等待队列
        waiting = deque(Request(i,list(p),len(p)) for i,p in enumerate(prompts))
        # 设置正在处理的请求、已完成请求、Target 前向次数、验证轮次数
        active, completed, target_calls, verify_calls = [], {}, 0, 0
        # 设置驻留请求峰值、验证批大小峰值、有请求运行时的补位次数
        peak_active, max_verify_batch, refill_rounds = 0, 0, 0
        # 等待 GPU 工作结束，使计时和完成响应准确
        torch.cuda.synchronize()
        # 从本轮开始重新统计显存峰值
        torch.cuda.reset_peak_memory_stats()
        # 设置生成计时起点
        started = time.perf_counter()
        # 设置上次进度输出时间
        last_progress = started
        # 执行操作并确保后续清理
        try:
            # 仍有等待或活跃请求时继续生成
            while waiting or active:
                # 初始化本轮新入驻请求
                new = []
                # 在驻留上限内接纳等待请求
                while waiting and len(active)+len(new) < max_batch_size:
                    # 向本轮新入驻请求追加本次内容
                    new.append(waiting.popleft())
                # 新请求先执行完整 Prompt 前向
                if new:
                    # 更新有请求运行时的补位次数
                    refill_rounds += bool(active)
                    # 设置模型或请求输出
                    outputs = self.target_forward(new,[r.tokens for r in new],pool)
                    # 更新 Target 前向次数
                    target_calls += 1
                    # 逐个处理投机请求、Target 逐行后继预测、新确认的 Target 五层特征
                    for req,(picks,features) in zip(new,outputs):
                        # 初始 Prefill 已写完全部 Prompt KV
                        req.computed = len(req.tokens)
                        # 首个输出已确定，但它自己的 Target KV 尚未计算
                        req.tokens.append(picks[-1])
                        # 初始 Prompt 的五层特征尚待注入 Draft 上下文
                        req.features = features
                        # 每个请求维护独立的 Draft 上下文缓存
                        req.draft_cache = DynamicCache()
                        # 诊断开启时再做额外检查
                        if getattr(self, 'logit_diagnostics', False):
                            # 向逐位置预测记录追加本次内容
                            req.prediction_trace.append(self.last_predictions[req.key][-1])
                        # 设置首 Token 时间
                        req.first_token_s = time.perf_counter()-started
                    # 向正在处理的请求追加本次内容
                    active.extend(new)
                # 设置驻留请求峰值
                peak_active = max(peak_active,len(active))
                # 设置已达输出上限的请求
                done = [r for r in active if len(r.outputs)==max_new_tokens]
                # 逐个处理投机请求
                for req in done:
                    # 设置完成时间
                    req.finish_s = time.perf_counter()-started
                    # 设置已完成请求
                    completed[req.key] = req
                    # 请求完成后归还全部页所有权
                    pool.release(req.key)
                    # 从正在处理的请求移除此项
                    active.remove(req)
                # 正在处理的请求未就绪或为空时执行此分支
                if not active:
                    # 跳过本项，继续下一次循环
                    continue
                # Draft 仍逐请求提议；AR 则只输入 Pending
                blocks = [self.propose(r) if speculative else [r.tokens[-1]] for r in active]
                # 多请求候选块在一次 Target 前向中验证
                outputs = self.target_forward(active,blocks,pool)
                # 更新 Target 前向次数
                target_calls += 1
                # 更新验证轮次数
                verify_calls += 1
                # 设置验证批大小峰值
                max_verify_batch = max(max_verify_batch,len(active))
                # 逐个处理投机请求、Pending 加候选的验证块、Target 对每行的后继预测、新确认的 Target 五层特征
                for req,block,(posterior,features) in zip(active,blocks,outputs):
                    # 只接受连续正确候选并保持 Pending 不变量
                    row = commit(req,block,posterior,max_new_tokens)
                    # 仅保留有效推进位置的 Target 特征供下轮 Draft 使用
                    req.features = features[:row['advance']]
                    # 诊断开启时再做额外检查
                    if getattr(self, 'logit_diagnostics', False):
                        # 向逐位置预测记录追加本次内容
                        req.prediction_trace.extend(self.last_predictions[req.key][:row['advance']])
                    # 归还拒绝部分跨出的整页，不清零仍保留的尾页
                    pool.trim(req.key,req.computed)
                    # 向执行轨迹追加本次内容
                    req.trace.append(dict(**row,computed_tokens=req.computed))
                # 逐个处理投机请求
                for req in list(active):
                    # 检查len(req.outputs)是否满足本分支条件
                    if len(req.outputs)==max_new_tokens:
                        # 设置完成时间
                        req.finish_s = time.perf_counter()-started
                        # 设置已完成请求
                        completed[req.key] = req
                        # 请求完成后归还全部页所有权
                        pool.release(req.key)
                        # 从正在处理的请求移除此项
                        active.remove(req)
                # 检查页池所有权没有重复或丢失
                pool.check()
                # 设置当前计时
                now = time.perf_counter()
                # 检查now - last_progress是否满足本分支条件
                if now-last_progress >= 20:
                    # 设置当前累计输出数
                    made = sum(len(r.outputs) for r in completed.values()) + sum(len(r.outputs) for r in active)
                    # 输出本轮进度或诊断信息
                    print(f'{getattr(self, "run_label", "generation")}: {len(completed)}/{len(prompts)} requests complete, {made} tokens, elapsed={now-started:.1f}s', flush=True)
                    # 设置上次进度输出时间
                    last_progress = now
            # 等待 GPU 工作结束，使计时和完成响应准确
            torch.cuda.synchronize()
            # 设置总生成耗时
            seconds = time.perf_counter()-started
            # 最后必须收齐全部输入请求的输出
            assert len(completed)==len(prompts)
            # 检查页数量、容量或所有权约束
            assert not pool.tables and len(pool.free)==pool.pages
            # 设置按编号排列的请求
            requests = [completed[i] for i in range(len(prompts))]
            # 设置执行轨迹
            trace = [x for r in requests for x in r.trace]
            # 汇总实际提交的 Draft 候选数
            accepted = sum(x['accepted_draft'] for x in trace)
            # 设置累计提出的候选数
            proposed = sum(x['proposed_draft'] for x in trace)
            # 返回各项统计与执行元数据
            return dict(summary=dict(requests=len(prompts),max_batch_size=max_batch_size,
                # 只统计实际提交的输出 Token 与整组耗时
                output_tokens=len(prompts)*max_new_tokens,total_s=seconds,
                # 总输出数除以整组耗时，得到总吞吐
                output_tokens_per_s=len(prompts)*max_new_tokens/seconds,
                # 分别报告 Target 总调用与验证调用数
                target_forward_calls=target_calls,verification_batch_calls=verify_calls,
                # 报告验证批大小和驻留请求峰值
                max_verification_batch=max_verify_batch,peak_active_requests=peak_active,
                # 区分请求补位次数与已提交候选数
                refill_while_active=refill_rounds,accepted_draft_tokens=accepted,
                # 报告提出量及真正提交的候选接受率
                proposed_draft_tokens=proposed,draft_acceptance_ratio=accepted/proposed if proposed else None,
                # 用各请求验证步骤的有效推进量计算均值
                average_verified_advance=sum(x['advance'] for x in trace)/len(trace) if trace else None,
                # 报告本轮 GPU 已分配显存峰值
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                # 报告页使用峰值与回滚释放的整页数
                peak_kv_pages=pool.peak,rollback_pages=pool.rollback_pages,
                # 页已全部归还，并附各请求的真实 Token 与轨迹
                all_pages_released=True),requests=[dict(request=r.key,token_ids=r.outputs,
                # 指定首 Token 时间
                first_token_s=r.first_token_s,finish_s=r.finish_s,trace=r.trace,
                # 指定逐位置预测记录
                prediction_trace=r.prediction_trace) for r in requests])
        # 无论成功或异常都执行清理
        finally:
            # 逐个处理请求编号或图形状
            for key in list(pool.tables):
                # 归还请求的全部页所有权
                pool.release(key)
            # 检查页池所有权没有重复或丢失
            pool.check()
            # 清除本次全局前向元数据，避免影响后续调用
            reset_context()
