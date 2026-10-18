"""Run a mixed varlen batch once; sample only completed prefills and decodes."""
# 导入 torch 所需接口
import torch
# 导入 reset_context 所需接口
from myvllm.utils import reset_context


# 将该调用设为无梯度推理
@torch.inference_mode()
# 执行混合窗口并只采样已就绪的请求
def run_chunked(runner, seqs):
    # 确认当前并行配置在支持范围内
    assert runner.world_size == 1
    # 执行操作并确保后续清理
    try:
        # 设置本轮输入 ID
        input_ids = runner.prepare_prefill(seqs)
        # 设置模型输出特征
        hidden = runner.model(input_ids)
        # 分别记录 Hidden 行、可采样请求和结果写回位置
        indices, ready, output_indices = [], [], []
        # 设置展平行游标
        cursor = 0
        # 逐个处理i、请求
        for i, seq in enumerate(seqs):
            # 越过当前请求在展平 Hidden 中占据的行
            cursor += seq._scheduled_end - seq._scheduled_start
            # 只有 Decode 或最后一个 Prefill 块允许采样
            if not seq._scheduled_prefill or seq._scheduled_end == len(seq):
                # 为可采样请求选择窗口最后一行特征
                indices.append(cursor - 1)
                # 向允许采样的请求列表追加本次内容
                ready.append(seq)
                # 向采样结果对应的请求位置追加本次内容
                output_indices.append(i)
        # 用 -1 标记本轮不应采样的请求，不是 PAD Token
        output = torch.full((len(seqs),), -1, dtype=torch.long,
                            # 指定目标设备
                            device=input_ids.device)
        # 至少有一个窗口已允许采样时才计算 Head
        if ready:
            # 设置行索引
            index = torch.tensor(indices, dtype=torch.long, device=hidden.device)
            # 设置可采样请求的特征
            selected_hidden = hidden.index_select(0, index)
            # chunk_lm_head_rows_v2: rows already selected; disable Prefill gathering.
            # 清除本次全局前向元数据，避免影响后续调用
            reset_context()
            # 设置词表分数
            logits = runner.model.compute_logits(selected_hidden)
            # 检查张量形状与本轮数据一致
            assert logits.ndim == 2 and logits.shape[0] == len(ready), (
                # 补充当前字段或错误说明
                "Chunk LM Head must return one logit row per sampled sequence")
            # 设置采样 Token 列表
            sampled = runner.sampler(logits, runner.prepare_sample(ready))
            # 设置绝对位置或结果索引
            positions = torch.tensor(output_indices, dtype=torch.long, device=output.device)
            # 把采样结果写回对应请求，其余仍为 -1
            output.index_copy_(0, positions, sampled.to(torch.long).reshape(-1))
        # 返回本轮返回值
        return output
    # 无论成功或异常都执行清理
    finally:
        # 清除本次全局前向元数据，避免影响后续调用
        reset_context()


# 衔接分块调度、模型前向和状态提交
def chunked_step(engine):
    # 设置本轮请求、是否包含 Prefill
    seqs, is_prefill = engine.scheduler.schedule()
    # 本轮请求未就绪或为空时执行此分支
    if not seqs:
        # 返回[]、0、是否包含 Prefill
        return [], 0, is_prefill
    # Pure decode keeps the existing CUDA Graph path. A mixed step is ONE
    # model forward through the paged varlen prefill path, including Q=1 decodes.
    # 设置执行入口
    method = "run_chunked" if is_prefill else "run"
    # 设置调用参数
    args = (seqs,) if is_prefill else (seqs, False)
    # 设置本轮执行结果
    result = engine.model_runner.call(method, *args)
    # 设置 Token 序列
    tokens = result.cpu().tolist()
    # 每个调度请求都必须有一个结果或 -1 哨兵
    assert len(tokens) == len(seqs)
    # 设置本轮允许提交输出的请求、采样 Token 列表
    completed, sampled = [], []
    # 设置本轮处理的 Token 数
    processed = 0
    # 逐个处理请求、当前输出 Token
    for seq, token in zip(seqs, tokens):
        # 设置窗口或分片终点
        end = seq._scheduled_end
        # 更新本轮处理的 Token 数
        processed += end - seq._scheduled_start
        # 设置已经计算的位置数
        seq.num_computed_tokens = end
        # Partial prompts must publish ONLY pages whose KV is already written.
        # 只发布本轮真正写完的完整页
        engine.scheduler.block_manager.mark_computed(seq, computed_tokens=end)
        # Prompt 尚未全部计算，只更新 KV 而不提交输出
        if seq._scheduled_prefill and end < len(seq):
            # 部分 Prefill 只能返回不采样哨兵，不能新增输出
            assert token == -1 and seq.num_completion_tokens == 0
        # 处理上述条件不成立的情况
        else:
            # 完整 Prefill 或 Decode 必须返回真实 Token ID
            assert token >= 0
            # 向本轮允许提交输出的请求追加本次内容
            completed.append(seq)
            # 向采样 Token 列表追加本次内容
            sampled.append(token)
    # 记录本轮真正追加了输出的请求
    engine.scheduler.last_schedule["sampled_seq_ids"] = [s.seq_id for s in completed]
    # 记录本步实际经过混合、Prefill 或普通 Decode 入口
    engine.scheduler.last_schedule["execution"] = (
        # 补充当前字段或错误说明
        "mixed_varlen" if engine.scheduler.last_schedule["mixed"]
        # 条件不成立时采用另一种取值
        else "prefill_varlen" if is_prefill else "existing_decode_path")
    # 仅对实际采样的请求追加输出并判断结束
    engine.scheduler.postprocess(completed, sampled)
    # 设置模型或请求输出
    outputs = [(s.seq_id, s.completion_token_ids) for s in completed if s.is_finished]
    # 返回完成请求、本步工作量与阶段标志
    return outputs, processed, is_prefill
