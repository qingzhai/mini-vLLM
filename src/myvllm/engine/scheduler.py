# 导入 deque 所需接口
from collections import deque
# 导入 Sequence, SequenceStatus 所需接口
from myvllm.engine.sequence import Sequence, SequenceStatus
# 导入 BlockManager 所需接口
from myvllm.engine.block_manager import BlockManager


# 管理等待队列、运行队列与单步预算
class Scheduler:
    # 初始化当前对象的状态
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        # block manager
        # 为调度器创建分页与前缀缓存管理器
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        # 设置单步 Token 预算
        self.max_num_batched_tokens = max_num_batched_tokens
        # 设置本路径的请求数量限制
        self.max_num_sequences = max_num_sequences
        # sequence queue
        # 声明等待队列
        self.waiting: deque[Sequence] = deque()
        # 声明驻留请求队列
        self.running: deque[Sequence] = deque()
        # 设置停止 Token
        self.eos = eos


    # 判断请求是否已经结束
    def is_finished(self):
        # 等待和运行队列都为空才算全部完成
        return len(self.waiting) == 0 and len(self.running) == 0
    
    # 把请求放入等待队列
    def add_sequence(self, sequence: Sequence):
        # Reject up front what the block manager could never satisfy, otherwise the
        # sequence sits in `waiting` forever and only surfaces as a stalled engine.
        # 设置 KV 总页数
        capacity = len(self.block_manager.blocks)
        # 检查sequence.num_blocks是否满足本分支条件
        if sequence.num_blocks > capacity:
            # 状态不满足要求时立即报错
            raise ValueError(
                # 报告当前请求需要的页数
                f"Sequence {sequence.seq_id} needs {sequence.num_blocks} blocks "
                # 报告输入长度与每页容量
                f"({len(sequence)} tokens at block_size={self.block_manager.block_size}) "
                # 报告实际 KV 总容量
                f"but the KV cache only holds {capacity}. "
                # 给出容量不足时的调整方向
                f"Raise max_cached_blocks or block_size, or shorten the prompt."
            # 结束当前调用或分组
            )
        # 向等待队列追加本次内容
        self.waiting.append(sequence)


    # 选择本轮请求与计算窗口
    def schedule(self) -> tuple[list[Sequence], bool]:
        # 初始化本步选中的请求
        scheduled_sequences = []
        # 设置本步已用 Token 数
        current_scheduled_tokens = 0
        # An empty schedule is only legitimate when this call freed blocks by
        # preempting, so the next call can make progress. See the guard below.
        # 设置是否发生抢占
        preempted = False
        # try schedule for prefilling from waiting queue if not exceeding limits
        # 原调度器限制本步 Prefill 数，不限制 running 总驻留数
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            # 设置请求
            seq = self.waiting[0]
            # KV 容量和本步 Token 预算都足够才接纳请求
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                # 设置请求
                seq = self.waiting.popleft() # remove from waiting
                # 先固定命中页再分配缺失页
                self.block_manager.allocate(seq)
                # 设置请求状态
                seq.status = SequenceStatus.RUNNING
                # 向驻留请求队列追加本次内容
                self.running.append(seq)
                # 向本步选中的请求追加本次内容
                scheduled_sequences.append(seq)
                # 更新本步已用 Token 数
                current_scheduled_tokens += len(seq)
            # 处理上述条件不成立的情况
            else:
                # 停止当前循环，保留已有进度
                break
        # 若已有可执行请求，则本步可以推进
        if scheduled_sequences:
            # 原调度器的本步只执行 Prefill
            return scheduled_sequences, True
        
        # try schedule for completion from running queue
        # 条件成立时继续选择本轮请求与计算窗口
        while self.running:
            # 依次尝试安排驻留请求的 Decode
            seq = self.running.popleft()
            # use can_append to check whether we can append one more token
            # 需要新页但页池不足时尝试抢占其他请求
            if not self.block_manager.can_append(seq):
                # 设置是否发生抢占
                preempted = True
                # 还有其他运行请求时可从队尾选择抢占对象
                if self.running:
                    # 当前请求先放回队首，再抢占其他请求
                    self.running.appendleft(seq)
                    # 抢占队尾请求，为当前请求释放 KV 页
                    self.preempt(self.running.pop())
                # 处理上述条件不成立的情况
                else:
                    # 释放页并把请求退回等待队列
                    self.preempt(seq)
                    # 停止当前循环，保留已有进度
                    break
            # 处理上述条件不成立的情况
            else:
                # 达到本步 Token 或请求数量上限就停止选取
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    # 达到单步上限时把尚未执行的请求放回队首
                    self.running.appendleft(seq)
                    # 停止当前循环，保留已有进度
                    break
                # append one token
                # 向block_manager追加本次内容
                self.block_manager.append(seq)
                # 向本步选中的请求追加本次内容
                scheduled_sequences.append(seq)
                # 更新本步已用 Token 数
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        # 若已有可执行请求，则本步可以推进
        if scheduled_sequences:
            # 恢复本轮已选请求在运行队列中的相对顺序
            self.running.extendleft(reversed(scheduled_sequences))
        # 仍有请求却没有调度或抢占进展时应报错
        elif not preempted and (self.waiting or self.running):
            # Nothing was scheduled and nothing was preempted, so no engine state
            # changed: every later schedule() would take the same decisions and
            # LLMEngine.generate() would spin forever. Fail loudly instead.
            # 状态不满足要求时立即报错
            raise RuntimeError(
                # 补充当前字段或错误说明
                "Scheduler made no progress: "
                # 报告等待与运行请求数量
                f"{len(self.waiting)} waiting and {len(self.running)} running sequences, "
                # 报告空闲页数量
                f"{len(self.block_manager.free_block_ids)} of "
                # 报告 KV 总页数
                f"{len(self.block_manager.blocks)} blocks free. "
                # 补充当前字段或错误说明
                "This means either a sequence that cannot fit in the KV cache, or "
                # 补充当前字段或错误说明
                "blocks leaked because their ref_count never returned to 0."
            # 结束当前调用或分组
            )

        # 返回本步选中的请求、False
        return scheduled_sequences, False


    # 释放页并把请求退回等待队列
    def preempt(self, seq: Sequence) -> None:
        # 归还请求持有的所有页引用
        self.block_manager.deallocate(seq)
        # 设置请求状态
        seq.status = SequenceStatus.WAITING
        # 抢占后优先重新尝试此请求
        self.waiting.appendleft(seq)        


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    # 追加采样 Token 并处理请求结束
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        # 逐个处理请求、token_id
        for seq, token_id in zip(seqs, token_ids):
            # 把本轮真实采样结果追加到请求
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            # 仅在未忽略 EOS 时使用 EOS 结束请求
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            # 达到固定输出数量即结束
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            # 配置了上下文上限时检查总长度
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            # 任一停止条件满足即结束请求并归还页引用
            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                # 设置请求状态
                seq.status = SequenceStatus.FINISHED
                # 归还请求持有的所有页引用
                self.block_manager.deallocate(seq)
                # 完成请求退出运行队列，下一步可以补位
                self.running.remove(seq)