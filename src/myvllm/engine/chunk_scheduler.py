"""Single-GPU chunk scheduling; decode-first batches may contain both phases."""
# 导入 Scheduler 所需接口
from myvllm.engine.scheduler import Scheduler
# 导入 SequenceStatus 所需接口
from myvllm.engine.sequence import SequenceStatus


# 在同一步内安排分块 Prefill 与 Decode
class ChunkScheduler(Scheduler):
    # 初始化当前对象的状态
    def __init__(self, original, chunk_size=256, decode_first=True):
        # 只能在原调度器尚无请求时接管队列
        assert not original.waiting and not original.running
        # 接管原调度器的空队列和页管理状态
        self.__dict__.update(original.__dict__)
        # 设置 Prefill 单块上限
        self.chunk_size = int(chunk_size)
        # 设置 Decode 优先开关
        self.decode_first = bool(decode_first)
        # 每个 Prefill 块至少处理一个 Token
        assert self.chunk_size >= 1
        # 驻留数和 Token 预算都必须为正
        assert self.max_num_sequences >= 1 and self.max_num_batched_tokens >= 1
        # 先将本步调度记录设为空
        self.last_schedule = None
        # 初始化执行轨迹
        self.trace = []

    # 把请求放入等待队列
    def add_sequence(self, seq):
        # 拒绝空 Prompt 或非正输出配额
        if len(seq) == 0 or seq.max_tokens < 1:
            # 状态不满足要求时立即报错
            raise ValueError("A nonempty prompt and max_tokens >= 1 are required")
        # 按部署模式或命令类型选择分支
        if seq.max_model_length is not None and len(seq) >= seq.max_model_length:
            # 状态不满足要求时立即报错
            raise ValueError("Prompt must leave room for at least one output token")
        # 把请求放入等待队列
        super().add_sequence(seq)

    # 选择本轮请求与计算窗口
    def schedule(self):
        # 取得当前调度器的块管理器
        bm = self.block_manager
        # max_num_sequences is also a true resident-request bound in this path.
        # 分块路径明确限制驻留请求总数
        while self.waiting and len(self.running) < self.max_num_sequences:
            # 设置请求
            seq = self.waiting[0]
            # bm.can_allocate(seq)未就绪或为空时执行此分支
            if not bm.can_allocate(seq):
                # 停止当前循环，保留已有进度
                break
            # 从等待队列移出已接纳请求
            self.waiting.popleft()
            # 先固定命中页再分配缺失页
            bm.allocate(seq)
            # 从真正可复用的 KV 末尾开始计算
            seq.num_computed_tokens = seq.num_cached_tokens
            # 设置请求状态
            seq.status = SequenceStatus.RUNNING
            # 向驻留请求队列追加本次内容
            self.running.append(seq)
        # 设置尚未产出首 Token 的请求
        prefills = [s for s in self.running if s.num_completion_tokens == 0]
        # 设置已进入 Decode 的请求
        decodes = [s for s in self.running if s.num_completion_tokens > 0]
        # 初始化本轮选中请求
        selected = []
        # 设置剩余 Token 预算
        budget = self.max_num_batched_tokens

        # 给一个 Decode 请求分配一个 Token 预算
        def add_decode(seq):
            # 共享外层的剩余 Token 预算
            nonlocal budget
            # 本轮预算或请求名额用尽时不再安排
            if budget == 0 or len(selected) >= self.max_num_sequences:
                # 结束当前操作
                return
            # 分块路径页耗尽时直接报错，没有继承抢占调度
            if not bm.can_append(seq):
                # 状态不满足要求时立即报错
                raise RuntimeError("Chunk scheduler exhausted KV pages; increase KV pool or lower concurrency")
            # 向bm追加本次内容
            bm.append(seq)
            # Decode 前只能有末尾一个 Token 尚未计算
            assert seq.num_computed_tokens == len(seq) - 1
            # Decode 窗口从末尾待计算 Token 开始
            seq._scheduled_start = len(seq) - 1
            # 右开窗口只包含一个 Decode Token
            seq._scheduled_end = len(seq)
            # 设置本轮窗口阶段
            seq._scheduled_prefill = False
            # 向本轮选中请求追加本次内容
            selected.append(seq)
            # 本请求 Decode 消耗一个 Token 预算
            budget -= 1

        # 用剩余预算截取 Prefill 窗口
        def add_prefill(seq):
            # 共享外层的剩余 Token 预算
            nonlocal budget
            # 本轮预算或请求名额用尽时不再安排
            if budget == 0 or len(selected) >= self.max_num_sequences:
                # 结束当前操作
                return
            # 设置窗口或分片起点
            start = seq.num_computed_tokens
            # 设置尚未处理的数量
            remaining = len(seq) - start
            # 已达输出上限的请求不得再次提交
            assert remaining > 0
            # The full-prefill control must not silently split a prompt to fit.
            # Full 对照不能为了适应预算而隐式分块
            if not self.decode_first and remaining > budget:
                # 本轮没有请求得到计算窗口
                if not selected:
                    # 状态不满足要求时立即报错
                    raise ValueError("Full-prefill control requires token budget >= uncached prompt length")
                # 结束当前操作
                return
            # 同时受剩余输入、块长和本步预算约束
            count = min(remaining, self.chunk_size, budget)
            # 设置本轮窗口起点
            seq._scheduled_start = start
            # 设置本轮窗口终点
            seq._scheduled_end = start + count
            # 设置本轮窗口阶段
            seq._scheduled_prefill = True
            # 向本轮选中请求追加本次内容
            selected.append(seq)
            # 扣除本次 Prefill 窗口消耗的预算
            budget -= count

        # Decode 优先模式先消耗单 Token 预算
        if self.decode_first:
            # 逐个处理请求
            for seq in decodes:
                # 给一个 Decode 请求分配一个 Token 预算
                add_decode(seq)
            # 逐个处理请求
            for seq in prefills:
                # 用剩余预算截取 Prefill 窗口
                add_prefill(seq)
        # 非 Decode 优先时，有 Prefill 就先执行 Prefill
        elif prefills:
            # 逐个处理请求
            for seq in prefills:
                # 用剩余预算截取 Prefill 窗口
                add_prefill(seq)
        # 处理上述条件不成立的情况
        else:
            # 逐个处理请求
            for seq in decodes:
                # 给一个 Decode 请求分配一个 Token 预算
                add_decode(seq)

        # 本轮没有请求得到计算窗口
        if not selected:
            # 有未完成请求却选不出窗口时立即报错
            if self.waiting or self.running:
                # 状态不满足要求时立即报错
                raise RuntimeError("Chunk scheduler made no progress; check KV capacity and token budget")
            # 返回[]、False
            return [], False
        # 逐个处理请求
        for seq in selected:
            # 完成请求退出运行队列，下一步可以补位
            self.running.remove(seq)
            # 向驻留请求队列追加本次内容
            self.running.append(seq)
        # 设置各请求窗口记录
        items = [dict(seq_id=s.seq_id,
                      # 指定本窗口的执行阶段
                      phase="prefill" if s._scheduled_prefill else "decode",
                      # 指定窗口或分片起点
                      start=s._scheduled_start, end=s._scheduled_end,
                      # 指定输入 Token 数
                      prompt_tokens=s.num_prompt_tokens,
                      # 指定本步之前的输出数
                      completion_tokens_before=s.num_completion_tokens)
                 # 将前述构造应用到每个元素
                 for s in selected]
        # 设置本步包含的阶段
        phases = {x["phase"] for x in items}
        # 设置本步调度记录
        self.last_schedule = dict(step=len(self.trace),
                                  # 指定Token 序列
                                  tokens=sum(x["end"] - x["start"] for x in items),
                                  # 记录本轮是否同时包含两种阶段
                                  mixed=len(phases) == 2, items=items)
        # 验证所有窗口合计不超过单步预算
        assert self.last_schedule["tokens"] <= self.max_num_batched_tokens
        # 向执行轨迹追加本次内容
        self.trace.append(self.last_schedule)
        # 任一窗口含 Prefill 就走分块执行入口
        return selected, "prefill" in phases
