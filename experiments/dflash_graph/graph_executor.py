"""Fixed-shape target verification graphs for the archived MinivLLM Qwen model.

Only B in {1, 2} and uniform Q in {1, 16} are captured. Every query uses its
absolute position and the existing per-query PagedAttention implementation.
Long initial prefills and nonuniform windows use the identical eager forward.
Layer hooks execute during capture; their tensor operations are recorded in the
graph and Python hooks are never needed during replay.

This helper is process-local and sequential. The model, KV cache and global
myvllm context must not be used concurrently by another engine.
"""
# 延迟解析类型注解
from __future__ import annotations

# 导入 dataclass 所需接口
from dataclasses import dataclass
# 导入 math 所需接口
import math
# 导入 Any 所需接口
from typing import Any

# 导入 torch 所需接口
import torch


# 兼容字典窗口与对象窗口
def _get(view, name):
    # 从字典键或对象属性读取同一个窗口字段
    return view[name] if isinstance(view, dict) else getattr(view, name)


# 构造逐 Query 的槽位、长度和页表
def build_query_metadata(views, *, max_length=2048, pages=64,
                         # 指定每页 Token 数
                         block_size=256, max_batch=2):
    """Build and validate host metadata, including queries crossing KV pages.

    token_ids holds the complete prefix plus the pending verification window;
    _scheduled_start/end delimit only the rows evaluated by this forward.
    """
    # 拒绝空批次或超过已支持请求数的批次
    if not views or len(views) > max_batch:
        # 状态不满足要求时立即报错
        raise ValueError(f'Expected 1..{max_batch} windows, got {len(views)}')
    # 设置页表固定宽度
    width = math.ceil(max_length / block_size)
    # 设置展平输入、KV 写入槽位、各 Query 的有效长度、各 Query 的页表、每请求的 Query 数
    ids, slots, lengths, tables, query_lengths = [], [], [], [], []
    # 逐个处理本请求的计算窗口
    for view in views:
        # 设置 Token 序列
        tokens = _get(view, 'token_ids')
        # 设置窗口或分片起点、窗口或分片终点
        start, end = int(_get(view, '_scheduled_start')), int(_get(view, '_scheduled_end'))
        # 设置物理页表
        table = [int(p) for p in _get(view, 'block_table')]
        # 窗口必须非空，且位于逻辑序列和上下文上限内
        if not 0 <= start < end <= min(len(tokens), max_length):
            # 状态不满足要求时立即报错
            raise ValueError(f'Invalid query window [{start}, {end}) for {len(tokens)} tokens')
        # 页表必须有足够容量寻址整个窗口
        if not 0 < len(table) <= width or end > len(table)*block_size:
            # 状态不满足要求时立即报错
            raise ValueError(f'Page table cannot address query end {end}: {table}')
        # 独占页表不能重复引用或越过物理页池
        if len(set(table)) != len(table) or any(p < 0 or p >= pages for p in table):
            # 状态不满足要求时立即报错
            raise ValueError(f'Invalid or repeated physical pages: {table}')
        # 设置补齐固定宽度的页表
        padded = table + [-1]*(width-len(table))
        # 向每请求的 Query 数追加本次内容
        query_lengths.append(end-start)
        # 将本请求 Query 追加到展平模型输入
        ids.extend(int(t) for t in tokens[start:end])
        # 逐个处理绝对 Token 位置
        for position in range(start, end):
            # 每行只允许读取到自身位置，保持 Target 因果性
            lengths.append(position+1)
            # 把本行逻辑位置转换为物理 KV 写入槽
            slots.append(table[position//block_size]*block_size + position % block_size)
            # 每个 Query 都携带所属请求的完整页表
            tables.append(padded.copy())
    # 只有各请求 Query 等长时才具有统一图桶
    uniform_q = query_lengths[0] if len(set(query_lengths)) == 1 else None
    # 返回逐 Query 对齐的输入与 KV 元数据
    return dict(input_ids=ids, slot_mapping=slots, context_lens=lengths,
                # 指定逐请求或逐 Query 物理页表
                block_tables=tables, query_lengths=query_lengths,
                # 指定本轮请求数
                batch=len(views), key=(len(views), uniform_q))


# 生成数据类的初始化与字段方法
@dataclass
# 保存一张图使用的固定地址张量
class _Buffers:
    # 声明本轮输入 ID
    input_ids: torch.Tensor
    # 声明各输入位置的 KV 写入槽
    slot_mapping: torch.Tensor
    # 声明有效历史长度
    context_lens: torch.Tensor
    # 声明逐请求或逐 Query 物理页表
    block_tables: torch.Tensor
    # 声明捕获的 CUDA 图
    graph: Any = None
    # 声明模型或请求输出
    outputs: dict | None = None
    # 声明图池显存增量
    private_allocated_bytes: int = 0


# 按固定 B/Q 桶捕获和重放 Target
class TargetGraphExecutor:
    """Own four independent graph pools and return owned outputs from each run.

    run returns picks [N], features [N, len(layer_ids)*H], hidden [N, H],
    and logits [N, V] only when diagnostic=True (otherwise logits is None).
    Returned tensors are clones, so a subsequent graph replay cannot mutate a
    request's retained features. Eager uses the same output ownership policy.
    """

    # 初始化当前对象的状态
    def __init__(self, model, layer_ids, device='cuda:0', max_length=2048,
                 # 指定物理页池总容量
                 pages=64, block_size=256, max_batch=2):
        # 设置 Target 模型
        self.model = model
        # 设置目标设备
        self.device = torch.device(device)
        # 设置采集特征的层索引
        self.layer_ids = tuple(int(i) for i in layer_ids)
        # 设置支持的上下文长度、物理页池总容量
        self.max_length, self.pages = int(max_length), int(pages)
        # 设置每页 Token 数、支持的最大请求数
        self.block_size, self.max_batch = int(block_size), int(max_batch)
        # 只支持最多两个请求，且各请求需有独立虚拟页
        if self.max_batch not in (1, 2) or self.pages < self.max_batch:
            # 状态不满足要求时立即报错
            raise ValueError('Graph experiment supports max_batch 1 or 2 and distinct dummy pages')
        # 受控窗口与每页容量至少能容纳 16 个位置
        if self.max_length < 16 or self.block_size < 16:
            # 状态不满足要求时立即报错
            raise ValueError('Graph experiment requires max_length and block_size >= 16')
        # 设置 Target 各层
        layers = model.model.layers
        # 特征层集合不能为空或包含重复层
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            # 状态不满足要求时立即报错
            raise ValueError('Feature layers must be nonempty and unique')
        # 拒绝不存在的模型层索引
        if any(i < 0 or i >= len(layers) for i in self.layer_ids):
            # 状态不满足要求时立即报错
            raise ValueError(f'Feature layer outside model: {self.layer_ids}')
        # 设置固定页表宽度
        self.table_width = math.ceil(self.max_length / self.block_size)
        # 初始化形状到图缓冲的映射
        self._entries = {}
        # 初始化本次前向层特征
        self._captured_features = {}
        # 初始化 Hook 句柄
        self._handles = []
        # 设置执行器关闭状态
        self._closed = False
        # 依次为需要的 Target 层安装特征 Hook
        for index in self.layer_ids:
            # 合并当前层输出与独立残差后采集特征
            def hook(module, args, output, index=index):
                # 层返回格式不符合 MinivLLM 约定时立即报错
                if not isinstance(output, tuple) or len(output) != 2:
                    # 状态不满足要求时立即报错
                    raise RuntimeError('Expected MinivLLM (x, residual) decoder output')
                # 设置当前特征、残差分支
                x, residual = output
                # 捕获完整层输出，后续可随图回放重新计算
                self._captured_features[index] = x + residual
            # 向Hook 句柄追加本次内容
            self._handles.append(layers[index].register_forward_hook(hook))
        # 清零执行计数
        self.reset_counters()

    # 为固定形状分配 GPU 输入缓冲
    def _allocate(self, rows):
        # 在指定设备上分配张量
        def empty(shape, dtype):
            # 返回本次计算结果
            return torch.empty(shape, dtype=dtype, device=self.device)
        # 返回本次计算结果
        return _Buffers(input_ids=empty((rows,), torch.int64),
                        # 指定各输入位置的 KV 写入槽
                        slot_mapping=empty((rows,), torch.int64),
                        # 指定有效历史长度
                        context_lens=empty((rows,), torch.int64),
                        # 指定逐请求或逐 Query 物理页表
                        block_tables=empty((rows, self.table_width), torch.int32))

    # 覆盖固定输入缓冲的全部内容
    def _fill(self, buffers, metadata):
        # All four inputs are replaced on every call; no stale page-table tails.
        # Host staging and copies intentionally remain outside graph capture.
        # 四类输入必须每轮全部更新
        for name in ('input_ids', 'slot_mapping', 'context_lens', 'block_tables'):
            # 设置固定 GPU 输入缓冲
            destination = getattr(buffers, name)
            # 先构造与目标缓冲类型相同的 CPU 输入
            source = torch.tensor(metadata[name], dtype=destination.dtype,
                                  # 在 CPU 构造暂存输入，随后复制到静态 GPU 地址
                                  device='cpu')
            # 新输入必须保持捕获时的固定形状
            if source.shape != destination.shape:
                # 状态不满足要求时立即报错
                raise RuntimeError(f'Static input shape changed: {name} {source.shape} != {destination.shape}')
            # 内容拷入捕获时的原地址，而非替换张量对象
            destination.copy_(source)

    # 执行同一条 Target 算术路径
    def _forward(self, buffers):
        # 导入 set_context, reset_context 所需接口
        from myvllm.utils import set_context, reset_context
        # 清空本次前向层特征
        self._captured_features.clear()
        # is_prefill=False supplies exact absolute RoPE positions without the
        # model's host-side cu_seqlens.cpu().tolist() branch. Attention remains
        # per-query causal PagedAttention for Q=1, Q=16 and initial full prefill.
        # 把本轮 KV 地址与有效长度传给模型
        set_context(is_prefill=False, slot_mapping=buffers.slot_mapping,
                    # 指定有效历史长度
                    context_lens=buffers.context_lens,
                    # 指定逐请求或逐 Query 物理页表
                    block_tables=buffers.block_tables)
        # 执行操作并确保后续清理
        try:
            # 设置模型输出特征
            hidden = self.model(buffers.input_ids)
            # 缺少任一指定层特征就不能继续验证
            if set(self._captured_features) != set(self.layer_ids):
                # 状态不满足要求时立即报错
                raise RuntimeError('Target feature hooks did not capture every selected layer')
            # 拼接五层特征，此张量运算也会被捕获进图
            features = torch.cat([self._captured_features[i] for i in self.layer_ids], dim=-1)
            # 设置词表分数
            logits = self.model.compute_logits(hidden)
            # 验证必须为每个 Query 输出一行 Logits
            if logits.ndim != 2 or logits.shape[0] != buffers.input_ids.shape[0]:
                # 状态不满足要求时立即报错
                raise RuntimeError('Verification LM head must produce one logit row per input row')
            # 最终词表计算必须走已验证的 FP32 Head
            if logits.dtype != torch.float32:
                # 状态不满足要求时立即报错
                raise RuntimeError(f'Expected the existing FP32 LM head, got {logits.dtype}')
            # 为每个 Query 位置选择 Target 的后继 Token
            picks = logits.argmax(dim=-1)
            # 返回逐位置预测与 Draft 所需层特征
            return dict(picks=picks, features=features, hidden=hidden, logits=logits)
        # 无论成功或异常都执行清理
        finally:
            # 清空本次前向层特征
            self._captured_features.clear()
            # 清除本次全局前向元数据，避免影响后续调用
            reset_context()

    # 将该调用设为无梯度推理
    @torch.inference_mode()
    # 预热并捕获四个 B/Q 形状桶
    def capture_all(self):
        """Capture four valid dummy shapes; caller must clear KV afterwards.

        Every graph gets a fresh private pool (no pool-sharing argument).
        Dummy windows start at zero and use distinct valid physical pages,
        so warmup/capture never accesses uninitialized prompt positions.
        """
        # 已关闭的执行器不能再使用图资源
        if self._closed:
            # 状态不满足要求时立即报错
            raise RuntimeError('Executor is closed')
        # 捕获真实 CUDA 图必须使用 CUDA 设备
        if self.device.type != 'cuda':
            # 状态不满足要求时立即报错
            raise RuntimeError('CUDA Graph capture requires an actual CUDA device')
        # 禁止在同一执行器中重复捕获已有图
        if any(entry.graph is not None for entry in self._entries.values()):
            # 状态不满足要求时立即报错
            raise RuntimeError('Graphs have already been captured')
        # 在指定资源或 CUDA 上下文内执行
        with torch.cuda.device(self.device):
            # 设置当前 CUDA Stream
            current = torch.cuda.current_stream(self.device)
            # 设置捕获用 Stream
            stream = torch.cuda.Stream(device=self.device)
            # 枚举 B=1 和 B=2 请求桶
            for batch in range(1, self.max_batch+1):
                # 分别捕获普通 AR 与投机验证形状
                for query in (1, 16):
                    # 设置 B/Q 图形状
                    key = (batch, query)
                    # 设置固定地址张量
                    buffers = self._allocate(batch*query)
                    # 设置每请求的验证窗口
                    views = [dict(token_ids=[0]*query, _scheduled_start=0,
                                  # 指定本轮窗口终点
                                  _scheduled_end=query, block_table=[b])
                             # 将前述构造应用到每个元素
                             for b in range(batch)]
                    # 构造当前虚拟输入的真实页表和因果长度
                    metadata = build_query_metadata(views, max_length=self.max_length,
                        # 指定物理页池总容量
                        pages=self.pages, block_size=self.block_size, max_batch=self.max_batch)
                    # 覆盖固定输入缓冲的全部内容
                    self._fill(buffers, metadata)
                    # 捕获流先等待主流完成输入准备
                    stream.wait_stream(current)
                    # Compile Triton and initialize GEMM state outside capture.
                    # 在指定资源或 CUDA 上下文内执行
                    with torch.cuda.stream(stream):
                        # 在图外预热算子与 GEMM 状态
                        for _ in range(3):
                            # 不使用图时执行完全相同的模型算术
                            outputs = self._forward(buffers)
                        # 移除不再需要的模型或请求输出
                        del outputs
                    # 主流等待预热流，确保编译在捕获前结束
                    current.wait_stream(stream)
                    # 等待 GPU 工作结束，使计时和完成响应准确
                    torch.cuda.synchronize(self.device)
                    # 记录图捕获前的已分配显存
                    before = torch.cuda.memory_allocated(self.device)
                    # 设置捕获的 CUDA 图
                    graph = torch.cuda.CUDAGraph()
                    # 记录 GPU 操作及依赖，Python 只在捕获时运行
                    with torch.cuda.graph(graph, stream=stream):
                        # 不使用图时执行完全相同的模型算术
                        outputs = self._forward(buffers)
                    # 主流等待捕获流完成
                    current.wait_stream(stream)
                    # 等待 GPU 工作结束，使计时和完成响应准确
                    torch.cuda.synchronize(self.device)
                    # 设置捕获的 CUDA 图、模型或请求输出
                    buffers.graph, buffers.outputs = graph, outputs
                    # 设置图池显存增量
                    buffers.private_allocated_bytes = max(0, torch.cuda.memory_allocated(self.device)-before)
                    # 设置形状到图缓冲的映射
                    self._entries[key] = buffers
                    # 输出本轮进度或诊断信息
                    print(f'TARGET CUDA GRAPH CAPTURED: batch={batch}, query={query}, '
                          # 报告图实际行数与固定页表宽度
                          f'rows={batch*query}, table_width={self.table_width}', flush=True)
        # 清零执行计数
        self.reset_counters()

    # 将该调用设为无梯度推理
    @torch.inference_mode()
    # 执行前向并返回本次结果
    def run(self, views, use_graph=False, diagnostic=False):
        # 已关闭的执行器不能再使用图资源
        if self._closed:
            # 状态不满足要求时立即报错
            raise RuntimeError('Executor is closed')
        # 先校验窗口并构造本轮全部动态元数据
        metadata = build_query_metadata(views, max_length=self.max_length,
            # 指定物理页池总容量
            pages=self.pages, block_size=self.block_size, max_batch=self.max_batch)
        # 设置 B/Q 图形状
        key = metadata['key']
        # 当前仅捕获 Q=1 和 Q=16
        eligible = key[1] in (1, 16)
        # 按完整 B/Q 组合查找固定缓冲
        buffers = self._entries.get(key) if eligible else None
        # 当前桶尚无静态输入缓冲
        if buffers is None:
            # 支持桶必须使用其已捕获的图，不能静默绕过
            if use_graph and eligible:
                # 状态不满足要求时立即报错
                raise RuntimeError(f'Graph {key} was not captured; call capture_all() before running')
            # 设置固定地址张量
            buffers = self._allocate(len(metadata['input_ids']))
            # Retain supported static shapes even in eager-only local tests.
            # 检查是否可使用相应 Graph
            if eligible:
                # 设置形状到图缓冲的映射
                self._entries[key] = buffers
        # 覆盖固定输入缓冲的全部内容
        self._fill(buffers, metadata)
        # 支持桶必须使用其已捕获的图，不能静默绕过
        if use_graph and eligible:
            # 支持桶未完成捕获时直接报错
            if buffers.graph is None:
                # 状态不满足要求时立即报错
                raise RuntimeError(f'Graph {key} has not been captured')
            # 用已更新的固定输入重新执行 GPU 图
            buffers.graph.replay()
            # 读取本次 Replay 刚写入的固定输出
            outputs = buffers.outputs
            # 更新各形状 Replay 计数
            self._graph_replays[self._key_name(key)] += 1
        # 处理上述条件不成立的情况
        else:
            # 不使用图时执行完全相同的模型算术
            outputs = self._forward(buffers)
            # 更新 Eager 调用计数
            self._eager_calls += 1
            # 本次要求 Graph 却走 Eager 时记录回退
            if use_graph:
                # 更新 Graph 回退计数
                self._fallbacks += 1
        # replay writes the captured output addresses again. Own each returned
        # feature tensor so retained per-request features cannot silently change.
        # 复制输出所有权，防止下次 Replay 覆盖请求保留状态
        result = {name: outputs[name].clone() for name in ('picks', 'features', 'hidden')}
        # 正式计时不复制大词表诊断张量
        result['logits'] = outputs['logits'].clone() if diagnostic else None
        # 返回本轮执行结果
        return result

    # 生成数据类的初始化与字段方法
    @staticmethod
    # 生成形状统计键
    def _key_name(key):
        # 返回 f'b{key[0]}-q{key[1]}'
        return f'b{key[0]}-q{key[1]}'

    # 清零执行计数
    def reset_counters(self):
        # 设置 Eager 调用计数、Graph 回退计数
        self._eager_calls, self._fallbacks = 0, 0
        # 设置各形状 Replay 计数
        self._graph_replays = {f'b{b}-q{q}': 0 for b in range(1, self.max_batch+1) for q in (1, 16)}

    # 返回本次执行的资源与调用统计
    def stats(self):
        # 设置本次捕获的层特征
        captured = [(key, value) for key, value in self._entries.items() if value.graph is not None]
        # 返回各项统计与执行元数据
        return dict(graph_replays=self._graph_replays.copy(),
                    # 汇总实际 Replay 和 Eager 次数
                    total_replays=sum(self._graph_replays.values()), eager_calls=self._eager_calls,
                    # 记录请求 Graph 却退回 Eager 的次数
                    fallbacks=self._fallbacks,
                    # 列出已经捕获的 B/Q 桶
                    captured_keys=[self._key_name(key) for key, _ in sorted(captured)],
                    # 汇总各图私有池的 allocated 增量
                    private_graph_allocated_bytes=sum(value.private_allocated_bytes for _, value in captured),
                    # 明确图池不共享，且采集指定层特征
                    private_graph_pools=True, feature_layers=list(self.layer_ids),
                    # 报告固定页表宽度
                    metadata_table_width=self.table_width,
                    # 说明返回张量已经复制为调用方独立所有
                    output_ownership='cloned in eager and graph paths')

    # 释放本执行器持有的资源
    def close(self):
        # 逐个处理层特征 Hook
        for handle in self._handles:
            # 从层特征 Hook移除此项
            handle.remove()
        # 清空Hook 句柄
        self._handles.clear()
        # 清空本次前向层特征
        self._captured_features.clear()
        # 清空形状到图缓冲的映射
        self._entries.clear()
        # 设置执行器关闭状态
        self._closed = True
