"""V4 speculative state machine with a shared eager/graph target executor."""
# 导入 SimpleNamespace 所需接口
from types import SimpleNamespace
# 导入 torch 所需接口
import torch
# 导入 DFlashRuntime 所需接口
from dflash_runtime import DFlashRuntime
# 导入 Window 所需接口
from state import Window


# 将投机状态机接到统一 Graph 执行器
class GraphDFlashRuntime(DFlashRuntime):
    # 初始化当前对象的状态
    def __init__(self, target, draft, executor):
        # The original experimental generate() requires an idle-engine contract.
        # This adapter owns the entire cache; ordinary engine.step is not used.
        # 设置模型执行器
        runner = SimpleNamespace(model=target.model, world_size=1,
            # 兼容基础运行时的单卡 Eager 初始化约束
            enforce_eager=True, block_size=256,
            # 指定实验配置
            config={'max_model_length': 2048, 'max_cached_blocks': target.pages})
        # 设置引擎适配对象
        engine = SimpleNamespace(model_runner=runner,
            # 声明普通调度器空闲，本运行时独占 KV
            scheduler=SimpleNamespace(is_finished=lambda: True,
                # 声明普通 BlockManager 没有活跃页引用
                block_manager=SimpleNamespace(used_block_ids=set())))
        # 初始化当前对象的状态
        super().__init__(engine, draft)
        # Executor owns its own hooks and graph-resident feature outputs.
        # 移除基础运行时 Hook，统一由 Graph 执行器采集特征
        super().close()
        # 设置 Target Graph 执行器
        self.executor = executor
        # 设置 Target Graph 开关
        self.graph_enabled = False
        # 设置有限值诊断开关
        self.validate_tensors = True
        # 设置 Top-2 诊断开关
        self.logit_diagnostics = False
        # 设置当前运行标签
        self.run_label = ''

    # 将各请求 Query 合并成一次 Target 前向
    def target_forward(self, reqs, queries, pool):
        # 设置每请求的验证窗口
        views = [Window(r, q, pool.reserve(r.key, r.computed+len(q)))
                 # 将前述构造应用到每个元素
                 for r, q in zip(reqs, queries)]
        # Initial prompt stays eager even if its length happens to equal 1/16.
        # 设置本轮执行结果
        result = self.executor.run(views,
            # 初始 Prefill 即使长度匹配也不进入 Graph
            use_graph=self.graph_enabled and all(r.computed > 0 for r in reqs),
            # 指定额外诊断开关
            diagnostic=self.validate_tensors)
        # 设置新确认的 Target 五层特征
        features = result['features']
        # 诊断开启时检查输出有限值
        if self.validate_tensors:
            # 逐个处理请求编号或图形状
            for key in ('features', 'hidden', 'logits'):
                # 检查张量没有 NaN 或无穷值
                assert torch.isfinite(result[key]).all().item(), 'Non-finite '+key
        # 设置 Target 逐行后继预测
        picks = result['picks'].cpu().tolist()
        # 每个验证位置必须恰好产生一个后继预测
        assert len(picks) == sum(map(len, queries))
        # 设置模型或请求输出、合并张量中的分片偏移
        outputs, offset = [], 0
        # 逐个处理本请求的 Query 块
        for query in queries:
            # 设置窗口或分片终点
            end = offset+len(query)
            # 向模型或请求输出追加本次内容
            outputs.append((picks[offset:end], features[offset:end]))
            # 设置合并张量中的分片偏移
            offset = end
        # 初始化逐请求 Top-2 诊断
        self.last_predictions = {}
        # 返回模型或请求输出
        return outputs

    # 释放本执行器持有的资源
    def close(self):
        # 移除基础运行时 Hook，统一由 Graph 执行器采集特征
        super().close()

    # 将该调用设为无梯度推理
    @torch.inference_mode()
    # 运行请求直到达到固定输出配额
    def generate(self, *args, **kwargs):
        # 清零执行计数
        self.executor.reset_counters()
        # 设置本轮执行结果
        result = super().generate(*args, **kwargs)
        # 记录真正的 Replay、Eager 与回退计数
        result['summary']['target_execution'] = self.executor.stats()
        # 报告本次是否启用了 Target Graph
        result['summary']['target_cuda_graph'] = self.graph_enabled
        # 明确本实现没有捕获 Draft 前向
        result['summary']['draft_cuda_graph'] = False
        # 返回本轮执行结果
        return result
