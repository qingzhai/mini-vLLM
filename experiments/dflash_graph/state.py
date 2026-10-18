"""Pure Python state accounting for experimental paged speculative decoding."""
# 导入 deque 所需接口
from collections import deque
# 导入 dataclass, field 所需接口
from dataclasses import dataclass, field

# 管理本运行时独占的 KV 页
class PagePool:
    # 初始化当前对象的状态
    def __init__(self, pages, block_size):
        # 设置页池总页数、每页 Token 数
        self.pages, self.block_size = pages, block_size
        # 交错分配物理页，避免误依赖连续页号
        self.free = deque(list(range(0,pages,2))+list(range(1,pages,2)))
        # 初始化请求到独占页表的映射
        self.tables = {}
        # 设置页使用峰值
        self.peak = 0
        # 设置回滚归还的整页数
        self.rollback_pages = 0
    # 按临时验证终点预留页
    def reserve(self, key, tokens):
        # 设置物理页表
        table = self.tables.setdefault(key, [])
        # 按有效或临时长度向上取整计算页需求
        needed = (tokens+self.block_size-1)//self.block_size
        # 检查页数量、容量或所有权约束
        assert len(self.free) >= max(needed-len(table),0), 'Speculative KV pool exhausted'
        # 页表不足时逐页补充分配
        while len(table) < needed:
            # 向物理页表追加本次内容
            table.append(self.free.popleft())
        # 设置页使用峰值
        self.peak = max(self.peak, self.pages-len(self.free))
        # 返回物理页表
        return table
    # 归还有效长度之外的整页
    def trim(self, key, tokens):
        # 设置物理页表
        table = self.tables[key]
        # 按有效或临时长度向上取整计算页需求
        needed = (tokens+self.block_size-1)//self.block_size
        # 物理页超过有效长度时逐页归还尾部
        while len(table) > needed:
            # 从页表尾部归还超出有效范围的整页
            self.free.append(table.pop())
            # 更新回滚归还的整页数
            self.rollback_pages += 1
    # 归还请求的全部页所有权
    def release(self, key):
        # 删除请求页表并归还全部页
        self.free.extend(self.tables.pop(key))
    # 检查每一页恰好位于空闲或占用集合
    def check(self):
        # 设置空闲与占用页的合集
        values = list(self.free)+[p for t in self.tables.values() for p in t]
        # 每个物理页只能出现一次，且总页数不丢失
        assert len(values)==self.pages and set(values)==set(range(self.pages))

# 生成数据类的初始化与字段方法
@dataclass
# 保存一个投机请求的 Token、KV 和特征状态
class Request:
    # 声明请求编号
    key: int
    # 声明 Token 序列
    tokens: list
    # 声明原始 Prompt 长度
    prompt_length: int
    # 声明有效 Target KV 长度
    computed: int = 0
    # 声明新确认的 Target 五层特征
    features: object = None
    # 声明本请求的 Draft KV
    draft_cache: object = None
    # 声明首 Token 时间
    first_token_s: float = 0.0
    # 声明完成时间
    finish_s: float = 0.0
    # 声明执行轨迹
    trace: list = field(default_factory=list)
    # 声明逐位置预测记录
    prediction_trace: list = field(default_factory=list)
    # 把方法暴露为只读属性
    @property
    # 只返回 Prompt 后的生成部分
    def outputs(self):
        # 按原始 Prompt 边界截出真实生成 Token
        return self.tokens[self.prompt_length:]

# 构造本次待计算窗口
class Window:
    # 初始化当前对象的状态
    def __init__(self, req, query, table):
        # 初始 Prompt 尚未计算，后续窗口必须仅留一个 Pending
        assert len(req.tokens) == req.computed+1 or req.computed == 0
        # 设置完整逻辑 Token 序列
        self.token_ids = req.tokens[:req.computed]+list(query)
        # 设置逻辑到物理页表
        self.block_table = table.copy()
        # 新请求或独占窗口的缓存前缀长度初始为零
        self.num_cached_tokens = 0
        # 设置本轮窗口起点
        self._scheduled_start = req.computed
        # 设置本轮窗口终点
        self._scheduled_end = req.computed+len(query)
        # Verification uses causal paged varlen attention, including Q=1 baseline.
        # 设置本轮窗口阶段
        self._scheduled_prefill = True
    # 返回当前逻辑长度
    def __len__(self):
        # 返回窗口所描述的完整逻辑序列长度
        return len(self.token_ids)

# 统计从头连续匹配的 Draft 候选
def accept_prefix(block, posterior):
    # 每个输入位置必须都有对应的 Target 后继预测
    assert len(block)==len(posterior) and block
    # 设置连续匹配候选数
    accepted = 0
    # 候选位置与前一个 Target 位置的后继预测对齐
    for draft, target in zip(block[1:], posterior[:-1]):
        # 遇到首个错误候选即停止连续接受
        if draft != target:
            # 停止当前循环，保留已有进度
            break
        # 更新连续匹配候选数
        accepted += 1
    # 返回连续匹配候选数
    return accepted

# 提交正确输出并保留一个 Pending Token
def commit(req, block, posterior, limit):
    """Maintain one pending token; reject all KV beyond the committed prefix."""
    # 末尾必须恰好保留一个尚未计算的真实 Token
    assert len(req.tokens)==req.computed+1
    # 验证块必须从当前真实 Pending 开始
    assert req.tokens[-1]==block[0]
    # 设置连续匹配候选数
    accepted = accept_prefix(block, posterior)
    # 计算还允许提交多少个输出 Token
    remaining = limit-len(req.outputs)
    # 已达输出上限的请求不得再次提交
    assert remaining > 0
    # 提交正确候选并补 Target 预测，最后按配额截断
    emitted = (list(block[1:accepted+1])+[posterior[accepted]])[:remaining]
    # 有效 KV 前进量等于本轮真正提交的输出数
    advance = len(emitted)
    # 将接受候选和补充预测加入真实输出
    req.tokens.extend(emitted)
    # KV 有效范围仅推进到已提交前缀，最末输出仍 Pending
    req.computed += advance
    # 提交后重新验证一个 Pending Token 的不变量
    assert req.computed == len(req.tokens)-1
    # 返回接受、提出和有效推进三种不同统计
    return dict(accepted_draft=min(accepted, advance), matched_draft_prefix=accepted, proposed_draft=len(block)-1,
                # 指定本轮有效推进长度
                advance=advance, rejected_draft=len(block)-1-accepted)
