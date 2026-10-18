# 导入 Enum, auto 所需接口
from enum import Enum, auto
# 导入 math 所需接口
import math
# 导入 count 所需接口
from itertools import count 
# 导入 SamplingParams 所需接口
from myvllm.sampling_parameters import SamplingParams
# 导入 copy 所需接口
from copy import copy


# 区分等待、运行和结束状态
class SequenceStatus(Enum):
    # 请求等待被接纳计算
    WAITING = auto()
    # 请求已经持有 KV 页并等待后续执行
    RUNNING = auto()
    # 请求达到停止条件，页引用已可归还
    FINISHED = auto()


# 保存普通引擎请求的逻辑状态
class Sequence:
    # 为新请求生成递增编号
    counter = count()

    # 初始化当前对象的状态
    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        # 设置每页 Token 数
        self.block_size = block_size # number of tokens per block
        # record sequence id
        # 设置请求编号
        self.seq_id = next(Sequence.counter)
        # status
        # 新请求先等待调度
        self.status = SequenceStatus.WAITING
        # token ids, need copy so that it is a new list, won't be affected by outside changes
        # 复制输入，避免与调用方共享可变列表
        self.token_ids = copy(token_ids)
        # last token
        # 设置末尾待计算 Token
        self.last_token = self.token_ids[-1] if self.token_ids else None
        # num_tokens, num_prompt_tokens
        # 设置序列总长度
        self.num_tokens = len(self.token_ids)
        # 保存原始 Prompt 边界，生成时不再改变
        self.num_prompt_tokens = len(self.token_ids)
        # num_cached_tokens = 0
        # 新请求或独占窗口的缓存前缀长度初始为零
        self.num_cached_tokens = 0
        # block_table
        # 初始化逻辑到物理页表
        self.block_table = []
        # sampling_params' related things
        # 设置采样温度
        self.temperature = sampling_params.temperature
        # 设置输出长度上限
        self.max_tokens = sampling_params.max_tokens
        # 设置是否忽略 EOS
        self.ignore_eos = sampling_params.ignore_eos
        # 设置上下文长度上限
        self.max_model_length = sampling_params.max_model_length

    # 返回当前逻辑长度
    def __len__(self):
        # 返回包含 Prompt 与生成结果的总长度
        return self.num_tokens

    # 按逻辑位置取 Token
    def __getitem__(self, idx):
        # 按逻辑下标返回一个 Token
        return self.token_ids[idx]

    # 把方法暴露为只读属性
    @property
    # 判断请求是否已经结束
    def is_finished(self):
        # 只有结束状态才向调用方报告完成
        return self.status == SequenceStatus.FINISHED

    # 把方法暴露为只读属性
    @property
    # 计算已生成 Token 数
    def num_completion_tokens(self):
        # 用总长度减去 Prompt 长度得到输出数
        return self.num_tokens - self.num_prompt_tokens

    # 把方法暴露为只读属性
    @property
    # 取得原始 Prompt
    def prompt_token_ids(self):
        # 按保存的 Prompt 边界取原始输入
        return self.token_ids[:self.num_prompt_tokens]

    # 把方法暴露为只读属性
    @property
    # 取得生成部分
    def completion_token_ids(self):
        # 取原始输入之后的所有真实生成 Token
        return self.token_ids[self.num_prompt_tokens:]

    # 把方法暴露为只读属性
    @property
    # 计算已缓存页数量
    def num_cached_blocks(self):
        # 把缓存前缀长度换算为页数
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    # 把方法暴露为只读属性
    @property
    # 按总长度向上取整计算页数
    def num_blocks(self):
        # 向上取整，为未填满的尾页也计一页
        return int(math.ceil(self.num_tokens / self.block_size))

    # 把方法暴露为只读属性
    @property
    # 计算最后一页使用的槽位数
    def last_block_num_tokens(self):
        # 减去前面整页容量，得到尾页已用槽数
        return self.num_tokens - max(self.num_blocks - 1, 0) * self.block_size

    # 取得一个逻辑页的 Token
    def block(self, i):
        # 拒绝超出逻辑页数的访问
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        # 最后一页可能未填满，需要单独取尾部
        if i == self.num_blocks - 1:
            # 返回尾页实际已有的 Token
            return self.token_ids[-self.last_block_num_tokens:]
        # 处理上述条件不成立的情况
        else:
            # 计算当前逻辑页的首 Token 下标
            start_idx = i * self.block_size
            # 计算完整逻辑页的右开终点
            end_idx = start_idx + self.block_size
            # 返回指定完整逻辑页的 Token
            return self.token_ids[start_idx : end_idx]

    # 追加一个真实输出 Token
    def append_token(self, token_id):
        # 向完整逻辑 Token 序列追加本次内容
        self.token_ids.append(token_id)
        # 设置末尾待计算 Token
        self.last_token = token_id
        # 追加输出后同步逻辑总长度
        self.num_tokens += 1 

    # 准备跨进程传输的请求状态
    def __getstate__(self):
        # 返回跨进程所需的最小请求状态
        return (
            # 补充准备跨进程传输的请求状态的字段
            self.num_tokens, 
            # 补充准备跨进程传输的请求状态的字段
            self.num_prompt_tokens, 
            # 补充准备跨进程传输的请求状态的字段
            self.num_cached_tokens, 
            # 补充准备跨进程传输的请求状态的字段
            self.block_table,
            # Prefill 传完整输入，Decode 仅需传最后一个 Token
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        # 结束当前调用或分组
        )

    # 恢复跨进程传入的请求状态
    def __setstate__(self, state):
        # 设置序列总长度、原始输入长度、可复用的前缀长度、逻辑到物理页表、last_token_or_ids
        (
            # 恢复逻辑总长度
            self.num_tokens,
            # 恢复原始 Prompt 边界
            self.num_prompt_tokens,
            # 恢复缓存前缀长度
            self.num_cached_tokens,
            # 恢复逻辑到物理页表
            self.block_table,
            # 接收完整 Prompt 或单个末尾 Token
            last_token_or_ids
        # 从传入状态解包这些字段
        ) = state
        # Check if this is prefill (num_completion_tokens == 0) or decode phase
        # 用恢复后的总长度判断是否已经进入 Decode
        num_completion_tokens = self.num_tokens - self.num_prompt_tokens
        # 检查num_completion_tokens是否满足本分支条件
        if num_completion_tokens == 0:
            # Prefill: last_token_or_ids is the full token_ids list
            # Prefill 恢复完整输入 Token
            self.token_ids = last_token_or_ids
        # 处理上述条件不成立的情况
        else:
            # Decode: last_token_or_ids is just the last token
            # Decode 仅恢复本轮所需的末尾 Token
            self.token_ids = [last_token_or_ids]
        # Restore last_token attribute
        # 设置末尾待计算 Token
        self.last_token = self.token_ids[-1] if self.token_ids else None
