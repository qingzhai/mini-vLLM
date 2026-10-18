"""Paged KV ownership, ready-only prefix reuse, and LRU of unreferenced pages."""
# 导入 deque 所需接口
from collections import deque
# 导入 blake2b 所需接口
from hashlib import blake2b
# 导入 struct 所需接口
import struct


# 保存单个物理页的身份与引用
class Block:
    # 初始化当前对象的状态
    def __init__(self, block_id):
        # 设置物理页号
        self.block_id = block_id
        # 用 -1 标记尚未建立有效前缀身份
        self.hash = -1
        # 初始没有活跃请求引用此页
        self.ref_count = 0
        # 清除该页原先登记的 Token 内容
        self.token_ids = []
        # 完整页 KV 尚未确认写完，不能共享
        self.ready = False

    # 登记本页的前缀身份与 Token
    def update(self, h, token_ids):
        # 设置前缀哈希
        self.hash = h
        # 设置本页登记的 Token
        self.token_ids = list(token_ids)

    # 清除旧身份并建立首次页引用
    def reset(self):
        # 用 -1 标记尚未建立有效前缀身份
        self.hash = -1
        # 新分配的物理页先由当前请求独占
        self.ref_count = 1
        # 清除该页原先登记的 Token 内容
        self.token_ids = []
        # 完整页 KV 尚未确认写完，不能共享
        self.ready = False


# 管理分页分配、前缀共享与空闲页驱逐
class BlockManager:
    # 初始化当前对象的状态
    def __init__(self, num_blocks, block_size):
        # 设置每页 Token 数
        self.block_size = block_size
        # 为每个物理页建立独立的身份记录
        self.blocks = [Block(i) for i in range(num_blocks)]
        # 用前缀身份定位已驻留物理页
        self.hash_to_block_id = {}
        # 所有物理页初始都可分配
        self.free_block_ids = deque(range(num_blocks))
        # 设置活跃引用的物理页集合
        self.used_block_ids = set()
        # 设置前缀缓存开关
        self.enable_prefix_caching = True
        # 设置统计信息
        self.stats = dict(cache_hit_tokens=0, cache_hit_blocks=0,
                          # 指定物理页重分配次数
                          allocations=0, evictions=0)

    # 把前缀身份与当前页 Token 链式哈希
    def compute_hash(self, token_ids, prefix_hash_value):
        # 设置哈希状态
        digest = blake2b(digest_size=8)
        # 首块以外把上一块哈希纳入身份
        if prefix_hash_value != -1:
            # 加入此前整段前缀的哈希
            digest.update(prefix_hash_value.to_bytes(8, "little"))
        # 将本页 Token 编码后加入哈希
        digest.update(struct.pack(f"<{len(token_ids)}I", *token_ids))
        # 将 64 位摘要转换为整数索引
        return int.from_bytes(digest.digest(), "little")

    # 查找从头连续就绪的整页前缀
    def _cached_prefix(self, seq):
        # 前缀缓存关闭时不查找历史页
        if not self.enable_prefix_caching:
            # 本次没有可复用内容
            return []
        # 设置连续命中的前缀页、当前前缀哈希
        hits, h = [], -1
        # Leave at least one uncached token; the last writable page stays private.
        # 至少留一个未缓存 Token，不共享可写尾页
        for i in range((len(seq) - 1) // self.block_size):
            # 设置 Token 序列
            tokens = seq.block(i)
            # 设置当前前缀哈希
            h = self.compute_hash(tokens, h)
            # 设置物理页号
            block_id = self.hash_to_block_id.get(h)
            # 当前前缀页不存在，连续命中到此为止
            if block_id is None:
                # 停止当前循环，保留已有进度
                break
            # 设置当前物理页记录
            block = self.blocks[block_id]
            # 未就绪或身份不符时停止前缀匹配
            if not block.ready or block.hash != h or block.token_ids != tokens:
                # 停止当前循环，保留已有进度
                break
            # 向连续命中的前缀页追加本次内容
            hits.append(block_id)
        # 返回连续命中的前缀页
        return hits

    # 扣除命中页后检查分配容量
    def can_allocate(self, seq):
        # 设置连续命中的前缀页
        hits = self._cached_prefix(seq)
        # 空闲命中页也必须先预留，不能算作可覆盖容量
        reserved_free = sum(self.blocks[i].ref_count == 0 for i in hits)
        # 缺失页只能使用未被命中页预留的空闲容量
        return seq.num_blocks - len(hits) <= len(self.free_block_ids) - reserved_free

    # 覆盖一个空闲页并淘汰旧缓存
    def _allocate_block(self, block_id):
        # 设置当前物理页记录
        block = self.blocks[block_id]
        # 只有没有活跃引用的页才允许覆盖
        assert block.ref_count == 0
        # 只有索引仍指向旧物理页时才删除旧身份
        if self.hash_to_block_id.get(block.hash) == block_id:
            # 覆盖物理页之前移除其旧缓存映射
            del self.hash_to_block_id[block.hash]
            # 更新旧缓存淘汰次数
            self.stats["evictions"] += 1
        # 固定此页，使其不再被当作空闲页分配
        self.free_block_ids.remove(block_id)
        # 将此页纳入活跃引用集合
        self.used_block_ids.add(block_id)
        # 重置页身份并建立第一个引用
        block.reset()
        # 更新物理页重分配次数
        self.stats["allocations"] += 1
        # 返回当前物理页记录
        return block

    # 归还零引用页并保留就绪内容
    def _deallocate_block(self, block_id):
        # 设置当前物理页记录
        block = self.blocks[block_id]
        # 确认该页刚失去最后引用，且尚未入空闲队列
        assert block.ref_count == 0 and block_id in self.used_block_ids
        # 从活跃引用的物理页集合移除此项
        self.used_block_ids.remove(block_id)
        # Keep computed full pages resident until their physical page is reused.
        # 不保留尚未写完的页身份
        if not block.ready:
            # 初始化本页登记的 Token
            block.token_ids = []
            # 设置前缀哈希
            block.hash = -1
        # 归还到队尾，之后优先覆盖更早空闲的页
        self.free_block_ids.append(block_id)

    # 先固定命中页再分配缺失页
    def allocate(self, seq):
        # 尚未分配的请求才能走首次分配
        assert not seq.block_table and seq.num_cached_tokens == 0
        # 分配前确认容量，避免覆盖活跃页
        assert self.can_allocate(seq), "Insufficient free KV pages"
        # 设置连续命中的前缀页
        hits = self._cached_prefix(seq)
        # Pin all hit pages before allocating misses, so they cannot be evicted.
        # 逐个处理物理页号
        for block_id in hits:
            # 设置当前物理页记录
            block = self.blocks[block_id]
            # 该页已无使用者，需要从空闲状态转回占用状态
            if block.ref_count == 0:
                # 固定此页，使其不再被当作空闲页分配
                self.free_block_ids.remove(block_id)
                # 将此页纳入活跃引用集合
                self.used_block_ids.add(block_id)
            # 命中页增加引用，保留原哈希和有效 KV
            block.ref_count += 1
            # 向逻辑到物理页表追加本次内容
            seq.block_table.append(block_id)
        # 设置可复用的前缀长度
        seq.num_cached_tokens = len(hits) * self.block_size
        # 更新命中的整页数
        self.stats["cache_hit_blocks"] += len(hits)
        # 更新命中的输入 Token 数
        self.stats["cache_hit_tokens"] += seq.num_cached_tokens
        # 为尚未命中的逻辑页逐一补充分配
        while len(seq.block_table) < seq.num_blocks:
            # 覆盖最早空闲的物理页
            block = self._allocate_block(self.free_block_ids[0])
            # 向逻辑到物理页表追加本次内容
            seq.block_table.append(block.block_id)

    # 只发布真正写完的完整 KV 页
    def mark_computed(self, seq, computed_tokens=None):
        # Called AFTER model execution and BEFORE appending the sampled token.
        # Only pages whose entire KV has actually been written become reusable.
        # 前缀缓存关闭时不查找历史页
        if not self.enable_prefix_caching:
            # 结束当前操作
            return
        # 设置当前前缀哈希
        h = -1
        # chunk_publish_limit_v1
        # 设置计算或输出边界
        limit = len(seq) if computed_tokens is None else computed_tokens
        # 发布范围不能超过当前已知 Token 序列
        assert 0 <= limit <= len(seq)
        # 只发布计算终点以内的完整页
        for i in range(limit // self.block_size):
            # 设置 Token 序列
            tokens = seq.block(i)
            # 设置当前前缀哈希
            h = self.compute_hash(tokens, h)
            # 设置当前物理页记录
            block = self.blocks[seq.block_table[i]]
            # 已发布的页必须保持原前缀身份
            if block.ready:
                # 已共享就绪页的身份不得在写入时改变
                assert block.hash == h and block.token_ids == tokens
            # 处理上述条件不成立的情况
            else:
                # 补充当前字典或状态对象的字段
                block.update(h, tokens)
                # 设置是否具备完整有效 KV
                block.ready = True
            # 发布这段完整前缀对应的物理页
            self.hash_to_block_id[h] = block.block_id

    # 归还请求持有的所有页引用
    def deallocate(self, seq):
        # Returning prefix pages last makes them recent in the free-page LRU.
        # 尾页先归还，让前缀页在空闲队列中更靠后
        for block_id in reversed(seq.block_table):
            # 设置当前物理页记录
            block = self.blocks[block_id]
            # 每次归还必须对应一个真实页引用
            assert block.ref_count > 0
            # 只归还当前请求的一份引用
            block.ref_count -= 1
            # 该页已无使用者，需要从空闲状态转回占用状态
            if block.ref_count == 0:
                # 归还零引用页并保留就绪内容
                self._deallocate_block(block_id)
        # 初始化逻辑到物理页表
        seq.block_table = []
        # 请求归还页后清空缓存前缀长度
        seq.num_cached_tokens = 0

    # 检查是否有空间容纳末尾 Token
    def can_append(self, seq):
        # 尾页仍有空间，或还能再分配一个空闲页
        return seq.num_blocks <= len(seq.block_table) or bool(self.free_block_ids)

    # 为增长后的序列补充分页
    def append(self, seq):
        # 为尚未命中的逻辑页逐一补充分配
        while len(seq.block_table) < seq.num_blocks:
            # 覆盖最早空闲的物理页
            block = self._allocate_block(self.free_block_ids[0])
            # 向逻辑到物理页表追加本次内容
            seq.block_table.append(block.block_id)

    # 清除无人引用的缓存索引
    def clear_cache(self):
        # 有请求持页时禁止清空缓存身份
        assert not self.used_block_ids, "Cannot clear a cache with live requests"
        # 清空哈希到物理页的索引
        self.hash_to_block_id.clear()
        # 逐个处理当前物理页记录
        for block in self.blocks:
            # 设置是否具备完整有效 KV
            block.ready = False
            # 设置前缀哈希
            block.hash = -1
            # 初始化本页登记的 Token
            block.token_ids = []
