from myvllm.layers import *
import torch 
import torch.nn as nn

# Qwen3Attention: 
# qkv projection
# if not qkv_bias: then rms_norm
# apply rotary embedding to q, k
# attention
# output projection
class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        block_size: int = 256,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()

        self.total_num_heads = num_heads
        self.num_heads = num_heads // self.tp_size

        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        # self.num_kv_heads is per-GPU value (divided by tp_size)
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.scale = scale

        self.qkv_projection = QKVColumnParallelLinear(
            input_size=hidden_size,  # Fixed: was head_dim * total_num_heads, should be hidden_size
            head_size=head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        self.qkv_bias = qkv_bias

        # Q and K norms as used in Qwen3
        self.q_norm = LayerNorm(torch.ones(head_dim), eps=rms_norm_epsilon)
        self.k_norm = LayerNorm(torch.ones(head_dim), eps=rms_norm_epsilon)

        self.rotary_emb = RotaryEmbedding(
            base=base,
            rotary_embedding=head_dim,
            max_position=max_position
        )

        self.attention = Attention(
            self.num_heads,
            head_dim,
            scale,
            self.num_kv_heads,
            block_size
        )

        self.o_proj = RowParallelLinear(
            input_size=head_dim * self.total_num_heads,
            output_size=hidden_size,
            bias=False,
        )

    # 执行当前模块前向
    def forward(
        # 接收当前实例
        self, 
        # 接收本次 Token 的特征
        x: torch.Tensor,
        # 接收每个 Token 的绝对位置
        positions: torch.Tensor,
    # 返回本层输出特征
    ) -> torch.Tensor:
        # Input: x shape (B, N, hidden_size) - REPLICATED on all GPUs

        # ===== QKV Projection (Column Parallel - THIS IS WHERE SHARDING HAPPENS) =====
        # Output shape PER GPU: (B, N, head_dim * (num_heads + 2*num_kv_heads))
        # where num_heads = total_num_heads/tp_size
        #       num_kv_heads = total_num_kv_heads/tp_size
        # 本卡只生成归属于自己的 Query 和 KV Heads
        qkv = self.qkv_projection(x)

        # ===== Split QKV =====
        # q_size = head_dim * num_heads           - Per-GPU size!
        # kv_size = head_dim * num_kv_heads       - Per-GPU size!
        # 按本卡宽度拆出 Q、K、V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Handle both batched (3D) and varlen (2D) inputs
        # Varlen: q shape: (total_tokens, q_size) where q_size = num_heads * head_dim
        # Batched: q shape: (B, N, q_size)
        # 检查q.dim()是否满足本分支条件
        if q.dim() == 2:
            # Varlen mode: (total_tokens, q_size) -> (total_tokens, num_heads, head_dim)
            # 设置 Query 特征
            q = q.view(-1, self.num_heads, self.head_dim)
            # 设置 Key 特征
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            # 设置 Value 特征
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 处理上述条件不成立的情况
        else:
            # Batched mode: (B, N, q_size) -> (B, N, num_heads, head_dim)
            # 设置批大小、序列长度
            B, N = q.size(0), q.size(1)
            # 设置 Query 特征
            q = q.view(B, N, self.num_heads, self.head_dim)
            # 设置 Key 特征
            k = k.view(B, N, self.num_kv_heads, self.head_dim)
            # 设置 Value 特征
            v = v.view(B, N, self.num_kv_heads, self.head_dim)

        # Apply Q and K norms - these are used in Qwen3 to stabilize attention
        # Applied to q and k because they participate in attention_weight computation
        # Removes possibility of large numbers that cause softmax instability
        # 检查qkv_bias是否满足本分支条件
        if self.qkv_bias is False:
            # 设置 Query 特征
            q = self.q_norm(q)
            # 设置 Key 特征
            k = self.k_norm(k)

        # 仅对 Q、K 注入绝对位置对应的旋转编码
        q, k = self.rotary_emb(positions, q, k)

        # 写入新 KV 并读取有效历史完成 Attention
        o = self.attention(q, k, v)
        # o shape: (B*N, num_heads, head_dim)     - Per-GPU, different heads per GPU

        # ===== Output Projection (Row Parallel - COMMUNICATION HAPPENS HERE by dist.all_reduce) =====
        # Row Parallel 汇合各卡 Head 对输出的贡献
        o = self.o_proj(o)
        # Input: (B*N, num_heads * head_dim) sharded across GPUs
        # Output: (B*N, hidden_size) REPLICATED on all GPUs (after all_reduce)

        # 返回所有 Head 汇合后的特征
        return o

# Qwen3MLP
# gate_up
# activateion
# gate_down
class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.gate_up = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
        )
        self.activation = SiluAndMul()
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )

    # 执行当前模块前向
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 本地 Gate/Up 与激活后，用 Row Down 汇合结果
        x = self.down_proj(self.activation(self.gate_up(x)))
        # 返回当前层或模型产生的特征
        return x


# Qwen3DecoderLayer
# input_layernorm, also consider residual
# self_attn
# layer_norm post attention
# mlp
class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        block_size: int = 256,
    ):
        super().__init__()
        gamma = torch.ones(hidden_size)
        self.input_layernorm = LayerNorm(gamma, eps=rms_norm_epsilon)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            block_size=block_size,
        )
        self.post_attention_layernorm = LayerNorm(gamma, eps=rms_norm_epsilon)
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias,
        )

    # 执行当前模块前向
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        # 残差分支已建立时执行此分支
        if residual is not None:
            # 设置当前特征、残差分支
            x, residual = self.input_layernorm(x, residual)
        # 处理上述条件不成立的情况
        else:
            # 设置残差分支
            residual = x  # Save BEFORE normalization
            # 设置当前特征
            x = self.input_layernorm(x)
        # Compute positions based on context (respecting sequence boundaries for batched prefill)
        # 导入 get_context 所需接口
        from myvllm.utils import get_context
        # 设置当前前向元数据
        context = get_context()
        # 按 Prefill 或 Decode 选择处理方式
        if context.is_prefill and context.cu_seqlens_q is not None:
            # For batched prefill, create positions that restart at 0 for each sequence
            # 初始化绝对位置或结果索引
            positions = []
            # 读取各请求在展平 Query 中的边界
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            # prefix_positions_v1
            # 读取包含缓存前缀的历史边界
            cu_seqlens_k = context.cu_seqlens_k.cpu().tolist()
            # 逐个处理i
            for i in range(len(cu_seqlens) - 1):
                # 算出本请求本轮实际 Query 数
                seq_len = cu_seqlens[i+1] - cu_seqlens[i]
                # 从总历史长度减去本轮长度，得到缓存前缀偏移
                cached_offset = cu_seqlens_k[i+1] - cu_seqlens_k[i] - seq_len
                # 向绝对位置或结果索引追加本次内容
                positions.extend(range(cached_offset, cached_offset + seq_len))
            # 设置绝对位置或结果索引
            positions = torch.tensor(positions, dtype=torch.long, device=x.device)
        # 按 Prefill 或 Decode 选择处理方式
        elif context.is_prefill:
            # For single sequence prefill, use sequential positions
            # 设置绝对位置或结果索引
            positions = torch.arange(x.size(0), device=x.device)
        # 处理上述条件不成立的情况
        else:
            # For decode, use context_lens - 1 as positions (current position for each sequence)
            # 用有效长度减一得到每个 Query 的绝对位置
            positions = context.context_lens - 1

        # 设置当前特征
        x = self.self_attn(x, positions=positions)
        # Residual connection always on for attention output
        # 设置当前特征、残差分支
        x, residual = self.post_attention_layernorm(x, residual)
        # 设置当前特征
        x = self.mlp(x)
        # 残差独立携带，采集完整层特征时需要相加
        return x, residual

# Qwen3Model
# embedding
# layers stack
# final layer norm
class Qwen3Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        block_size: int = 256,
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim = hidden_size
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                rms_norm_epsilon=rms_norm_epsilon,
                qkv_bias=qkv_bias,
                base=base,
                max_position=max_position,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
                block_size=block_size,
            ) for _ in range(num_layers)
        ])
        gamma = torch.ones(hidden_size)
        self.norm = LayerNorm(gamma, eps=rms_norm_epsilon)

    # 执行当前模块前向
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 设置当前特征
        x = self.embed_tokens(input_ids)
        # 先将残差分支设为空
        residual = None
        # 逐个处理layer
        for layer in self.layers:
            # 设置当前特征、残差分支
            x, residual = layer(x, residual)
        # 设置当前特征、_
        x, _ = self.norm(x, residual)
        # 返回当前层或模型产生的特征
        return x



# Qwen3ForCausalLM
# add lm_head on top of Qwen3Model
class Qwen3ForCausalLM(nn.Module):
    packed_module_mapping = {
        "q_proj": ('q_proj', 'q'),
        "k_proj": ('k_proj', 'k'),
        "v_proj": ('v_proj', 'v'),
        "gate_up": ('gate_up_proj', '0'),
        "gate_down": ('gate_down_proj', '1'),
    }
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        tie_word_embeddings: bool = False,
        block_size: int = 256,
    ):
        super().__init__()
        head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.model = Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
            block_size=block_size,
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    # 执行当前模块前向
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 设置当前特征
        x = self.model(input_ids)
        # 返回当前层或模型产生的特征
        return x 

    # 把 Hidden State 投影为词表分数
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 设置词表分数
        logits = self.lm_head(hidden_states)
        # 返回完整词表的预测分数
        return logits

if __name__ == "__main__":
    model = Qwen3ForCausalLM(
        vocab_size=50257,
        hidden_size=768,
        num_heads=12,
        head_dim=64,
        intermediate_size=3072,
        num_layers=2,
    )
    input_ids = torch.randint(0, 50257, (2, 16)).cuda()
    output = model(input_ids)