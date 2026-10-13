import torch.nn as nn
import torch 

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    original_dtype = x.dtype
    x_fp32 = x.float()
    cos = cos.to(device=x.device, dtype=torch.float32)
    sin = sin.to(device=x.device, dtype=torch.float32)

    if x.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif x.dim() == 4:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:
        raise ValueError(f"Unsupported RoPE input shape: {x.shape}")

    x1, x2 = x_fp32.chunk(2, dim=-1)
    output = torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1,
    )
    return output.to(dtype=original_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self, 
        base:int,
        rotary_embedding: int, 
        max_position: int = 2048,
        is_llama3: bool = False,
        # the following params are only used in llama3.2
        llama3_rope_factor: float = 32.0,
        llama3_rope_high_freq_factor: float = 4.0,
        llama3_rope_low_freq_factor: float = 1.0,
        llama3_rope_original_max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.base = base
        # how many dimensions to apply rotary embedding
        self.rotary_embedding = rotary_embedding
        # max position that the long context can reach
        self.max_position = max_position
        self.inv_freq = 1/(base ** (torch.arange(0, self.rotary_embedding, 2)/self.rotary_embedding))

        if is_llama3:
            # specifically for llama3.2
            import math
            inv_freq = self.inv_freq
            # no smooth if low_freq_factor == high_freq_factor
            wave_len = 2 * math.pi / inv_freq
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            else:
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / llama3_rope_factor + smooth
                inv_freq = factor * inv_freq
            self.inv_freq = inv_freq

        positions = torch.arange(self.max_position).float()
        # (max_position, rotary_embedding/2)
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # (max_position, rotary_embedding)
        cos_sin_cache = torch.cat([cos, sin], dim=-1)
        self.register_buffer("cos_sin_cache", cos_sin_cache)

    @torch.compile
    # tell the position index of the token
    # apply rotary embedding to query and key
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]  # (seq_len, rotary_embedding)
        cos, sin = cos_sin.chunk(2, dim=-1)
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin)
        )


if __name__ == "__main__":
    base = 5
    # how many dimensions to apply rotary embedding
    rotary_dim = 16
    # maximum position that the long context can reach
    max_position = 100
    print(torch.arange(0, rotary_dim, 2))
    print(base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    print(inv_freq)

    t = torch.arange(max_position).float()

    freqs = torch.einsum("i,j -> ij", t, inv_freq)

    print(freqs.size())

    print(freqs[2])

