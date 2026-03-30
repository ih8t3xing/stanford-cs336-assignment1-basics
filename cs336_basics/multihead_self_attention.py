import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None,
                 use_flash: bool = False, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.use_flash = use_flash

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (..., seq_len, d_model)
            token_positions: (..., seq_len) int, defaults to [0, 1, ..., seq_len-1]
        Returns:
            (..., seq_len, d_model)
        """
        *batch, seq_len, _ = x.shape
        h, d_k = self.num_heads, self.d_k

        # Project and split into heads: (..., seq_len, d_model) -> (..., h, seq_len, d_k)
        def split_heads(t: Tensor) -> Tensor:
            return t.view(*batch, seq_len, h, d_k).transpose(-2, -3)

        Q = split_heads(self.q_proj(x))  # (..., h, seq_len, d_k)
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        if self.use_flash:
            # Uses FlashAttention kernel when available (CUDA), falls back to efficient impl
            attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        else:
            # Causal mask: lower-triangular, True = keep
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Merge heads: (..., seq_len, d_model)
        attn_out = attn_out.transpose(-2, -3).contiguous().view(*batch, seq_len, self.d_model)

        return self.o_proj(attn_out)
