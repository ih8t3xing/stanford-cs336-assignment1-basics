import torch
import torch.nn as nn

from cs336_basics.multihead_self_attention import CausalMultiHeadSelfAttention
from cs336_basics.positionwise_feedforward import PositionwiseFeedForward
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RoPE


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        use_rmsnorm: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)
        if use_rmsnorm:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ffn = PositionwiseFeedForward(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-norm transformer block:
            y = x + Attention(RMSNorm(x))
            z = y + FFN(RMSNorm(y))
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
