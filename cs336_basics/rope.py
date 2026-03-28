import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # θ_i = 1 / (theta^(2i / d_k)) for i = 0, ..., d_k//2 - 1
        i = torch.arange(0, d_k // 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (theta ** (2 * i / d_k))  # (d_k//2,)

        # Precompute cos/sin for all positions: (max_seq_len, d_k//2)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, d_k//2)

        self.register_buffer("cos", angles.cos())  # (max_seq_len, d_k//2)
        self.register_buffer("sin", angles.sin())  # (max_seq_len, d_k//2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        cos = self.cos[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k//2)

        # Split x into two halves along last dim
        x1 = x[..., 0::2]  # (..., seq_len, d_k//2) — even indices
        x2 = x[..., 1::2]  # (..., seq_len, d_k//2) — odd indices

        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out
