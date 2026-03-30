import torch
import torch.nn as nn

from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        use_rmsnorm: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta=rope_theta, max_seq_len=context_length, use_rmsnorm=use_rmsnorm, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) integer token indices
        Returns:
            (batch, seq_len, vocab_size) logits
        """
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
