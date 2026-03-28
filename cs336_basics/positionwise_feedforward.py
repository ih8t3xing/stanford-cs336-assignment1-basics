from torch import nn
import torch


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, dtype=None, device=None):
        super().__init__()
        if d_ff is None:
            # d_ff ≈ (8/3) * d_model, rounded up to nearest multiple of 64
            d_ff = int((8 / 3) * d_model)
            d_ff = (d_ff // 64) * 64

        factory_kwargs = {"dtype": dtype, "device": device}
        self.w1 = nn.Linear(d_model, d_ff, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, **factory_kwargs)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(W1 x) * (W3 x), then project back
        # SiLU(z) = z * sigmoid(z)
        gate = self.w1(x)
        silu = gate * torch.sigmoid(gate)
        return self.w2(silu * self.w3(x))
