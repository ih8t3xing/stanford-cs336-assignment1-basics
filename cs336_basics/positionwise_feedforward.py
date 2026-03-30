from torch import nn
import torch


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, use_swiglu: bool = True, dtype=None, device=None):
        super().__init__()
        self.use_swiglu = use_swiglu
        if d_ff is None:
            if use_swiglu:
                # d_ff ≈ (8/3) * d_model, rounded down to nearest multiple of 64
                d_ff = int((8 / 3) * d_model)
                d_ff = (d_ff // 64) * 64
            else:
                d_ff = 4 * d_model

        factory_kwargs = {"dtype": dtype, "device": device}
        self.w1 = nn.Linear(d_model, d_ff, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, **factory_kwargs)
        if use_swiglu:
            self.w3 = nn.Linear(d_model, d_ff, bias=False, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        silu = gate * torch.sigmoid(gate)
        if self.use_swiglu:
            return self.w2(silu * self.w3(x))
        else:
            return self.w2(silu)
