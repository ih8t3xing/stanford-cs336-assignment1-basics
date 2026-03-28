from typing import Iterable
import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads) + 1e-6)
    clip_coef = max_l2_norm / total_norm
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)
