import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Scaled dot-product attention.

    Args:
        Q: (..., seq_len, d_k)
        K: (..., seq_len, d_k)
        V: (..., seq_len, d_v)
        mask: (seq_len, seq_len) boolean, True = keep, False = mask out
    Returns:
        (..., seq_len, d_v)
    """
    d_k = Q.shape[-1]
    # (..., queries, keys)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)

    return attn @ V
